import cv2
import time
import os
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch

# Cap CPU thread pools so PyTorch/OpenCV do not saturate low-power CPUs
torch.set_num_threads(2)
cv2.setNumThreads(2)

# Set OpenCV FFMPEG options for RTSP over TCP with short probe timeout
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;3000000"

from ultralytics import YOLO
from backend.config import load_settings, save_settings, SystemSettings, BASE_DIR
from backend.tables_manager import tables_manager
from backend.occupancy_tracker import OccupancyTracker
from backend.database import get_kpis, get_recent_events
from backend.video_generator import generate_sample_restaurant_video, SAMPLE_VIDEO_PATH
from backend.firebase_sync import firebase_manager

class CVEngine:
    def __init__(self):
        self.settings: SystemSettings = load_settings()
        self.tracker = OccupancyTracker(
            debounce_sec=self.settings.debounce_seconds,
            menu_alert_sec=self.settings.menu_alert_seconds,
        )
        self.model: Optional[YOLO] = None

        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._viewer_lock = threading.Lock()
        self.active_stream_viewers = 0
        self.running = False

        self.capture_thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None

        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_clean_frame: Optional[np.ndarray] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_mjpeg_bytes: Optional[bytes] = None

        self.latest_table_statuses: List[Dict[str, Any]] = []
        self.latest_events: List[Dict[str, Any]] = []
        self.latest_person_boxes: List[Tuple[float, float, float, float]] = []
        self.latest_person_confs: List[float] = []
        self.latest_detections: List[Dict[str, Any]] = []

        self.is_connected = False
        self.is_switching = False
        self.connection_status = "Initializing"
        self.actual_inference_fps = 0.0
        self.actual_stream_fps = 0.0
        self.last_inference_time = 0.0
        self.last_firebase_sync = 0.0

        # Load YOLO model
        self._load_yolo()

    def _load_yolo(self):
        model_name = self.settings.model_name or "yolov8n-pose.pt"
        print(f"[CVEngine] Loading YOLO model: {model_name}...")
        try:
            new_model = YOLO(model_name)
            with self._model_lock:
                self.model = new_model
            print("[CVEngine] YOLO model loaded successfully.")
        except Exception as e:
            print(f"[CVEngine] Error loading model {model_name}: {e}. Retrying default yolov8n.pt...")
            with self._model_lock:
                self.model = YOLO("yolov8n.pt")

    def update_settings(self, new_settings: SystemSettings):
        with self._lock:
            old_url = self.settings.rtsp_url
            old_source = self.settings.video_source
            old_model = self.settings.model_name
            self.settings = new_settings
            self.tracker.set_debounce(new_settings.debounce_seconds)
            self.tracker.set_menu_alert_seconds(new_settings.menu_alert_seconds)

        if old_model != new_settings.model_name:
            self._load_yolo()

        if old_source != new_settings.video_source or old_url != new_settings.rtsp_url:
            self.switch_to_source(new_settings.rtsp_url)

    def switch_channel(self, channel: int):
        """Seamlessly switch RTSP camera channel without disconnecting clients."""
        with self._lock:
            self.is_switching = True
            self.settings.ossia_channel = channel
            self.settings.video_source = "rtsp"
            stype = "main" if self.settings.ossia_stream == 0 else "sub"
            new_url = f"rtsp://{self.settings.ossia_user}:{self.settings.ossia_pass}@{self.settings.ossia_ip}:{self.settings.ossia_port}/chID={channel}&streamType={stype}&linkType=tcp"
            self.settings.rtsp_url = new_url
            save_settings(self.settings)

        print(f"[CVEngine] Switching to Channel {channel}: {new_url}")
        self.switch_to_source(new_url)

    def switch_to_source(self, new_source: str):
        self.is_switching = True
        # Clear detected people to avoid ghost bounding boxes on new angle
        with self._lock:
            self.latest_person_boxes = []

    def start(self):
        if self.running:
            return
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()

        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()
        print("[CVEngine] Capture & Inference workers started.")

    def stop(self):
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)
        self.capture_thread = None
        self.inference_thread = None
        print("[CVEngine] Workers stopped.")

    def _get_capture_source(self) -> Tuple[Any, str]:
        source_type = self.settings.video_source.lower()
        if source_type == "rtsp":
            url = self.settings.rtsp_url
            return url, f"RTSP (Ch {self.settings.ossia_channel})"
        elif source_type == "file":
            path = Path(self.settings.video_file_path)
            if not path.is_absolute():
                path = BASE_DIR / path
            if not path.exists():
                generate_sample_restaurant_video()
                path = SAMPLE_VIDEO_PATH
            return str(path), f"Video File: {path.name}"
        else: # "sample"
            if not SAMPLE_VIDEO_PATH.exists():
                generate_sample_restaurant_video()
            return str(SAMPLE_VIDEO_PATH), "Sample Restaurant Demo"

    def _capture_worker(self):
        """Dedicated high-speed frame grabber; yields Main Stream when Multi-Cam AI handles idle monitoring."""
        while self.running:
            with self._viewer_lock:
                has_viewers = self.active_stream_viewers > 0

            if self.settings.multi_cam_ai and self.settings.video_source == "rtsp" and not has_viewers:
                self.is_connected = True
                self.is_switching = False
                self.connection_status = "Multi-Cam AI Active"
                time.sleep(0.5)
                continue

            source, label = self._get_capture_source()
            print(f"[CVEngine Grabber] Connecting to {label}...")
            self.connection_status = f"Connecting to {label}"
            self.is_connected = False

            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if isinstance(source, str) and source.startswith("rtsp://"):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print(f"[CVEngine Grabber] Failed to open {label}. Retrying in 2s...")
                self.connection_status = "Connection Failed"
                time.sleep(2.0)
                continue

            self.is_connected = True
            self.is_switching = False
            self.connection_status = "Online / Streaming"
            print(f"[CVEngine Grabber] Connected to {label}")

            last_decode_time = 0.0
            while self.running and not self.is_switching:
                now = time.time()
                with self._viewer_lock:
                    has_viewers = self.active_stream_viewers > 0

                if self.settings.multi_cam_ai and self.settings.video_source == "rtsp" and not has_viewers:
                    break

                # Throttle YUV->BGR decoding: ~12 FPS when watching live, ~4 FPS when idle
                decode_interval = 0.08 if has_viewers else 0.25
                if (now - last_decode_time) < decode_interval:
                    if not cap.grab():
                        if self.settings.video_source in ("file", "sample"):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        print("[CVEngine Grabber] Frame grab dropped, reconnecting...")
                        break
                    time.sleep(0.03)
                    continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    if self.settings.video_source in ("file", "sample"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("[CVEngine Grabber] Frame read dropped, reconnecting...")
                        break

                last_decode_time = now
                with self._frame_lock:
                    self.latest_raw_frame = frame

                time.sleep(0.02)

            cap.release()
            time.sleep(0.5)

    def _inference_worker(self):
        """Runs YOLO detection on 640px scaled frame, updates tracker, and renders overlay."""
        sample_start_time = time.time()
        fps_timer = time.time()
        stream_frame_count = 0

        while self.running:
            target_delay = 1.0 / max(1.0, self.settings.target_fps)
            loop_start = time.time()
            try:
                with self._viewer_lock:
                    has_viewers = self.active_stream_viewers > 0

                # When Multi-Cam AI monitors all calibrated cameras and no Single Focus stream is open, yield AI & sync state
                if self.settings.multi_cam_ai and self.settings.video_source == "rtsp" and not has_viewers:
                    statuses = self.tracker.get_all_statuses()
                    with self._lock:
                        self.latest_table_statuses = statuses
                    now = time.time()
                    if (now - self.last_firebase_sync) >= 5.0:
                        try:
                            firebase_manager.sync_live_state(self.get_live_state())
                            self.last_firebase_sync = now
                        except Exception:
                            pass
                    time.sleep(0.5)
                    continue

                raw_frame = None
                with self._frame_lock:
                    if self.latest_raw_frame is not None:
                        raw_frame = self.latest_raw_frame.copy()

                if raw_frame is None or self.is_switching:
                    # Generate switching / waiting placeholder frame
                    placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
                    msg = f"Connecting to Camera {self.settings.ossia_channel}..." if self.is_switching else "Connecting to Camera Stream..."
                    cv2.putText(placeholder, msg, (380, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 220, 255), 2, cv2.LINE_AA)
                    with self._lock:
                        self.latest_annotated_frame = placeholder
                        ret, jpeg = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ret:
                            self.latest_mjpeg_bytes = jpeg.tobytes()
                    time.sleep(0.1)
                    continue

                h, w = raw_frame.shape[:2]

                # 1. Run YOLO / Pose inference on 640px image for pinpoint accuracy
                detections = self._run_detection_fast(raw_frame, sample_start_time)
                person_boxes = [d["box"] for d in detections]
                self.latest_detections = detections
                self.latest_person_boxes = person_boxes
                self.latest_person_confs = [d["conf"] for d in detections]

                # 2. Update occupancy state machine with dual-threshold hysteresis + tabletop item check
                statuses, events = self.tracker.update_detections(
                    detections,
                    w,
                    h,
                    active_channel=self.settings.ossia_channel,
                    strict_conf=self.settings.confidence_threshold,
                    frame=raw_frame,
                )

                with self._lock:
                    self.latest_table_statuses = statuses
                    if events:
                        self.latest_events.extend(events)
                        if len(self.latest_events) > 50:
                            self.latest_events = self.latest_events[-50:]

                with self._viewer_lock:
                    has_viewers = self.active_stream_viewers > 0

                # 3 & 4. Only annotate, resize, and JPEG-encode when a browser is watching live stream
                if has_viewers:
                    annotated = self._annotate_frame(raw_frame, statuses, person_boxes)
                    if w > 1280:
                        scale = 1280.0 / w
                        stream_preview = cv2.resize(annotated, (1280, int(h * scale)), interpolation=cv2.INTER_LINEAR)
                    else:
                        stream_preview = annotated

                    ret, jpeg = cv2.imencode(".jpg", stream_preview, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    with self._lock:
                        self.latest_clean_frame = raw_frame
                        self.latest_annotated_frame = annotated
                        if ret:
                            self.latest_mjpeg_bytes = jpeg.tobytes()
                    stream_frame_count += 1
                else:
                    with self._lock:
                        self.latest_clean_frame = raw_frame

                now = time.time()
                elapsed_inf = max(0.001, now - loop_start)
                self.actual_inference_fps = round(1.0 / elapsed_inf, 1)

                if (now - fps_timer) >= 1.0:
                    self.actual_stream_fps = round(stream_frame_count / (now - fps_timer), 1)
                    stream_frame_count = 0
                    fps_timer = now

                # 5. Firebase periodic sync (every 5 seconds)
                if (now - self.last_firebase_sync) >= 5.0:
                    try:
                        state = self.get_live_state()
                        firebase_manager.sync_live_state(state)
                        self.last_firebase_sync = now
                    except Exception:
                        pass

                # Throttle to configured target inference FPS
                sleep_time = max(0.01, target_delay - elapsed_inf)
                time.sleep(sleep_time)
            except Exception as e:
                print(f"[CVEngine Inference Worker Error]: {e}")
                time.sleep(0.5)

    def _extract_pinpoint_anchor(
        self,
        box: Tuple[float, float, float, float],
        kpts_xy: Optional[Any],
        kpts_conf: Optional[Any],
        frame_h: int,
    ) -> Tuple[Tuple[float, float], Dict[str, Tuple[float, float]]]:
        """Compute a precise seat/body anchor using pose keypoints (hips -> shoulders -> face) when available."""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        vis_pts: Dict[str, Tuple[float, float]] = {}

        if kpts_xy is not None and kpts_conf is not None and len(kpts_xy) >= 13:
            def _pt(idx: int, min_c: float = 0.25):
                c = float(kpts_conf[idx])
                px, py = float(kpts_xy[idx][0]), float(kpts_xy[idx][1])
                if c >= min_c and (px > 1.0 or py > 1.0):
                    return (px, py)
                return None

            nose = _pt(0, 0.25)
            l_sh, r_sh = _pt(5, 0.25), _pt(6, 0.25)
            l_wrist, r_wrist = _pt(9, 0.22), _pt(10, 0.22)
            l_hip, r_hip = _pt(11, 0.25), _pt(12, 0.25)

            if nose:
                vis_pts["nose"] = nose
            if l_sh:
                vis_pts["l_sh"] = l_sh
            if r_sh:
                vis_pts["r_sh"] = r_sh
            if l_wrist:
                vis_pts["l_wrist"] = l_wrist
            if r_wrist:
                vis_pts["r_wrist"] = r_wrist
            if l_hip:
                vis_pts["l_hip"] = l_hip
            if r_hip:
                vis_pts["r_hip"] = r_hip

            # 1. Prefer visible hip midpoint (exact seated center)
            hips = [p for p in (l_hip, r_hip) if p is not None]
            if hips:
                hx = sum(p[0] for p in hips) / len(hips)
                hy = sum(p[1] for p in hips) / len(hips)
                return (hx, hy), vis_pts

            # 2. Next prefer shoulder midpoint shifted slightly down to chest/seat
            shs = [p for p in (l_sh, r_sh) if p is not None]
            if shs:
                sx = sum(p[0] for p in shs) / len(shs)
                sy = sum(p[1] for p in shs) / len(shs)
                box_h = max(10.0, y2 - y1)
                return (sx, min(y2, sy + box_h * 0.42)), vis_pts

            # 3. Fallback to face/nose shifted slightly down
            if nose:
                return (nose[0], min(y2, nose[1] + frame_h * 0.045)), vis_pts

        # Standard box fallback (82% down box)
        return (cx, y1 + (y2 - y1) * 0.82), vis_pts

    def _run_detection_fast(self, frame: np.ndarray, sample_start_time: float) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        h_f, w_f = frame.shape[:2]
        min_w = w_f * 0.02
        min_h = h_f * 0.04
        # Run internally at 0.20 so seated hysteresis keeps diners locked even when leaning/shifting
        internal_conf = min(0.20, float(self.settings.confidence_threshold))

        if self.model is not None:
            try:
                with self._model_lock:
                    results = self.model.predict(
                        source=frame,
                        imgsz=640,
                        classes=[0],  # Class 0 = person
                        conf=internal_conf,
                        verbose=False,
                        device="cpu"
                    )
                for r in results:
                    boxes = r.boxes
                    has_kpts = getattr(r, "keypoints", None) is not None and r.keypoints is not None
                    for idx, box in enumerate(boxes):
                        b = box.xyxy[0].cpu().numpy()
                        conf_val = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0
                        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                        if (x2 - x1) < min_w or (y2 - y1) < min_h:
                            continue

                        k_xy = r.keypoints.xy[idx].cpu().numpy() if (has_kpts and idx < len(r.keypoints.xy)) else None
                        k_cf = r.keypoints.conf[idx].cpu().numpy() if (has_kpts and r.keypoints.conf is not None and idx < len(r.keypoints.conf)) else None
                        anchor, vis_pts = self._extract_pinpoint_anchor((x1, y1, x2, y2), k_xy, k_cf, h_f)

                        detections.append({
                            "box": (x1, y1, x2, y2),
                            "conf": conf_val,
                            "anchor": anchor,
                            "keypoints": vis_pts,
                        })
            except Exception as e:
                print(f"[CVEngine Detection Error]: {e}")

        # Synthetic fallback simulation if in sample mode
        if self.settings.video_source == "sample" and len(detections) == 0:
            t_sec = (time.time() - sample_start_time) % 30.0
            if 5.0 <= t_sec < 25.0:
                for b, c in [((575, 330, 615, 390), 0.92), ((665, 330, 705, 390), 0.89)]:
                    detections.append({"box": b, "conf": c, "anchor": ((b[0] + b[2]) / 2.0, b[1] + (b[3] - b[1]) * 0.82), "keypoints": {}})
            if t_sec >= 10.0:
                for b, c in [((170, 390, 210, 450), 0.91), ((270, 390, 310, 450), 0.88), ((220, 480, 260, 540), 0.87)]:
                    detections.append({"box": b, "conf": c, "anchor": ((b[0] + b[2]) / 2.0, b[1] + (b[3] - b[1]) * 0.82), "keypoints": {}})

        return detections

    def append_events(self, events: List[Dict[str, Any]]):
        if not events:
            return
        with self._lock:
            self.latest_events.extend(events)
            if len(self.latest_events) > 50:
                self.latest_events = self.latest_events[-50:]

    def render_annotated_jpeg(self, frame: np.ndarray, channel: int, detections: List[Dict[str, Any]]) -> Optional[bytes]:
        """Render annotated JPEG snapshot for any camera channel (used by Telegram alerts & /photo)."""
        if frame is None:
            return None
        h, w = frame.shape[:2]
        statuses = self.tracker.get_all_statuses(w, h)
        boxes = [d["box"] for d in detections]
        annotated = self._annotate_frame(frame.copy(), statuses, boxes, channel=channel, detections=detections)
        ret, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes() if ret else None

    def _annotate_frame(
        self,
        frame: np.ndarray,
        statuses: List[Dict[str, Any]],
        person_boxes: List[Tuple[float, float, float, float]],
        channel: Optional[int] = None,
        detections: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        overlay = frame.copy()
        h, w = frame.shape[:2]
        now = time.time()
        target_ch = channel if channel is not None else self.settings.ossia_channel

        # 1. Draw Table Polygons with status colors (only for tables belonging to target_ch)
        for s in statuses:
            if s.get("camera_channel", 1) != target_ch:
                continue
            pts = np.array(s["polygon_px"], np.int32).reshape((-1, 1, 2))
            status = s["status"]

            if status == "FREE":
                fill_color = (40, 180, 60)      # Emerald Green
                border_color = (50, 220, 80)
            elif status == "OCCUPIED":
                fill_color = (40, 50, 230)      # Crimson Red
                border_color = (60, 70, 255)
            else: # NEEDS_SERVICE
                pulse = (np.sin(now * 6.0) + 1.0) / 2.0
                fill_color = (20, 160, 240)     # Amber/Yellow
                border_color = (30, 200, 255)

            cv2.fillPoly(overlay, [pts], fill_color)
            cv2.polylines(frame, [pts], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

        # Blend polygon translucent overlay
        cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

        # 2. Draw Table Information Badges
        for s in statuses:
            if s.get("camera_channel", 1) != target_ch:
                continue
            pts = np.array(s["polygon_px"], np.int32)
            if len(pts) == 0:
                continue
            center_x = int(np.mean(pts[:, 0]))
            center_y = int(np.mean(pts[:, 1]))

            status = s["status"]
            tid = s["id"]
            headcount = s["headcount"]
            capacity = s["capacity"]
            timer = s["elapsed_formatted"]

            if status == "FREE":
                badge_border = (46, 204, 113)
                badge_text = f"{tid.upper()} | FREE"
                sub_text = f"Capacity: {capacity}"
            elif status == "OCCUPIED":
                badge_border = (231, 76, 60)
                badge_text = f"{tid.upper()} | {timer}"
                sub_text = f"Guests: {headcount}/{capacity}"
            else: # NEEDS_SERVICE
                badge_border = (243, 156, 18)
                badge_text = f"{tid.upper()} | ALERT!"
                sub_text = f"Wait: {timer} ({headcount}p)"

            bx, by = center_x - 70, center_y - 22
            bw, bh = 140, 44
            bx = max(10, min(w - bw - 10, bx))
            by = max(10, min(h - bh - 10, by))

            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (15, 18, 24), -1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), badge_border, 1, cv2.LINE_AA)
            cv2.putText(frame, badge_text, (bx + 8, by + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (250, 250, 250), 1, cv2.LINE_AA)
            cv2.putText(frame, sub_text, (bx + 8, by + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 200, 220), 1, cv2.LINE_AA)

        # 3. Draw pinpoint pose skeleton / anchor markers (replaces giant distracting boxes when pose is available)
        dets = detections if detections is not None else getattr(self, "latest_detections", [])
        if dets:
            for d in dets:
                x1, y1, x2, y2 = d["box"]
                ax, ay = int(d["anchor"][0]), int(d["anchor"][1])
                conf_val = d["conf"]
                kpts = d.get("keypoints") or {}

                # Draw subtle skeleton & face dot when Pose keypoints exist
                if kpts:
                    if "nose" in kpts:
                        nx, ny = int(kpts["nose"][0]), int(kpts["nose"][1])
                        cv2.circle(frame, (nx, ny), 5, (255, 220, 60), -1, cv2.LINE_AA)
                    if "l_sh" in kpts and "r_sh" in kpts:
                        cv2.line(frame, (int(kpts["l_sh"][0]), int(kpts["l_sh"][1])),
                                 (int(kpts["r_sh"][0]), int(kpts["r_sh"][1])), (0, 230, 255), 2, cv2.LINE_AA)
                    if "l_hip" in kpts and "r_hip" in kpts:
                        cv2.line(frame, (int(kpts["l_hip"][0]), int(kpts["l_hip"][1])),
                                 (int(kpts["r_hip"][0]), int(kpts["r_hip"][1])), (0, 230, 255), 2, cv2.LINE_AA)
                    for wk in ("l_wrist", "r_wrist"):
                        if wk in kpts:
                            cv2.circle(frame, (int(kpts[wk][0]), int(kpts[wk][1])), 4, (80, 255, 140), -1, cv2.LINE_AA)
                else:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), 1, cv2.LINE_AA)

                # Pinpoint seated anchor dot + confidence label
                dot_color = (0, 230, 255) if conf_val >= self.settings.confidence_threshold else (140, 180, 200)
                cv2.circle(frame, (ax, ay), 6, dot_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (ax, ay), 10, dot_color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"{conf_val:.2f}", (ax + 12, ay + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, dot_color, 2, cv2.LINE_AA)
        else:
            confs = getattr(self, "latest_person_confs", [])
            for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
                cx = int((x1 + x2) / 2.0)
                anchor_y = int(y1 + (y2 - y1) * 0.82)
                cv2.circle(frame, (cx, anchor_y), 6, (0, 230, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, anchor_y), 10, (0, 230, 255), 2, cv2.LINE_AA)
                if idx < len(confs):
                    cv2.putText(frame, f"{confs[idx]:.2f}", (cx + 12, anchor_y + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 2, cv2.LINE_AA)

        # 4. Stream HUD Top Bar
        hud_bg = (14, 18, 24)
        cv2.rectangle(frame, (0, 0), (w, 36), hud_bg, -1)
        cv2.line(frame, (0, 36), (w, 36), (40, 50, 65), 1)

        dot_color = (50, 220, 80) if self.is_connected else (40, 50, 230)
        cv2.circle(frame, (18, 18), 5, dot_color, -1, cv2.LINE_AA)

        hud_title = f"VISIONDINE AI • Cam {target_ch} • {self.connection_status}"
        cv2.putText(frame, hud_title, (32, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 230, 245), 1, cv2.LINE_AA)

        diners_count = sum(s['headcount'] for s in statuses)
        hud_stats = f"AI: {self.actual_inference_fps:.1f} FPS | Stream: {self.actual_stream_fps:.1f} FPS | Diners: {diners_count}"
        cv2.putText(frame, hud_stats, (w - 380, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 200, 255), 1, cv2.LINE_AA)

        return frame

    def get_snapshot(self, clean: bool = True) -> Optional[bytes]:
        with self._lock:
            raw = self.latest_clean_frame.copy() if self.latest_clean_frame is not None else None
            annotated = self.latest_annotated_frame if self.latest_annotated_frame is not None else None
            statuses = list(self.latest_table_statuses)
            boxes = list(self.latest_person_boxes)

        if clean:
            frame = raw
        else:
            # Always render fresh annotated frame on demand (even when web viewer is closed)
            frame = self._annotate_frame(raw, statuses, boxes) if raw is not None else annotated

        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for Camera Stream...", (420, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buffer.tobytes() if ret else None

    def generate_mjpeg_stream(self):
        """Yield MJPEG frames continuously at configured stream_fps without disconnecting."""
        target_delay = 1.0 / max(10.0, self.settings.stream_fps)
        with self._viewer_lock:
            self.active_stream_viewers += 1
        try:
            while self.running:
                start_t = time.time()
                jpeg_bytes = None
                with self._lock:
                    jpeg_bytes = self.latest_mjpeg_bytes

                if jpeg_bytes is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                    )

                elapsed = time.time() - start_t
                sleep_t = max(0.03, target_delay - elapsed)
                time.sleep(sleep_t)
        finally:
            with self._viewer_lock:
                self.active_stream_viewers = max(0, self.active_stream_viewers - 1)

    def get_live_state(self) -> Dict[str, Any]:
        if self.settings.multi_cam_ai:
            statuses = self.tracker.get_all_statuses()
            with self._lock:
                events = list(self.latest_events)
        else:
            with self._lock:
                statuses = list(self.latest_table_statuses)
                events = list(self.latest_events)

        total_tables = len(statuses)
        occupied_tables = sum(1 for s in statuses if s["status"] in ("OCCUPIED", "NEEDS_SERVICE"))
        available_tables = total_tables - occupied_tables
        active_diners = sum(s["headcount"] for s in statuses)
        service_alerts = [s for s in statuses if s["status"] == "NEEDS_SERVICE"]

        db_kpis = get_kpis()

        return {
            "timestamp": time.time(),
            "connection": {
                "is_connected": self.is_connected,
                "status": self.connection_status,
                "inference_fps": self.actual_inference_fps,
                "stream_fps": self.actual_stream_fps,
                "video_source": self.settings.video_source,
                "camera_channel": self.settings.ossia_channel
            },
            "kpis": {
                "total_tables": total_tables,
                "occupied_tables": occupied_tables,
                "available_tables": available_tables,
                "active_diners": active_diners,
                "avg_turnover_minutes": db_kpis["avg_turnover_minutes"],
                "completed_sessions_today": db_kpis["completed_sessions_today"],
                "total_diners_today": db_kpis["total_diners_today"],
                "pending_alerts_count": len(service_alerts)
            },
            "tables": statuses,
            "alerts": service_alerts,
            "recent_events": events[-15:]
        }

# Global singleton
cv_engine = CVEngine()
