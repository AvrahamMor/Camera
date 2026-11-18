import cv2
import time
import os
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

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
        self.tracker = OccupancyTracker(debounce_sec=self.settings.debounce_seconds)
        self.model: Optional[YOLO] = None

        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
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
        model_name = self.settings.model_name or "yolov8n.pt"
        print(f"[CVEngine] Loading YOLO model: {model_name}...")
        try:
            self.model = YOLO(model_name)
            print("[CVEngine] YOLO model loaded successfully.")
        except Exception as e:
            print(f"[CVEngine] Error loading model {model_name}: {e}. Retrying default yolov8n.pt...")
            self.model = YOLO("yolov8n.pt")

    def update_settings(self, new_settings: SystemSettings):
        with self._lock:
            old_url = self.settings.rtsp_url
            old_source = self.settings.video_source
            old_model = self.settings.model_name
            self.settings = new_settings
            self.tracker.set_debounce(new_settings.debounce_seconds)

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
        """Dedicated high-speed frame grabber to prevent RTSP socket buffer buildup."""
        while self.running:
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

            while self.running and not self.is_switching:
                ret, frame = cap.read()
                if not ret or frame is None:
                    if self.settings.video_source in ("file", "sample"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("[CVEngine Grabber] Frame read dropped, reconnecting...")
                        break

                with self._frame_lock:
                    self.latest_raw_frame = frame

                # Small sleep to match source framerate and prevent CPU maxing
                time.sleep(0.015)

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

            # 1. Run YOLO inference on 640px image for ultra-fast CPU inference (<30ms)
            person_boxes = self._run_detection_fast(raw_frame, sample_start_time)
            self.latest_person_boxes = person_boxes

            # 2. Update occupancy state machine (filtered to active camera channel)
            statuses, events = self.tracker.update(person_boxes, w, h, active_channel=self.settings.ossia_channel)

            with self._lock:
                self.latest_table_statuses = statuses
                if events:
                    self.latest_events.extend(events)
                    if len(self.latest_events) > 50:
                        self.latest_events = self.latest_events[-50:]

            # 3. Annotate frame
            annotated = self._annotate_frame(raw_frame, statuses, person_boxes)

            # 4. Downscale for web MJPEG streaming (1280px max width for silky smooth 20+ FPS)
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

            now = time.time()
            elapsed_inf = max(0.001, now - loop_start)
            self.actual_inference_fps = round(1.0 / elapsed_inf, 1)

            stream_frame_count += 1
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

    def _run_detection_fast(self, frame: np.ndarray, sample_start_time: float) -> List[Tuple[float, float, float, float]]:
        person_boxes: List[Tuple[float, float, float, float]] = []

        if self.model is not None:
            try:
                # Run YOLOv8 on imgsz=640 (fastest CPU mode, ~25-35ms)
                results = self.model.predict(
                    source=frame,
                    imgsz=640,
                    classes=[0], # Class 0 = person
                    conf=self.settings.confidence_threshold,
                    verbose=False,
                    device="cpu"
                )
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0].cpu().numpy()
                        person_boxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
            except Exception as e:
                print(f"[CVEngine Detection Error]: {e}")

        # Synthetic fallback simulation if in sample mode
        if self.settings.video_source == "sample" and len(person_boxes) == 0:
            t_sec = (time.time() - sample_start_time) % 30.0
            if 5.0 <= t_sec < 25.0:
                person_boxes.append((575, 330, 615, 390))
                person_boxes.append((665, 330, 705, 390))
            if t_sec >= 10.0:
                person_boxes.append((170, 390, 210, 450))
                person_boxes.append((270, 390, 310, 450))
                person_boxes.append((220, 480, 260, 540))

        return person_boxes

    def _annotate_frame(
        self,
        frame: np.ndarray,
        statuses: List[Dict[str, Any]],
        person_boxes: List[Tuple[float, float, float, float]]
    ) -> np.ndarray:
        overlay = frame.copy()
        h, w = frame.shape[:2]
        now = time.time()

        # 1. Draw Table Polygons with status colors (only for tables belonging to active camera)
        for s in statuses:
            if s.get("camera_channel", 1) != self.settings.ossia_channel:
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
            if s.get("camera_channel", 1) != self.settings.ossia_channel:
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

        # 3. Draw person feet/anchor indicators
        for (x1, y1, x2, y2) in person_boxes:
            cx = int((x1 + x2) / 2.0)
            bottom_y = int(y2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (80, 140, 180), 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, bottom_y), 4, (0, 230, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, bottom_y), 8, (0, 230, 255), 1, cv2.LINE_AA)

        # 4. Stream HUD Top Bar
        hud_bg = (14, 18, 24)
        cv2.rectangle(frame, (0, 0), (w, 36), hud_bg, -1)
        cv2.line(frame, (0, 36), (w, 36), (40, 50, 65), 1)

        dot_color = (50, 220, 80) if self.is_connected else (40, 50, 230)
        cv2.circle(frame, (18, 18), 5, dot_color, -1, cv2.LINE_AA)

        hud_title = f"VISIONDINE AI • Cam {self.settings.ossia_channel} • {self.connection_status}"
        cv2.putText(frame, hud_title, (32, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 230, 245), 1, cv2.LINE_AA)

        diners_count = sum(s['headcount'] for s in statuses)
        hud_stats = f"AI: {self.actual_inference_fps:.1f} FPS | Stream: {self.actual_stream_fps:.1f} FPS | Diners: {diners_count}"
        cv2.putText(frame, hud_stats, (w - 380, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 200, 255), 1, cv2.LINE_AA)

        return frame

    def get_snapshot(self, clean: bool = True) -> Optional[bytes]:
        with self._lock:
            frame = self.latest_clean_frame if clean else self.latest_annotated_frame
        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for Camera Stream...", (420, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buffer.tobytes() if ret else None

    def generate_mjpeg_stream(self):
        """Yield MJPEG frames continuously at configured stream_fps without disconnecting."""
        target_delay = 1.0 / max(10.0, self.settings.stream_fps)
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
            sleep_t = max(0.015, target_delay - elapsed)
            time.sleep(sleep_t)

    def get_live_state(self) -> Dict[str, Any]:
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
