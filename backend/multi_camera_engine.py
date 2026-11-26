import cv2
import time
import os
import threading
from typing import List, Dict, Tuple, Optional, Callable, Any
import numpy as np

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;2000000"

from backend.config import load_settings, save_settings, SystemSettings
from backend.tables_manager import tables_manager


class CameraGrabber:
    """Lightweight RTSP sub-stream grabber driven by active demand or Multi-Cam AI channels."""

    def __init__(
        self,
        channel: int,
        rtsp_url: str,
        should_connect_fn: Callable[[int], bool],
        is_fast_decode_fn: Callable[[], bool],
    ):
        self.channel = channel
        self.rtsp_url = rtsp_url
        self.should_connect_fn = should_connect_fn
        self.is_fast_decode_fn = is_fast_decode_fn
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.is_connected = False
        self.last_frame_time = 0.0
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        self.thread = None

    def _loop(self):
        while self.running:
            if not self.should_connect_fn(self.channel):
                self.is_connected = False
                time.sleep(0.5)
                continue

            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self.is_connected = False
                time.sleep(3.0)
                continue

            self.is_connected = True
            last_decode = 0.0
            while self.running and self.should_connect_fn(self.channel):
                now = time.time()
                # ~8 FPS when Matrix viewer is open; ~1.2 FPS for lightweight Multi-Cam AI background monitoring
                decode_interval = 0.12 if self.is_fast_decode_fn() else 0.85
                if (now - last_decode) < decode_interval:
                    if not cap.grab():
                        self.is_connected = False
                        break
                    time.sleep(0.05)
                    continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    self.is_connected = False
                    break

                last_decode = now
                with self.lock:
                    self.latest_frame = frame
                    self.last_frame_time = now
                    self.is_connected = True

                time.sleep(0.03)

            cap.release()
            self.is_connected = False
            time.sleep(0.5)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.latest_frame


class MultiCameraEngine:
    """
    Orchestrates the 3x3 restaurant matrix (Channels 1-8 + status tile) AND
    lightweight round-robin Multi-Cam AI monitoring across all calibrated table cameras.
    """

    def __init__(self):
        self.settings: SystemSettings = load_settings()
        self.channels = list(range(1, 9))
        self.grabbers: Dict[int, CameraGrabber] = {}

        self.running = False
        self.matrix_thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None

        self.latest_matrix_frame: Optional[np.ndarray] = None
        self.latest_matrix_jpeg: Optional[bytes] = None

        self.focused_channel = self.settings.ossia_channel or 1
        if self.focused_channel not in self.channels:
            self.focused_channel = 1

        self.lock = threading.Lock()
        self._viewer_lock = threading.Lock()
        self.active_viewers = 0
        self.last_demand_time = 0.0

        self.camera_person_boxes: Dict[int, List[Tuple[float, float, float, float]]] = {
            ch: [] for ch in self.channels
        }
        self.camera_detections: Dict[int, List[Dict[str, Any]]] = {
            ch: [] for ch in self.channels
        }

        # Shared AI pipeline wired from app startup (no duplicate model in RAM)
        self.model = None
        self.model_lock: Optional[threading.Lock] = None
        self.tracker = None
        self.detect_fn: Optional[Callable[[np.ndarray], List[Dict[str, Any]]]] = None
        self.event_sink_fn: Optional[Callable[[List[Dict[str, Any]]], None]] = None
        self.is_single_viewer_active_fn: Optional[Callable[[], bool]] = None

        self._init_grabbers()

    def set_shared_model(self, model, model_lock: threading.Lock):
        self.model = model
        self.model_lock = model_lock

    def set_shared_pipeline(
        self,
        model,
        model_lock: threading.Lock,
        tracker,
        detect_fn: Callable[[np.ndarray], List[Dict[str, Any]]],
        event_sink_fn: Callable[[List[Dict[str, Any]]], None],
        is_single_viewer_active_fn: Optional[Callable[[], bool]] = None,
    ):
        self.model = model
        self.model_lock = model_lock
        self.tracker = tracker
        self.detect_fn = detect_fn
        self.event_sink_fn = event_sink_fn
        self.is_single_viewer_active_fn = is_single_viewer_active_fn

    def update_settings(self, new_settings: SystemSettings):
        creds_changed = (
            new_settings.ossia_ip != self.settings.ossia_ip
            or new_settings.ossia_port != self.settings.ossia_port
            or new_settings.ossia_user != self.settings.ossia_user
            or new_settings.ossia_pass != self.settings.ossia_pass
        )
        self.settings = new_settings
        if creds_changed:
            was_running = self.running
            if was_running:
                self.stop()
            self._init_grabbers()
            if was_running:
                self.start()

    def get_ai_channels(self) -> List[int]:
        """Return only camera channels that actually have calibrated tables (avoids wasting bandwidth)."""
        if not self.settings.multi_cam_ai:
            return []
        chs = sorted({
            int(getattr(t, "camera_channel", 1))
            for t in tables_manager.get_all()
            if 1 <= int(getattr(t, "camera_channel", 1)) <= 8
        })
        return chs

    def touch_demand(self):
        with self._viewer_lock:
            self.last_demand_time = time.time()

    def has_active_viewers(self) -> bool:
        with self._viewer_lock:
            return self.active_viewers > 0 or (time.time() - self.last_demand_time) < 45.0

    def has_matrix_viewers_only(self) -> bool:
        with self._viewer_lock:
            return self.active_viewers > 0

    def _should_grabber_connect(self, channel: int) -> bool:
        if self.has_active_viewers():
            return True
        if self.settings.multi_cam_ai and self.settings.video_source == "rtsp":
            return channel in self.get_ai_channels()
        return False

    def get_snapshot_frame(self, channel: int) -> Optional[np.ndarray]:
        self.touch_demand()
        grabber = self.grabbers.get(channel)
        if not grabber:
            return None
        deadline = time.time() + 2.5
        while time.time() < deadline:
            frame = grabber.get_frame()
            if frame is not None and (time.time() - grabber.last_frame_time) < 10.0:
                return frame
            time.sleep(0.15)
        return grabber.get_frame()

    def _init_grabbers(self):
        user = self.settings.ossia_user or "admin"
        passw = self.settings.ossia_pass or ""
        ip = self.settings.ossia_ip or "192.168.1.100"
        port = self.settings.ossia_port or 554

        for ch in self.channels:
            url = f"rtsp://{user}:{passw}@{ip}:{port}/chID={ch}&streamType=sub&linkType=tcp"
            self.grabbers[ch] = CameraGrabber(
                channel=ch,
                rtsp_url=url,
                should_connect_fn=self._should_grabber_connect,
                is_fast_decode_fn=self.has_matrix_viewers_only,
            )

    def start(self):
        if self.running:
            return
        self.running = True

        for grabber in self.grabbers.values():
            grabber.start()

        self.matrix_thread = threading.Thread(target=self._matrix_composer_loop, daemon=True)
        self.matrix_thread.start()

        self.inference_thread = threading.Thread(target=self._multi_cam_inference_loop, daemon=True)
        self.inference_thread.start()
        print("[MultiCamEngine] Matrix & Multi-Cam AI engine started.")

    def stop(self):
        self.running = False
        for grabber in self.grabbers.values():
            grabber.stop()
        if self.matrix_thread and self.matrix_thread.is_alive():
            self.matrix_thread.join(timeout=1.5)
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.5)
        print("[MultiCamEngine] All workers stopped.")

    def switch_focus_channel(self, channel: int):
        if channel in self.channels:
            with self.lock:
                self.focused_channel = channel
                self.settings.ossia_channel = channel
                save_settings(self.settings)
            print(f"[MultiCamEngine] Switched focus to Channel {channel}")

    def _matrix_composer_loop(self):
        """Stitches channels 1-8 into a 3x3 grid at ~8 FPS ONLY when a viewer is connected."""
        tile_w, tile_h = 426, 240
        blank_tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

        while self.running:
            if not self.has_matrix_viewers_only():
                time.sleep(0.4)
                continue

            start_t = time.time()
            tiles = {}
            ai_channels = set(self.get_ai_channels())

            for ch in self.channels:
                grabber = self.grabbers.get(ch)
                frame = grabber.get_frame() if grabber else None

                if frame is not None:
                    tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
                else:
                    tile = blank_tile.copy()
                    cv2.putText(tile, f"Cam {ch} Offline", (130, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 120), 1, cv2.LINE_AA)

                is_focused = (ch == self.focused_channel)
                border_color = (60, 120, 255) if is_focused else (40, 48, 60)
                cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), border_color, 2 if is_focused else 1)

                tag_w = 104 if ch in ai_channels else 90
                cv2.rectangle(tile, (6, 6), (tag_w, 26), (15, 20, 28), -1)
                cv2.rectangle(tile, (6, 6), (tag_w, 26), (60, 75, 95), 1)
                lbl = f"CAM {ch:02d} AI" if ch in ai_channels else f"CAM {ch:02d}"
                cv2.putText(tile, lbl, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 240, 255), 1, cv2.LINE_AA)

                dot_c = (50, 220, 80) if (grabber and grabber.is_connected) else (40, 50, 220)
                cv2.circle(tile, (tag_w - 10, 16), 4, dot_c, -1, cv2.LINE_AA)

                with self.lock:
                    boxes = self.camera_person_boxes.get(ch, [])
                for b in boxes:
                    bx1, by1, bx2, by2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    cv2.rectangle(tile, (bx1, by1), (bx2, by2), (80, 180, 240), 1, cv2.LINE_AA)

                tiles[ch] = tile

            # Tile 9: Information & Status Card
            tile9 = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            cv2.rectangle(tile9, (0, 0), (tile_w - 1, tile_h - 1), (35, 45, 60), 1)
            cv2.rectangle(tile9, (10, 10), (tile_w - 10, 38), (22, 30, 42), -1)
            cv2.putText(tile9, "BASHERT - RESTAURANT AI", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(tile9, f"AI Monitoring: {len(ai_channels)} Cams Active", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 230, 245), 1, cv2.LINE_AA)
            cv2.putText(tile9, "YOLOv8n-Pose + Menu Alerts: ON", (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 225, 120), 1, cv2.LINE_AA)
            cv2.putText(tile9, "Google Cloud + Telegram: Live", (20, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 185, 70), 1, cv2.LINE_AA)
            cv2.putText(tile9, f"Clock: {time.strftime('%H:%M:%S')}", (20, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 205, 225), 1, cv2.LINE_AA)
            cv2.putText(tile9, "Click any camera to zoom in", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 145, 165), 1, cv2.LINE_AA)

            row1 = np.hstack([tiles[1], tiles[2], tiles[3]])
            row2 = np.hstack([tiles[4], tiles[5], tiles[6]])
            row3 = np.hstack([tiles[7], tiles[8], tile9])
            matrix_frame = np.vstack([row1, row2, row3])

            cv2.rectangle(matrix_frame, (0, 0), (matrix_frame.shape[1], 28), (12, 16, 22), -1)
            cv2.line(matrix_frame, (0, 28), (matrix_frame.shape[1], 28), (40, 50, 65), 1)
            cv2.putText(matrix_frame, "VISIONDINE AI • LIVE RESTAURANT 8-CAMERA MATRIX (CHANNELS 1-8) • OSSIA VMS",
                        (18, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 220, 245), 1, cv2.LINE_AA)
            cv2.putText(matrix_frame, f"Multi-Cam AI: {len(ai_channels)} Cams | Focus: CAM {self.focused_channel}",
                        (matrix_frame.shape[1] - 360, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 220, 150), 1, cv2.LINE_AA)

            ret, jpeg = cv2.imencode(".jpg", matrix_frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            with self.lock:
                self.latest_matrix_frame = matrix_frame
                if ret:
                    self.latest_matrix_jpeg = jpeg.tobytes()

            elapsed = time.time() - start_t
            time.sleep(max(0.02, (1.0 / 8.0) - elapsed))

    def _multi_cam_inference_loop(self):
        """
        Round-robin YOLO detection across calibrated table cameras (every ~0.33s per step).
        Updates shared OccupancyTracker for each camera channel with minimal CPU load.
        """
        idx = 0
        while self.running:
            try:
                matrix_open = self.has_matrix_viewers_only()
                ai_mode = bool(self.settings.multi_cam_ai and self.settings.video_source == "rtsp")
                ai_channels = self.get_ai_channels() if ai_mode else []

                if not matrix_open and not ai_channels:
                    time.sleep(0.4)
                    continue

                target_pool = self.channels if matrix_open else ai_channels
                if not target_pool or self.model is None or self.detect_fn is None:
                    time.sleep(0.4)
                    continue

                ch = target_pool[idx % len(target_pool)]
                idx += 1

                # Avoid duplicating work on the focused channel if Single Focus viewer is actively running it
                if self.is_single_viewer_active_fn and self.is_single_viewer_active_fn() and ch == self.settings.ossia_channel:
                    time.sleep(0.05)
                    continue

                grabber = self.grabbers.get(ch)
                frame = grabber.get_frame() if grabber else None

                if frame is not None:
                    detections = self.detect_fn(frame)
                    h_nat, w_nat = frame.shape[:2]
                    scale_x = 426.0 / max(1.0, float(w_nat))
                    scale_y = 240.0 / max(1.0, float(h_nat))

                    scaled_boxes = [
                        (
                            d["box"][0] * scale_x,
                            d["box"][1] * scale_y,
                            d["box"][2] * scale_x,
                            d["box"][3] * scale_y,
                        )
                        for d in detections
                    ]

                    with self.lock:
                        self.camera_detections[ch] = detections
                        self.camera_person_boxes[ch] = scaled_boxes

                    if ai_mode and ch in ai_channels and self.tracker is not None:
                        _, events = self.tracker.update_detections(
                            detections,
                            w_nat,
                            h_nat,
                            active_channel=ch,
                            strict_conf=self.settings.confidence_threshold,
                            frame=frame,
                        )
                        if events and self.event_sink_fn:
                            self.event_sink_fn(events)

                time.sleep(0.33)  # ~3 inferences/sec total keeps CPU cool across all restaurant tables
            except Exception as e:
                print(f"[MultiCam Inference Error]: {e}")
                time.sleep(0.5)

    def generate_matrix_stream(self):
        """Yield MJPEG stream for the 3x3 composite 8-camera grid."""
        with self._viewer_lock:
            self.active_viewers += 1
        try:
            while self.running:
                start_t = time.time()
                with self.lock:
                    jpeg_bytes = self.latest_matrix_jpeg

                if jpeg_bytes is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                    )

                elapsed = time.time() - start_t
                time.sleep(max(0.03, (1.0 / 8.0) - elapsed))
        finally:
            with self._viewer_lock:
                self.active_viewers = max(0, self.active_viewers - 1)


# Global singleton
multi_camera_engine = MultiCameraEngine()
