import cv2
import time
import os
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;2000000"

from backend.config import load_settings, save_settings, SystemSettings, BASE_DIR
from backend.tables_manager import tables_manager
from backend.occupancy_tracker import OccupancyTracker
from backend.database import get_kpis, get_recent_events
from backend.firebase_sync import firebase_manager

class CameraGrabber:
    """Non-blocking background frame grabber for a single RTSP sub-stream channel."""
    def __init__(self, channel: int, rtsp_url: str):
        self.channel = channel
        self.rtsp_url = rtsp_url
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
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self.is_connected = False
                time.sleep(2.0)
                continue

            self.is_connected = True
            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    self.is_connected = False
                    break

                with self.lock:
                    self.latest_frame = frame
                    self.last_frame_time = time.time()
                    self.is_connected = True

                time.sleep(0.02) # ~25-30 FPS grab rate

            cap.release()
            time.sleep(1.0)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None


class MultiCameraEngine:
    """
    Orchestrates the 3x3 restaurant matrix (Channels 1-9),
    running real-time CV analytics across all cameras simultaneously.
    """
    def __init__(self):
        self.settings: SystemSettings = load_settings()
        self.channels = list(range(1, 9)) # Restrict strictly to 1..8 (Restaurant dining)
        self.grabbers: Dict[int, CameraGrabber] = {}

        self.running = False
        self.matrix_thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None

        self.latest_matrix_frame: Optional[np.ndarray] = None
        self.latest_matrix_jpeg: Optional[bytes] = None

        # Focused single camera view
        self.focused_channel = self.settings.ossia_channel or 1
        if self.focused_channel not in self.channels:
            self.focused_channel = 1

        self.lock = threading.Lock()
        self.tracker = OccupancyTracker(debounce_sec=self.settings.debounce_seconds)

        self.camera_person_boxes: Dict[int, List[Tuple[float, float, float, float]]] = {
            ch: [] for ch in self.channels
        }

        # Initialize grabbers
        self._init_grabbers()

        # Load YOLO model
        self.model = None
        self._load_yolo()

    def _load_yolo(self):
        from ultralytics import YOLO
        try:
            self.model = YOLO(self.settings.model_name or "yolov8n.pt")
            print("[MultiCamEngine] YOLO loaded successfully.")
        except Exception as e:
            print(f"[MultiCamEngine] YOLO load error: {e}")

    def _init_grabbers(self):
        user = self.settings.ossia_user or "admin"
        passw = self.settings.ossia_pass or ""
        ip = self.settings.ossia_ip or "192.168.1.100"
        port = self.settings.ossia_port or 554

        for ch in self.channels:
            url = f"rtsp://{user}:{passw}@{ip}:{port}/chID={ch}&streamType=sub&linkType=tcp"
            self.grabbers[ch] = CameraGrabber(channel=ch, rtsp_url=url)

    def start(self):
        if self.running:
            return
        self.running = True

        # Start all 9 camera grabbers
        for ch, grabber in self.grabbers.items():
            grabber.start()

        # Start 3x3 matrix stitching thread
        self.matrix_thread = threading.Thread(target=self._matrix_composer_loop, daemon=True)
        self.matrix_thread.start()

        # Start AI multi-camera inference loop
        self.inference_thread = threading.Thread(target=self._multi_cam_inference_loop, daemon=True)
        self.inference_thread.start()
        print("[MultiCamEngine] 3x3 Matrix & Grabbers started.")

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
        """Stitches channels 1-9 into a single 3x3 grid (1278x720) at ~15-20 FPS."""
        tile_w, tile_h = 426, 240
        blank_tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

        while self.running:
            start_t = time.time()
            tiles = {}

            for ch in self.channels:
                grabber = self.grabbers.get(ch)
                frame = grabber.get_frame() if grabber else None

                if frame is not None:
                    # Resize to tile dimensions
                    tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
                else:
                    tile = blank_tile.copy()
                    cv2.putText(tile, f"Cam {ch} Offline", (130, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 120), 1, cv2.LINE_AA)

                # Draw camera badge & border on tile
                is_focused = (ch == self.focused_channel)
                border_color = (60, 120, 255) if is_focused else (40, 48, 60)
                cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), border_color, 2 if is_focused else 1)

                # Channel HUD Tag
                cv2.rectangle(tile, (6, 6), (90, 26), (15, 20, 28), -1)
                cv2.rectangle(tile, (6, 6), (90, 26), (60, 75, 95), 1)
                tag = f"CAM {ch:02d}"
                cv2.putText(tile, tag, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 240, 255), 1, cv2.LINE_AA)

                # Status dot
                dot_c = (50, 220, 80) if (grabber and grabber.is_connected) else (40, 50, 220)
                cv2.circle(tile, (78, 16), 4, dot_c, -1, cv2.LINE_AA)

                # Draw bounding boxes if any detected on this camera
                with self.lock:
                    boxes = self.camera_person_boxes.get(ch, [])
                for b in boxes:
                    # Scale boxes from native frame to tile
                    # b is (x1, y1, x2, y2)
                    bx1, by1, bx2, by2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    # Native sub-stream size is roughly 704x480 or 352x288
                    cv2.rectangle(tile, (bx1, by1), (bx2, by2), (80, 180, 240), 1, cv2.LINE_AA)

                tiles[ch] = tile

            # Tile 9: Information & Status Card (since restaurant uses Cams 1-8)
            tile9 = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            cv2.rectangle(tile9, (0, 0), (tile_w - 1, tile_h - 1), (35, 45, 60), 1)
            cv2.rectangle(tile9, (10, 10), (tile_w - 10, 38), (22, 30, 42), -1)
            cv2.putText(tile9, "BASHERT - RESTAURANT AI", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(tile9, "Active Cameras: 8 (Ch 1 - 8)", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 230, 245), 1, cv2.LINE_AA)
            cv2.putText(tile9, "AI Inference: Real-Time Active", (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 225, 120), 1, cv2.LINE_AA)
            cv2.putText(tile9, "Google Cloud: Synced Live", (20, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 185, 70), 1, cv2.LINE_AA)
            curr_clock = time.strftime('%H:%M:%S')
            cv2.putText(tile9, f"Clock: {curr_clock}", (20, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 205, 225), 1, cv2.LINE_AA)
            cv2.putText(tile9, "Click any camera to focus", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 145, 165), 1, cv2.LINE_AA)

            # Assemble 3x3 rows
            row1 = np.hstack([tiles[1], tiles[2], tiles[3]])
            row2 = np.hstack([tiles[4], tiles[5], tiles[6]])
            row3 = np.hstack([tiles[7], tiles[8], tile9])
            matrix_frame = np.vstack([row1, row2, row3])

            # Top Matrix Header Bar
            cv2.rectangle(matrix_frame, (0, 0), (matrix_frame.shape[1], 28), (12, 16, 22), -1)
            cv2.line(matrix_frame, (0, 28), (matrix_frame.shape[1], 28), (40, 50, 65), 1)

            header_text = "VISIONDINE AI • LIVE RESTAURANT 8-CAMERA MATRIX (CHANNELS 1-8) • OSSIA VMS"
            cv2.putText(matrix_frame, header_text, (18, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 220, 245), 1, cv2.LINE_AA)

            stats_text = f"Live 8 Cameras Active | Focus: CAM {self.focused_channel}"
            cv2.putText(matrix_frame, stats_text, (matrix_frame.shape[1] - 340, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 220, 150), 1, cv2.LINE_AA)

            ret, jpeg = cv2.imencode(".jpg", matrix_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

            with self.lock:
                self.latest_matrix_frame = matrix_frame
                if ret:
                    self.latest_matrix_jpeg = jpeg.tobytes()

            elapsed = time.time() - start_t
            sleep_t = max(0.015, (1.0 / 15.0) - elapsed) # ~15 FPS composite stream
            time.sleep(sleep_t)

    def _multi_cam_inference_loop(self):
        """Runs fast YOLO detection round-robin across channels 1..9 (~15ms per camera)."""
        idx = 0
        while self.running:
            ch = self.channels[idx % len(self.channels)]
            idx += 1

            grabber = self.grabbers.get(ch)
            frame = grabber.get_frame() if grabber else None

            if frame is not None and self.model is not None:
                try:
                    # Fast inference on sub-stream frame (takes ~15-20ms)
                    results = self.model.predict(
                        source=frame,
                        imgsz=384,
                        classes=[0], # Person
                        conf=self.settings.confidence_threshold,
                        verbose=False,
                        device="cpu"
                    )
                    detected = []
                    h_nat, w_nat = frame.shape[:2]
                    scale_x = 426.0 / max(1.0, float(w_nat))
                    scale_y = 240.0 / max(1.0, float(h_nat))

                    for r in results:
                        for box in r.boxes:
                            b = box.xyxy[0].cpu().numpy()
                            # Scale box to tile dimensions (426x240)
                            detected.append((
                                float(b[0]) * scale_x,
                                float(b[1]) * scale_y,
                                float(b[2]) * scale_x,
                                float(b[3]) * scale_y
                            ))

                    with self.lock:
                        self.camera_person_boxes[ch] = detected
                except Exception as e:
                    pass

            time.sleep(0.05) # Small sleep between cameras

    def generate_matrix_stream(self):
        """Yield MJPEG stream for the 3x3 composite 9-camera grid."""
        while self.running:
            start_t = time.time()
            jpeg_bytes = None
            with self.lock:
                jpeg_bytes = self.latest_matrix_jpeg

            if jpeg_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )

            elapsed = time.time() - start_t
            time.sleep(max(0.02, (1.0 / 15.0) - elapsed))

# Global singleton
multi_camera_engine = MultiCameraEngine()
