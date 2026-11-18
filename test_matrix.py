import cv2
import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.config import load_settings

cfg = load_settings()
ip = cfg.ossia_ip or "192.168.1.100"
pwd = cfg.ossia_pass or "password"
user = cfg.ossia_user or "admin"
port = cfg.ossia_port or 554

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;2000000"

print("Connecting to channels 1-9 sub-streams...", flush=True)
caps = {}
for ch in range(1, 10):
    url = f"rtsp://{user}:{pwd}@{ip}:{port}/chID={ch}&streamType=sub&linkType=tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    caps[ch] = cap

print("Reading 1 frame from each channel...", flush=True)
frames = {}
for ch, cap in caps.items():
    ret, frame = cap.read()
    if ret and frame is not None:
        frames[ch] = cv2.resize(frame, (426, 240))
    else:
        # Placeholder
        black = np.zeros((240, 426, 3), dtype=np.uint8)
        cv2.putText(black, f"Cam {ch} Offline", (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        frames[ch] = black

# Create 3x3 stitched grid (1278x720)
row1 = np.hstack([frames[1], frames[2], frames[3]])
row2 = np.hstack([frames[4], frames[5], frames[6]])
row3 = np.hstack([frames[7], frames[8], frames[9]])
grid = np.vstack([row1, row2, row3])

print(f"Stitched grid shape: {grid.shape}", flush=True)
ret, jpeg = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 75])
print(f"JPEG size: {len(jpeg) // 1024} KB", flush=True)

for cap in caps.values():
    cap.release()
print("Test completed successfully!", flush=True)
