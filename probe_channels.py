import cv2
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.config import load_settings

cfg = load_settings()
ip = cfg.ossia_ip or "192.168.1.100"
pwd = cfg.ossia_pass or "password"
user = cfg.ossia_user or "admin"
port = cfg.ossia_port or 554

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;2000000"

print("Starting scan...", flush=True)
for ch in range(1, 17):
    url = f"rtsp://{user}:{pwd}@{ip}:{port}/chID={ch}&streamType=main&linkType=tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    opened = cap.isOpened()
    ret = False
    shape = None
    if opened:
        ret, frame = cap.read()
        if ret and frame is not None:
            shape = frame.shape
    cap.release()
    print(f"Channel {ch:2d}: opened={opened}, read_frame={ret}, shape={shape}", flush=True)
print("Scan complete.", flush=True)
