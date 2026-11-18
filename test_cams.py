import urllib.request
import urllib.error
import re
import cv2
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.config import load_settings

cfg = load_settings()
ip = cfg.ossia_ip or "192.168.1.160"
pwd = cfg.ossia_pass or "password"
user = cfg.ossia_user or "admin"

# Common Provision IP camera paths
candidates = [
    f"rtsp://{user}:{pwd}@{ip}:554/live/ch0",
    f"rtsp://{user}:{pwd}@{ip}:554/live/ch1",
    f"rtsp://{user}:{pwd}@{ip}:554/ch01/0",
    f"rtsp://{user}:{pwd}@{ip}:554/ch01/1",
    f"rtsp://{user}:{pwd}@{ip}:554/ch1/0",
    f"rtsp://{user}:{pwd}@{ip}:554/ch1/1",
    f"rtsp://{user}:{pwd}@{ip}:554/profile1",
    f"rtsp://{user}:{pwd}@{ip}:554/profile2",
    f"rtsp://{user}:{pwd}@{ip}:554/onvif1",
    f"rtsp://{user}:{pwd}@{ip}:554/onvif2",
    f"rtsp://{user}:{pwd}@{ip}:554/Streaming/Channels/101",
    f"rtsp://{user}:{pwd}@{ip}:554/Streaming/Channels/102",
    f"rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel=1&subtype=0",
    f"rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel=1&subtype=1",
    f"rtsp://{user}:{pwd}@{ip}:554/h264/ch1/main/av_stream",
    f"rtsp://{user}:{pwd}@{ip}:554/h264/ch1/sub/av_stream",
    f"rtsp://{user}:{pwd}@{ip}:554/unicast/c1/s0/live",
    f"rtsp://{user}:{pwd}@{ip}:554/unicast/c1/s1/live",
]

found = False
for url in candidates:
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
    ok = cap.isOpened()
    ret = False
    if ok:
        ret, frame = cap.read()
    cap.release()
    print(f"{url} -> opened: {ok}, read: {ret}")
    if ok and ret:
        print(f"*** SUCCESS ON {ip}! ***")
        cv2.imwrite("camera_success.jpg", frame)
        found = True
        break

if not found:
    print("Testing other camera IPs: 161, 162, 170...")
    for octet in [161, 162, 170]:
        ip_c = f"192.168.1.{octet}"
        for path in ["/ch01/0", "/ch01/1", "/unicast/c1/s0/live", "/unicast/c1/s1/live", "/live/ch0"]:
            u = f"rtsp://{user}:{pwd}@{ip_c}:554{path}"
            cap = cv2.VideoCapture(u)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)
            ok = cap.isOpened()
            ret = False
            if ok:
                ret, frame = cap.read()
            cap.release()
            if ok and ret:
                print(f"*** SUCCESS ON {ip_c}! URL: {u} ***")
                cv2.imwrite(f"camera_{octet}_success.jpg", frame)
                found = True
                break
        if found:
            break
