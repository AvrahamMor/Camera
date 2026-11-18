import cv2
import numpy as np
import math
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_VIDEO_DIR = BASE_DIR / "sample_videos"
SAMPLE_VIDEO_PATH = SAMPLE_VIDEO_DIR / "demo_restaurant.mp4"

def draw_person(frame, x, y, angle=0, shirt_color=(45, 110, 220), hair_color=(30, 30, 40), is_seated=False):
    """
    Draws a realistic top-down human figure:
    - Shoulders, arms, torso (shirt)
    - Head, hair
    - If standing: shoes/feet visible
    """
    scale = 0.85
    # Shoulders ellipse
    shoulder_w = int(28 * scale)
    shoulder_h = int(14 * scale)
    cv2.ellipse(frame, (int(x), int(y)), (shoulder_w, shoulder_h), angle, 0, 360, shirt_color, -1, cv2.LINE_AA)
    cv2.ellipse(frame, (int(x), int(y)), (shoulder_w, shoulder_h), angle, 0, 360, (20, 20, 20), 1, cv2.LINE_AA)

    # Head circle
    head_r = int(11 * scale)
    cv2.circle(frame, (int(x), int(y)), head_r, (190, 215, 240), -1, cv2.LINE_AA) # skin tone
    cv2.circle(frame, (int(x), int(y - 2 * scale)), int(head_r * 0.9), hair_color, -1, cv2.LINE_AA) # hair

    if not is_seated:
        # Subtle feet / shoes in movement direction
        rad = math.radians(angle + 90)
        foot_dist = 18 * scale
        fx = int(x + foot_dist * math.cos(rad))
        fy = int(y + foot_dist * math.sin(rad))
        cv2.circle(frame, (fx - 4, fy), 4, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(frame, (fx + 4, fy), 4, (30, 30, 30), -1, cv2.LINE_AA)

def generate_sample_restaurant_video(output_path: Path = SAMPLE_VIDEO_PATH, duration_sec: int = 30, fps: int = 20):
    """
    Generates a 1280x720 video simulating a restaurant floor with tables and patrons.
    Loops smoothly for continuous testing.
    """
    SAMPLE_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 1280, 720
    total_frames = duration_sec * fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Pre-render static restaurant floor
    base_floor = np.zeros((height, width, 3), dtype=np.uint8)
    # Warm hardwood plank tiles
    base_floor[:] = (26, 38, 52) # Dark warm brown
    for y in range(0, height, 40):
        cv2.line(base_floor, (0, y), (width, y), (20, 30, 42), 1)
    for x in range(0, width, 160):
        for y in range(0, height, 40):
            offset = 80 if (y // 40) % 2 == 1 else 0
            cv2.line(base_floor, (x + offset, y), (x + offset, y + 40), (20, 30, 42), 1)

    # Ambient wall / bar area at top
    cv2.rectangle(base_floor, (0, 0), (width, 80), (18, 22, 28), -1)
    cv2.putText(base_floor, "KITCHEN & SERVICE BAR AREA", (460, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 140, 160), 2, cv2.LINE_AA)
    cv2.line(base_floor, (0, 80), (width, 80), (70, 80, 95), 2)

    # Tables layout (matching tables_config.json):
    # Table 01: [0.08, 0.45] to [0.32, 0.85] -> (102, 324) to (409, 612)
    # Table 02: [0.40, 0.35] to [0.60, 0.68] -> (512, 252) to (768, 489)
    # Table 03: [0.68, 0.45] to [0.92, 0.85] -> (870, 324) to (1177, 612)
    # Table 04: [0.15, 0.15] to [0.35, 0.38] -> (192, 108) to (448, 273)
    # Table 05: [0.65, 0.15] to [0.90, 0.38] -> (832, 108) to (1152, 273)

    tables_static = [
        {"id": "T01", "rect": (140, 380, 220, 180), "shape": "rect", "color": (50, 75, 110)},
        {"id": "T02", "center": (640, 370), "radius": 75, "shape": "circle", "color": (55, 80, 115)},
        {"id": "T03", "rect": (920, 380, 220, 180), "shape": "rect", "color": (50, 75, 110)},
        {"id": "T04", "rect": (230, 140, 150, 100), "shape": "rect", "color": (45, 70, 105)},
        {"id": "T05", "rect": (880, 140, 220, 100), "shape": "rect", "color": (45, 70, 105)},
    ]

    for t in tables_static:
        if t["shape"] == "rect":
            rx, ry, rw, rh = t["rect"]
            # Drop shadow
            cv2.rectangle(base_floor, (rx + 6, ry + 6), (rx + rw + 6, ry + rh + 6), (12, 16, 22), -1)
            # Table surface
            cv2.rectangle(base_floor, (rx, ry), (rx + rw, ry + rh), t["color"], -1)
            cv2.rectangle(base_floor, (rx, ry), (rx + rw, ry + rh), (110, 140, 180), 2)
            # Table label
            cv2.putText(base_floor, t["id"], (rx + rw // 2 - 25, ry + rh // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 240), 2, cv2.LINE_AA)
        elif t["shape"] == "circle":
            cx, cy = t["center"]
            r = t["radius"]
            cv2.circle(base_floor, (cx + 5, cy + 5), r, (12, 16, 22), -1)
            cv2.circle(base_floor, (cx, cy), r, t["color"], -1)
            cv2.circle(base_floor, (cx, cy), r, (110, 140, 180), 2)
            cv2.putText(base_floor, t["id"], (cx - 25, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 240), 2, cv2.LINE_AA)

    print(f"[VideoGenerator] Generating {total_frames} frames ({duration_sec}s at {fps} FPS)...")

    for f_idx in range(total_frames):
        t_sec = f_idx / fps
        frame = base_floor.copy()

        # Dynamic simulation events:
        # Table 02: Couple arrives at t=3s, sits t=5s..24s, leaves at t=25s..29s
        if t_sec < 3.0:
            pass
        elif 3.0 <= t_sec < 5.0:
            prog = (t_sec - 3.0) / 2.0
            p1_x = int(640 - 150 * (1 - prog))
            p1_y = int(680 - 310 * prog)
            p2_x = int(680 - 120 * (1 - prog))
            p2_y = int(680 - 310 * prog)
            draw_person(frame, p1_x, p1_y, angle=-90, shirt_color=(60, 160, 240))
            draw_person(frame, p2_x, p2_y, angle=-90, shirt_color=(230, 80, 120))
        elif 5.0 <= t_sec < 25.0:
            # Seated at Table 02 (left and right of round table)
            draw_person(frame, 595, 370, angle=0, shirt_color=(60, 160, 240), is_seated=True)
            draw_person(frame, 685, 370, angle=180, shirt_color=(230, 80, 120), is_seated=True)
        elif 25.0 <= t_sec < 29.0:
            prog = (t_sec - 25.0) / 4.0
            p1_x = int(595 - 200 * prog)
            p1_y = int(370 + 320 * prog)
            p2_x = int(685 - 200 * prog)
            p2_y = int(370 + 320 * prog)
            draw_person(frame, p1_x, p1_y, angle=90, shirt_color=(60, 160, 240))
            draw_person(frame, p2_x, p2_y, angle=90, shirt_color=(230, 80, 120))

        # Table 01: Party of 3 enters at t=7s, sits t=10s..30s
        if 7.0 <= t_sec < 10.0:
            prog = (t_sec - 7.0) / 3.0
            x_pos = int(50 + (200 - 50) * prog)
            y_pos = int(680 - (680 - 470) * prog)
            draw_person(frame, x_pos, y_pos, angle=-75, shirt_color=(70, 200, 100))
            draw_person(frame, x_pos + 30, y_pos + 20, angle=-75, shirt_color=(210, 160, 40))
        elif t_sec >= 10.0:
            # 3 people seated at booth T01
            draw_person(frame, 190, 430, angle=90, shirt_color=(70, 200, 100), is_seated=True)
            draw_person(frame, 290, 430, angle=90, shirt_color=(210, 160, 40), is_seated=True)
            draw_person(frame, 240, 520, angle=-90, shirt_color=(180, 80, 210), is_seated=True)

        # Table 04: Solo guest arrives at t=14s, sits at t=16s..30s
        if 14.0 <= t_sec < 16.0:
            prog = (t_sec - 14.0) / 2.0
            x_pos = int(305)
            y_pos = int(30 + (190 - 30) * prog)
            draw_person(frame, x_pos, y_pos, angle=90, shirt_color=(240, 120, 50))
        elif t_sec >= 16.0:
            draw_person(frame, 305, 190, angle=90, shirt_color=(240, 120, 50), is_seated=True)

        # Waiter moving between kitchen and tables at t=18s..24s
        if 18.0 <= t_sec < 24.0:
            prog = (t_sec - 18.0) / 6.0
            # Moves from kitchen (550, 90) to Table 02 (640, 270) then back
            if prog < 0.5:
                w_x = int(550 + (640 - 550) * (prog / 0.5))
                w_y = int(90 + (270 - 90) * (prog / 0.5))
            else:
                w_x = int(640 - (640 - 550) * ((prog - 0.5) / 0.5))
                w_y = int(270 - (270 - 90) * ((prog - 0.5) / 0.5))
            # Waiter in crisp black/white uniform
            draw_person(frame, w_x, w_y, angle=90 if prog < 0.5 else -90, shirt_color=(245, 245, 250), hair_color=(10, 10, 10))

        # Add timestamp & camera stream info watermark
        timestamp_str = f"OSSIA VMS / CAM-01 [RESTAURANT MAIN FLOOR] | {t_sec:04.1f}s"
        cv2.putText(frame, timestamp_str, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 200, 220), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"[VideoGenerator] Generated sample video successfully: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_sample_restaurant_video()
