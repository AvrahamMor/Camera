# VisionDine AI • Real-Time Restaurant Table Occupancy & Service Analytics

VisionDine AI is an enterprise-grade Computer Vision platform designed for restaurants, bars, and hospitality venues. Powered by Ultralytics YOLOv8 and FastAPI, VisionDine AI connects directly to IP camera networks (such as **Provision-ISR / Ossia VMS NVRs**) to deliver real-time table occupancy tracking, service delay alerts, customer flow KPIs, and cloud synchronization.

---

## 🌟 Key Features

### 1. 8-Camera Multi-Stream Matrix Composer
- **Multi-Camera Edge Engine**: Simultanously captures sub-streams from up to 8 Provision-ISR camera channels (Channels 1–8) with individual reconnection loops.
- **3x3 Live Matrix Feed (`/api/video_feed/matrix`)**: Dynamically stitches all active camera feeds into a single consolidated 3x3 video stream, with an integrated restaurant branding and live system status clock tile.
- **Low-Latency MJPEG Streaming**: Efficient, lightweight MJPEG output compatible with any modern browser or mobile display.

### 2. High-Accuracy AI Inference
- **Ultralytics YOLOv8 (`yolov8n.pt`)**: Real-time person detection filtered to class 0 (`person`).
- **Bottom-Center Foot Anchor Point**: Computes the ground-contact coordinate `((x1 + x2)/2, y2)` for each bounding box, eliminating false positives caused by leaning diners, high chairs, or angled ceiling views.
- **FPS Throttling**: Decoupled inference engine runs AI at 3–5 FPS while maintaining a fluid 15 FPS video stream.
- **Debounce Buffer**: 4-second temporal buffer prevents spurious state transitions when waiters or passing patrons walk past tables.

### 3. Interactive Web-Based Table Zone (ROI) Calibration
- **Per-Camera Calibration Toolbar**: Seamlessly switch between camera feeds to calibrate tables assigned to specific viewpoints.
- **HTML5 Canvas Polygon Editor**:
  - **Left Click**: Add vertices around table boundaries and seating zones.
  - **Drag**: Fine-tune point coordinates.
  - **Right Click / Save**: Complete and persist polygon.
  - **Delete / Clear**: Edit or remove configured tables.
- **Resolution-Independent Coordinates**: Automatically normalizes polygon vertices to `[0.0 .. 1.0]`, ensuring persistent calibration across changing camera resolutions (720p, 1080p, 2K, 4K).

### 4. Cloud & Telegram Integration
- **Google Cloud / Firebase Firestore Sync**: Automatically synchronizes live floor state, table occupancy, headcount, and urgent alerts to `restaurants/main_branch/live/state` every 5 seconds for remote tablet and mobile owner apps.
- **Automated Telegram Bot Alerts**: Dispatches instantaneous alerts with table numbers, elapsed minutes, and headcount directly to waitstaff and manager Telegram groups when tables exceed service thresholds.

### 5. Table State Machine & Service Monitoring
- 🟢 **`FREE`** (Green): Table is unoccupied.
- 🔴 **`OCCUPIED`** (Red): Diners are seated. Tracks live session duration (`MM:SS`) and active headcount.
- 🟡 **`NEEDS_SERVICE`** (Flashing Amber): Triggered when occupancy exceeds the configurable threshold without waitstaff presence. Includes quick staff acknowledgement (`/api/tables/{id}/acknowledge_service`).

---

## 📁 Project Architecture

```
Camera/
├── backend/
│   ├── app.py                     # FastAPI application, REST endpoints, MJPEG streams & WebSocket
│   ├── config.py                  # System configuration loader, validator, and RTSP URL generator
│   ├── cv_engine.py               # Single-camera capture and YOLO inference pipeline
│   ├── database.py                # SQLite + SQLAlchemy models (OccupancySession, TableEvent)
│   ├── firebase_sync.py           # Google Cloud Firebase Firestore real-time cloud synchronizer
│   ├── multi_camera_engine.py      # Multi-camera parallel grabbers & 3x3 matrix composer
│   ├── occupancy_tracker.py       # Point-in-polygon logic, debounce window, and state machine
│   ├── tables_manager.py          # Thread-safe CRUD manager for config/tables_config.json
│   ├── telegram_alerts.py         # Telegram bot alert dispatcher for urgent service notices
│   └── video_generator.py         # Synthetic restaurant simulation video generator for testing
├── frontend/
│   ├── index.html                 # Single-page dashboard application (Tailwind CSS + Lucide Icons)
│   ├── css/
│   │   └── style.css              # Custom styles, animations, and dark-theme glows
│   └── js/
│       ├── app.js                 # WebSocket sync, live state, KPI cards, settings & tab router
│       ├── calibration.js         # Interactive ROI canvas polygon editor with camera selector
│       └── analytics.js           # Chart.js analytics visualizations and event audit log
├── config/
│   ├── settings.example.json      # Template for NVR credentials and inference settings
│   ├── firebase_credentials.example.json # Template for Firebase Admin SDK service account key
│   ├── telegram_config.example.json      # Template for Telegram bot alerts configuration
│   └── tables_config.json         # Table zones, seat capacities, and camera assignments
├── requirements.txt               # Python package dependencies
├── run.py                         # Single-command server launcher
└── README.md                      # Documentation
```

---

## 🔒 Security & Secrets Management

To ensure maximum security on public or shared repositories:
- All sensitive credentials, passwords, private keys, database dumps, and camera snapshots are excluded via `.gitignore`.
- Configuration templates (`*.example.json`) are provided.

### Setup Credentials Locally:
1. **NVR / Camera Credentials**:
   Copy `config/settings.example.json` to `config/settings.json` and fill in your NVR IP and password:
   ```powershell
   cp config/settings.example.json config/settings.json
   ```

2. **Firebase Firestore Cloud Sync (Optional)**:
   Place your Firebase Service Account private key JSON in:
   ```
   config/firebase_credentials.json
   ```

3. **Telegram Bot Alerts (Optional)**:
   Configure your bot token and chat ID via the dashboard **Camera & Settings** tab, or manually in:
   ```
   config/telegram_config.json
   ```

---

## 🚀 Quick Start Guide

### 1. Requirements
- Python 3.10+
- Modern Web Browser (Google Chrome, Microsoft Edge, Firefox, Safari)

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/AvrahamMor/Camera.git
cd Camera

# Install dependencies
python -m pip install -r requirements.txt
```

### 3. Launching VisionDine AI
Run the unified launcher:
```powershell
python run.py
```

Options:
- `python run.py --no-browser`: Start server in headless/background mode without opening a browser window.
- `python run.py --port 8085 --host 0.0.0.0`: Specify custom host and port.
- `python run.py --source sample`: Run with synthetic simulated restaurant footage.
- `python run.py --source rtsp`: Connect to live NVR RTSP stream.

---

## 📹 NVR & Camera Stream Configuration

### Provision-ISR / Ossia VMS RTSP Format
Provision-ISR NVRs broadcast RTSP video over TCP on port **554**:
```
rtsp://<user>:<password>@<ip>:554/chID=<channel>&streamType=<stream>&linkType=tcp
```
- `chID`: Camera channel number (1 to 16).
- `streamType`: `sub` (Recommended for AI processing) or `main` (Full HD/4K).
- `linkType`: `tcp` (Guarantees no packet loss).

---

## 🔌 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web dashboard application |
| `GET` | `/api/video_feed` | Live MJPEG stream with table zone overlays |
| `GET` | `/api/video_feed/matrix` | Live 3x3 multi-camera matrix feed (Channels 1–8 + Status Tile) |
| `GET` | `/api/snapshot` | Snapshot for calibration (`?channel=X&clean=true`) |
| `GET` | `/api/live` | Current table status, active alerts, and real-time KPIs |
| `WS`  | `/ws/live` | WebSocket connection for 500ms real-time dashboard sync |
| `GET` | `/api/tables` | List all configured table zones and polygons |
| `POST`| `/api/tables` | Create or update table zone polygon |
| `DELETE`| `/api/tables/{id}` | Delete table zone |
| `POST`| `/api/tables/{id}/acknowledge_service` | Acknowledge table service alert |
| `GET` | `/api/settings` | Read current system configuration |
| `POST`| `/api/settings` | Update system configuration |
| `POST`| `/api/settings/test_rtsp` | Test connection to RTSP stream |
| `GET` | `/api/settings/telegram` | Get Telegram alert settings |
| `POST`| `/api/settings/telegram` | Save Telegram alert settings & send test message |
| `GET` | `/api/analytics/kpis` | Summary KPIs (total sessions, turnover time, diners) |
| `GET` | `/api/analytics/hourly` | 24-hour occupancy breakdown |
| `GET` | `/api/analytics/tables` | Per-table turnover statistics |
| `GET` | `/api/analytics/events` | Historical audit log of dining events |

---

## 🛡️ License
Proprietary & Confidential • Developed for Bashert Restaurant.
