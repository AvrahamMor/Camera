import asyncio
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Response, status
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2

from backend.config import load_settings, save_settings, SystemSettings, BASE_DIR
from backend.tables_manager import tables_manager, TableModel
from backend.cv_engine import cv_engine
from backend.multi_camera_engine import multi_camera_engine
from backend.database import (
    init_db, get_kpis, get_recent_events, get_hourly_analytics,
    get_table_turnover_analytics, acknowledge_service
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[Server] Initializing database...")
    init_db()
    print("[Server] Starting Computer Vision engines...")
    cv_engine.start()
    multi_camera_engine.start()
    yield
    # Shutdown
    print("[Server] Stopping Computer Vision engines...")
    cv_engine.stop()
    multi_camera_engine.stop()

app = FastAPI(
    title="Restaurant Table Occupancy & Service Analytics",
    description="Real-Time Computer Vision Floor Analytics for Ossia VMS & RTSP Streams",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSockets
connected_websockets: List[WebSocket] = []

# --- Video Streaming & Snapshot Endpoints ---

@app.get("/api/camera/channels")
def get_camera_channels():
    """Return available restaurant NVR channels (strictly channels 1 to 8)."""
    current_ch = cv_engine.settings.ossia_channel or 1
    channels = []
    for ch in range(1, 9):
        channels.append({
            "channel": ch,
            "name": f"מצלמה {ch}",
            "is_current": (ch == current_ch)
        })
    return {"current_channel": current_ch, "channels": channels}

@app.post("/api/camera/switch")
def switch_camera_channel(payload: Dict[str, Any]):
    """Switch active camera channel on the fly seamlessly (1-8)."""
    new_ch = int(payload.get("channel", 1))
    if not (1 <= new_ch <= 8):
        raise HTTPException(status_code=400, detail="Channel must be between 1 and 8 for the restaurant")

    cv_engine.switch_channel(new_ch)
    multi_camera_engine.switch_focus_channel(new_ch)
    return {"success": True, "channel": new_ch, "rtsp_url": cv_engine.settings.rtsp_url}

@app.get("/api/video_feed")
def video_feed(mode: str = "matrix"):
    """MJPEG stream: 'matrix' for multi-camera grid (cams 1-8), or 'single' for focused single camera."""
    if mode == "matrix":
        return StreamingResponse(
            multi_camera_engine.generate_matrix_stream(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    return StreamingResponse(
        cv_engine.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/video_feed/matrix")
def video_feed_matrix():
    """Dedicated multi-camera matrix MJPEG stream (cams 1-8)."""
    return StreamingResponse(
        multi_camera_engine.generate_matrix_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/snapshot")
def get_snapshot(channel: int = 0, clean: bool = True):
    """Grab a high-resolution snapshot for the interactive ROI calibration tool."""
    if 1 <= channel <= 8:
        grabber = multi_camera_engine.grabbers.get(channel)
        if grabber:
            frame = grabber.get_frame()
            if frame is not None:
                ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ret:
                    return Response(content=buf.tobytes(), media_type="image/jpeg")

    jpeg_bytes = cv_engine.get_snapshot(clean=clean)
    if not jpeg_bytes:
        raise HTTPException(status_code=503, detail="Camera stream unavailable")
    return Response(content=jpeg_bytes, media_type="image/jpeg")

@app.get("/api/stream/status")
def stream_status():
    return {
        "is_connected": cv_engine.is_connected,
        "status": cv_engine.connection_status,
        "inference_fps": cv_engine.actual_inference_fps,
        "stream_fps": cv_engine.actual_stream_fps,
        "video_source": cv_engine.settings.video_source
    }

# --- Live State & WebSocket ---

@app.get("/api/live")
def get_live_state():
    return cv_engine.get_live_state()

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            state = cv_engine.get_live_state()
            await websocket.send_json(state)
            await asyncio.sleep(0.5)  # 2 updates per second over WS
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)

# --- Table ROI Calibration Endpoints ---

@app.get("/api/tables", response_model=List[TableModel])
def get_tables():
    return tables_manager.get_all()

@app.post("/api/tables", response_model=TableModel)
def add_table(table: TableModel):
    return tables_manager.add_or_update(table)

@app.put("/api/tables/{table_id}", response_model=TableModel)
def update_table(table_id: str, table: TableModel):
    table.id = table_id
    return tables_manager.add_or_update(table)

@app.delete("/api/tables/{table_id}")
def delete_table(table_id: str):
    success = tables_manager.delete(table_id)
    if not success:
        raise HTTPException(status_code=404, detail="Table not found")
    return {"message": "Table deleted successfully"}

@app.post("/api/tables/reset")
def reset_tables():
    return tables_manager.reset_defaults()

# --- Service Alert Acknowledgment ---

@app.post("/api/tables/{table_id}/acknowledge_service")
def ack_service(table_id: str):
    success = cv_engine.tracker.acknowledge_service(table_id)
    if not success:
        raise HTTPException(status_code=404, detail="Table not found")
    return {"message": f"Service acknowledged for table {table_id}"}

# --- Settings & RTSP Connection Test ---

@app.get("/api/settings", response_model=SystemSettings)
def get_settings():
    return cv_engine.settings

@app.post("/api/settings", response_model=SystemSettings)
def update_settings(new_settings: SystemSettings):
    save_settings(new_settings)
    cv_engine.update_settings(new_settings)
    return cv_engine.settings

@app.post("/api/settings/test_rtsp")
def test_rtsp_connection(payload: Dict[str, str]):
    rtsp_url = payload.get("rtsp_url", "")
    if not rtsp_url:
        raise HTTPException(status_code=400, detail="rtsp_url is required")

    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    connected = cap.isOpened()
    ret = False
    if connected:
        ret, frame = cap.read()
    cap.release()

    if connected and ret:
        return {"success": True, "message": "Successfully connected to RTSP stream and captured frame!"}
    elif connected and not ret:
        return {"success": False, "message": "Connected to RTSP stream, but failed to read video frames. Check stream codec."}
    else:
        return {"success": False, "message": "Could not connect to RTSP stream. Verify camera IP, port 554, and credentials."}

# --- Telegram & Firebase Integrations Endpoints ---

@app.get("/api/settings/telegram")
def get_telegram_settings():
    from backend.telegram_alerts import telegram_alerts
    return {
        "bot_token": telegram_alerts.bot_token,
        "chat_id": telegram_alerts.chat_id,
        "enabled": telegram_alerts.enabled
    }

@app.post("/api/settings/telegram")
def save_telegram_settings(payload: Dict[str, Any]):
    from backend.telegram_alerts import telegram_alerts
    bot_token = payload.get("bot_token", "").strip()
    chat_id = str(payload.get("chat_id", "")).strip()
    enabled = bool(payload.get("enabled", True))
    telegram_alerts.save_config(bot_token, chat_id, enabled)
    test_ok = False
    if bot_token and chat_id:
        test_ok = telegram_alerts._post_telegram("🤖 *חיבור ל-VisionDine AI בוצע בהצלחה!*\nמערכת התראות שולחנות מוכנה לשליחת הודעות.")
    return {"success": True, "test_sent": test_ok}

@app.get("/api/settings/firebase")
def get_firebase_status():
    from backend.firebase_sync import firebase_manager
    return {
        "configured": firebase_manager.is_configured
    }

# --- Analytics Endpoints ---

@app.get("/api/analytics/kpis")
def analytics_kpis():
    return get_kpis()

@app.get("/api/analytics/hourly")
def analytics_hourly():
    return get_hourly_analytics()

@app.get("/api/analytics/tables")
def analytics_tables():
    return get_table_turnover_analytics()

@app.get("/api/analytics/events")
def analytics_events(limit: int = 50):
    return get_recent_events(limit=limit)

# --- Frontend Static Files Serving ---

FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(
            str(index_file),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
        )
    return {"message": "Frontend index.html not found"}
