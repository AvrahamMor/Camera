import asyncio
import os
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, List
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import cv2

from backend.config import load_settings, save_settings, SystemSettings, BASE_DIR
from backend.tables_manager import tables_manager, TableModel
from backend.cv_engine import cv_engine
from backend.multi_camera_engine import multi_camera_engine
from backend.database import (
    init_db, get_kpis, get_recent_events, get_hourly_analytics,
    get_table_turnover_analytics
)
from backend.security import (
    auth_manager,
    COOKIE_NAME,
    MASK_PLACEHOLDER,
    SESSION_TTL_SECONDS,
    is_local_request,
    is_private_network_ip,
    is_masked_value,
    mask_secret,
    mask_rtsp_url,
    validate_rtsp_url,
    validate_video_file_path,
    validate_model_name,
)

PUBLIC_PATHS = {
    "/login",
    "/api/auth/status",
    "/api/auth/login",
    "/api/auth/setup",
    "/api/auth/logout",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    from backend.telegram_alerts import telegram_alerts
    print("[Server] Initializing database...")
    init_db()
    print("[Server] Starting Computer Vision engines...")
    multi_camera_engine.set_shared_model(cv_engine.model, cv_engine._model_lock)
    cv_engine.start()
    multi_camera_engine.start()
    telegram_alerts.bind_handlers(
        state_fn=cv_engine.get_live_state,
        ack_fn=cv_engine.tracker.acknowledge_service,
        snapshot_fn=lambda: cv_engine.get_snapshot(clean=False),
    )
    telegram_alerts.start_polling()
    yield
    # Shutdown
    print("[Server] Stopping Computer Vision engines...")
    telegram_alerts.stop_polling()
    cv_engine.stop()
    multi_camera_engine.stop()


# Disable public Swagger/ReDoc/OpenAPI docs in production
app = FastAPI(
    title="Restaurant Table Occupancy & Service Analytics",
    description="Real-Time Computer Vision Floor Analytics for Ossia VMS & RTSP Streams",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Active WebSockets
connected_websockets: List[WebSocket] = []


def _extract_client_ip(request_or_ws) -> str:
    return request_or_ws.client.host if request_or_ws.client else ""


def _apply_security_headers(response: Response, path: str) -> Response:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if path.startswith("/api/") or path == "/login":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    path = request.url.path
    client_ip = _extract_client_ip(request)

    # 1. Network perimeter check: restrict to private LAN / localhost / Tailscale unless disabled
    if auth_manager.lan_only and not is_private_network_ip(client_ip):
        resp = JSONResponse({"detail": "Access restricted to private network"}, status_code=403)
        return _apply_security_headers(resp, path)

    # 2. Same-origin CSRF check on mutating HTTP methods
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        host = request.headers.get("host", "")
        origin = request.headers.get("origin")
        referer = request.headers.get("referer")
        if origin and urlsplit(origin).netloc != host:
            resp = JSONResponse({"detail": "Cross-origin request blocked (CSRF protection)"}, status_code=403)
            return _apply_security_headers(resp, path)
        if not origin and referer and urlsplit(referer).netloc != host:
            resp = JSONResponse({"detail": "Cross-origin referer blocked (CSRF protection)"}, status_code=403)
            return _apply_security_headers(resp, path)

    # 3. Authentication enforcement for every non-public path
    if path not in PUBLIC_PATHS:
        token = request.cookies.get(COOKIE_NAME)
        session = auth_manager.verify_session_token(token)
        if not session:
            if path.startswith("/api/"):
                resp = JSONResponse({"detail": "Authentication required"}, status_code=401)
                return _apply_security_headers(resp, path)
            resp = RedirectResponse(url="/login", status_code=302)
            return _apply_security_headers(resp, path)
        request.state.user = session.get("u", "admin")

    response = await call_next(request)
    return _apply_security_headers(response, path)


def _set_session_cookie(response: Response, username: str) -> None:
    token = auth_manager.create_session_token(username)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        path="/",
    )


def _restore_masked_rtsp_password(candidate_url: str, fallback_pass: str) -> str:
    """If an RTSP URL contains the mask placeholder, substitute the real stored password."""
    if not candidate_url or MASK_PLACEHOLDER not in candidate_url:
        return candidate_url
    try:
        parts = urlsplit(candidate_url)
        if parts.password == MASK_PLACEHOLDER:
            real_pw = fallback_pass or ""
            stored_parts = urlsplit(cv_engine.settings.rtsp_url or "")
            if not real_pw and stored_parts.password:
                real_pw = stored_parts.password
            user = parts.username or ""
            host = parts.hostname or ""
            port = f":{parts.port}" if parts.port else ""
            cred = f"{user}:{real_pw}@" if (user or real_pw) else ""
            from urllib.parse import urlunsplit
            return urlunsplit((parts.scheme, f"{cred}{host}{port}", parts.path, parts.query, parts.fragment))
    except Exception:
        pass
    return candidate_url


# --- Authentication Endpoints ---

@app.get("/login")
def serve_login():
    login_file = BASE_DIR / "frontend" / "login.html"
    if login_file.exists():
        return FileResponse(str(login_file))
    return {"message": "Login page missing"}


@app.get("/api/auth/status")
def auth_status(request: Request):
    client_ip = _extract_client_ip(request)
    session = auth_manager.verify_session_token(request.cookies.get(COOKIE_NAME))
    return {
        "authenticated": bool(session),
        "setup_complete": auth_manager.is_setup_complete,
        "is_local": is_local_request(client_ip),
        "username": session.get("u") if session else None,
    }


@app.post("/api/auth/setup")
def auth_setup(payload: Dict[str, Any], request: Request, response: Response):
    client_ip = _extract_client_ip(request)
    setup_code = str(payload.get("setup_code", "")).strip()
    allowed, err_msg = auth_manager.verify_setup_permission(client_ip, setup_code)
    if not allowed:
        raise HTTPException(status_code=403, detail=err_msg)

    username = str(payload.get("username", "admin")).strip() or "admin"
    password = str(payload.get("password", ""))
    ok, pw_err = auth_manager.set_password(password, username=username)
    if not ok:
        raise HTTPException(status_code=400, detail=pw_err)

    _set_session_cookie(response, username)
    return {"success": True, "username": username}


@app.post("/api/auth/login")
def auth_login(payload: Dict[str, Any], request: Request, response: Response):
    client_ip = _extract_client_ip(request)
    allowed, retry_after = auth_manager.check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"יותר מדי ניסיונות כניסה כושלים. נסה שוב בעוד {retry_after} שניות.",
        )

    username = str(payload.get("username", "admin")).strip()
    password = str(payload.get("password", ""))
    if not auth_manager.verify_credentials(username, password):
        auth_manager.record_failure(client_ip)
        raise HTTPException(status_code=401, detail="שם משתמש או סיסמה שגויים")

    auth_manager.clear_failures(client_ip)
    _set_session_cookie(response, username)
    return {"success": True, "username": username}


@app.post("/api/auth/logout")
def auth_logout(response: Response):
    response.delete_cookie(key=COOKIE_NAME, path="/")
    return {"success": True}


@app.post("/api/auth/change_password")
def auth_change_password(payload: Dict[str, Any], request: Request, response: Response):
    current_password = str(payload.get("current_password", ""))
    new_password = str(payload.get("new_password", ""))
    username = getattr(request.state, "user", "admin")

    if not auth_manager.verify_credentials(username, current_password):
        raise HTTPException(status_code=401, detail="הסיסמה הנוכחית שגויה")

    ok, err = auth_manager.set_password(new_password, username=username)
    if not ok:
        raise HTTPException(status_code=400, detail=err)

    _set_session_cookie(response, username)
    return {"success": True, "message": "הסיסמה עודכנה בהצלחה"}


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
    return {"success": True, "channel": new_ch}


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
        frame = multi_camera_engine.get_snapshot_frame(channel)
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
    client_ip = _extract_client_ip(websocket)
    if auth_manager.lan_only and not is_private_network_ip(client_ip):
        await websocket.close(code=1008)
        return

    session = auth_manager.verify_session_token(websocket.cookies.get(COOKIE_NAME))
    if not session:
        await websocket.close(code=1008)
        return

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


# --- Settings & RTSP Connection Test (Secrets Masked) ---

def _masked_settings_copy(settings: SystemSettings) -> SystemSettings:
    copy = settings.model_copy(deep=True)
    copy.ossia_pass = mask_secret(copy.ossia_pass)
    copy.rtsp_url = mask_rtsp_url(copy.rtsp_url)
    return copy


@app.get("/api/settings", response_model=SystemSettings)
def get_settings():
    return _masked_settings_copy(cv_engine.settings)


@app.post("/api/settings", response_model=SystemSettings)
def update_settings(new_settings: SystemSettings):
    # Preserve existing password if client submitted masked placeholder or blank when masked
    if is_masked_value(new_settings.ossia_pass) or (new_settings.ossia_pass == "" and cv_engine.settings.ossia_pass):
        new_settings.ossia_pass = cv_engine.settings.ossia_pass

    new_settings.rtsp_url = _restore_masked_rtsp_password(new_settings.rtsp_url, new_settings.ossia_pass or "")

    # Validate inputs against SSRF / path traversal
    if new_settings.video_source not in ("sample", "rtsp", "file"):
        raise HTTPException(status_code=400, detail="מקור וידאו לא חוקי")

    if new_settings.video_source == "rtsp" or new_settings.rtsp_url:
        ok, err = validate_rtsp_url(new_settings.rtsp_url)
        if not ok:
            raise HTTPException(status_code=400, detail=err)

    if new_settings.video_source == "file":
        ok, err = validate_video_file_path(new_settings.video_file_path, BASE_DIR)
        if not ok:
            raise HTTPException(status_code=400, detail=err)

    if not validate_model_name(new_settings.model_name):
        raise HTTPException(status_code=400, detail="שם קובץ מודל AI אינו תקין")

    save_settings(new_settings)
    cv_engine.update_settings(new_settings)
    return _masked_settings_copy(cv_engine.settings)


@app.post("/api/settings/test_rtsp")
def test_rtsp_connection(payload: Dict[str, str]):
    rtsp_url = payload.get("rtsp_url", "").strip()
    if not rtsp_url:
        raise HTTPException(status_code=400, detail="rtsp_url is required")

    rtsp_url = _restore_masked_rtsp_password(rtsp_url, cv_engine.settings.ossia_pass or "")
    ok, err = validate_rtsp_url(rtsp_url)
    if not ok:
        raise HTTPException(status_code=400, detail=err)

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


# --- Telegram & Firebase Integrations Endpoints (Secrets Masked) ---

@app.get("/api/settings/telegram")
def get_telegram_settings():
    from backend.telegram_alerts import telegram_alerts
    return {
        "bot_token": mask_secret(telegram_alerts.bot_token),
        "has_token": bool(telegram_alerts.bot_token),
        "chat_id": telegram_alerts.chat_id,
        "enabled": telegram_alerts.enabled
    }


@app.post("/api/settings/telegram/detect_chat")
def detect_telegram_chat(payload: Dict[str, Any]):
    from backend.telegram_alerts import telegram_alerts
    incoming_token = str(payload.get("bot_token", "")).strip()
    token = telegram_alerts.bot_token if (not incoming_token or is_masked_value(incoming_token)) else incoming_token
    return telegram_alerts.detect_latest_chat(token)


@app.post("/api/settings/telegram")
def save_telegram_settings(payload: Dict[str, Any]):
    from backend.telegram_alerts import telegram_alerts
    incoming_token = str(payload.get("bot_token", "")).strip()
    bot_token = telegram_alerts.bot_token if (not incoming_token or is_masked_value(incoming_token)) else incoming_token
    chat_id = str(payload.get("chat_id", "")).strip()
    enabled = bool(payload.get("enabled", True))

    telegram_alerts.save_config(bot_token, chat_id, enabled)
    test_ok = False
    if bot_token and chat_id and payload.get("send_test", True):
        test_ok = telegram_alerts._post_telegram("🤖 *חיבור ל-VisionDine AI בוצע בהצלחה!*\nמערכת התראות שולחנות מוכנה לשליחת הודעות.")
    return {"success": True, "test_sent": test_ok}


@app.get("/api/settings/firebase")
def get_firebase_status():
    from backend.firebase_sync import firebase_manager
    return {
        "configured": firebase_manager.is_configured
    }


@app.post("/api/system/git_update")
def system_git_update():
    """Pull latest updates from GitHub origin/main and trigger watchdog restart."""
    if not (BASE_DIR / ".git").exists():
        raise HTTPException(
            status_code=400,
            detail="תיקיית הפרויקט במחשב זה לא חוברה עדיין ל-Git (הורדה כ-ZIP). חבר אותה פעם אחת ל-Git כדי לאפשר עדכון בלחיצה.",
        )

    try:
        res = subprocess.run(
            ["git", "pull", "origin", "main"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
        if res.returncode != 0:
            raise HTTPException(status_code=500, detail=f"שגיאת Git: {out.strip()}")

        if "Already up to date." in res.stdout:
            return {"success": True, "updated": False, "message": "השרת כבר מעודכן לגרסה האחרונה!", "output": out.strip()}

        def _delayed_exit():
            import time
            time.sleep(1.5)
            os._exit(0)  # Watchdog loop in start_24_7.bat restarts cleanly in 5s

        threading.Thread(target=_delayed_exit, daemon=True).start()
        return {
            "success": True,
            "updated": True,
            "message": "הגרסה החדשה ירדה בהצלחה! השרת מתאתחל כעת (יחזור תוך ~6 שניות).",
            "output": out.strip(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"עדכון נכשל: {e}")


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
    safe_limit = max(1, min(500, limit))
    return get_recent_events(limit=safe_limit)


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
