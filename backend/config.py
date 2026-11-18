import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
TABLES_FILE = CONFIG_DIR / "tables_config.json"
DATABASE_FILE = BASE_DIR / "restaurant_analytics.db"

class SystemSettings(BaseModel):
    video_source: str = Field(default="sample", description="'sample', 'rtsp', or 'file'")
    rtsp_url: str = Field(default="rtsp://admin:123456@192.168.1.100:554/unicast/c1/s0/live")
    rtsp_transport: str = Field(default="tcp", description="'tcp' or 'udp'")
    video_file_path: str = Field(default="sample_videos/demo_restaurant.mp4")
    model_name: str = Field(default="yolov8n.pt", description="YOLO model path or weight name")
    confidence_threshold: float = Field(default=0.35, ge=0.1, le=1.0)
    target_fps: float = Field(default=5.0, ge=1.0, le=30.0, description="CV inference FPS throttle")
    stream_fps: float = Field(default=15.0, ge=5.0, le=30.0, description="MJPEG output streaming FPS")
    debounce_seconds: float = Field(default=4.0, ge=0.5, le=30.0, description="Debounce buffer time")
    service_alert_threshold_seconds: int = Field(default=180, ge=10, description="Seconds before service alert")
    reconnect_interval_seconds: float = Field(default=5.0, ge=1.0)
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8085)

    # Optional helper fields for Ossia VMS / Provision-ISR generator
    ossia_ip: Optional[str] = "192.168.1.100"
    ossia_port: Optional[int] = 554
    ossia_user: Optional[str] = "admin"
    ossia_pass: Optional[str] = "123456"
    ossia_channel: Optional[int] = 1
    ossia_stream: Optional[int] = 0

def load_settings() -> SystemSettings:
    """Load settings from JSON file or return defaults."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return SystemSettings(**data)
        except Exception as e:
            print(f"[Config] Error loading settings.json: {e}. Using defaults.")
    
    settings = SystemSettings()
    save_settings(settings)
    return settings

def save_settings(settings: SystemSettings) -> None:
    """Save settings back to JSON file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings.model_dump(), f, indent=2)

def build_ossia_rtsp_url(ip: str, port: int, user: str, password: str, channel: int = 1, stream: int = 0) -> str:
    """
    Build official RTSP URL for Provision-ISR / Ossia VMS.
    Channel: 1..N
    Stream: 0 (Main Stream), 1 (Sub Stream, Recommended for CV)
    """
    user_cred = f"{user}:{password}@" if user else ""
    stype = "main" if str(stream) in ("0", "main") else "sub"
    return f"rtsp://{user_cred}{ip}:{port}/chID={channel}&streamType={stype}&linkType=tcp"
