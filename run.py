import argparse
import sys
import webbrowser
import uvicorn
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from backend.config import load_settings, save_settings
from backend.video_generator import generate_sample_restaurant_video, SAMPLE_VIDEO_PATH

def print_banner(host: str, port: int, source: str):
    banner = f"""
================================================================================
    VISIONDINE AI • REAL-TIME RESTAURANT TABLE OCCUPANCY & ANALYTICS
    Designed for Ossia VMS (Provision-ISR) & Computer Vision Edge Streams
================================================================================
  * Dashboard URL : http://localhost:{port}
  * Stream Feed   : http://localhost:{port}/api/video_feed
  * Video Source  : {source.upper()}
  * API Docs      : http://localhost:{port}/docs
================================================================================
    Press Ctrl+C to stop the server.
"""
    print(banner)

def main():
    parser = argparse.ArgumentParser(description="Launch VisionDine AI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8085, help="Bind port (default: 8085)")
    parser.add_argument("--source", type=str, choices=["sample", "rtsp", "file"], default=None, help="Override video source")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")
    args = parser.parse_args()

    # Load and optionally override settings
    settings = load_settings()
    if args.source:
        settings.video_source = args.source
    settings.server_host = args.host
    settings.server_port = args.port
    save_settings(settings)

    # Ensure sample video exists if in sample mode
    if settings.video_source == "sample" and not SAMPLE_VIDEO_PATH.exists():
        print("[Setup] Generating sample restaurant test video...")
        generate_sample_restaurant_video()

    print_banner(args.host, args.port, settings.video_source)

    if not args.no_browser:
        # Open browser in a separate thread after a short pause
        import threading
        import time
        def open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{args.port}")
        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(
        "backend.app:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
