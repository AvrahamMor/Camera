import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    HAS_FIREBASE = True
except ImportError:
    HAS_FIREBASE = False

BASE_DIR = Path(__file__).resolve().parent.parent
CREDENTIALS_PATH = BASE_DIR / "config" / "firebase_credentials.json"

class FirebaseManager:
    def __init__(self):
        self.app = None
        self.db = None
        self.is_configured = False
        self._init_firebase()

    def _init_firebase(self):
        if not HAS_FIREBASE:
            print("[Firebase] firebase-admin package is not installed.")
            return

        if not CREDENTIALS_PATH.exists():
            print("[Firebase] No credentials file found at config/firebase_credentials.json. Operating in local mode.")
            return

        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(str(CREDENTIALS_PATH))
                self.app = firebase_admin.initialize_app(cred)
            else:
                self.app = firebase_admin.get_app()

            self.db = firestore.client()
            self.is_configured = True
            print("[Firebase] Successfully connected to Firebase Cloud Firestore!")
        except Exception as e:
            print(f"[Firebase] Initialization error: {e}")
            self.is_configured = False

    def sync_live_state(self, state: Dict[str, Any]):
        """Sync live restaurant floor state and active alerts to Firebase Firestore."""
        if not self.is_configured or not self.db:
            return

        try:
            doc_ref = self.db.collection("restaurants").document("main_branch").collection("live").document("state")
            doc_ref.set({
                "timestamp": firestore.SERVER_TIMESTAMP,
                "kpis": state.get("kpis", {}),
                "active_diners": state.get("kpis", {}).get("active_diners", 0),
                "pending_alerts_count": state.get("kpis", {}).get("pending_alerts_count", 0),
                "tables": state.get("tables", []),
                "alerts": state.get("alerts", [])
            }, merge=True)
        except Exception as e:
            print(f"[Firebase Sync Error] sync_live_state: {e}")

    def log_session(self, session_data: Dict[str, Any]):
        """Log completed dining session to Cloud Firestore."""
        if not self.is_configured or not self.db:
            return

        try:
            self.db.collection("restaurants").document("main_branch").collection("sessions").add(session_data)
        except Exception as e:
            print(f"[Firebase Sync Error] log_session: {e}")

    def log_event(self, event_data: Dict[str, Any]):
        """Log table occupancy/service event to Cloud Firestore."""
        if not self.is_configured or not self.db:
            return

        try:
            self.db.collection("restaurants").document("main_branch").collection("events").add(event_data)
        except Exception as e:
            print(f"[Firebase Sync Error] log_event: {e}")

# Global singleton
firebase_manager = FirebaseManager()
