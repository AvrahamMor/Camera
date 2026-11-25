import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "telegram_config.json"


def is_valid_bot_token(token: str) -> bool:
    """Validate Telegram bot token format strictly (prevents SSRF / malformed URLs)."""
    return bool(re.fullmatch(r"\d{6,15}:[A-Za-z0-9_-]{20,64}", (token or "").strip()))


class TelegramAlertManager:
    def __init__(self):
        self.bot_token = ""
        self.chat_id = ""
        self.enabled = False
        self._lock = threading.RLock()
        self._poll_running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._last_update_id = 0

        # Runtime callbacks wired from app startup
        self._state_fn: Optional[Callable[[], Dict[str, Any]]] = None
        self._ack_fn: Optional[Callable[[str], bool]] = None
        self._snapshot_fn: Optional[Callable[[], Optional[bytes]]] = None

        self.load_config()

    def load_config(self):
        with self._lock:
            if CONFIG_PATH.exists():
                try:
                    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        self.bot_token = cfg.get("bot_token", "").strip()
                        self.chat_id = str(cfg.get("chat_id", "")).strip()
                        self.enabled = bool(self.bot_token and self.chat_id and cfg.get("enabled", True))
                except Exception as e:
                    print(f"[Telegram] Failed to load config: {e}")

    def save_config(self, bot_token: str, chat_id: str, enabled: bool = True):
        with self._lock:
            self.bot_token = bot_token.strip()
            self.chat_id = str(chat_id).strip()
            self.enabled = bool(self.bot_token and self.chat_id and enabled)
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "bot_token": self.bot_token,
                    "chat_id": self.chat_id,
                    "enabled": self.enabled
                }, f, indent=2)

    def bind_handlers(
        self,
        state_fn: Callable[[], Dict[str, Any]],
        ack_fn: Callable[[str], bool],
        snapshot_fn: Callable[[], Optional[bytes]],
    ):
        self._state_fn = state_fn
        self._ack_fn = ack_fn
        self._snapshot_fn = snapshot_fn

    def start_polling(self):
        if self._poll_running:
            return
        self._poll_running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        print("[Telegram] Interactive 2-way bot listener started.")

    def stop_polling(self):
        self._poll_running = False
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)

    def detect_latest_chat(self, candidate_token: str = "") -> Dict[str, Any]:
        """Query Telegram getUpdates to find the latest user/group chat_id automatically."""
        token = (candidate_token or self.bot_token).strip()
        if not is_valid_bot_token(token):
            return {"found": False, "detail": "פורמט ה-Bot Token אינו תקין"}

        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params={"limit": 20, "timeout": 1},
                timeout=6,
            )
            data = resp.json()
            if not data.get("ok"):
                return {"found": False, "detail": data.get("description", "Token שגוי או לא פעיל")}

            results = data.get("result", [])
            for upd in reversed(results):
                msg = upd.get("message") or upd.get("edited_message") or upd.get("channel_post")
                if not msg and "callback_query" in upd:
                    msg = upd["callback_query"].get("message")
                if not msg and "my_chat_member" in upd:
                    msg = upd["my_chat_member"]

                if msg and "chat" in msg:
                    chat = msg["chat"]
                    chat_id = str(chat["id"])
                    label = chat.get("title") or chat.get("username") or chat.get("first_name") or chat_id
                    return {"found": True, "chat_id": chat_id, "chat_label": label}

            return {
                "found": False,
                "detail": "לא נמצאה הודעה עדיין. היכנס לבוט בטלגרם, לחץ Start ושלח הודעה אחת (למשל 'היי'), ואז לחץ שוב על זיהוי.",
            }
        except Exception as e:
            return {"found": False, "detail": f"שגיאת חיבור לטלגרם: {e}"}

    def send_alert(self, table_name: str, elapsed_str: str, headcount: int = 1, table_id: str = "") -> bool:
        """Send urgent service alert with inline Acknowledge button."""
        with self._lock:
            enabled, token, chat_id = self.enabled, self.bot_token, self.chat_id
        if not enabled or not token or not chat_id:
            return False

        message = (
            f"🔔 *התראת שירות דחופה!*\n"
            f"📍 *שולחן:* {table_name}\n"
            f"👥 *סועדים:* {headcount}\n"
            f"⏳ *ממתינים למלצר כבר:* `{elapsed_str}` דקות\n\n"
            f"⚡ _נא לגשת בהקדם!_"
        )

        reply_markup = None
        if table_id:
            reply_markup = {
                "inline_keyboard": [[
                    {"text": "✅ טופל / קיבל שירות", "callback_data": f"ack:{table_id}"}
                ]]
            }

        return self._post_telegram(message, reply_markup=reply_markup)

    def _post_telegram(self, text: str, chat_id_override: Optional[str] = None, reply_markup: Optional[Dict[str, Any]] = None) -> bool:
        with self._lock:
            token = self.bot_token
            target_chat = chat_id_override or self.chat_id
        if not is_valid_bot_token(token) or not target_chat:
            return False

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload: Dict[str, Any] = {
            "chat_id": target_chat,
            "text": text,
            "parse_mode": "Markdown",
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup

        try:
            resp = requests.post(url, json=payload, timeout=5)
            return resp.status_code == 200
        except Exception as e:
            print(f"[Telegram Alert Error] {e}")
            return False

    def send_photo(self, photo_bytes: bytes, caption: str = "", chat_id_override: Optional[str] = None) -> bool:
        with self._lock:
            token = self.bot_token
            target_chat = chat_id_override or self.chat_id
        if not is_valid_bot_token(token) or not target_chat or not photo_bytes:
            return False

        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendPhoto",
                data={"chat_id": target_chat, "caption": caption},
                files={"photo": ("snapshot.jpg", photo_bytes, "image/jpeg")},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception as e:
            print(f"[Telegram Photo Error] {e}")
            return False

    def _poll_loop(self):
        """Lightweight long-polling loop for authorized commands (/status, /photo, inline acks)."""
        while self._poll_running:
            with self._lock:
                enabled = self.enabled
                token = self.bot_token
                allowed_chat = self.chat_id

            if not enabled or not is_valid_bot_token(token) or not allowed_chat:
                time.sleep(3.0)
                continue

            try:
                resp = requests.get(
                    f"https://api.telegram.org/bot{token}/getUpdates",
                    params={"offset": self._last_update_id + 1, "timeout": 15},
                    timeout=20,
                )
                if resp.status_code != 200:
                    time.sleep(4.0)
                    continue

                data = resp.json()
                for upd in data.get("result", []):
                    self._last_update_id = max(self._last_update_id, upd.get("update_id", 0))
                    self._handle_update(upd, token, allowed_chat)
            except Exception:
                time.sleep(4.0)

    def _handle_update(self, upd: Dict[str, Any], token: str, allowed_chat: str):
        # 1. Handle Inline Button callback ("ack:<table_id>")
        if "callback_query" in upd:
            cq = upd["callback_query"]
            msg = cq.get("message") or {}
            chat_id = str((msg.get("chat") or {}).get("id", ""))
            if chat_id != allowed_chat:
                return
            data = cq.get("data", "")
            if data.startswith("ack:") and self._ack_fn:
                table_id = data.split(":", 1)[1]
                ok = self._ack_fn(table_id)
                requests.post(
                    f"https://api.telegram.org/bot{token}/answerCallbackQuery",
                    json={"callback_query_id": cq["id"], "text": "אושר שירות!" if ok else "השולחן לא נמצא"},
                    timeout=4,
                )
                if ok and msg.get("message_id"):
                    orig_text = msg.get("text", "")
                    requests.post(
                        f"https://api.telegram.org/bot{token}/editMessageText",
                        json={
                            "chat_id": chat_id,
                            "message_id": msg["message_id"],
                            "text": f"{orig_text}\n\n✅ *טופל ואושר מהנייד!*",
                            "parse_mode": "Markdown",
                        },
                        timeout=4,
                    )
            return

        # 2. Handle Text Commands strictly from authorized chat_id
        msg = upd.get("message")
        if not msg:
            return
        chat_id = str((msg.get("chat") or {}).get("id", ""))
        if chat_id != allowed_chat:
            return

        text = (msg.get("text") or "").strip().lower()
        if text in ("/start", "/help", "עזרה"):
            self._post_telegram(
                "🤖 *VisionDine AI — בוט ניהול מסעדה*\n\n"
                "פקודות זמינות:\n"
                "• `/status` — דוח תפוסת שולחנות וסועדים בזמן אמת\n"
                "• `/photo` — צילום חי ממצלמת המסעדה\n\n"
                "בנוסף, בכל התראת שירות תוכלו ללחוץ ישירות על *✅ טופל*.",
                chat_id_override=chat_id,
            )
        elif text in ("/status", "סטטוס"):
            if not self._state_fn:
                return
            st = self._state_fn()
            kpis = st.get("kpis", {})
            occupied = [t for t in st.get("tables", []) if t.get("status") in ("OCCUPIED", "NEEDS_SERVICE")]
            lines = [
                "📊 *סטטוס מסעדה בזמן אמת*",
                f"🪑 תפוסים: *{kpis.get('occupied_tables', 0)}/{kpis.get('total_tables', 0)}*",
                f"👥 סועדים כרגע: *{kpis.get('active_diners', 0)}*",
                f"🔔 ממתינים לשירות: *{kpis.get('pending_alerts_count', 0)}*",
            ]
            if occupied:
                lines.append("\n*שולחנות פעילים:*")
                for t in occupied[:12]:
                    icon = "🟡" if t.get("status") == "NEEDS_SERVICE" else "🔴"
                    lines.append(f"{icon} {t.get('name')} — {t.get('headcount')} סועדים ({t.get('elapsed_formatted')})")
            self._post_telegram("\n".join(lines), chat_id_override=chat_id)
        elif text in ("/photo", "תמונה"):
            if self._snapshot_fn:
                img = self._snapshot_fn()
                if img:
                    self.send_photo(img, caption="📸 צילום חי מ-VisionDine AI", chat_id_override=chat_id)
                else:
                    self._post_telegram("⚠️ המצלמה אינה זמינה כרגע.", chat_id_override=chat_id)


telegram_alerts = TelegramAlertManager()
