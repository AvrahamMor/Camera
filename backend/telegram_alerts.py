import urllib.request
import urllib.parse
import json
from typing import Optional
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "telegram_config.json"

class TelegramAlertManager:
    def __init__(self):
        self.bot_token = ""
        self.chat_id = ""
        self.enabled = False
        self.load_config()

    def load_config(self):
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
        self.bot_token = bot_token.strip()
        self.chat_id = str(chat_id).strip()
        self.enabled = bool(self.bot_token and self.chat_id and enabled)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "bot_token": self.bot_token,
                "chat_id": self.chat_id,
                "enabled": self.enabled
            }, f, indent=2)

    def send_alert(self, table_name: str, elapsed_str: str, headcount: int = 1):
        """Send urgent service alert to restaurant waiters/manager Telegram chat."""
        if not self.enabled or not self.bot_token or not self.chat_id:
            return False

        message = (
            f"🔔 *התראת שירות דחופה!*\n"
            f"📍 *שולחן:* {table_name}\n"
            f"👥 *סועדים:* {headcount}\n"
            f"⏳ *ממתינים למלצר כבר:* `{elapsed_str}` דקות\n\n"
            f"⚡ _נא לגשת בהקדם!_"
        )

        return self._post_telegram(message)

    def _post_telegram(self, text: str) -> bool:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"[Telegram Alert Error] {e}")
            return False

telegram_alerts = TelegramAlertManager()
