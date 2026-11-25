import base64
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

BASE_DIR = Path(__file__).resolve().parent.parent
AUTH_FILE = BASE_DIR / "config" / "auth_config.json"

COOKIE_NAME = "visiondine_session"
MASK_PLACEHOLDER = "********"
PBKDF2_ITERATIONS = 310_000
SESSION_TTL_SECONDS = 12 * 3600  # 12 hours

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 15 * 60  # 15 minutes


def is_local_request(client_host: Optional[str]) -> bool:
    """Return True if the client IP is loopback (127.0.0.1 / ::1)."""
    if not client_host:
        return False
    if client_host in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        ip = ipaddress.ip_address(client_host)
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        return ip.is_loopback
    except ValueError:
        return False


def is_private_network_ip(client_host: Optional[str]) -> bool:
    """Allow loopback, RFC1918 LAN (192.168.x.x, 10.x.x.x, 172.16-31.x.x), and Tailscale CGNAT (100.64.0.0/10)."""
    if not client_host:
        return False
    if client_host == "testclient":
        return True
    try:
        ip = ipaddress.ip_address(client_host)
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if ip.is_loopback or ip.is_private or ip.is_link_local:
            return True
        if isinstance(ip, ipaddress.IPv4Address) and ip in ipaddress.ip_network("100.64.0.0/10"):
            return True
        return False
    except ValueError:
        return False


def mask_secret(value: Optional[str]) -> str:
    """Return a fixed mask if secret is set, else empty string."""
    return MASK_PLACEHOLDER if value else ""


def is_masked_value(value: Optional[str]) -> bool:
    """Check whether a value is the placeholder mask (unchanged by client)."""
    return value is not None and value.strip() == MASK_PLACEHOLDER


def mask_rtsp_url(rtsp_url: str) -> str:
    """Replace embedded password in rtsp://user:pass@host/... with ********."""
    if not rtsp_url:
        return ""
    try:
        parts = urlsplit(rtsp_url)
        if parts.password:
            user = parts.username or ""
            host = parts.hostname or ""
            port = f":{parts.port}" if parts.port else ""
            netloc = f"{user}:{MASK_PLACEHOLDER}@{host}{port}"
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        pass
    return rtsp_url


def validate_rtsp_url(rtsp_url: str) -> Tuple[bool, str]:
    """Validate that an RTSP URL uses rtsp:// or rtsps:// and has no control chars (prevents SSRF/file read)."""
    if not rtsp_url or any(c in rtsp_url for c in ("\r", "\n", "\x00")):
        return False, "כתובת RTSP אינה תקינה"
    try:
        parts = urlsplit(rtsp_url.strip())
    except Exception:
        return False, "פורמט כתובת RTSP שגוי"

    if parts.scheme.lower() not in ("rtsp", "rtsps"):
        return False, "מותר להזין כתובת בפרוטוקול rtsp:// או rtsps:// בלבד"
    if not parts.hostname:
        return False, "חסרה כתובת שרת/IP ב-RTSP"
    return True, ""


def validate_video_file_path(file_path: str, base_dir: Path = BASE_DIR) -> Tuple[bool, str]:
    """Ensure local video file stays strictly inside sample_videos/ (prevents path traversal)."""
    if not file_path:
        return False, "נתיב קובץ וידאו ריק"
    allowed_dir = (base_dir / "sample_videos").resolve()
    try:
        candidate = Path(file_path)
        resolved = candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()
        if not resolved.is_relative_to(allowed_dir):
            return False, "מותר להשתמש בקובצי וידאו מתוך תיקיית sample_videos בלבד"
    except Exception:
        return False, "נתיב קובץ וידאו אינו תקין"
    return True, ""


def validate_model_name(model_name: str) -> bool:
    """Allowlist safe YOLO weight filenames without directory traversal."""
    return bool(re.fullmatch(r"[a-zA-Z0-9_.-]+\.(pt|onnx)", (model_name or "").strip()))


class AuthManager:
    """Handles password hashing, signed session cookies, rate limiting, and setup."""

    def __init__(self, auth_file: Path = AUTH_FILE):
        self.auth_file = auth_file
        self._lock = threading.RLock()
        self._failed_attempts: Dict[str, list] = {}
        self._lockouts: Dict[str, float] = {}
        self.config: Dict[str, Any] = {}
        self._load_or_init()

    def _load_or_init(self) -> None:
        with self._lock:
            self.auth_file.parent.mkdir(parents=True, exist_ok=True)
            if self.auth_file.exists():
                try:
                    with open(self.auth_file, "r", encoding="utf-8") as f:
                        self.config = json.load(f)
                except Exception as e:
                    print(f"[Security] Failed reading auth_config.json: {e}")
                    self.config = {}

            changed = False
            if not self.config.get("secret_key"):
                self.config["secret_key"] = secrets.token_hex(32)
                changed = True
            if "username" not in self.config:
                self.config["username"] = "admin"
                changed = True
            if "password_hash" not in self.config:
                self.config["password_hash"] = ""
                changed = True
            if "salt" not in self.config:
                self.config["salt"] = ""
                changed = True
            if "lan_only" not in self.config:
                self.config["lan_only"] = True
                changed = True

            # Generate one-time setup code while password is unset (so remote LAN setup stays secure)
            if not self.config["password_hash"] and not self.config.get("setup_code"):
                setup_code = secrets.token_urlsafe(8)
                self.config["setup_code"] = setup_code
                changed = True
                print("=" * 72)
                print("[Security] FIRST-TIME ADMIN PASSWORD SETUP REQUIRED")
                print(f"[Security] Open http://localhost:8085 from this PC, OR use setup code: {setup_code}")
                print("=" * 72)

            if changed:
                self._save()

    def _save(self) -> None:
        self.auth_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.auth_file, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

    @property
    def is_setup_complete(self) -> bool:
        with self._lock:
            return bool(self.config.get("password_hash") and self.config.get("salt"))

    @property
    def lan_only(self) -> bool:
        with self._lock:
            return bool(self.config.get("lan_only", True))

    def _hash_password(self, password: str, salt_hex: str) -> str:
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            PBKDF2_ITERATIONS,
        )
        return dk.hex()

    def set_password(self, new_password: str, username: str = "admin") -> Tuple[bool, str]:
        if len(new_password) < 8:
            return False, "הסיסמה חייבת להכיל לפחות 8 תווים"
        clean_user = re.sub(r"[^a-zA-Z0-9_.-]", "", username.strip()) or "admin"
        salt_hex = secrets.token_hex(16)
        pw_hash = self._hash_password(new_password, salt_hex)
        with self._lock:
            self.config["username"] = clean_user
            self.config["salt"] = salt_hex
            self.config["password_hash"] = pw_hash
            self.config.pop("setup_code", None)
            # Rotate signing key on password change to invalidate old sessions
            self.config["secret_key"] = secrets.token_hex(32)
            self._save()
        return True, ""

    def verify_setup_permission(self, client_host: Optional[str], setup_code: str = "") -> Tuple[bool, str]:
        with self._lock:
            if self.is_setup_complete:
                return False, "סיסמת המנהל כבר הוגדרה"
            if is_local_request(client_host):
                return True, ""
            expected_code = self.config.get("setup_code", "")
            if expected_code and setup_code and hmac.compare_digest(setup_code.strip(), expected_code):
                return True, ""
            return False, "להגדרה מרחוק יש להזין את קוד ההגדרה (setup_code) מקובץ config/auth_config.json או logs/server.log, או לפתוח http://localhost:8085 ישירות ממחשב המסעדה"

    def check_rate_limit(self, client_ip: str) -> Tuple[bool, int]:
        """Return (is_allowed, retry_after_seconds)."""
        now = time.time()
        with self._lock:
            locked_until = self._lockouts.get(client_ip, 0)
            if locked_until > now:
                return False, int(locked_until - now) + 1
            if client_ip in self._lockouts:
                del self._lockouts[client_ip]
            return True, 0

    def record_failure(self, client_ip: str) -> None:
        now = time.time()
        with self._lock:
            attempts = [ts for ts in self._failed_attempts.get(client_ip, []) if now - ts < 300]
            attempts.append(now)
            self._failed_attempts[client_ip] = attempts
            if len(attempts) >= MAX_LOGIN_ATTEMPTS:
                self._lockouts[client_ip] = now + LOCKOUT_SECONDS
                self._failed_attempts[client_ip] = []
                print(f"[Security] IP {client_ip} locked out for {LOCKOUT_SECONDS}s after {MAX_LOGIN_ATTEMPTS} failed logins.")

    def clear_failures(self, client_ip: str) -> None:
        with self._lock:
            self._failed_attempts.pop(client_ip, None)
            self._lockouts.pop(client_ip, None)

    def verify_credentials(self, username: str, password: str) -> bool:
        with self._lock:
            if not self.is_setup_complete:
                return False
            stored_user = self.config.get("username", "admin")
            stored_hash = self.config.get("password_hash", "")
            salt_hex = self.config.get("salt", "")

        if not hmac.compare_digest(username.strip(), stored_user):
            return False
        candidate_hash = self._hash_password(password, salt_hex)
        return hmac.compare_digest(candidate_hash, stored_hash)

    def create_session_token(self, username: str) -> str:
        now = int(time.time())
        payload = {"u": username, "iat": now, "exp": now + SESSION_TTL_SECONDS}
        payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode("ascii").rstrip("=")
        with self._lock:
            secret = bytes.fromhex(self.config["secret_key"])
        sig = hmac.new(secret, payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
        return f"{payload_b64}.{sig}"

    def verify_session_token(self, token: Optional[str]) -> Optional[Dict[str, Any]]:
        if not token or "." not in token:
            return None
        try:
            payload_b64, sig = token.split(".", 1)
            with self._lock:
                secret = bytes.fromhex(self.config["secret_key"])
            expected_sig = hmac.new(secret, payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expected_sig):
                return None
            padded = payload_b64 + "=" * (-len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
            if int(payload.get("exp", 0)) < int(time.time()):
                return None
            return payload
        except Exception:
            return None


auth_manager = AuthManager()
