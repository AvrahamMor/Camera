import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.security import auth_manager


def main():
    print("=" * 60)
    print("  VisionDine AI - Admin Password Setup / Reset")
    print("=" * 60)
    username = input("Username [admin]: ").strip() or "admin"
    pw1 = getpass.getpass("New password (min 8 chars): ")
    pw2 = getpass.getpass("Confirm password: ")
    if pw1 != pw2:
        print("Error: Passwords do not match.")
        sys.exit(1)

    ok, err = auth_manager.set_password(pw1, username=username)
    if not ok:
        print(f"Error: {err}")
        sys.exit(1)

    print(f"Success! Admin password updated for '{username}'.")


if __name__ == "__main__":
    main()
