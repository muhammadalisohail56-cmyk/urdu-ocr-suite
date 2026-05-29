"""
auth.py
=======
Lightweight, dependency-free session auth for exactly two users.

Credentials live in environment variables `USER1_CREDENTIALS` and
`USER2_CREDENTIALS`, each formatted as ``username:password``. Sessions are
stateless HMAC-signed tokens (stdlib only) stored in an HttpOnly cookie, so no
session store is needed — perfect for a single-process Replit deployment.

Security notes
--------------
* Passwords are compared with `hmac.compare_digest` (constant time).
* Tokens are signed with `SESSION_SECRET`; set it in the environment for
  persistent sessions across restarts (otherwise a random per-process secret is
  used and everyone is logged out on restart).
* For stronger storage you may put a SHA-256 hex digest of the password after
  the colon and prefix it with ``sha256:`` — see `verify_login`.
"""

from __future__ import annotations

import os
import hmac
import json
import time
import base64
import hashlib
import secrets
from typing import Optional, Dict

from fastapi import Request, WebSocket

COOKIE_NAME = "ocr_session"
TOKEN_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(60 * 60 * 12)))  # 12h

# Persistent secret if provided, else random (sessions reset on restart).
_SECRET = (os.getenv("SESSION_SECRET") or secrets.token_hex(32)).encode("utf-8")


def _load_credentials() -> Dict[str, str]:
    """Parse the two credential env vars into {username: password_spec}."""
    creds: Dict[str, str] = {}
    for var in ("USER1_CREDENTIALS", "USER2_CREDENTIALS"):
        raw = os.getenv(var, "").strip()
        if not raw or ":" not in raw:
            continue
        username, _, password = raw.partition(":")
        username = username.strip()
        if username:
            creds[username] = password
    return creds


def verify_login(username: str, password: str) -> bool:
    """Constant-time credential check. Supports plaintext or ``sha256:<hex>``."""
    creds = _load_credentials()
    expected = creds.get(username)
    if expected is None:
        # Still do a dummy compare to reduce username-enumeration timing signal.
        hmac.compare_digest(password, password)
        return False
    if expected.startswith("sha256:"):
        digest = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(digest, expected[len("sha256:"):])
    return hmac.compare_digest(password, expected)


def _sign(payload_b64: str) -> str:
    sig = hmac.new(_SECRET, payload_b64.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode("ascii").rstrip("=")


def create_token(username: str) -> str:
    payload = {"u": username, "exp": int(time.time()) + TOKEN_TTL_SECONDS}
    payload_b64 = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return f"{payload_b64}.{_sign(payload_b64)}"


def verify_token(token: Optional[str]) -> Optional[str]:
    """Return the username for a valid, unexpired token, else None."""
    if not token or "." not in token:
        return None
    payload_b64, _, sig = token.partition(".")
    if not hmac.compare_digest(sig, _sign(payload_b64)):
        return None
    try:
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, json.JSONDecodeError):
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    return payload.get("u")


def current_user(request: Request) -> Optional[str]:
    """Username from the request's session cookie, or None."""
    return verify_token(request.cookies.get(COOKIE_NAME))


def current_user_ws(websocket: WebSocket) -> Optional[str]:
    """Username from a websocket's session cookie, or None."""
    return verify_token(websocket.cookies.get(COOKIE_NAME))
