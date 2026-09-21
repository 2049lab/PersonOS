"""AK/SK request signing between services (a self-hosted AK/SK scheme following common
internal security guidance).

- Algorithm: HMAC-SHA256 (signature in hex). The guidance doesn't mandate one algorithm;
  we fix it to this.
- Request headers: X-Access-Key (the AK, in the clear) / X-Timestamp (unix seconds) /
  X-Nonce / X-Signature / X-Signed-Headers.
- The canonical string covers: METHOD + path + sorted query + the signed headers + the
  SHA-256 of the body + timestamp + nonce.
- The "minimum set" of headers that must be signed: content-type, x-user-token — if they
  appear in the request (non-empty) they must be part of the signature, so they can't be
  stripped or tampered with.
- Server-side flow: all fields present -> timestamp within the window -> look up the AK
  -> nonce replay check -> rebuild the canonical string -> constant-time comparison. Any
  failure is reported to the caller as a single "authentication failed"; the real reason
  goes to the internal log.
- **Unconditionally enforced**: no env bypass of any kind (which avoids the security
  smell of "one environment variable turns auth off"); tests bypass it explicitly with
  dependency_overrides.
- The AK -> SK map is read from the PERSONOS_AKSK_MAP environment variable (JSON) and
  cached; an SK never leaves the process and is never persisted.

Server-side verification and caller-side signing share canonical_string/sign, so the two
sides agree by construction (see sign_request).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time

from fastapi import HTTPException, Request
from loguru import logger

from personos.config import get_secret, settings

_WINDOW = int(os.environ.get("PERSONOS_SIGN_WINDOW_S", "300"))   # Timestamp window (seconds)
_MIN_SIGNED = ("content-type", "x-user-token")                  # Minimum set of headers to sign (if present, they must be signed)

_aksk_cache: dict | None = None
_mem_nonce: dict[str, float] = {}                               # In-memory fallback when there is no Redis (single replica / tests)


class _SigError(Exception):
    """Signature verification failed (the internal reason; externally it is always
    folded into a 401 authentication failure)."""


# -- Shared: the canonical string and the signature (used by both the server and the
# caller, so the two can't drift apart) --

def canonical_string(method: str, path: str, query_items, signed_headers: dict,
                     body: bytes, timestamp: str, nonce: str) -> str:
    """Canonicalize a request into the string to be signed. query_items is a list of
    (k, v) pairs; signed_headers maps header name -> value."""
    q = "&".join(f"{k}={v}" for k, v in sorted(query_items or []))
    hb = "\n".join(f"{k}:{(v or '').strip()}" for k, v in sorted(signed_headers.items()))
    body_hash = hashlib.sha256(body or b"").hexdigest()
    return "\n".join([method.upper(), path, q, hb, body_hash, str(timestamp), nonce])


def sign(sk: str, canonical: str) -> str:
    return hmac.new(sk.encode(), canonical.encode(), hashlib.sha256).hexdigest()


def sign_request(ak: str, sk: str, method: str, path: str, *,
                 query_items=None, headers: dict | None = None, body: bytes = b"") -> dict:
    """For callers: compute the signature headers a request should carry. Pass the
    headers you are actually sending (at least content-type / x-user-token)."""
    ts = str(int(time.time()))
    nonce = secrets.token_hex(16)
    hdrs = {k.lower(): v for k, v in (headers or {}).items()}
    signed = sorted(h for h in _MIN_SIGNED if hdrs.get(h))
    canon = canonical_string(method, path, list(query_items or []),
                             {h: hdrs.get(h, "") for h in signed}, body, ts, nonce)
    return {"X-Access-Key": ak, "X-Timestamp": ts, "X-Nonce": nonce,
            "X-Signature": sign(sk, canon), "X-Signed-Headers": ",".join(signed)}


# -- Server side: AK -> SK, nonce, verification --

def _load_aksk() -> dict:
    """The AK -> SK map (JSON), read from the secret store or env. The SK stays in
    memory and is never leaked.

    Only a non-empty result is cached: if the credentials haven't been provisioned yet
    on the first load (empty), we don't cache and retry next time — so once the
    credentials are in place it heals without a restart (this avoids "start the service
    first, fill in the map later" being permanently stuck on a cached empty map).
    """
    global _aksk_cache
    if _aksk_cache:                                # Use the cache only if it's non-empty
        return _aksk_cache
    raw = get_secret("PERSONOS_AKSK_MAP", required=False)
    try:
        m = json.loads(raw) if raw else {}
    except Exception:                              # noqa: BLE001  Malformed -> treat as no credentials (reject everything)
        logger.error("failed to parse PERSONOS_AKSK_MAP, every signature will be rejected")
        m = {}
    if m:
        _aksk_cache = m                            # Cache only once we got something non-empty
    return m


def _reset_cache() -> None:
    """For tests: clear the AK/SK and nonce caches."""
    global _aksk_cache
    _aksk_cache = None
    _mem_nonce.clear()


def _seen_nonce(ak: str, nonce: str, ttl: int) -> bool:
    """Atomically record AK+nonce; if it already exists this is a replay and we return
    True. Redis is used when configured (multi-replica), otherwise we fall back to
    memory."""
    if settings.redis_cluster:
        try:
            from personos.storage.redis_client import get_redis, key
            ok = get_redis().set(key("nonce", ak, nonce), "1", nx=True, ex=ttl)
            return not ok
        except Exception:                          # noqa: BLE001  Redis failed -> fall back to memory (best effort)
            pass
    now = time.time()
    for k, exp in list(_mem_nonce.items()):
        if exp < now:
            _mem_nonce.pop(k, None)
    mk = f"{ak}:{nonce}"
    if mk in _mem_nonce:
        return True
    _mem_nonce[mk] = now + ttl
    return False


def _verify(method: str, path: str, query_items, headers, body: bytes) -> str:
    """Verify one request and return the caller's AK; any failed step raises
    _SigError."""
    ak = headers.get("x-access-key", "")
    ts = headers.get("x-timestamp", "")
    nonce = headers.get("x-nonce", "")
    sig = headers.get("x-signature", "")
    signed_raw = headers.get("x-signed-headers", "")   # May be empty (when none of the minimum-set headers are present), so it isn't required
    if not (ak and ts and nonce and sig):
        raise _SigError("missing signature headers")
    try:
        tsi = int(ts)
    except ValueError:
        raise _SigError("invalid timestamp")
    if abs(time.time() - tsi) > _WINDOW:
        raise _SigError("timestamp outside the allowed window")
    signed = sorted({h.strip().lower() for h in signed_raw.split(",") if h.strip()})
    # If a security-relevant header is present in the request it must be part of the
    # signature, so an attacker can't strip it and replay the request.
    for h in _MIN_SIGNED:
        if headers.get(h) and h not in signed:
            raise _SigError(f"required signed header {h} was not included in the signature")
    sk = _load_aksk().get(ak)
    if not sk:
        raise _SigError(f"unknown or disabled AK: {ak}")
    if _seen_nonce(ak, nonce, _WINDOW):
        raise _SigError("replayed nonce")
    canon = canonical_string(method, path, query_items,
                             {h: headers.get(h, "") for h in signed}, body, ts, nonce)
    if not hmac.compare_digest(sign(sk, canon), sig):
        raise _SigError("signature mismatch")
    return ak


async def verify_signature(request: Request) -> None:
    """FastAPI dependency: mounted on the /api/v1 router, it verifies **every** request
    unconditionally (no env bypass of any kind).

    When a test needs to bypass it, override it explicitly through FastAPI's own
    mechanism: app.dependency_overrides[verify_signature] = lambda: None.
    """
    body = await request.body()                    # Starlette caches it, so later Pydantic parsing is unaffected
    try:
        ak = _verify(request.method, request.url.path,
                     list(request.query_params.multi_items()), request.headers, body)
    except _SigError as e:
        logger.warning(f"AK/SK signature verification failed path={request.url.path}: {e}")   # Log the real reason internally
        raise HTTPException(status_code=401, detail="authentication failed")          # One message for everyone, never revealing whether the AK exists
    request.state.caller_ak = ak
