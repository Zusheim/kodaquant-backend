# core/rate_limiter.py
"""
Rate limiter en memoria (sliding window) para mitigar fuerza bruta en
/register y /login. Sin dependencias externas, sin estado compartido entre
procesos -- válido para un solo worker. Si se escala a múltiples
workers/instancias, migrar la key a Redis (INCR + EXPIRE) manteniendo la
misma firma de `enforce_rate_limit`.
"""
import time
from collections import defaultdict
from threading import Lock

from fastapi import HTTPException, Request, status

from core.config import settings

_attempts: dict[str, list[float]] = defaultdict(list)
_lock = Lock()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def enforce_rate_limit(request: Request, bucket: str) -> None:
    """Lanza 429 si `bucket:ip` excede el límite en la ventana configurada."""
    key = f"{bucket}:{_client_ip(request)}"
    now = time.time()
    window = settings.RATE_LIMIT_WINDOW_SECONDS
    max_attempts = settings.RATE_LIMIT_MAX_ATTEMPTS

    with _lock:
        timestamps = [t for t in _attempts[key] if now - t < window]
        if len(timestamps) >= max_attempts:
            retry_after = int(window - (now - timestamps[0]))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Demasiados intentos. Intenta de nuevo más tarde.",
                headers={"Retry-After": str(max(retry_after, 1))},
            )
        timestamps.append(now)
        _attempts[key] = timestamps