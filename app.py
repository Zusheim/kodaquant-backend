import os

import uvicorn

# Bridge between the local dev command:
#   uvicorn main:app --reload --reload-exclude "venv*" --port 8080 --loop asyncio
# and the Docker/HF Spaces container. "main:app" is passed as an import
# string (not the app object) because uvicorn requires a string to spawn
# extra worker subprocesses when WEB_CONCURRENCY > 1.

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        # HF Spaces Docker routes traffic to the port declared as
        # app_port in the README frontmatter (7860). PORT is honored
        # too so the same image runs unmodified on platforms that
        # inject it (Render, Railway, etc.).
        port=int(os.getenv("PORT", "7860")),
        # Kept identical to the local dev command on purpose — do not
        # switch to uvloop without re-validating the TensorFlow/Keras
        # inference path under it first.
        loop="asyncio",
        # Defaults to 1 on purpose: main.py's `_radar_cache` is an
        # in-memory, process-local TTL cache. Raising WEB_CONCURRENCY
        # above 1 gives each worker its own cache (correct but wasteful
        # — duplicated Keras inference per worker) until that cache is
        # moved to a shared store (e.g. Redis).
        workers=int(os.getenv("WEB_CONCURRENCY", "1")),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )