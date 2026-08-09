import logging
import os
import time

import uvicorn

# Bridge between the local dev command:
#   uvicorn main:app --reload --reload-exclude "venv*" --port 8080 --loop asyncio
# and the Docker/HF Spaces container. "main:app" is passed as an import
# string (not the app object) because uvicorn requires a string to spawn
# extra worker subprocesses when WEB_CONCURRENCY > 1.

logger = logging.getLogger("kodaquant.launcher")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "info").upper())

# --- Por qué este archivo cambió (CrashLoopBackOff) -----------------------
# Los logs mostraban: "Uvicorn running..." -> "Shutting down" ->
# "Finished server process [1]" — SIN traceback. Esa secuencia es el
# shutdown *graceful* normal de uvicorn tras recibir SIGTERM DESPUÉS de que
# el ASGI startup terminó con éxito. No es una excepción de nuestro código;
# es una señal externa (muy probablemente la propia HF: ZeroGPU necesita
# inspeccionar el Blocks montado en main.py — ver `_zerogpu_probe` — para
# detectar el uso de @spaces.GPU antes de confirmar el tier de hardware, y
# eso implica levantar y volver a bajar el proceso).
#
# El bug real estaba acá: el loop anterior solo reintentaba ante OSError
# (bind de puerto). Un `uvicorn.run()` que retorna limpio por SIGTERM caía
# en el `break` y el script terminaba (exit 0) — el contenedor se quedaba
# sin servidor, HF lo reiniciaba desde cero, y volvía a pisar la misma
# señal: CrashLoopBackOff, aunque nuestro código nunca "crasheó".
#
# Fix: este proceso jamás debe terminar por su cuenta. CUALQUIER salida de
# uvicorn.run() — limpia o por excepción — relanza el servidor. El backoff
# solo crece si el proceso muere rápido y seguido (bug real persistente);
# si llegó a servir tráfico un rato, se resetea, para no perder tiempo en
# el caso normal de "una señal aislada y listo".


def _run_forever() -> None:
    port = int(os.getenv("PORT", "7860"))
    workers = int(os.getenv("WEB_CONCURRENCY", "1"))
    log_level = os.getenv("LOG_LEVEL", "info")

    attempt = 0
    consecutive_fast_exits = 0
    while True:
        attempt += 1
        started_at = time.monotonic()
        try:
            logger.info("Lanzando uvicorn (intento %d) en 0.0.0.0:%d", attempt, port)
            uvicorn.run(
                "main:app",
                host="0.0.0.0",
                # HF Spaces Docker routes traffic to the port declared as
                # app_port in the README frontmatter (7860). PORT is honored
                # too so the same image runs unmodified on platforms that
                # inject it (Render, Railway, etc.).
                port=port,
                # Kept identical to the local dev command on purpose — do not
                # switch to uvloop without re-validating the TensorFlow/Keras
                # inference path under it first.
                loop="asyncio",
                # Defaults to 1 on purpose: main.py's `_radar_cache` is an
                # in-memory, process-local TTL cache, and ZeroGPU's context
                # handling assumes a single worker. Raising WEB_CONCURRENCY
                # above 1 gives each worker its own cache (correct but
                # wasteful) until that cache moves to a shared store.
                workers=workers,
                log_level=log_level,
            )
            uptime = time.monotonic() - started_at
            logger.warning(
                "uvicorn.run() volvió limpio tras %.1fs sin excepción "
                "(señal externa, ej. probe de ZeroGPU o un redeploy). "
                "Relanzando.",
                uptime,
            )
        except OSError as exc:
            uptime = time.monotonic() - started_at
            logger.warning(
                "uvicorn.run() lanzó OSError tras %.1fs (%s) — el puerto %d "
                "probablemente sigue en TIME_WAIT de la instancia anterior.",
                uptime, exc, port,
            )
        except Exception:
            uptime = time.monotonic() - started_at
            logger.exception(
                "uvicorn.run() falló de forma inesperada tras %.1fs — relanzando.",
                uptime,
            )

        # Solo penalizamos con backoff creciente las salidas RÁPIDAS y
        # seguidas (indicio de un bug real de arranque). Si el proceso
        # llegó a estar arriba un rato sirviendo tráfico, no hay motivo
        # para frenar el próximo intento.
        if uptime < 10:
            consecutive_fast_exits += 1
        else:
            consecutive_fast_exits = 0
        delay = min(3 * consecutive_fast_exits, 30) or 1
        logger.info("Reintentando en %ds...", delay)
        time.sleep(delay)


if __name__ == "__main__":
    _run_forever()