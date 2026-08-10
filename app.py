import logging
import os
import time

import gradio as gr
import spaces
import uvicorn

from main import app as _fastapi_app

# --- Por qué este archivo cambió (ronda 2) ---------------------------------
#
# Error 1: "No @spaces.GPU function detected during startup"
#   El validador de ZeroGPU escanea el AST de `app_file` (app.py, según
#   `app_file: app.py` en el README) buscando un @spaces.GPU conectado a un
#   componente Gradio a nivel raíz de ESE archivo. El probe vivía en
#   main.py; acá solo se lo referenciaba como el string runtime "main:app",
#   invisible para ese scan estático. Fix: el probe pasa a vivir
#   directamente en app.py (ver `_zerogpu_startup_probe` más abajo). El
#   probe original en main.py (`/zerogpu-probe`) queda intacto — no hace
#   daño, no se tocó main.py.
#
# Error 2: Errno 98 (address already in use) en el puerto 7860
#   No es HF pre-bindeando el puerto por su cuenta: una app FastAPI montada
#   (`gr.mount_gradio_app`) no tiene ningún ".launch()" propio, alguien
#   tiene que abrir el socket con un servidor ASGI real — de ahí que se
#   siga necesitando uvicorn acá abajo. La causa real de la colisión fue el
#   supervisor `while True` de la ronda anterior: relanzaba uvicorn
#   INSTANTÁNEAMENTE ante CUALQUIER salida de uvicorn.run(), incluida una
#   limpia (sin excepción). Si el pase de detección de ZeroGPU efectivamente
#   termina este proceso y HF arranca por su cuenta una invocación NUEVA de
#   app.py para "el arranque real" (tal como ya sospechaba el comentario
#   original de este archivo), ese relanzamiento instantáneo competía por
#   el puerto contra la invocación nueva. Fix: se vuelve al comportamiento
#   original — reintentar SOLO ante OSError de bind (TIME_WAIT transitorio,
#   típico de un redeploy), y si el proceso termina limpio, se lo deja
#   terminar sin pelear por el puerto.

logger = logging.getLogger("kodaquant.launcher")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "info").upper())


@spaces.GPU()
def _zerogpu_startup_probe() -> str:
    """
    Dummy conectada a un evento de Gradio a nivel raíz de app.py (app_file)
    para que el escaneo AST de ZeroGPU la detecte durante el build y
    reserve el tier de GPU. KodaQuant (FastAPI, montado abajo) es quien
    realmente la usa en las peticiones reales — este probe no hace
    inferencia, solo existe para que el hipervisor la encuentre.
    """
    return "ok"


with gr.Blocks() as _zerogpu_demo:
    _status = gr.Textbox(value="pending", visible=False)
    # `.load()` dispara sola al inicializarse el Blocks — no depende de que
    # un usuario haga click (a diferencia del probe con botón en main.py).
    _zerogpu_demo.load(_zerogpu_startup_probe, None, _status)

# Nombre final `app`: tanto el hipervisor de HF como el import string
# "app:app" de más abajo necesitan encontrar exactamente esta variable acá.
# Ruta distinta a la de main.py ("/zerogpu-probe") para no pisarla.
app = gr.mount_gradio_app(_fastapi_app, _zerogpu_demo, path="/zerogpu-launch-probe")


def _run() -> None:
    port = int(os.getenv("PORT", "7860"))
    workers = int(os.getenv("WEB_CONCURRENCY", "1"))
    log_level = os.getenv("LOG_LEVEL", "info")

    _last_exc: OSError | None = None
    for attempt in range(1, 6):
        try:
            logger.info("Lanzando uvicorn (intento %d/5) en 0.0.0.0:%d", attempt, port)
            uvicorn.run(
                "app:app",
                host="0.0.0.0",
                port=port,
                loop="asyncio",
                workers=workers,
                log_level=log_level,
            )
            # Salida limpia (sin excepción): probablemente el pase de
            # detección de ZeroGPU u otro ciclo de vida legítimo del
            # hipervisor de HF. No competimos por el puerto de nuevo — se
            # deja terminar; si corresponde un relanzamiento real, que lo
            # dispare HF (o un redeploy), no una segunda instancia nuestra
            # peleando por el mismo bind.
            logger.info("uvicorn.run() terminó sin excepción — cerrando sin reintentar.")
            return
        except OSError as exc:
            _last_exc = exc
            logger.warning(
                "OSError bindeando el puerto %d (%s) — probable TIME_WAIT de "
                "una instancia previa. Reintentando en 3s.", port, exc,
            )
            time.sleep(3)
    if _last_exc is not None:
        raise _last_exc


if __name__ == "__main__":
    _run()