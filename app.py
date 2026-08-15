"""
Entrypoint de Hugging Face Spaces (app_file: app.py).

ROUND 4 -- FIX del estado zombi "Starting..."/"Building..." permanente en
el dashboard de HF (ver el bloque de comentarios "RONDA 4" en main.py,
junto a `_zerogpu_probe`, para la auditoría completa línea a línea contra
gradio==6.22.0).

Resumen: `Server.launch()` (rondas 1-3) invocaba `app.launch()` SIN
`app_kwargs`. Internamente, `Blocks.launch()` -> `App.create_app()`
reasigna `app.router.lifespan_context` a partir de
`app_kwargs.get("lifespan")` — nunca del `lifespan=` pasado al
constructor `Server(...)` ni de ningún `@app.on_event("startup")`
registrado antes de `.launch()`. Sin `app_kwargs={"lifespan": lifespan}`
explícito ACÁ, el hook de arranque real (log de "ASGI startup complete" +
warm-up del probe de ZeroGPU) queda descartado en silencio — causa raíz
real de que el hypervisor de HF nunca viera la señal de arranque que
esperaba y el Space quedara colgado en "Starting...".

`main.app` sigue siendo un `gradio.Server` (FastAPI real que ZeroGPU
reconoce nativamente, ver ROUND 3) — `.launch()` sigue reemplazando al
`uvicorn.run()` manual, maneja el ciclo de vida completo y detecta el
entorno de Spaces automáticamente. Lo único que cambia acá es el
`app_kwargs={"lifespan": lifespan}` explícito.
"""

from main import app, lifespan

if __name__ == "__main__":
    app.launch(app_kwargs={"lifespan": lifespan})