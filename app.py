"""
Entrypoint de Hugging Face Spaces (app_file: app.py).

ROUND 3 — se elimina el patrón `gr.mount_gradio_app` + supervisor de
uvicorn manual (rondas 1 y 2). Ese patrón exportaba como `app` un FastAPI
plano con Gradio montado como sub-app en un path escondido — el
hypervisor de ZeroGPU nunca lo reconoce como demo Gradio nativo, sin
importar qué hubiera adentro de esa sub-app ("No @spaces.GPU function
detected during startup" persistente).

`main.app` ahora es un `gradio.Server` (ver main.py): un FastAPI real con
las mismas rutas, CORS y middleware de KodaQuant, pero del tipo que
Gradio/ZeroGPU reconocen nativamente. `.launch()` reemplaza al
`uvicorn.run()` manual — ya no hay que bindear el puerto a mano, manejar
Errno 98 ni reintentar ante OSError; Server.launch() maneja el ciclo de
vida completo (y detecta el entorno de Spaces automáticamente, igual que
Blocks.launch()).
"""

from main import app

if __name__ == "__main__":
    app.launch()