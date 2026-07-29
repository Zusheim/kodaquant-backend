# api/online_learning.py
"""
Endpoint disparador del ciclo diario de online learning (Fase 1). Pensado
para ser golpeado por un cron externo (Render Cron Job, GitHub Actions
schedule, cron de VPS con `curl`) UNA vez al día, después del cierre de
mercado (NYSE cierra 21:00 UTC / 16:00 ET) — NO por el frontend ni por
usuarios finales, por eso vive detrás de un secreto compartido en vez de
`verify_api_credits` (esto no consume créditos de ningún usuario, es
mantenimiento interno del modelo).
"""

import os

from fastapi import APIRouter, Header, HTTPException

from services.online_learning import run_daily_online_learning_cycle

router = APIRouter()

# Debe fijarse en el entorno de producción (`CRON_SECRET=...`) — sin él,
# el endpoint rechaza TODO request (fail-closed, nunca fail-open).
_CRON_SECRET = os.getenv("CRON_SECRET")


@router.post("/api/v1/quanti/online-learning/run")
async def trigger_online_learning_cycle(x_cron_secret: str | None = Header(default=None)):
    if not _CRON_SECRET or x_cron_secret != _CRON_SECRET:
        raise HTTPException(status_code=403, detail="CRON_SECRET ausente o inválido.")

    # SÍNCRONO Y POTENCIALMENTE LARGO (descarga de mercado + fit de Keras) —
    # se corre tal cual dentro del handler porque este endpoint es de uso
    # exclusivo del cron (1 invocación/día), no del tráfico de usuarios; no
    # justifica la complejidad de un job queue para este volumen.
    report = run_daily_online_learning_cycle()
    return {"status": "success" if not report.get("errors") else "partial", "report": report}