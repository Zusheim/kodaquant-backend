from dotenv import load_dotenv
import asyncio
import contextlib
import logging
import threading
import time
load_dotenv()
from fastapi import Depends, HTTPException, Request
from gradio import Server
import spaces
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from services.currency_engine import convert_to_usd
from services.prediccion import generate_predictions
from services.risk_manager import calculate_portfolio
from api.auth import router as auth_router
from core.security import (
    verify_api_credits,
    AuthenticatedUser,
    get_allowed_origins,   # <-- CAMBIO: lista cerrada de orígenes CORS
    ALLOWED_METHODS,
    ALLOWED_HEADERS,
    StrictOriginMiddleware,  # <-- FIX: bloquea Origin no permitido a nivel app
                              #     (gradio.Server pisa nuestro CORSMiddleware
                              #     en prod -- ver detalle en core/security.py)
)
from api.payments import router as payments_router
from api.online_learning import router as online_learning_router
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from services.quanti_engine import DEFAULT_FORECAST_HORIZON_DAYS
from services.quanti_engine import generate_quanti_strategy_stream
from services.quanti_engine import _forecast_asset
from services.quanti_engine import _normalize_language 
from services.quanti_engine import _derive_capital_signal
from api.quota_test_routes import router as quota_test_router
from api.quanti_chat import router as quanti_chat_router
from api.support import router as support_router
from core.config import settings

logger = logging.getLogger("kodaquant.main")

# NOTA: `lifespan` NO se pasa acá (`Server(lifespan=...)`) a propósito —
# `Blocks.launch()`/`App.create_app()` lo pisa igual que a `@app.on_event`
# (ver el bloque de comentarios "RONDA 4" junto al probe de ZeroGPU, más
# abajo). El único punto que gradio realmente respeta es
# `app.launch(app_kwargs={"lifespan": lifespan})` en app.py.
app = Server(title="KodaQuant Terminal", version="4.0")
app.include_router(payments_router)
app.include_router(auth_router, prefix="/api/v1/auth")
app.include_router(online_learning_router)
if settings.ENV != "production":
    app.include_router(quota_test_router)
app.include_router(quanti_chat_router)
app.include_router(support_router)

_VALID_EXPERIENCE = {"beginner", "advanced"}
_VALID_ANALYSIS_MODE = {"specific", "discovery"}
_VALID_RISK_PROFILE = {"conservador", "moderado", "agresivo"}

class QuantiConsultRequest(BaseModel):
    """
    Solo `capital` es realmente obligatorio — todo lo demás tiene default
    seguro y se NORMALIZA en lugar de rechazar la request con 422 ante
    variaciones menores (mayúsculas, catálogo, strings vacíos). Alias
    camelCase 1:1 con lo que envía App.jsx.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    capital: float = Field(..., gt=0, alias="capital")
    currency: str = Field(default="USD", alias="currency")
    experience_level: str = Field(default="beginner", alias="experienceLevel")
    analysis_mode: str = Field(default="discovery", alias="analysisMode")
    target_asset: Optional[str] = Field(default=None, alias="targetAsset")
    forecast_horizon_days: int = Field(default=DEFAULT_FORECAST_HORIZON_DAYS, alias="forecastHorizonDays")
    risk_profile: str = Field(default="moderado", alias="riskProfile")
    risk_score: int = Field(default=50, alias="riskScore")
    language: str = Field(default="en", alias="language")

    @field_validator("capital", mode="before")
    @classmethod
    def _coerce_capital(cls, v):
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError("capital debe ser un número mayor a 0")

    @field_validator("currency", mode="before")
    @classmethod
    def _normalize_currency(cls, v):
        if not isinstance(v, str) or not v.strip():
            return "USD"
        return v.strip().upper()[:3]

    @field_validator("experience_level", mode="before")
    @classmethod
    def _normalize_experience(cls, v):
        v = str(v).strip().lower() if v else "beginner"
        return v if v in _VALID_EXPERIENCE else "beginner"

    @field_validator("analysis_mode", mode="before")
    @classmethod
    def _normalize_analysis_mode(cls, v):
        v = str(v).strip().lower() if v else "discovery"
        return v if v in _VALID_ANALYSIS_MODE else "discovery"

    @field_validator("target_asset", mode="before")
    @classmethod
    def _normalize_target_asset(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return str(v).strip().upper()

    @field_validator("forecast_horizon_days", mode="before")
    @classmethod
    def _normalize_horizon(cls, v):
        try:
            days = int(v)
        except (TypeError, ValueError):
            return DEFAULT_FORECAST_HORIZON_DAYS
        return days if days > 0 else DEFAULT_FORECAST_HORIZON_DAYS

    @field_validator("risk_profile", mode="before")
    @classmethod
    def _normalize_risk_profile_field(cls, v):
        v = str(v).strip().lower() if v else "moderado"
        return v if v in _VALID_RISK_PROFILE else "moderado"

    @field_validator("risk_score", mode="before")
    @classmethod
    def _normalize_risk_score(cls, v):
        try:
            score = int(v)
        except (TypeError, ValueError):
            return 50
        return max(0, min(100, score))

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language_field(cls, v):
        # Reusa la MISMA normalización que quanti_engine (_SUPPORTED_LANGUAGES
        # = {"en","es"}, default "en"), para que nunca haya drift entre el
        # schema de entrada y lo que el LLM realmente honra.
        return _normalize_language(v)

class StrategyRequest(BaseModel):
    budget_usd: float = Field(..., gt=0)
    experience_level: Literal["beginner", "intermediate", "advanced"] = "beginner"

@app.post("/strategy")
async def strategy_mock(payload: StrategyRequest):
    return {
        "status": "success",
        "amounts": {
            "riesgo_usd": 4000,
            "reserva_usd": 6000,
            "riesgo_pct": "40.0%",
            "reserva_pct": "60.0%",
        },
        "guidelines": [
            "Conexión exitosa con el servidor KodaQuant.",
            "El motor está en línea y operando.",
        ],
    }

# FIX CRÍTICO (Alpha Seeker nunca corría): este mock estático alimentaba
# `/api/v1/quanti/consult` con un `top_assets` FIJO donde BTC-USD siempre
# tenía el `risk_score` más alto (0.78) — `_resolve_plan_b_ticker`
# (quanti_engine.py) lo elegía SIEMPRE, sin importar el forecast real de
# cada ciclo. Reemplazado por `_get_radar_data`, que llama al pipeline
# REAL (`generate_predictions`, ver services/prediccion.py) con caché TTL.

_RADAR_CACHE_TTL_SECONDS = 60.0
_radar_cache: dict[str, tuple[float, dict]] = {}


async def _get_radar_data(budget_usd: float) -> dict:
    """
    Radar real (Alpha Seeker) para `/api/v1/quanti/consult` — sustituye el
    mock hardcodeado. Cacheado con TTL: escanea el universo completo (10
    tickers, Keras + Monte Carlo vía `_forecast_asset`) y el resultado NO
    depende de `budget_usd` (ese solo escala montos de display en
    `recommendations`; el ranking real de `top_assets` es independiente),
    así que recomputarlo en cada request dispararía ~10 inferencias Keras
    extra por usuario concurrente sin necesidad. `generate_predictions`
    nunca lanza excepción (Circuit Breaker propio, devuelve
    status="error" con listas vacías) — safe de propagar tal cual.
    """
    now = time.monotonic()
    cached = _radar_cache.get("global")
    if cached is not None and (now - cached[0]) < _RADAR_CACHE_TTL_SECONDS:
        return cached[1]

    data = await generate_predictions(budget_usd)
    if data.get("status") != "success":
        logger.warning("generate_predictions degradado en /consult: %s", data.get("error"))
    _radar_cache["global"] = (now, data)
    return data

app.add_middleware(
    CORSMiddleware,
    # Lista cerrada, resuelta en core/security.py. En ENVIRONMENT=production
    # esto se reduce a *solo* los dominios de Firebase Hosting (+
    # settings.FRONTEND_URL); localhost nunca se cuela a producción.
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=ALLOWED_METHODS,     # GET, POST, OPTIONS — no "*"
    allow_headers=ALLOWED_HEADERS,     # Authorization, Content-Type — no "*"
    expose_headers=["Content-Type", "Cache-Control"],
)

# FIX CRÍTICO — gradio.Server SIEMPRE agrega su propio CustomCORSMiddleware
# en .launch() (después de este punto, ver app.py), y ese middleware queda
# por FUERA de nuestro CORSMiddleware de arriba sin importar el orden acá.
# Su regla solo restringe origen si el Host es localhost -- en *.hf.space
# NUNCA aplica esa restricción, así que termina permitiendo cualquier
# Origin con credentials=true, pisando get_allowed_origins(). Verificado
# contra gradio==6.22.0 (ver core/security.py::StrictOriginMiddleware para
# el detalle completo). Esta capa rechaza (403) el request ANTES de que
# llegue al router, así que ningún header que agregue Gradio después
# importa: la request nunca se procesó.
app.add_middleware(StrictOriginMiddleware)


# FIX — blindaje JSON-only. Ninguna respuesta de esta API debe poder
# renderizar como HTML, sin importar qué falle. Starlette por default
# puede devolver su propia página de error para excepciones no atrapadas;
# estos handlers garantizan que 404/422/500 y cualquier excepción no
# manejada siempre vuelvan como JSON, consistente con lo que espera
# quantiService.js (response.json() en cada llamada).
@app.exception_handler(StarletteHTTPException)
async def _json_http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def _json_validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def _json_unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Excepción no manejada en %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor."},
    )

# --- ZeroGPU probe -----------------------------------------------------
# ROUND 3 — causa raíz real del "No @spaces.GPU function detected during
# startup": `gr.mount_gradio_app(fastapi_app, blocks, path=...)` devuelve
# la instancia FastAPI de ENTRADA (fastapi_app), mutada in place — nunca
# un gr.Blocks/Interface. El hypervisor de ZeroGPU necesita reconocer el
# objeto `app` exportado como demo Gradio nativo; al recibir un FastAPI
# puro con Gradio montado como sub-app en un path escondido, no lo
# reconoce como tal — pase lo que pase adentro de esa sub-app (Blocks con
# GPU function real por .load() o .click(), no importa), y mata el
# proceso. Esto es válido tanto para el mount en app.py (ronda 2) como
# para este mismo patrón acá en main.py (bug preexistente, mismo defecto).
#
# Fix: `gradio.Server` (gradio>=6.22, ver
# https://gradio.app/main/guides/server-mode) es un FastAPI real —
# `app.add_middleware`, `app.include_router`, rutas @app.get/@app.post,
# todo lo que ya usa este archivo sigue funcionando sin tocarlo — pero ES
# el tipo de objeto que Gradio/ZeroGPU reconocen nativamente como servidor
# Gradio. `@app.api()` registra la función en la tabla de endpoints de
# Gradio (mismo registro que el hypervisor inspecciona), sin necesidad de
# gr.Blocks ni de mount_gradio_app. Nunca se llama desde el frontend — solo
# necesita existir registrada para que el hypervisor la detecte al
# build/startup. KodaQuant sigue forzando toda la inferencia Keras real a
# CPU (ver DEVICE GUARD arriba); este endpoint no hace inferencia real.
@app.api(name="zerogpu_probe")
@spaces.GPU()
def _zerogpu_probe() -> str:
    return "ok"


# FIX 2026-08-15 (RONDA 4) — CAUSA RAÍZ REAL del estado zombi "Starting...".
#
# Auditoría línea a línea de gradio==6.22.0 (gradio/routes.py::App.create_app,
# gradio/blocks.py::Blocks.launch, gradio/server.py::Server.launch):
#
#   Server.launch()  ->  blocks.launch(_app=self, app_kwargs=app_kwargs, ...)
#   Blocks.launch()  ->  App.create_app(self, app=_app, app_kwargs=app_kwargs, ...)
#   App.create_app(): como `app` (=self, nuestra instancia YA construida)
#     NO es None, toma la rama:
#         app.router.lifespan_context = create_lifespan_handler(
#             app_kwargs.get("lifespan", None), *delete_cache
#         )
#
# Esa línea PISA `app.router.lifespan_context` con un handler construido a
# partir de `app_kwargs` — el dict que se le pasa a `.launch()`, NUNCA el
# `lifespan=` que se le pasó al CONSTRUCTOR `Server(...)`. Como `app.py`
# (rondas 1-3) llamaba `app.launch()` sin `app_kwargs`, `app_kwargs.get(
# "lifespan")` daba `None` -> el lifespan real quedaba DESCARTADO en
# silencio. Y como `router.lifespan_context` se reasigna DIRECTO (no se
# compone con la implementación default de Starlette que sí lee
# `router.on_startup`), cualquier `@app.on_event("startup")` registrado
# antes de `.launch()` (como el que vivía acá) también queda huérfano —
# explica exacto por qué "KodaQuant ASGI startup complete" NUNCA aparecía
# en los logs de producción pese a que el resto del pipeline sí corría.
#
# FIX REAL: el lifespan tiene que viajar como `app_kwargs={"lifespan": lifespan}`
# en la LLAMADA a `.launch()` (ver app.py) — es el ÚNICO punto de la cadena
# que `create_app` efectivamente respeta. `lifespan` se define acá y se
# exporta para que app.py lo importe.
def _warm_zerogpu_probe_sync() -> None:
    """
    Red de seguridad INDEPENDIENTE de todo el ciclo de vida ASGI/Gradio de
    arriba: corre en un hilo daemon disparado en el momento del IMPORT de
    este módulo — mucho antes de que exista `router`/`lifespan_context`
    alguno para pisar. Garantiza al menos una ejecución real de
    `_zerogpu_probe` durante el arranque del proceso sin importar qué haga
    `Server.launch()` internamente ni qué versión de gradio la sirva.
    """
    try:
        result = _zerogpu_probe()
        logger.info("ZeroGPU probe warm-up OK (hilo de import, red de seguridad): %s", result)
    except Exception as exc:
        logger.warning("ZeroGPU probe warm-up (hilo de import) falló (no crítico): %r", exc)


threading.Thread(
    target=_warm_zerogpu_probe_sync, daemon=True, name="zerogpu-warmup-import"
).start()


@contextlib.asynccontextmanager
async def lifespan(app):
    """
    Lifespan real de la app — DEBE inyectarse vía
    `app.launch(app_kwargs={"lifespan": lifespan})` en app.py (ver bloque
    de comentarios arriba). Pasarlo al constructor `Server(lifespan=...)`
    o registrar `@app.on_event("startup")` NO alcanza: ambos caminos son
    pisados por `Blocks.launch()`/`App.create_app()` antes de que el
    Space quede realmente sirviendo tráfico.
    """
    # Uvicorn solo loguea el genérico "Application startup complete.". Este
    # marcador con timestamp deja explícito en los logs del Space cuánto
    # tardó el cold start real (imports de TensorFlow/Keras + montaje de
    # gradio) — clave para diferenciar "todo bien, algo externo mandó
    # SIGTERM" de "el arranque fue tan lento que algo lo mató por timeout".
    logger.info("KodaQuant ASGI startup complete — rutas + probe de ZeroGPU montados.")

    async def _warm_zerogpu_probe_async() -> None:
        # Segunda ejecución del probe, esta vez DENTRO del ciclo de vida
        # ASGI real (útil si el hilo de import de arriba corrió antes de
        # que `device-api.zero` estuviera listo para recibirlo). Fire-and-
        # forget con timeout: si ZeroGPU no responde o el tier actual no lo
        # tiene habilitado, esto NUNCA debe retrasar ni tumbar el arranque.
        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _zerogpu_probe), timeout=15.0
            )
            logger.info("ZeroGPU probe warm-up OK (lifespan ASGI): %s", result)
        except Exception as exc:
            logger.warning("ZeroGPU probe warm-up (lifespan ASGI) falló (no crítico): %r", exc)

    warmup_task = asyncio.create_task(_warm_zerogpu_probe_async())
    try:
        yield
    finally:
        warmup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await warmup_task

@app.get("/")
async def root():
    return {"status": "online", "system": "KodaQuant Terminal", "version": "4.0"}

@app.get("/api/v1/convert")
async def test_conversion(amount: float, currency: str):
    try:
        data = await convert_to_usd(amount, currency.upper())
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/v1/radar")
async def test_radar(usd_budget: float):
    try:
        data = await generate_predictions(usd_budget)
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/v1/quanti/forecast")
async def test_ml_forecast(ticker: str = "SPY"):
    # `data` es el dict completo que arma `_forecast_asset` — ya incluye
    # `predicted_path` (array plano día-a-día) junto con `forecast`
    # (mismo dato, con date/price/kind). No requiere transformación aquí:
    # serializar el dict completo es suficiente para que el frontend lo lea.
    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, _forecast_asset, ticker)
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/v1/quanti/consult")
async def quanti_consult(
    payload: QuantiConsultRequest,
    current_user: AuthenticatedUser = Depends(verify_api_credits),
):
    try:
        conversion_report = await convert_to_usd(payload.capital, payload.currency)
        budget_usd = conversion_report["usd_amount"]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error de conversión de divisa: {exc}")

    radar_data = await _get_radar_data(budget_usd)
    # Auto-Allocation: copia superficial — NUNCA mutar en sitio el dict que
    # devuelve `_get_radar_data` (está cacheado con TTL y compartido entre
    # requests concurrentes). `_derive_capital_signal` es puro/determinista
    # sobre datos ya calculados, así que enriquecerlo acá es seguro y no
    # requiere invalidar ni recomputar el caché del radar.
    radar_data = {**radar_data, "signal": _derive_capital_signal(radar_data)}

    try:
        portfolio_allocation = await calculate_portfolio(budget_usd, radar_data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en risk_manager: {exc}")

    # `generate_quanti_strategy_stream` emite `chart_data.plan_a` / `plan_b`
    # como el dict completo de `_forecast_asset` (ver quanti_engine.py), así
    # que `predicted_path` viaja automáticamente en el evento `meta` del SSE
    # — no hace falta tocar el stream para exponerlo.
    return StreamingResponse(
        generate_quanti_strategy_stream(
            budget_usd=budget_usd,
            portfolio_allocation=portfolio_allocation,
            radar_data=radar_data,
            experience_level=payload.experience_level,
            analysis_mode=payload.analysis_mode,
            target_asset=payload.target_asset,
            forecast_horizon_days=payload.forecast_horizon_days,
            risk_profile=payload.risk_profile,
            risk_score=payload.risk_score,
            language=payload.language,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )