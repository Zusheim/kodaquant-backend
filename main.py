from dotenv import load_dotenv
import asyncio
load_dotenv()
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
)
from api.payments import router as payments_router
from api.online_learning import router as online_learning_router
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from services.quanti_engine import DEFAULT_FORECAST_HORIZON_DAYS
from services.quanti_engine import generate_quanti_strategy_stream
from services.quanti_engine import _forecast_asset
from services.quanti_engine import _normalize_language 
from api.quota_test_routes import router as quota_test_router
from api.quanti_chat import router as quanti_chat_router
from api.support import router as support_router
from core.config import settings

app = FastAPI(title="KodaQuant Terminal", version="4.0")
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

def _get_market_radar_data() -> dict:
    return {
        "volatility_index": 18.4,
        "trend": "bullish",
        "top_assets": [
            {"symbol": "SPY", "risk_score": 0.32},
            {"symbol": "QQQ", "risk_score": 0.41},
            {"symbol": "BTC-USD", "risk_score": 0.78},
        ],
    }

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

    radar_data = _get_market_radar_data()

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