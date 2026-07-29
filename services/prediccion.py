"""
services/prediccion.py

Módulo de generación de predicciones de mercado para KodaQuant Terminal.

Responsable de orquestar el pipeline de inferencia:
  1. Carga de contexto macro.
  2. Carga de modelo(s) LSTM/BiLSTM (.keras) y scalers (.pkl) [PENDIENTE].
  3. Generación de recomendaciones accionables con métricas de error.

Estado actual: capa de simulación (mock) para desarrollo de frontend/backend
mientras el pipeline de ML real (ETL + modelo entrenado) se integra.
"""

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger("kodaquant.prediccion")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE ARTEFACTOS DE ML (PENDIENTE DE INTEGRACIÓN)
# ---------------------------------------------------------------------------
# TODO: Rutas a los artefactos serializados del modelo real.
# MODEL_PATH = "models/bilstm_multivariate.keras"
# SCALER_FEATURES_PATH = "models/scaler_features.pkl"   # Fit SOLO en train (X)
# SCALER_TARGET_PATH = "models/scaler_target.pkl"       # Fit SOLO en train (y)
#
# TODO: Carga perezosa (lazy load) de modelo/scalers a nivel de módulo,
# para no recargar en cada request. Ejemplo:
#
# _model = None
# _scaler_features = None
# _scaler_target = None
#
# def _load_artifacts():
#     global _model, _scaler_features, _scaler_target
#     if _model is None:
#         from tensorflow.keras.models import load_model
#         import joblib
#         _model = load_model(MODEL_PATH)
#         _scaler_features = joblib.load(SCALER_FEATURES_PATH)
#         _scaler_target = joblib.load(SCALER_TARGET_PATH)
#     return _model, _scaler_features, _scaler_target
#
# TODO: Fetch de ventana viva (ej. últimos 60 días) vía yfinance,
# construcción de indicadores técnicos (RSI, EMA, MACD), transform con
# _scaler_features, inferencia con _model, e inverse_transform con
# _scaler_target para obtener el precio proyectado en escala real.

# Margen de error fijo del modelo (MAE histórico de validación).
# TODO: reemplazar por el MAE real calculado en la fase de validación del modelo.
FIXED_MAE_MARGIN: float = 4.55

# Universo simulado de tickers candidatos para el mock.
_MOCK_UNIVERSE: List[Dict[str, Any]] = [
    {"ticker": "NVDA", "price": 132.45, "growth": 6.8, "action": "COMPRA FUERTE"},
    {"ticker": "AAPL", "price": 214.32, "growth": 2.3, "action": "COMPRA LIGERA"},
    {"ticker": "AMZN", "price": 198.77, "growth": 4.1, "action": "COMPRA FUERTE"},
    {"ticker": "MSFT", "price": 441.20, "growth": 1.9, "action": "COMPRA LIGERA"},
    {"ticker": "GOOGL", "price": 178.05, "growth": 3.4, "action": "COMPRA LIGERA"},
]


def _build_macro_context() -> str:
    """
    Construye el texto de contexto macroeconómico.

    TODO: Reemplazar por datos reales (ej. VIX vía yfinance ^VIX,
    índice DXY del dólar) en lugar del texto simulado.
    """
    vix_sim = round(random.uniform(13.5, 22.0), 2)
    dxy_sim = round(random.uniform(100.5, 106.8), 2)

    if vix_sim < 16:
        vix_comment = "señal de complacencia y apetito por riesgo"
    elif vix_sim < 20:
        vix_comment = "volatilidad moderada, sentimiento neutral"
    else:
        vix_comment = "aversión al riesgo elevándose en el mercado"

    return (
        f"El VIX se ubica en niveles de {vix_sim}, reflejando {vix_comment}. "
        f"El índice del Dólar (DXY) simulado opera en {dxy_sim}, "
        "un factor clave para el apetito de riesgo en renta variable "
        "tecnológica. El contexto macro sugiere cautela táctica con "
        "sesgo selectivo hacia calidad."
    )


def _build_recommendations(usd_budget: float) -> List[Dict[str, Any]]:
    """
    Construye la lista de recomendaciones simuladas.

    TODO: Sustituir por salida real del modelo BiLSTM/LSTM:
      - price -> último close real (yfinance)
      - growth -> proyección inversa-transformada del modelo
      - action -> regla de negocio sobre el growth proyectado vs. umbral
    """
    picks = random.sample(_MOCK_UNIVERSE, k=3)
    recommendations: List[Dict[str, Any]] = []

    for pick in picks:
        allocation_pct = round(random.uniform(0.15, 0.45), 2)
        allocation_usd = round(usd_budget * allocation_pct, 2)
        projected_price = round(pick["price"] * (1 + pick["growth"] / 100), 2)

        recommendations.append(
            {
                "ticker": pick["ticker"],
                "action": pick["action"],
                "current_price_usd": pick["price"],
                "projected_growth_pct": pick["growth"],
                "projected_price_usd": projected_price,
                "suggested_allocation_usd": allocation_usd,
                "suggested_allocation_pct": allocation_pct,
            }
        )

    return recommendations


async def generate_predictions(usd_budget: float) -> Dict[str, Any]:
    """
    Genera el reporte de predicciones de mercado para KodaQuant Terminal.

    Args:
        usd_budget: Presupuesto en USD del usuario, usado para dimensionar
                    las asignaciones sugeridas por recomendación.

    Returns:
        Diccionario con estructura de grado institucional:
        {
            "status": "success" | "error",
            "generated_at": ISO-8601 timestamp,
            "usd_budget": float,
            "macro_context": str,
            "recommendations": [ {...}, ... ],
            "mae_margin": float,
            "error": str | None
        }
    """
    try:
        if usd_budget is None or usd_budget <= 0:
            raise ValueError("usd_budget debe ser un valor numérico positivo.")

        # Simula latencia de inferencia real (I/O-bound: carga de modelo,
        # fetch de datos de mercado, forward pass, etc.).
        await asyncio.sleep(0.3)

        # TODO: Sustituir estas dos llamadas por el pipeline real:
        #   1. _load_artifacts() -> modelo + scalers
        #   2. fetch_live_window(ticker) -> últimos 60 días vía yfinance
        #   3. build_technical_indicators(df) -> RSI, EMA, MACD
        #   4. scaler_features.transform(...) -> input escalado
        #   5. model.predict(...) -> salida escalada
        #   6. scaler_target.inverse_transform(...) -> precio real proyectado
        macro_context = _build_macro_context()
        recommendations = _build_recommendations(usd_budget)

        return {
            "status": "success",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "usd_budget": usd_budget,
            "macro_context": macro_context,
            "recommendations": recommendations,
            "mae_margin": FIXED_MAE_MARGIN,
            "error": None,
        }

    except Exception as exc:
        logger.exception("Error generando predicciones en generate_predictions")
        return {
            "status": "error",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "usd_budget": usd_budget,
            "macro_context": None,
            "recommendations": [],
            "mae_margin": FIXED_MAE_MARGIN,
            "error": str(exc),
        }


if __name__ == "__main__":
    # Bloque de prueba manual: python -m services.prediccion
    async def _main() -> None:
        result = await generate_predictions(usd_budget=1000.0)
        import json

        print(json.dumps(result, indent=2, ensure_ascii=False))

    asyncio.run(_main())