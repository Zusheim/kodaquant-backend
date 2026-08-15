"""
services/prediccion.py

Radar de mercado — KodaQuant Terminal V5. Orquesta el pipeline REAL de
inferencia expuesto por quanti_engine.py (Keras 3 + Monte Carlo + MC
Dropout, ver _forecast_asset) sobre todo el universo entrenado
(REGIME_TICKERS), traduce el edge direccional de cada especialista a
acción operativa (action) y dimensiona el capital sugerido
(allocation_pct) en proporción a la confianza estadística real del propio
forecast — cero simulación, cero cifras inventadas.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from services.quanti_engine import (
    DEFAULT_FORECAST_HORIZON_DAYS,
    PLAN_A_TICKER,
    REGIME_TICKERS,
    _forecast_asset,
    _safe_sentiment,
)

logger = logging.getLogger("kodaquant.prediccion")

# Universo real escaneado — mismos tickers entrenados por los especialistas
# V5 (equity_specialist + crypto_specialist); REGIME_TICKERS es la única
# fuente de verdad (ver quanti_engine.py).
_RADAR_UNIVERSE: List[str] = [t for tickers in REGIME_TICKERS.values() for t in tickers]
_CRYPTO_TICKERS = frozenset(REGIME_TICKERS.get("crypto_specialist", ()))

TOP_N_RECOMMENDATIONS = 3

# FIX 2026-08-15 -- ver ROOT CAUSE de "degradación total del radar" en el
# docstring de services/market_data.py. Antes de esto, `_scan_universe`
# despachaba los 10 tickers de REGIME_TICKERS los 10 A LA VEZ (cada uno con
# inferencia Keras CPU-only real + I/O de red), maximizando tanto la
# contención de CPU como la ventana en la que todos pedían los mismos 5
# macro tickers compartidos al mismo tiempo -- el single-flight lock nuevo
# en market_data.py ya colapsa esa ráfaga a 1 request por macro ticker,
# pero acotar la concurrencia acá es la segunda capa: con esto, el primer
# ticker despachado resuelve (y cachea) el macro context casi en solitario,
# y el resto del universo entra de a oleadas encontrando caché ya tibia.
_SCAN_CONCURRENCY = 4

# Alpha Seeker — candidatos reales para "Quanti's Choice" (selección
# automática de Plan B en modo discovery, ver _resolve_plan_b_ticker en
# quanti_engine.py). Tope de candidatos expuestos al motor de allocation.
_ALPHA_SEEKER_MAX_CANDIDATES = 5

# Umbrales de retorno proyectado (edge) -> acción operativa.
_STRONG_BUY_THRESHOLD_PCT = 4.0
_LIGHT_BUY_THRESHOLD_PCT = 1.0
_SELL_THRESHOLD_PCT = -1.0

# Cotas de asignación de capital por pick, moduladas por confianza real.
_MIN_ALLOCATION_PCT = 0.10
_MAX_ALLOCATION_PCT = 0.45
# Cripto: la misma razón señal/ruido implica menor confianza estructural
# (mayor volatilidad realizada) -> se amortigua el allocation_pct.
_CRYPTO_CONFIDENCE_DAMPENING = 0.6
_CONFIDENCE_SCORE_CAP = 1.5  # techo de normalización de la razón edge/banda

# Fallback SOLO si el universo entero falla en yfinance/Keras (degradación
# total del radar) — nunca sustituye una predicción real disponible.
_FALLBACK_MAE_MARGIN = 4.55


def _action_for_return(predicted_return_pct: float) -> str:
    if predicted_return_pct >= _STRONG_BUY_THRESHOLD_PCT:
        return "COMPRA FUERTE"
    if predicted_return_pct >= _LIGHT_BUY_THRESHOLD_PCT:
        return "COMPRA LIGERA"
    if predicted_return_pct <= _SELL_THRESHOLD_PCT:
        return "VENTA"
    return "MANTENER"


def _confidence_score(forecast: Dict[str, Any]) -> float:
    """
    Razón señal/ruido: |edge proyectado| / ancho medio de la banda P5-P95
    (ambos ya calculados por _forecast_asset vía Monte Carlo real). Edge
    grande + banda angosta = alta confianza; edge chico + banda ancha =
    ruido. Se deriva 100% de la salida del modelo, cero cifras inventadas.

    Penalización OOD (ver `real_ood_dampening` en quanti_engine.py,
    auditoría SPY oob_frac=45.6% en T+1): `real_ood_dampening_applied`
    es el MISMO factor que ya recortó la magnitud de `predicted_return_pct`
    en origen porque el input real de T+1 ya caía fuera del rango de
    entrenamiento del `feature_scaler` — se reaplica acá porque un input
    OOD no solo degrada la magnitud de la señal, también la CONFIANZA que
    merece esa señal (el band width por sí solo no lo captura: una banda
    angosta sobre un input extrapolado sigue siendo una banda angosta
    calculada por una red operando fuera de lo que aprendió). Con datos
    in-distribution (`real_ood_dampening_applied` ausente o 1.0, forecasts
    generados antes de este campo) esto es un no-op exacto — no cambia el
    comportamiento previo.
    """
    points = forecast.get("forecast") or []
    last_price = forecast.get("last_price") or 0.0
    if not points or last_price <= 0:
        return 0.0

    avg_band_pct = sum(
        (p["upper_bound"] - p["lower_bound"]) / last_price * 100 for p in points
    ) / len(points)
    if avg_band_pct <= 0:
        return 0.0

    raw_score = abs(forecast.get("predicted_return_pct", 0.0)) / avg_band_pct
    ood_factor = float(forecast.get("real_ood_dampening_applied", 1.0))
    return raw_score * ood_factor


def _allocation_pct_for(ticker: str, forecast: Dict[str, Any]) -> float:
    normalized = min(_confidence_score(forecast), _CONFIDENCE_SCORE_CAP) / _CONFIDENCE_SCORE_CAP
    allocation = _MIN_ALLOCATION_PCT + normalized * (_MAX_ALLOCATION_PCT - _MIN_ALLOCATION_PCT)

    if ticker in _CRYPTO_TICKERS:
        allocation *= _CRYPTO_CONFIDENCE_DAMPENING

    return round(allocation, 4)


def _build_macro_context(spy_sentiment: Dict[str, Any], btc_sentiment: Dict[str, Any]) -> str:
    """
    Contexto macro derivado de get_market_sentiment (momentum de retornos
    log REALES vía yfinance, mismo mecanismo que usa quanti_engine) sobre
    el ancla de renta variable (PLAN_A_TICKER) y el proxy de apetito por
    riesgo cripto — reemplaza el texto VIX/DXY simulado.
    """
    spy_score = spy_sentiment.get("sentiment_score", 0.0)
    spy_label = spy_sentiment.get("sentiment_label", "Neutral")
    btc_score = btc_sentiment.get("sentiment_score", 0.0)
    btc_label = btc_sentiment.get("sentiment_label", "Neutral")

    if btc_score > 0.2:
        risk_appetite = "apetito por riesgo elevado"
    elif btc_score < -0.2:
        risk_appetite = "aversión al riesgo dominante"
    else:
        risk_appetite = "apetito por riesgo mixto"

    return (
        f"El benchmark de renta variable ({PLAN_A_TICKER}) registra sentimiento "
        f"{spy_label} (momentum normalizado: {spy_score:+.2f}). El proxy de "
        f"apetito por riesgo cripto (BTC-USD) marca sentimiento {btc_label} "
        f"(momentum normalizado: {btc_score:+.2f}), vía KodaQuant Sentinel — "
        f"ambas lecturas derivadas de retornos logarítmicos realizados, sin "
        f"cifras simuladas. El régimen combinado sugiere {risk_appetite} en "
        "el universo escaneado por el radar."
    )


def _derive_mae_margin(candidates: List[Tuple[str, Dict[str, Any]]]) -> float:
    """
    Margen de error real: promedio del semiancho de la banda P5-P95 en el
    día terminal proyectado sobre los picks recomendados — reemplaza el
    MAE fijo simulado por dispersión Monte Carlo genuina de cada forecast.
    """
    margins = []
    for _, forecast in candidates:
        points = forecast.get("forecast") or []
        if not points:
            continue
        terminal = points[-1]
        margins.append((terminal["upper_bound"] - terminal["lower_bound"]) / 2)

    if not margins:
        return _FALLBACK_MAE_MARGIN
    return round(sum(margins) / len(margins), 2)


def _rank_alpha_seeker_candidates(
    valid_forecasts: List[Tuple[str, Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Alpha Seeker: universo REAL de candidatos para "Quanti's Choice".

    Distinto de `recommendations` (top movers en valor ABSOLUTO — incluye
    VENTAs, correcto para el scanner general que debe mostrar todo el
    mercado). `top_assets` es el contrato que consume
    `_resolve_plan_b_ticker` (quanti_engine.py) para decidir en QUÉ activo
    invertir el capital de riesgo: solo admite edge direccional POSITIVO y
    de grado de compra real (`predicted_return_pct >= _LIGHT_BUY_THRESHOLD_PCT`)
    — un algoritmo de "mejor oportunidad" nunca puede devolver un activo en
    caída. Ranking por Alpha Score = retorno CON signo * (1 + razón
    señal/ruido) — nunca abs(), y nunca un campo fantasma tipo
    `risk_score` que nadie completa. 100% derivado de `_forecast_asset`,
    cero cifras inventadas.
    """
    candidates = []
    for ticker, forecast in valid_forecasts:
        predicted_return_pct = forecast.get("predicted_return_pct", 0.0)
        if predicted_return_pct < _LIGHT_BUY_THRESHOLD_PCT:
            continue  # sin edge de compra real -> nunca candidato a "mejor oportunidad"
        confidence = _confidence_score(forecast)
        candidates.append(
            {
                "symbol": ticker,
                "action": _action_for_return(predicted_return_pct),
                "predicted_return_pct": predicted_return_pct,
                "confidence_score": round(confidence, 4),
                "data_reliability": forecast.get("forecast_reliability", "normal"),
            }
        )

    candidates.sort(
        key=lambda c: c["predicted_return_pct"] * (1 + c["confidence_score"]),
        reverse=True,
    )
    return candidates[:_ALPHA_SEEKER_MAX_CANDIDATES]


async def _scan_universe() -> Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, Any], Dict[str, Any]]:
    """
    Despacha _forecast_asset (síncrona/bloqueante, Keras) para TODO
    _RADAR_UNIVERSE vía executor + get_market_sentiment (async, Circuit
    Breaker propio) en paralelo — mismo patrón que
    quanti_engine._build_investment_plans.
    """
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(_SCAN_CONCURRENCY)

    async def _safe_forecast(ticker: str):
        async with semaphore:  # FIX 2026-08-15 — ver _SCAN_CONCURRENCY arriba
            try:
                return await loop.run_in_executor(None, _forecast_asset, ticker, DEFAULT_FORECAST_HORIZON_DAYS)
            except Exception as exc:  # noqa: BLE001 — un ticker caído no tumba el radar completo
                logger.warning("Forecast radar falló para %s: %r", ticker, exc)
                return None

    *forecasts, spy_sentiment, btc_sentiment = await asyncio.gather(
        *(_safe_forecast(t) for t in _RADAR_UNIVERSE),
        _safe_sentiment(PLAN_A_TICKER),
        _safe_sentiment("BTC-USD"),
    )

    valid = [
        (ticker, forecast)
        for ticker, forecast in zip(_RADAR_UNIVERSE, forecasts)
        if forecast is not None and not forecast.get("forecast_incomplete")
    ]
    return valid, spy_sentiment, btc_sentiment


async def generate_predictions(usd_budget: float) -> Dict[str, Any]:
    """
    Genera el reporte de predicciones de mercado con inferencia REAL del
    especialista Keras V5 correspondiente (equity/crypto, ver
    quanti_engine._forecast_asset) — cero mock. Estructura de retorno
    intacta para no romper frontend/endpoints existentes.
    """
    try:
        if usd_budget is None or usd_budget <= 0:
            raise ValueError("usd_budget debe ser un valor numérico positivo.")

        valid_forecasts, spy_sentiment, btc_sentiment = await _scan_universe()
        if not valid_forecasts:
            raise RuntimeError(
                "El pipeline Keras/yfinance no devolvió ningún forecast válido "
                "para el universo V5 — degradación total del radar."
            )

        # Alpha Seeker: candidatos reales para "Quanti's Choice" — se deriva
        # ANTES del sort de abajo (ese sort es para `recommendations`, el
        # scanner general de top movers; este es un ranking independiente,
        # solo-positivo, para decidir en qué activo invertir Plan B).
        top_assets = _rank_alpha_seeker_candidates(valid_forecasts)

        # Ranking por convicción real: edge absoluto ponderado por razón
        # señal/ruido (_confidence_score), no solo magnitud del movimiento.
        valid_forecasts.sort(
            key=lambda item: abs(item[1]["predicted_return_pct"]) * (1 + _confidence_score(item[1])),
            reverse=True,
        )
        top_candidates = valid_forecasts[:TOP_N_RECOMMENDATIONS]

        recommendations = []
        for ticker, forecast in top_candidates:
            allocation_pct = _allocation_pct_for(ticker, forecast)
            recommendations.append(
                {
                    "ticker": ticker,
                    "action": _action_for_return(forecast["predicted_return_pct"]),
                    "current_price_usd": forecast["last_price"],
                    "projected_growth_pct": forecast["predicted_return_pct"],
                    "projected_price_usd": forecast["predicted_price"],
                    "suggested_allocation_usd": round(usd_budget * allocation_pct, 2),
                    "suggested_allocation_pct": allocation_pct,
                    # Transparencia OOD (ver _confidence_score/quanti_engine.py
                    # real_ood_dampening): "degraded" ya está reflejado en
                    # suggested_allocation_pct (recortado vía _confidence_score
                    # más arriba) — estos dos campos son para que el frontend
                    # explique EL PORQUÉ del recorte, no solo lo aplique en
                    # silencio. Default "normal"/0.0 para forecasts generados
                    # antes de este campo (backward-compatible).
                    "data_reliability": forecast.get("forecast_reliability", "normal"),
                    "real_data_ood_frac": forecast.get("real_data_ood_frac", 0.0),
                }
            )

        return {
            "status": "success",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "usd_budget": usd_budget,
            "macro_context": _build_macro_context(spy_sentiment, btc_sentiment),
            "recommendations": recommendations,
            "top_assets": top_assets,
            "mae_margin": _derive_mae_margin(top_candidates),
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
            "top_assets": [],
            "mae_margin": _FALLBACK_MAE_MARGIN,
            "error": str(exc),
        }


if __name__ == "__main__":
    # Prueba manual: python -m services.prediccion
    async def _main() -> None:
        result = await generate_predictions(usd_budget=1000.0)
        import json

        print(json.dumps(result, indent=2, ensure_ascii=False))

    asyncio.run(_main())