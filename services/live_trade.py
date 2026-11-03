"""
live_trade.py -- Inferencia diaria + órdenes a Alpaca (paper trading)
=============================================================================
Cierra el círculo: toma el modelo YA entrenado y validado
(`services/kodaquant_models/equity_specialist/model_v5.keras`, producido por
`python ml_engine/train_kodaquant_v5.py`), corre inferencia sobre el día de
HOY con el mismo pipeline exacto de features que el entrenamiento
(`tkv5.download_all`/`tkv5.engineer_asset` -- cero drift train/serve), y
envía órdenes a tu cuenta paper de Alpaca con el mismo sizing por
vol-target que ya validamos en `pnl_backtest.py`.

SEGURIDAD: corre en modo DRY-RUN por default -- imprime qué órdenes
mandaría, pero NO llama a Alpaca. Hace falta `--live` explícito para que
efectivamente envíe órdenes (a tu cuenta *paper*, sin plata real, pero aun
así conviene el freno).

REQUISITOS (una sola vez):
    1. python ml_engine/train_kodaquant_v5.py   -- si todavía no corriste
       el entrenamiento de producción completo (walk_forward_eval.py y
       pnl_backtest.py entrenan modelos DESCARTABLES por fold -- ninguno
       queda guardado en disco. Este es el que sí hay que generar una vez).
    2. export ALPACA_API_KEY=tu_key
       export ALPACA_SECRET_KEY=tu_secret
       (Dashboard de Alpaca -> "Generate New Key", con la cuenta en modo
       Paper Trading activado.)

USO:
    python services/live_trade.py                                  # dry-run equity_specialist (default, seguro)
    python services/live_trade.py --regime crypto_specialist        # dry-run V22 (target_mode="triple_barrier")
    python services/live_trade.py --live              # envía órdenes de verdad (a paper)
    python services/live_trade.py --vol-target-annual 0.15
    python services/live_trade.py --regime crypto_specialist --min-confidence 0.5
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import requests

# --- RESOLUCIÓN ABSOLUTA DE IMPORTS (mismo bootstrap que el resto del repo) -
def _bootstrap_project_root() -> Path:
    here = Path(__file__).resolve()
    for ancestor in (here.parent, *here.parents):
        if (ancestor / "ml_engine").is_dir() and (ancestor / "services").is_dir():
            return ancestor
    return here.parent.parent


_PROJECT_ROOT = _bootstrap_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import como módulo -- el `if __name__ == "__main__"` de train_kodaquant_v5.py
# NO se ejecuta acá, es seguro traer sus funciones/constantes sin disparar
# un entrenamiento completo.
import ml_engine.train_kodaquant_v5 as tkv5  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
logger = logging.getLogger("kodaquant.live_trade")

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
VOL_LOOKBACK_DAYS = 20
MAX_LEVERAGE = 3.0

# V22 — meta-labeling (López de Prado, cap. 3), MISMO criterio y MISMO
# valor que `quanti_engine.TB_MIN_SIGNAL_CONFIDENCE`
# (services/quanti_engine.py) — duplicado a propósito acá, no importado:
# este script vive del lado `ml_engine/` (importa `tkv5` directo, ver
# `_bootstrap_project_root`), nunca del lado `services/` donde vive
# quanti_engine.py — acoplar ambos módulos por un solo float no vale el
# import cruzado. Si calibrás uno con pnl_backtest.py/walk_forward_eval.py
# (ver MANUAL DE IGNICIÓN), replicá el número acá también.
TB_MIN_SIGNAL_CONFIDENCE = 0.45


def _alpaca_headers() -> dict:
    import os
    key, secret = os.environ.get("ALPACA_API_KEY", ""), os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError(
            "Faltan ALPACA_API_KEY / ALPACA_SECRET_KEY en el entorno. "
            "Generalas en el dashboard de Alpaca (modo Paper Trading) y "
            "exportalas antes de correr este script."
        )
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _load_production_model(regime_name: str):
    model_dir = tkv5.MODELS_ROOT / regime_name
    model_path, scalers_path = model_dir / "model_v5.keras", model_dir / "scalers_dict.pkl"
    if not model_path.exists() or not scalers_path.exists():
        raise RuntimeError(
            f"No existe un modelo de producción entrenado en {model_dir}. "
            f"Corré primero: python ml_engine/train_kodaquant_v5.py"
        )
    model = tkv5.keras.models.load_model(model_path, compile=False)
    with open(scalers_path, "rb") as f:
        payload = pickle.load(f)
    logger.info("Modelo cargado: %s (entrenado con tickers=%s)", model_path, payload["tickers"])
    return model, payload


def predict_today_regression(model, payload: dict) -> tuple[dict[str, float], dict]:
    """
    target_mode="regression" (equity_specialist) — CERO cambios de
    comportamiento respecto al script original; solo ahora también
    devuelve `all_market_data` (el MISMO que usó para featurizar, con
    force_refresh=True) para que `__main__` deje de re-descargar una
    segunda copia potencialmente desalineada antes de llamar a
    `size_orders_regression` (ver FIX abajo, `__main__`).

    Devuelve {ticker: r_hat} -- log-return de mañana predicho, en la
    unidad real (ya invertido el StandardScaler del target), para cada
    ticker. Reusa `download_all`/`engineer_asset` de train_kodaquant_v5.py
    tal cual -- garantiza que el feature engineering de HOY sea IDÉNTICO
    al que vio el modelo en entrenamiento (ninguna transformación nueva
    definida acá).
    """
    tickers, lookback = payload["tickers"], payload["lookback"]
    feature_cols = ["LOG_RETURN_1D"] + payload["tech_cols"] + payload["macro_tickers"]

    logger.info("Descargando datos frescos (force_refresh=True) para inferencia de hoy...")
    all_market_data = tkv5.download_all(
        tickers, payload["macro_tickers"], tkv5.PERIOD, cache_tag=f"{payload['regime_name']}_live",
        force_refresh=True,
    )

    predictions: dict[str, float] = {}
    for ticker in tickers:
        df_asset = tkv5.engineer_asset(all_market_data, ticker, payload["macro_tickers"])
        if len(df_asset) < lookback:
            logger.warning("%s: historia insuficiente hoy (%d < %d) -- se salta.", ticker, len(df_asset), lookback)
            continue

        window = df_asset[feature_cols].values[-lookback:]
        window_scaled = payload["feature_scalers"][ticker].transform(window)
        X = window_scaled.reshape(1, lookback, len(feature_cols)).astype(np.float32)
        asset_id = np.array([[payload["asset_to_id"][ticker]]], dtype=np.int32)

        y_pred_scaled = model.predict([X, asset_id], verbose=0)
        r_hat = float(payload["target_scalers"][ticker].inverse_transform(y_pred_scaled[:, 0:1])[0, 0])
        predictions[ticker] = r_hat

        realized_vol = float(df_asset["LOG_RETURN_1D"].tail(VOL_LOOKBACK_DAYS).std())
        logger.info("  %-6s  r_hat=%+.4f  vol_%dd=%.4f  último_close=$%.2f",
                    ticker, r_hat, VOL_LOOKBACK_DAYS, realized_vol, df_asset["RAW_CLOSE"].iloc[-1])

    return predictions, all_market_data


def predict_today_triple_barrier(model, payload: dict, min_confidence: float) -> tuple[dict[str, dict], dict]:
    """
    V22 — target_mode="triple_barrier" (crypto_specialist). UN forward
    pass determinista (`model.predict(..., verbose=0)`, MISMO contrato que
    `train_kodaquant_v5.evaluate_asset_triple_barrier` — nunca
    autoregresivo: la cabeza softmax clasifica el DESENLACE COMPLETO de la
    apuesta -- SL/TIMEOUT/TP dentro de `max_holding_days` -- no un paso
    aislado que tenga sentido encadenar día a día, ver docstring de
    `services/quanti_engine.py::_forecast_asset_triple_barrier` para el
    razonamiento completo).

    Devuelve {ticker: {...}} con el detalle real de clasificación
    (`predicted_class`, `confidence`, `probabilities`, `actionable`,
    `expected_return_pct`, niveles de barrera reales) — `actionable=False`
    (clase TIMEOUT, o confianza de clasificación bajo `min_confidence`,
    meta-labeling real, ver TB_MIN_SIGNAL_CONFIDENCE) es la señal para que
    `size_orders_triple_barrier` NO genere ninguna orden para ese ticker,
    nunca una convicción inventada.
    """
    tickers, lookback = payload["tickers"], payload["lookback"]
    feature_cols = ["LOG_RETURN_1D"] + payload["tech_cols"] + payload["macro_tickers"]
    tb_params = payload.get("tb_params") or {}
    pt_mult = float(tb_params.get("pt_mult", 2.0))
    sl_mult = float(tb_params.get("sl_mult", 2.0))
    vol_lookback = int(tb_params.get("vol_lookback", 20))
    class_names = list(payload.get("tb_class_names") or ["SL", "TIMEOUT", "TP"])

    logger.info("Descargando datos frescos (force_refresh=True) para inferencia de hoy...")
    all_market_data = tkv5.download_all(
        tickers, payload["macro_tickers"], tkv5.PERIOD, cache_tag=f"{payload['regime_name']}_live",
        force_refresh=True,
    )

    predictions: dict[str, dict] = {}
    for ticker in tickers:
        df_asset = tkv5.engineer_asset(all_market_data, ticker, payload["macro_tickers"])
        if len(df_asset) < lookback:
            logger.warning("%s: historia insuficiente hoy (%d < %d) -- se salta.", ticker, len(df_asset), lookback)
            continue

        window = df_asset[feature_cols].values[-lookback:]
        window_scaled = payload["feature_scalers"][ticker].transform(window)
        X = window_scaled.reshape(1, lookback, len(feature_cols)).astype(np.float32)
        asset_id = np.array([[payload["asset_to_id"][ticker]]], dtype=np.int32)

        probs = model.predict([X, asset_id], verbose=0)[0]  # (3,) softmax [SL, TIMEOUT, TP]
        if probs.shape != (3,):
            logger.warning("%s: salida del modelo con shape inesperado %s -- se salta (sin clase inventada).",
                            ticker, probs.shape)
            continue

        predicted_idx = int(np.argmax(probs))
        predicted_class = class_names[predicted_idx]
        confidence = float(probs[predicted_idx])
        actionable = predicted_class != "TIMEOUT" and confidence >= min_confidence

        last_price = float(df_asset["RAW_CLOSE"].iloc[-1])
        # Volatilidad realizada actual -- MISMA fórmula exacta que
        # `compute_triple_barrier_labels` (train_kodaquant_v5.py): std de
        # LOG_RETURN_1D en una ventana de `vol_lookback` días.
        vol_t = float(df_asset["LOG_RETURN_1D"].tail(vol_lookback).std())
        vol_t = vol_t if vol_t > 0 else 1e-6
        barrier_upper = last_price * float(np.exp(pt_mult * vol_t))
        barrier_lower = last_price * float(np.exp(-sl_mult * vol_t))

        p_sl, p_timeout, p_tp = float(probs[0]), float(probs[1]), float(probs[2])
        # Esperanza del retorno bajo la distribución de clases real del
        # modelo sobre los niveles de barrera reales -- MISMA fórmula que
        # `quanti_engine._forecast_asset_triple_barrier` (cero cifra
        # inventada; TIMEOUT aporta 0.0 -- ver docstring de esa función
        # para por qué). Solo informativo/logging acá, `size_orders_
        # triple_barrier` no la usa para el sizing (ver esa función).
        expected_return_pct = (p_tp * pt_mult * vol_t - p_sl * sl_mult * vol_t) * 100.0

        predictions[ticker] = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": {"SL": p_sl, "TIMEOUT": p_timeout, "TP": p_tp},
            "actionable": actionable,
            "expected_return_pct": expected_return_pct,
            "last_price": last_price,
            "barrier_upper": barrier_upper,
            "barrier_lower": barrier_lower,
        }
        logger.info(
            "  %-8s  clase=%-7s  confianza=%5.1f%%  P(SL/TIMEOUT/TP)=%.2f/%.2f/%.2f  "
            "E[retorno]=%+.2f%%  accionable=%-5s  último_close=$%.2f",
            ticker, predicted_class, confidence * 100, p_sl, p_timeout, p_tp,
            expected_return_pct, str(actionable), last_price,
        )

    return predictions, all_market_data


def size_orders_regression(predictions: dict[str, float], all_market_data, payload: dict,
                            equity: float, vol_target_annual: float) -> list[dict]:
    """Mismo esquema de sizing que `pnl_backtest.simulate_portfolio`
    (Directiva: cero cambios a la matemática del modelo -- esto es
    asignación de capital, una capa aparte) -- tamaño inversamente
    proporcional a la volatilidad realizada reciente de CADA activo,
    igual-ponderado como base entre los tickers activos hoy."""
    vol_target_daily = vol_target_annual / np.sqrt(252)
    n_active = len(predictions)
    orders = []
    for ticker, r_hat in predictions.items():
        df_asset = tkv5.engineer_asset(all_market_data, ticker, payload["macro_tickers"])
        realized_vol = float(df_asset["LOG_RETURN_1D"].tail(VOL_LOOKBACK_DAYS).std())
        last_price = float(df_asset["RAW_CLOSE"].iloc[-1])
        if realized_vol <= 0 or np.isnan(realized_vol):
            continue
        size_mult = min(vol_target_daily / realized_vol, MAX_LEVERAGE)
        notional = (equity / n_active) * size_mult
        orders.append({
            "symbol": ticker, "side": "buy" if r_hat > 0 else "sell",
            "notional": round(notional, 2), "r_hat": r_hat, "last_price": last_price,
        })
    return orders


def size_orders_triple_barrier(predictions: dict[str, dict], equity: float) -> list[dict]:
    """
    V22 -- sizing para target_mode="triple_barrier". MISMO criterio que
    `pnl_backtest.simulate_portfolio_triple_barrier`: cada señal
    accionable es un TRADE DISCRETO completo (entrada hoy, salida al
    tocar barrera o vencer `max_holding_days`), nunca un turnover diario
    -- vol-target NO aplica acá (ver aviso en `__main__`, mismo criterio
    que `--vol-target-annual` en pnl_backtest.py), igual-ponderado entre
    las señales accionables del día. Tickers con `actionable=False`
    (TIMEOUT, o confianza bajo `TB_MIN_SIGNAL_CONFIDENCE`) NO generan
    ninguna orden -- meta-labeling real, nunca un descarte silencioso ni
    una convicción inventada.
    """
    actionable = {t: p for t, p in predictions.items() if p["actionable"]}
    if not actionable:
        return []

    n_active = len(actionable)
    orders = []
    for ticker, pred in actionable.items():
        notional = equity / n_active
        orders.append({
            "symbol": ticker,
            "side": "buy" if pred["predicted_class"] == "TP" else "sell",
            "notional": round(notional, 2),
            # r_hat acá es E[log-return] bajo la distribución de clases (no
            # un r_hat de un solo paso, ver predict_today_triple_barrier) --
            # se preserva el nombre de campo por compatibilidad con el log
            # de __main__, que es agnóstico a target_mode.
            "r_hat": pred["expected_return_pct"] / 100.0,
            "last_price": pred["last_price"],
            "predicted_class": pred["predicted_class"],
            "confidence": pred["confidence"],
        })
    return orders


def get_account_equity() -> float:
    resp = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers=_alpaca_headers(), timeout=10)
    resp.raise_for_status()
    return float(resp.json()["equity"])


def get_current_side(symbol: str) -> str | None:
    resp = requests.get(f"{ALPACA_BASE_URL}/v2/positions/{symbol}", headers=_alpaca_headers(), timeout=10)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    qty = float(resp.json()["qty"])
    return "buy" if qty > 0 else "sell"


def submit_order(order: dict) -> None:
    """Sizing en `notional` (dólares) para longs -- Alpaca no permite
    notional en shorts (requiere `qty` entera), así que para sell se
    convierte a acciones enteras al precio actual."""
    current_side = get_current_side(order["symbol"])
    if current_side is not None and current_side != order["side"]:
        logger.info("  %s: posición actual (%s) contraria a la señal -- cerrando primero.",
                    order["symbol"], current_side)
        requests.delete(f"{ALPACA_BASE_URL}/v2/positions/{order['symbol']}",
                         headers=_alpaca_headers(), timeout=10).raise_for_status()

    payload = {"symbol": order["symbol"], "side": order["side"], "type": "market", "time_in_force": "day"}
    if order["side"] == "buy":
        payload["notional"] = str(order["notional"])
    else:
        payload["qty"] = str(max(1, int(order["notional"] / order["last_price"])))

    resp = requests.post(f"{ALPACA_BASE_URL}/v2/orders", headers=_alpaca_headers(), json=payload, timeout=10)
    resp.raise_for_status()
    logger.info("  Orden enviada: %s", resp.json().get("id"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia diaria + órdenes a Alpaca (paper trading).")
    parser.add_argument("--regime", default="equity_specialist", choices=list(tkv5.REGIMES))
    parser.add_argument("--vol-target-annual", type=float, default=0.15,
                         help="SOLO target_mode='regression' (equity_specialist). Se ignora explícitamente "
                              "en target_mode='triple_barrier' (crypto_specialist) -- ver size_orders_triple_barrier.")
    parser.add_argument("--min-confidence", type=float, default=TB_MIN_SIGNAL_CONFIDENCE,
                         help="SOLO target_mode='triple_barrier'. Piso de confianza de clasificación "
                              "(softmax [SL, TIMEOUT, TP]) por debajo del cual la señal se descarta -- "
                              "meta-labeling, López de Prado cap. 3. Calibrá este número con "
                              "pnl_backtest.py/walk_forward_eval.py antes de subirlo/bajarlo en vivo "
                              "(ver MANUAL DE IGNICIÓN).")
    parser.add_argument("--live", action="store_true",
                         help="Sin esto: dry-run, solo imprime las órdenes. Con esto: las envía a Alpaca de verdad "
                              "(a tu cuenta PAPER -- sin plata real, pero es el freno de seguridad del script).")
    args = parser.parse_args()

    model, payload = _load_production_model(args.regime)
    target_mode = payload.get("target_mode", "regression")

    # V22 — despacho por target_mode (ver docstring de módulo /
    # `services/quanti_engine.py` para el razonamiento completo de por qué
    # la cabeza softmax de Triple Barrier necesita un camino de
    # predicción/sizing propio, nunca el autoregresivo de regresión).
    #
    # FIX (drift train/serve en el sizing): la versión anterior descargaba
    # `all_market_data` DOS veces -- una acá (force_refresh=False, para
    # `size_orders`) y otra dentro de `predict_today` (force_refresh=True,
    # para featurizar la predicción) -- pudiendo quedar `size_orders`
    # leyendo un caché más viejo que el que realmente vio el modelo. Ahora
    # `predict_today_*` devuelve el MISMO `all_market_data` que usó para
    # predecir, y `size_orders_*` reusa exactamente ese objeto -- una sola
    # descarga, cero desalineación posible entre precio/vol usados para
    # predecir y usados para dimensionar la orden.
    if target_mode == "triple_barrier":
        if args.vol_target_annual != 0.15:  # el usuario tocó el flag explícitamente
            logger.warning("--vol-target-annual se ignora en target_mode='triple_barrier' "
                            "(ver size_orders_triple_barrier).")
        predictions, all_market_data = predict_today_triple_barrier(model, payload, args.min_confidence)
        if not predictions:
            logger.error("Sin predicciones -- no se genera ninguna orden.")
            raise SystemExit(1)
        equity = get_account_equity() if args.live else 100_000.0
        orders = size_orders_triple_barrier(predictions, equity)
    else:
        predictions, all_market_data = predict_today_regression(model, payload)
        if not predictions:
            logger.error("Sin predicciones -- no se genera ninguna orden.")
            raise SystemExit(1)
        equity = get_account_equity() if args.live else 100_000.0  # placeholder solo para el preview en dry-run
        orders = size_orders_regression(predictions, all_market_data, payload, equity, args.vol_target_annual)

    logger.info("=" * 78)
    logger.info("%s -- regimen=%s (target_mode=%s) -- equity=$%.2f -- %d orden(es)%s",
                "ÓRDENES A ENVIAR (--live)" if args.live else "DRY-RUN (ninguna orden real)",
                args.regime, target_mode, equity, len(orders),
                "" if args.live else " -- agregá --live para enviarlas de verdad")
    for o in orders:
        logger.info("  %-6s  %-4s  $%9.2f  (r_hat=%+.4f, precio=$%.2f)",
                    o["symbol"], o["side"].upper(), o["notional"], o["r_hat"], o["last_price"])
    logger.info("=" * 78)

    if args.live:
        for o in orders:
            submit_order(o)
        logger.info("Listo -- revisá tu dashboard de Alpaca (modo Paper) para confirmar las posiciones.")