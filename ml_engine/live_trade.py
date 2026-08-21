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
    python services/live_trade.py                    # dry-run (default, seguro)
    python services/live_trade.py --live              # envía órdenes de verdad (a paper)
    python services/live_trade.py --vol-target-annual 0.15 --cost-aware
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


def predict_today(model, payload: dict) -> dict[str, float]:
    """Devuelve {ticker: r_hat} -- log-return de mañana predicho, en la
    unidad real (ya invertido el StandardScaler del target), para cada
    ticker. Reusa `download_all`/`engineer_asset` de train_kodaquant_v5.py
    tal cual -- garantiza que el feature engineering de HOY sea IDÉNTICO
    al que vio el modelo en entrenamiento (ninguna transformación nueva
    definida acá)."""
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

    return predictions


def size_orders(predictions: dict[str, float], all_market_data, payload: dict,
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
    parser.add_argument("--regime", default="equity_specialist")
    parser.add_argument("--vol-target-annual", type=float, default=0.15)
    parser.add_argument("--live", action="store_true",
                         help="Sin esto: dry-run, solo imprime las órdenes. Con esto: las envía a Alpaca de verdad "
                              "(a tu cuenta PAPER -- sin plata real, pero es el freno de seguridad del script).")
    args = parser.parse_args()

    model, payload = _load_production_model(args.regime)
    all_market_data = tkv5.download_all(
        payload["tickers"], payload["macro_tickers"], tkv5.PERIOD,
        cache_tag=f"{args.regime}_live", force_refresh=False,  # ya se refrescó en predict_today
    )
    predictions = predict_today(model, payload)
    if not predictions:
        logger.error("Sin predicciones -- no se genera ninguna orden.")
        raise SystemExit(1)

    equity = get_account_equity() if args.live else 100_000.0  # placeholder solo para el preview en dry-run
    orders = size_orders(predictions, all_market_data, payload, equity, args.vol_target_annual)

    logger.info("=" * 78)
    logger.info("%s -- equity=$%.2f -- %d orden(es)%s",
                "ÓRDENES A ENVIAR (--live)" if args.live else "DRY-RUN (ninguna orden real)",
                equity, len(orders), "" if args.live else " -- agregá --live para enviarlas de verdad")
    for o in orders:
        logger.info("  %-6s  %-4s  $%9.2f  (r_hat=%+.4f, precio=$%.2f)",
                    o["symbol"], o["side"].upper(), o["notional"], o["r_hat"], o["last_price"])
    logger.info("=" * 78)

    if args.live:
        for o in orders:
            submit_order(o)
        logger.info("Listo -- revisá tu dashboard de Alpaca (modo Paper) para confirmar las posiciones.")