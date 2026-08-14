# services/online_learning.py
"""
Continuous / Self-Correcting Online Learning — KodaQuant Terminal V5
====================================================================
Ciclo diario de auto-corrección MULTI-RÉGIMEN: cada especialista V5
(`kodaquant_models/equity_specialist/`, `.../crypto_specialist/`, ver
`REGIME_TICKERS`/`_regime_for_ticker` en quanti_engine.py) se fine-tunea
de forma INDEPENDIENTE, con su propio micro-batch, su propio modelo y su
propio bundle de scalers — nunca mezclados entre sí (dimensiones/escalas
de features distintas entre equity y cripto).

CICLO (1x/día, post-cierre de mercado):
  1. Por cada ticker con predicción pendiente de ayer (ledger SQLite):
     descarga el cierre real (Twelve Data + fallback Stooq, ver
     services/market_data.py — yfinance retirado del proyecto), calcula
     el error vs. lo predicho, y resuelve a qué régimen pertenece
     (`_regime_for_ticker`).
  2. Agrupa los samples en un micro-batch POR RÉGIMEN.
  3. Por cada régimen con batch no vacío: carga una copia fresca de
     `model_v5.keras` (grafo de entrenamiento normal, no el bridge MC
     Dropout de inferencia), recompila con `DirectionalHuberLoss` (misma
     loss del entrenamiento original) + LR bajo de fine-tuning, corre una
     micro-época, guarda atómicamente en su propia carpeta e invalida el
     cache de `_get_keras_model()`.
  4. Marca en el ledger SOLO los tickers cuyo régimen fine-tuneó con
     éxito (si cripto falla pero equity no, equity se marca aprendido y
     cripto queda pendiente para el próximo ciclo).
  5. Re-loguea la predicción T+1 de HOY para MAÑANA, para todo el
     universo (ambos regímenes), cerrando el círculo.

Nunca debe fallar de forma que tumbe el proceso del scheduler: cada etapa
está aislada en try/except con logging explícito.
"""

import os
import sqlite3
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import numpy as np
import pandas as pd

from services.market_data import fetch_close_history

from services.quanti_engine import (
    REGIME_TICKERS,
    BahdanauAttention,
    DirectionalHuberLoss,
    _fetch_feature_window,
    _forecast_asset,
    _get_keras_model,
    _get_scalers,
    _model_path,
    _regime_for_ticker,
    directional_accuracy_metric,
    inference_lock,
    keras,
    MODELS_ROOT,
)

# ---------------------------------------------------------------------------
# Ledger — SQLite local, único y compartido entre regímenes (una fila por
# ticker/target_date; el ticker ya determina su régimen vía
# `_regime_for_ticker`, la columna `regime` lo deja explícito para
# auditoría/reportes sin tener que re-resolverlo).
# ---------------------------------------------------------------------------

_LEDGER_PATH = MODELS_ROOT / "online_learning_ledger.sqlite3"

# Fine-tuning: LR deliberadamente ~100x menor que el de entrenamiento largo
# original — un micro-batch diario es señal muy pequeña frente al histórico
# con que se entrenó cada especialista; un LR alto destruiría ese
# conocimiento en una sola pasada (catastrophic forgetting).
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_WEIGHT_DECAY = 1e-5


def _utc_today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _ledger_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_LEDGER_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_ledger (
                ticker               TEXT NOT NULL,
                target_date          TEXT NOT NULL,
                regime               TEXT NOT NULL DEFAULT '',
                anchor_date          TEXT NOT NULL,
                anchor_price         REAL NOT NULL,
                predicted_price      REAL NOT NULL,
                predicted_log_return REAL NOT NULL,
                logged_at            TEXT NOT NULL,
                actual_price         REAL,
                actual_log_return    REAL,
                error_log_return     REAL,
                learned_at           TEXT,
                PRIMARY KEY (ticker, target_date)
            )
            """
        )
        try:
            # Migración defensiva: ledgers V4 preexistentes no tienen esta
            # columna — CREATE TABLE IF NOT EXISTS no la agrega a una
            # tabla ya existente.
            conn.execute("ALTER TABLE prediction_ledger ADD COLUMN regime TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # columna ya existe
        yield conn
        conn.commit()
    finally:
        conn.close()


def _iter_universe_by_regime() -> Iterator[tuple[str, str]]:
    """
    (ticker, regime) para TODO el universo entrenado. La fuente de verdad
    es `asset_to_id` de cada `scalers_dict.pkl` (no `REGIME_TICKERS` a
    secas) — un régimen sin artefactos cargables se omite con log, nunca
    tumba el resto del ciclo.
    """
    for regime in REGIME_TICKERS:
        try:
            scalers = _get_scalers(regime)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ [online_learning] No se pudieron cargar los scalers del régimen '{regime}': {exc!r}")
            continue
        for ticker in scalers["asset_to_id"]:
            yield ticker, regime


# ---------------------------------------------------------------------------
# Paso A — loguear la predicción de HOY para MAÑANA (cierra el círculo)
# ---------------------------------------------------------------------------

def log_next_day_prediction(ticker: str) -> dict[str, Any] | None:
    """
    Corre un forecast real de 1 paso (`_forecast_asset`, mismo motor que ve
    el usuario, CERO lógica duplicada; resuelve su propio régimen
    internamente) y deja constancia en el ledger de qué se predijo para la
    sesión siguiente.
    """
    try:
        regime = _regime_for_ticker(ticker)
    except ValueError as exc:
        print(f"⚠️ [online_learning] '{ticker}' no pertenece a ningún régimen V5 conocido: {exc!r}")
        return None

    try:
        result = _forecast_asset(ticker, steps=1)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ [online_learning] No se pudo generar el forecast T+1 de {ticker} para loguear: {exc!r}")
        return None

    if result.get("forecast_incomplete") or not result.get("forecast"):
        print(f"⚠️ [online_learning] Forecast T+1 de {ticker} incompleto — no se loguea (evita ensuciar el ledger).")
        return None

    anchor_price = float(result["last_price"])
    point = result["forecast"][0]
    predicted_price = float(point["expected_path"])
    predicted_log_return = float(np.log(predicted_price / anchor_price))
    target_date = point["date"]
    anchor_date = result["historical"][-1]["date"] if result.get("historical") else _utc_today_str()

    with _ledger_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO prediction_ledger
                (ticker, target_date, regime, anchor_date, anchor_price,
                 predicted_price, predicted_log_return, logged_at,
                 actual_price, actual_log_return, error_log_return, learned_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT actual_price FROM prediction_ledger WHERE ticker=? AND target_date=?), NULL),
                    COALESCE((SELECT actual_log_return FROM prediction_ledger WHERE ticker=? AND target_date=?), NULL),
                    COALESCE((SELECT error_log_return FROM prediction_ledger WHERE ticker=? AND target_date=?), NULL),
                    COALESCE((SELECT learned_at FROM prediction_ledger WHERE ticker=? AND target_date=?), NULL))
            """,
            (
                ticker, target_date, regime, anchor_date, anchor_price,
                predicted_price, predicted_log_return, _utc_now_iso(),
                ticker, target_date, ticker, target_date, ticker, target_date, ticker, target_date,
            ),
        )

    return {
        "ticker": ticker, "target_date": target_date, "regime": regime, "anchor_date": anchor_date,
        "anchor_price": anchor_price, "predicted_price": predicted_price,
        "predicted_log_return": predicted_log_return,
    }


def log_next_day_predictions_for_universe() -> dict[str, Any]:
    logged, failed = [], []
    for ticker, _regime in _iter_universe_by_regime():
        row = log_next_day_prediction(ticker)
        (logged if row else failed).append(ticker)
    return {"logged": logged, "failed": failed}


# ---------------------------------------------------------------------------
# Paso B — reconstruir la ventana EXACTA que vio el modelo en `anchor_date`
# ---------------------------------------------------------------------------

def _reconstruct_scaled_sample(
    ticker: str, anchor_date: str, anchor_price: float, actual_price: float,
) -> tuple[np.ndarray, float] | None:
    """
    Devuelve (ventana_normalizada (lookback, n_features), r_real_escalado)
    lista para entrar al `.fit()` del especialista de `ticker`, o None si
    no se pudo reconstruir (datos insuficientes, régimen desconocido, etc.).
    """
    try:
        regime = _regime_for_ticker(ticker)
        scalers = _get_scalers(regime)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ [online_learning] No se pudo resolver régimen/scalers de {ticker}: {exc!r}")
        return None

    feature_scaler = scalers["feature_scalers"][ticker]
    target_scaler = scalers["target_scalers"][ticker]
    lookback = scalers["lookback"]
    macro_tickers = scalers["macro_tickers"]

    try:
        _close_df, raw_features = _fetch_feature_window(
            ticker, macro_tickers, lookback, as_of_date=anchor_date,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ [online_learning] No se pudo reconstruir la ventana de {ticker} @ {anchor_date}: {exc!r}")
        return None

    if raw_features.shape[0] < lookback:
        print(f"⚠️ [online_learning] Ventana incompleta para {ticker} @ {anchor_date} (feriado/gap) — se omite hoy.")
        return None

    scaled_window = feature_scaler.transform(raw_features).astype(np.float32)

    r_real = float(np.log(actual_price / anchor_price))
    r_real_scaled = float(target_scaler.transform(np.array([[r_real]]))[0, 0])
    return scaled_window, r_real_scaled


def _fetch_actual_close(ticker: str, target_date: str) -> float | None:
    """
    Cierre real de `target_date` -- vía `fetch_close_history`
    (services/market_data.py: Twelve Data + fallback Stooq, misma caché en
    disco que ya usa `_fetch_feature_window`/`get_market_sentiment`;
    yfinance retirado del proyecto). Reemplaza 1:1 el antiguo
    `yf.download(ticker, start=..., end=..., auto_adjust=True, ...)` --
    incluso reutiliza la descarga ya cacheada de este mismo ticker si el
    ciclo diario corrió después de un forecast reciente, cero llamada de
    red extra en ese caso.
    """
    try:
        closes = fetch_close_history(ticker, min_days=10)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ [online_learning] fetch_close_history falló pidiendo el cierre real de {ticker} @ {target_date}: {exc!r}")
        return None
    if closes.empty:
        return None
    target_ts = pd.Timestamp(target_date)
    if target_ts not in closes.index:
        return None
    return float(closes.loc[target_ts])


# ---------------------------------------------------------------------------
# Paso C.1 — fine-tuning de UN especialista sobre su propio micro-batch
# ---------------------------------------------------------------------------

def _fine_tune_regime(
    regime: str, X_batch: list[np.ndarray], asset_id_batch: list[int], y_batch: list[float],
) -> float:
    """
    Carga fresca de `kodaquant_models/<regime>/model_v5.keras` (grafo de
    entrenamiento normal, NO el bridge MC Dropout de inferencia), bajo el
    mismo `inference_lock` que serializa forward passes de usuarios reales
    contra el modelo cacheado de ESE régimen. Recompila con
    `DirectionalHuberLoss` (misma loss del entrenamiento original) + LR
    bajo de fine-tuning. Guardado atómico en la carpeta del propio
    régimen + invalidación de `_get_keras_model` cache.
    """
    model_path = _model_path(regime)
    X = np.stack(X_batch, axis=0)
    asset_ids = np.array(asset_id_batch, dtype=np.int32).reshape(-1, 1)
    y = np.array(y_batch, dtype=np.float32).reshape(-1, 1)

    with inference_lock:
        finetune_model = keras.models.load_model(
            model_path,
            custom_objects={
                "BahdanauAttention": BahdanauAttention,
                "DirectionalHuberLoss": DirectionalHuberLoss,
                "directional_accuracy_metric": directional_accuracy_metric,
            },
            compile=False,
        )
        finetune_model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=FINE_TUNE_LEARNING_RATE, weight_decay=FINE_TUNE_WEIGHT_DECAY,
            ),
            loss=DirectionalHuberLoss(),
            metrics=["mae", directional_accuracy_metric],
        )
        history = finetune_model.fit(
            [X, asset_ids], y, epochs=1, batch_size=len(X_batch), shuffle=True, verbose=0,
        )

        # Guardado ATÓMICO: nunca se sobrescribe el .keras del régimen en sitio.
        tmp_path = f"{model_path}.tmp"
        finetune_model.save(tmp_path)
        os.replace(tmp_path, model_path)

        # Invalida el cache (todas las claves/regímenes) — la siguiente
        # `_forecast_asset()` real recarga el `.keras` recién fine-tuneado
        # del régimen que corresponda.
        _get_keras_model.cache_clear()

    return float(history.history["loss"][-1])


# ---------------------------------------------------------------------------
# Paso C.2 — el ciclo completo, multi-régimen
# ---------------------------------------------------------------------------

def run_daily_online_learning_cycle() -> dict[str, Any]:
    """
    Entry point para un cron/APScheduler diario post-cierre. Idempotente
    por régimen: una fila del ledger solo se marca aprendida si el
    fine-tuning de SU régimen tuvo éxito; si un régimen falla, sus filas
    quedan pendientes para el próximo ciclo sin afectar al otro régimen.
    """
    report: dict[str, Any] = {
        "learned": [], "skipped": [], "errors": [],
        "loss_by_regime": {}, "loss": None,
    }
    today = _utc_today_str()

    universe = [ticker for ticker, _regime in _iter_universe_by_regime()]
    if not universe:
        report["errors"].append("No se pudo cargar ningún especialista V5 (equity/crypto) — sin universo.")
        return report

    with _ledger_conn() as conn:
        rows = conn.execute(
            """
            SELECT ticker, target_date, anchor_date, anchor_price, predicted_log_return
            FROM prediction_ledger
            WHERE target_date = ? AND actual_price IS NULL
            """,
            (today,),
        ).fetchall()

    if not rows:
        report["skipped"] = universe
        report["errors"].append(f"Sin predicciones pendientes de ayer para hoy ({today}) — nada que aprender aún.")
        return report

    # Micro-batches SEPARADOS por régimen — nunca se mezclan equity y
    # cripto en el mismo `.fit()` (dimensiones/escalas de features distintas).
    batches: dict[str, dict[str, list]] = {}
    pending_updates: list[dict[str, Any]] = []

    for ticker, target_date, anchor_date, anchor_price, predicted_log_return in rows:
        try:
            regime = _regime_for_ticker(ticker)
        except ValueError as exc:
            report["errors"].append(f"{ticker}: {exc!r}")
            continue

        try:
            actual_price = _fetch_actual_close(ticker, target_date)
            if actual_price is None:
                report["skipped"].append(ticker)
                continue

            sample = _reconstruct_scaled_sample(ticker, anchor_date, anchor_price, actual_price)
            if sample is None:
                report["skipped"].append(ticker)
                continue
            scaled_window, r_real_scaled = sample

            scalers = _get_scalers(regime)
            asset_id = scalers["asset_to_id"][ticker]

            bucket = batches.setdefault(regime, {"X": [], "asset_id": [], "y": []})
            bucket["X"].append(scaled_window)
            bucket["asset_id"].append(asset_id)
            bucket["y"].append(r_real_scaled)

            r_real = float(np.log(actual_price / anchor_price))
            pending_updates.append({
                "ticker": ticker, "target_date": target_date, "regime": regime,
                "actual_price": actual_price, "actual_log_return": r_real,
                "error_log_return": r_real - predicted_log_return,
            })
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"{ticker}: {exc!r}\n{traceback.format_exc(limit=3)}")

    if not batches:
        report["errors"].append("Ningún ticker tuvo datos completos hoy — ciclo sin fine-tuning.")
        return report

    learned_tickers: set[str] = set()
    for regime, bucket in batches.items():
        try:
            report["loss_by_regime"][regime] = _fine_tune_regime(
                regime, bucket["X"], bucket["asset_id"], bucket["y"],
            )
            learned_tickers.update(u["ticker"] for u in pending_updates if u["regime"] == regime)
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(
                f"Fine-tuning del régimen '{regime}' falló, su modelo NO se tocó: "
                f"{exc!r}\n{traceback.format_exc(limit=5)}"
            )

    successful_updates = [u for u in pending_updates if u["ticker"] in learned_tickers]
    if not successful_updates:
        report["errors"].append("Ningún régimen completó el fine-tuning — el ledger no se marca como aprendido.")
        return report

    with _ledger_conn() as conn:
        for u in successful_updates:
            conn.execute(
                """
                UPDATE prediction_ledger
                SET actual_price = ?, actual_log_return = ?, error_log_return = ?,
                    learned_at = ?, regime = ?
                WHERE ticker = ? AND target_date = ?
                """,
                (u["actual_price"], u["actual_log_return"], u["error_log_return"],
                 _utc_now_iso(), u["regime"], u["ticker"], u["target_date"]),
            )
    report["learned"] = [u["ticker"] for u in successful_updates]
    report["mean_abs_error_log_return"] = float(
        np.mean([abs(u["error_log_return"]) for u in successful_updates])
    )
    if report["loss_by_regime"]:
        report["loss"] = float(np.mean(list(report["loss_by_regime"].values())))

    # Cierra el círculo: loguea la predicción de HOY para MAÑANA con los
    # modelos YA fine-tuneados, para todo el universo (ambos regímenes).
    report["next_day_logging"] = log_next_day_predictions_for_universe()

    return report