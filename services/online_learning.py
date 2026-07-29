# services/online_learning.py
"""
Continuous / Self-Correcting Online Learning — KodaQuant Terminal
====================================================================
Ciclo diario de auto-corrección para `attention_bilstm_global.keras`.

NOTA DE ARQUITECTURA — por qué este archivo NO exporta ONNX: la petición
original de esta fase asumía un runtime ONNX ("recompila quanti.onnx").
Ese runtime fue retirado del pipeline real (ver el docstring de
`quanti_engine.py`: "CERO ONNX Runtime en este proceso" — la inferencia ya
carga el `.keras` nativo in-process vía Keras 3/TF). Re-exportar ONNX aquí
reintroduciría una dependencia que el resto del sistema activamente evitó.
Este módulo respeta la arquitectura real: fine-tunea el `.keras` vivo,
lo guarda atómicamente, e invalida el `@lru_cache` de
`_get_keras_model()` para que la SIGUIENTE inferencia cargue los pesos
recién actualizados — sin reiniciar el proceso, sin ONNX.

CICLO (pensado para correr 1x/día vía cron / APScheduler, DESPUÉS del
cierre de mercado):

  1. Por cada ticker del universo entrenado (`scalers['asset_to_id']`):
     a. Busca en el ledger (SQLite local) qué se predijo AYER para HOY
        (`log_next_day_predictions_for_universe`, corrida al final del
        ciclo anterior, deja esa fila).
     b. Descarga el cierre REAL de hoy (yfinance).
     c. Calcula el error: delta entre el log-return real y el
        log-return que el modelo predijo, ambos anclados al MISMO
        precio base (`anchor_price`) — así el error no se contamina si
        el precio ancla se movió entre el momento de predicción y hoy.
  2. Reconstruye, por ticker, la ventana de features EXACTA que el
     modelo vio al predecir (mismo feature engineering, cortado a esa
     fecha — ver `as_of_date` en `_fetch_feature_window`), la agrupa en
     un micro-batch (uno por ticker con dato nuevo hoy).
  3. Carga una copia FRESCA del modelo (no el "bridge" de MC Dropout de
     inferencia — un grafo de entrenamiento normal), la recompila con
     un optimizer de fine-tuning (LR bajo, para no destruir lo
     aprendido en el entrenamiento largo original — "catastrophic
     forgetting"), y corre UNA micro-época de `.fit()` sobre el batch
     del día (backpropagation dinámico real, no un ajuste heurístico).
  4. Guarda el `.keras` actualizado de forma ATÓMICA (escribe a un
     archivo temporal + `os.replace`, nunca sobrescribe el archivo
     vivo directamente — evita dejar un modelo corrupto a medio
     escribir si el proceso muere a mitad del guardado).
  5. Invalida el cache de `_get_keras_model()`: la siguiente request de
     inferencia del Command Center ya sirve con el cerebro actualizado.
  6. Marca las filas del ledger como aprendidas (actual_price,
     actual_log_return, error_log_return, learned_at).
  7. Re-loguea la predicción T+1 de HOY para MAÑANA (con el modelo ya
     fine-tuneado), cerrando el círculo para el ciclo de mañana.

Este archivo NUNCA debe fallar de forma que tumbe el proceso del
scheduler: cada etapa está aislada en try/except con logging explícito
y el ciclo sigue con los tickers restantes si uno falla.
"""

import os
import sqlite3
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yfinance as yf

from services.quanti_engine import (
    ML_MODEL_PATH,
    BahdanauAttention,
    _fetch_feature_window,
    _forecast_asset,
    _get_keras_model,
    _get_scalers,
    inference_lock,
    keras,
)

# ---------------------------------------------------------------------------
# Ledger — SQLite local, cero infraestructura extra. Vive junto al modelo
# (mismo directorio que `attention_bilstm_global.keras` / `scalers.pkl`),
# así que sobrevive redeploys que preserven ese volumen igual que el modelo.
# ---------------------------------------------------------------------------

_LEDGER_PATH = Path(os.path.dirname(ML_MODEL_PATH)) / "online_learning_ledger.sqlite3"

# Fine-tuning: LR deliberadamente ~100x menor que el `0.001` del
# entrenamiento largo original (ver `entrenamiento.ipynb::build_model`) —
# un micro-batch de N tickers (N <= tamaño del universo, normalmente 10) es
# una señal MUY pequeña frente a los 10y de historia con que se entrenó el
# modelo; un LR alto aquí destruiría ese conocimiento en una sola pasada
# (catastrophic forgetting) en vez de corregirlo incrementalmente.
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
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Paso A — loguear la predicción de HOY para MAÑANA (cierra el círculo)
# ---------------------------------------------------------------------------

def log_next_day_prediction(ticker: str) -> dict[str, Any] | None:
    """
    Corre un forecast real de 1 paso (reutiliza `_forecast_asset` tal cual
    — mismo motor, mismo Monte Carlo/MC-Dropout que ve el usuario, CERO
    lógica de predicción duplicada) y deja constancia en el ledger de qué
    se predijo para la sesión siguiente. `predicted_price` es la mediana
    (`expected_path`, P50) del forecast de 1 paso — la misma cifra que
    `_forecast_asset` expone como `predicted_price` cuando steps=1.
    """
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
    # `anchor_date`: última sesión real ANTES del target — se recupera del
    # último punto de `historical` (mismo ancla que usó `_forecast_asset`).
    anchor_date = result["historical"][-1]["date"] if result.get("historical") else _utc_today_str()

    with _ledger_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO prediction_ledger
                (ticker, target_date, anchor_date, anchor_price,
                 predicted_price, predicted_log_return, logged_at,
                 actual_price, actual_log_return, error_log_return, learned_at)
            VALUES (?, ?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT actual_price FROM prediction_ledger WHERE ticker=? AND target_date=?), NULL),
                    COALESCE((SELECT actual_log_return FROM prediction_ledger WHERE ticker=? AND target_date=?), NULL),
                    COALESCE((SELECT error_log_return FROM prediction_ledger WHERE ticker=? AND target_date=?), NULL),
                    COALESCE((SELECT learned_at FROM prediction_ledger WHERE ticker=? AND target_date=?), NULL))
            """,
            (
                ticker, target_date, anchor_date, anchor_price,
                predicted_price, predicted_log_return, _utc_now_iso(),
                ticker, target_date, ticker, target_date, ticker, target_date, ticker, target_date,
            ),
        )

    return {
        "ticker": ticker, "target_date": target_date, "anchor_date": anchor_date,
        "anchor_price": anchor_price, "predicted_price": predicted_price,
        "predicted_log_return": predicted_log_return,
    }


def log_next_day_predictions_for_universe() -> dict[str, Any]:
    scalers = _get_scalers()
    logged, failed = [], []
    for ticker in scalers["asset_to_id"]:
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
    lista para entrar al `.fit()`, o None si no se pudo reconstruir (datos
    de mercado insuficientes ese día — feriado, ticker deslistado, etc.).
    """
    scalers = _get_scalers()
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
    Cierre real de `target_date`. Ventana de descarga corta (+/- unos días)
    en vez de re-descargar 2y — este llamado es por-ticker y diario.
    """
    start = (pd.Timestamp(target_date) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(target_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ [online_learning] yfinance falló pidiendo el cierre real de {ticker} @ {target_date}: {exc!r}")
        return None
    if raw.empty:
        return None
    close_col = raw["Close"][ticker] if isinstance(raw["Close"], pd.DataFrame) else raw["Close"]
    target_ts = pd.Timestamp(target_date)
    if target_ts not in close_col.index:
        # Sesión aún no cerrada / feriado no calendarizado — se reintenta en
        # el próximo ciclo, la fila del ledger sigue con actual_price NULL.
        return None
    return float(close_col.loc[target_ts])


# ---------------------------------------------------------------------------
# Paso C — el ciclo completo
# ---------------------------------------------------------------------------

def run_daily_online_learning_cycle() -> dict[str, Any]:
    """
    Entry point pensado para un cron/APScheduler diario post-cierre. Idempotente:
    si se corre dos veces el mismo día, la segunda pasada no encuentra filas
    pendientes (actual_price ya no es NULL) y no vuelve a fine-tunear con el
    mismo dato — evita overfitting al re-aprender el mismo día dos veces.
    """
    report: dict[str, Any] = {"learned": [], "skipped": [], "errors": [], "loss": None}
    today = _utc_today_str()

    try:
        scalers = _get_scalers()
        universe = list(scalers["asset_to_id"].keys())
    except Exception as exc:  # noqa: BLE001
        report["errors"].append(f"No se pudieron cargar los scalers: {exc!r}")
        return report

    X_batch: list[np.ndarray] = []
    asset_id_batch: list[int] = []
    y_batch: list[float] = []
    pending_updates: list[dict[str, Any]] = []  # para el UPDATE del ledger tras el fit exitoso

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

    for ticker, target_date, anchor_date, anchor_price, predicted_log_return in rows:
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

            asset_id = scalers["asset_to_id"][ticker]
            X_batch.append(scaled_window)
            asset_id_batch.append(asset_id)
            y_batch.append(r_real_scaled)

            r_real = float(np.log(actual_price / anchor_price))
            pending_updates.append({
                "ticker": ticker, "target_date": target_date,
                "actual_price": actual_price, "actual_log_return": r_real,
                "error_log_return": r_real - predicted_log_return,
            })
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"{ticker}: {exc!r}\n{traceback.format_exc(limit=3)}")

    if not X_batch:
        report["errors"].append("Ningún ticker tuvo datos completos hoy — ciclo sin fine-tuning.")
        return report

    # --- Fine-tuning: modelo fresco (grafo de entrenamiento normal, NO el
    # bridge de MC Dropout de inferencia), bajo el mismo lock que serializa
    # las llamadas de inferencia — así un `.fit()` nunca corre a la vez que
    # un forward pass de un usuario real contra el modelo cacheado. ---
    try:
        X = np.stack(X_batch, axis=0)
        asset_ids = np.array(asset_id_batch, dtype=np.int32).reshape(-1, 1)
        y = np.array(y_batch, dtype=np.float32).reshape(-1, 1)

        with inference_lock:
            finetune_model = keras.models.load_model(
                ML_MODEL_PATH, custom_objects={"BahdanauAttention": BahdanauAttention}, compile=False,
            )
            finetune_model.compile(
                optimizer=keras.optimizers.AdamW(
                    learning_rate=FINE_TUNE_LEARNING_RATE, weight_decay=FINE_TUNE_WEIGHT_DECAY,
                ),
                loss=keras.losses.Huber(),
                metrics=["mae"],
            )
            history = finetune_model.fit(
                [X, asset_ids], y, epochs=1, batch_size=len(X_batch), shuffle=True, verbose=0,
            )

            # Guardado ATÓMICO: nunca se sobrescribe `ML_MODEL_PATH` en sitio.
            tmp_path = f"{ML_MODEL_PATH}.tmp"
            finetune_model.save(tmp_path)
            os.replace(tmp_path, ML_MODEL_PATH)

            # Invalida el cache — la siguiente `_forecast_asset()` real
            # recarga el `.keras` recién fine-tuneado (y reconstruye el
            # bridge de MC Dropout sobre los pesos nuevos).
            _get_keras_model.cache_clear()

        report["loss"] = float(history.history["loss"][-1])
    except Exception as exc:  # noqa: BLE001
        report["errors"].append(f"Fine-tuning falló, modelo NO se tocó: {exc!r}\n{traceback.format_exc(limit=5)}")
        return report

    # --- Marca el ledger como aprendido (solo si el fit + guardado tuvieron éxito) ---
    with _ledger_conn() as conn:
        for u in pending_updates:
            conn.execute(
                """
                UPDATE prediction_ledger
                SET actual_price = ?, actual_log_return = ?, error_log_return = ?, learned_at = ?
                WHERE ticker = ? AND target_date = ?
                """,
                (u["actual_price"], u["actual_log_return"], u["error_log_return"], _utc_now_iso(),
                 u["ticker"], u["target_date"]),
            )
    report["learned"] = [u["ticker"] for u in pending_updates]
    report["mean_abs_error_log_return"] = float(
        np.mean([abs(u["error_log_return"]) for u in pending_updates])
    )

    # --- Cierra el círculo: loguea la predicción de HOY para MAÑANA con el
    # modelo YA fine-tuneado, para que el ciclo de mañana tenga su baseline. ---
    log_report = log_next_day_predictions_for_universe()
    report["next_day_logging"] = log_report

    return report