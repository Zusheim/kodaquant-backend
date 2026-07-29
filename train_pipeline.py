#!/usr/bin/env python3
"""
train_pipeline.py — KodaQuant Terminal MLOps
==============================================
Demonio/cronjob de reentrenamiento incremental con validación estricta
"Champion vs. Challenger" para el modelo global Attention-BiLSTM
(`attention_bilstm_global.keras` + `scalers.pkl`) que consume
`services/quanti_engine.py` en producción.

DISEÑO — DECISIONES CRÍTICAS
-----------------------------
1. CERO reimplementación paralela del feature engineering. Se importan
   `_compute_rsi` / `_compute_macd` / `BahdanauAttention` DIRECTAMENTE
   desde `services.quanti_engine` — el mismo código que usa el motor de
   inferencia en vivo. Esto hace estructuralmente imposible que el
   preprocesamiento de entrenamiento diverja del de inferencia (la causa
   #1 de "training/serving skew" en sistemas de ML reales).

2. Los `feature_scalers` / `target_scalers` NUNCA se re-ajustan (`.fit`)
   aquí — solo `.transform()` / `.inverse_transform()`. El fine-tuning es
   incremental sobre el mismo espacio de escalado con el que el Champion
   fue entrenado originalmente en Colab. Si algún día cambia el esquema
   de escalado, el reentrenamiento completo sigue siendo responsabilidad
   del notebook de Colab, no de este script.

3. El modelo es GLOBAL (un único grafo, embedding `asset_id` por ticker),
   no un modelo por ticker. El drift se mide por ticker y se agrega; el
   fine-tuning combina TODOS los tickers del universo en un solo dataset.

4. `attention_bilstm_global.keras` tiene un nodo `dropout_regularizer`
   horneado con `training=False` (ver docstring de
   `quanti_engine._build_mc_dropout_bridge`) — ese mismo bug de
   congelamiento afecta también a `.fit()`, no solo a la inferencia. Este
   script reconstruye el tramo final del grafo con `training=None`
   (simbólico, NO hardcodeado a True como el bridge de inferencia) para
   que Keras propague el modo real automáticamente: dropout activo
   durante `.fit()`, apagado durante `.evaluate()`/`.predict()`.

Requisitos locales (macOS, Python 3.12): mismos que `quanti_engine.py`
(`keras`, `tensorflow`, `pandas`, `numpy`, `yfinance`, `scikit-learn`) más
`scipy` para el test estadístico Champion vs. Challenger.

Uso:
    python train_pipeline.py                     # ciclo normal (respeta drift)
    python train_pipeline.py --force-retrain      # ignora el chequeo de drift
    python train_pipeline.py --months 18 --lr 5e-6
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import pickle
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from scipy import stats
except ImportError as exc:  # noqa: BLE001
    raise ImportError(
        "train_pipeline.py requiere scipy para el test estadístico pareado "
        "Champion vs. Challenger (ttest_rel). Instala con: pip install scipy"
    ) from exc

# ---------------------------------------------------------------------------
# Bootstrap de path — hace visible el paquete `services/` sin importar desde
# qué cwd dispare el cron (`* * * * * cd /ruta && python train_pipeline.py`
# vs `python /ruta/train_pipeline.py` desde otro directorio).
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    from services.quanti_engine import (  # noqa: E402
        BahdanauAttention,
        ML_MODEL_PATH,
        ML_SCALERS_PATH,
        _MC_DROPOUT_DOWNSTREAM_LAYERS,
        _MC_DROPOUT_LAYER_NAME,
        _compute_macd,
        _compute_rsi,
        keras,
    )
except ImportError as exc:  # noqa: BLE001
    raise ImportError(
        "No se pudo importar services.quanti_engine. train_pipeline.py debe "
        "vivir en la raíz del backend (mismo nivel que la carpeta `services/`) "
        f"y requiere que Keras 3 cargue correctamente ahí. Detalle: {exc!r}"
    ) from exc


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MLOpsConfig:
    """Parámetros del ciclo MLOps. Todos con default institucional razonable."""

    model_path: Path = Path(ML_MODEL_PATH)
    scalers_path: Path = Path(ML_SCALERS_PATH)
    state_path: Path = Path(ML_MODEL_PATH).parent / "mlops_state.json"
    backup_dir: Path = Path(ML_MODEL_PATH).parent / "mlops_backups"
    log_dir: Path = _THIS_DIR / "logs"

    # Data ingestion
    training_history_months: int = 24

    # Drift detection ("el vigilante")
    holdout_days: int = 15
    drift_relative_threshold: float = 0.05       # 5% de desviación media vs. baseline
    drift_hard_mae_limit: float | None = None     # límite duro opcional, escala log-return

    # Incremental training ("el challenger")
    validation_split_days: int = 15
    finetune_learning_rate: float = 1e-5
    finetune_max_epochs: int = 50
    finetune_batch_size: int = 64
    early_stopping_patience: int = 5

    # Champion vs. Challenger (A/B matemático)
    promotion_min_relative_improvement: float = 0.0   # exige mejora estrictamente >0 además de p-value
    promotion_p_value_threshold: float = 0.05

    random_seed: int = 42


# ---------------------------------------------------------------------------
# Logging institucional — [TIMESTAMP] - [LEVEL] - [MODULE] - Message
# ---------------------------------------------------------------------------

def _configure_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("kodaquant.mlops")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(
            fmt="[%(asctime)s] - [%(levelname)s] - [%(module)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(log_dir / "mlops_pipeline.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    return logger


# ---------------------------------------------------------------------------
# KodaQuantMLOps
# ---------------------------------------------------------------------------

class KodaQuantMLOps:
    """
    Orquestador del ciclo de vida del modelo Bi-LSTM: ingestión, detección
    de drift, fine-tuning incremental, comparación estadística Champion vs.
    Challenger y promoción atómica a producción.
    """

    def __init__(self, config: MLOpsConfig | None = None) -> None:
        self.config = config or MLOpsConfig()
        self.logger = _configure_logging(self.config.log_dir)
        self.scalers: dict[str, Any] | None = None
        self.state: dict[str, Any] = {}
        np.random.seed(self.config.random_seed)

    # ------------------------------------------------------------------ #
    # Estado persistente (baseline de MAE entre corridas)
    # ------------------------------------------------------------------ #

    def _load_state(self) -> dict[str, Any]:
        if self.config.state_path.exists():
            try:
                with open(self.config.state_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                self.logger.warning(
                    "No se pudo leer %s (%s) — se reinicia el estado del pipeline.",
                    self.config.state_path, exc,
                )
        return {"baseline_mae": {}, "last_run": None, "history": []}

    def _save_state(self) -> None:
        self.config.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.config.state_path.with_suffix(".tmp.json")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.config.state_path)

    # ------------------------------------------------------------------ #
    # Carga de artefactos de producción
    # ------------------------------------------------------------------ #

    def _load_scalers(self) -> dict[str, Any]:
        if not self.config.scalers_path.exists():
            raise FileNotFoundError(f"No se encontró scalers.pkl en {self.config.scalers_path}")
        with open(self.config.scalers_path, "rb") as fh:
            return pickle.load(fh)

    def _load_champion_raw(self):
        """Carga el grafo Champion TAL CUAL vive en producción (sin bridge)."""
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo Champion en {self.config.model_path}")
        return keras.models.load_model(
            str(self.config.model_path),
            custom_objects={"BahdanauAttention": BahdanauAttention},
            compile=False,
        )

    def _make_trainable_graph(self, model):
        """
        Repara el nodo `dropout_regularizer` congelado en `training=False`
        para que sea entrenable: reconstruye únicamente el tramo final del
        grafo (`dropout_regularizer -> post_fusion_dense -> return_head`),
        reutilizando los MISMOS objetos de capa (mismos pesos, cero
        duplicación), con un nodo NUEVO donde `training=None` — a
        diferencia del bridge de inferencia de `quanti_engine.py` (que lo
        hornea en `True` para siempre), aquí se deja simbólico para que
        `.fit()` y `.evaluate()`/`.predict()` reciban el modo correcto
        automáticamente. Si el layer no existe en el grafo (arquitectura
        futura sin ese nodo congelado), se devuelve el modelo intacto.
        """
        try:
            dropout_layer = model.get_layer(_MC_DROPOUT_LAYER_NAME)
        except ValueError:
            return model

        try:
            fusion_tensor = dropout_layer.input
        except AttributeError:
            return model

        x = dropout_layer(fusion_tensor, training=None)
        for layer_name in _MC_DROPOUT_DOWNSTREAM_LAYERS:
            x = model.get_layer(layer_name)(x)

        return keras.Model(inputs=model.input, outputs=x, name=f"{model.name}_trainable")

    # ------------------------------------------------------------------ #
    # 1. DATA INGESTION & PREPROCESSING — replica exacta de quanti_engine
    # ------------------------------------------------------------------ #

    def _download_history(self, tickers: list[str], months: int) -> pd.DataFrame:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - pd.DateOffset(months=months)
        raw = yf.download(
            tickers,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        close = raw["Close"][tickers].ffill().dropna()
        if close.empty:
            raise RuntimeError(
                f"yfinance no devolvió datos utilizables para {tickers} "
                f"(desde {start_date.date()} hasta {end_date.date()})."
            )
        return close

    def _engineer_features(
        self, close: pd.DataFrame, ticker: str, macro_tickers: list[str]
    ) -> pd.DataFrame:
        """
        Bit a bit igual a `_fetch_feature_window` de quanti_engine.py:
        mismas funciones (`_compute_rsi`, `_compute_macd` IMPORTADAS, no
        reimplementadas), mismas columnas, mismo orden — condición
        necesaria para que `feature_scaler.transform()` reciba las
        columnas en la posición con la que fue entrenado.
        """
        df = close.copy()
        df["RSI_14"] = _compute_rsi(df[ticker])
        df["EMA_20"] = df[ticker].ewm(span=20, adjust=False).mean()
        macd_line, signal_line = _compute_macd(df[ticker])
        df["MACD"] = macd_line
        df["MACD_SIGNAL"] = signal_line
        df = df.ffill().dropna()

        feature_cols = [ticker, "RSI_14", "EMA_20", "MACD", "MACD_SIGNAL"] + macro_tickers
        return df[feature_cols]

    def _build_windows(
        self, feature_df: pd.DataFrame, ticker: str, lookback: int
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Ventanas deslizantes 3D `(samples, lookback, features)`. El target
        de la ventana que TERMINA en el índice `i` es el log-return real
        `día i -> día i+1` — exactamente la magnitud que predice el modelo
        en producción (ver docstring de `_forecast_asset`).
        """
        values = feature_df.values.astype(np.float64)
        close_prices = feature_df[ticker].to_numpy(dtype=np.float64)
        log_returns = np.diff(np.log(close_prices))  # len = n - 1

        n = values.shape[0]
        X: list[np.ndarray] = []
        y: list[float] = []
        dates: list[Any] = []

        for i in range(lookback - 1, n - 1):
            X.append(values[i - lookback + 1: i + 1])
            y.append(log_returns[i])
            dates.append(feature_df.index[i + 1])

        if not X:
            return (
                np.empty((0, lookback, values.shape[1])),
                np.empty((0,)),
                pd.DatetimeIndex([]),
            )

        return np.stack(X), np.asarray(y, dtype=np.float64), pd.DatetimeIndex(dates)

    def _prepare_ticker_dataset(
        self, ticker: str, macro_tickers: list[str], lookback: int, months: int
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        symbols = list(dict.fromkeys([ticker] + macro_tickers))
        close = self._download_history(symbols, months)
        feature_df = self._engineer_features(close, ticker, macro_tickers)
        return self._build_windows(feature_df, ticker, lookback)

    def _scale_dataset(
        self, X_raw: np.ndarray, y_raw: np.ndarray, ticker: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """SOLO `.transform()` — jamás se reajustan los scalers de producción."""
        feature_scaler = self.scalers["feature_scalers"][ticker]
        target_scaler = self.scalers["target_scalers"][ticker]
        n_samples, lookback, n_features = X_raw.shape

        X_scaled = (
            feature_scaler.transform(X_raw.reshape(-1, n_features))
            .reshape(n_samples, lookback, n_features)
            .astype(np.float32)
        )
        y_scaled = target_scaler.transform(y_raw.reshape(-1, 1)).reshape(-1).astype(np.float32)
        return X_scaled, y_scaled

    def _inverse_transform_targets(self, y_scaled: np.ndarray, asset_ids: np.ndarray) -> np.ndarray:
        """Inverse-transform por fila respetando el `target_scaler` propio de cada ticker."""
        id_to_ticker = {v: k for k, v in self.scalers["asset_to_id"].items()}
        flat_ids = asset_ids.reshape(-1)
        y_real = np.empty(flat_ids.shape[0], dtype=np.float64)

        for asset_id in np.unique(flat_ids):
            ticker = id_to_ticker[int(asset_id)]
            mask = flat_ids == asset_id
            scaler = self.scalers["target_scalers"][ticker]
            y_real[mask] = scaler.inverse_transform(
                y_scaled.reshape(-1)[mask].reshape(-1, 1)
            ).reshape(-1)

        return y_real

    # ------------------------------------------------------------------ #
    # 2. STATISTICAL DRIFT DETECTION — el vigilante
    # ------------------------------------------------------------------ #

    def evaluate_champion_drift(self, champion_raw) -> dict[str, dict[str, float]]:
        """
        Evalúa el Champion (determinista, `training=False` vía `.predict()`)
        sobre los últimos `holdout_days` reales de CADA ticker del universo
        entrenado. Devuelve MAE/MSE en escala REAL de log-return
        (post `inverse_transform`) por ticker, más un agregado `__global__`
        ponderado por número de muestras (no promedio simple de promedios).
        """
        lookback = self.scalers["lookback"]
        macro_tickers = self.scalers["macro_tickers"]
        asset_to_id = self.scalers["asset_to_id"]

        results: dict[str, dict[str, float]] = {}
        pooled_abs_err: list[np.ndarray] = []
        drift_check_months = max(3, self.config.training_history_months // 4)

        for ticker in asset_to_id:
            try:
                X_raw, y_raw, _dates = self._prepare_ticker_dataset(
                    ticker, macro_tickers, lookback, months=drift_check_months
                )
            except Exception as exc:  # noqa: BLE001 — un ticker caído no debe tumbar el chequeo global
                self.logger.warning("Drift check omitido para %s: %s", ticker, exc)
                continue

            if len(X_raw) == 0:
                continue
            if len(X_raw) < self.config.holdout_days:
                self.logger.warning(
                    "Drift check: %s solo tiene %d ventanas (< holdout_days=%d) — se usa todo lo disponible.",
                    ticker, len(X_raw), self.config.holdout_days,
                )

            X_recent = X_raw[-self.config.holdout_days:]
            y_recent = y_raw[-self.config.holdout_days:]

            X_scaled, _y_scaled = self._scale_dataset(X_recent, y_recent, ticker)
            asset_id = asset_to_id[ticker]
            asset_tensor = np.full((len(X_scaled), 1), asset_id, dtype=np.int32)

            y_pred_scaled = np.asarray(
                champion_raw.predict([X_scaled, asset_tensor], verbose=0)
            ).reshape(-1, 1)
            target_scaler = self.scalers["target_scalers"][ticker]
            y_pred_real = target_scaler.inverse_transform(y_pred_scaled).reshape(-1)

            mae = float(mean_absolute_error(y_recent, y_pred_real))
            mse = float(mean_squared_error(y_recent, y_pred_real))
            results[ticker] = {"mae": mae, "mse": mse, "n_samples": float(len(y_recent))}
            pooled_abs_err.append(np.abs(y_recent - y_pred_real))

        if pooled_abs_err:
            pooled = np.concatenate(pooled_abs_err)
            results["__global__"] = {
                "mae": float(np.mean(pooled)),
                "mse": float(np.mean(pooled ** 2)),
                "n_samples": float(pooled.size),
            }

        return results

    def detect_drift(self, drift_metrics: dict[str, dict[str, float]]) -> tuple[bool, str]:
        """
        No reentrena por defecto. Solo dispara si el MAE global actual
        supera el threshold DINÁMICO (`baseline * (1 + drift_relative_threshold)`)
        o el límite duro opcional. Sin baseline previo, se inicializa y NO
        se dispara en la primera corrida (nada con qué comparar todavía).
        """
        global_metrics = drift_metrics.get("__global__")
        if not global_metrics:
            return False, "Sin datos suficientes de ningún ticker — se omite el ciclo."

        current_mae = global_metrics["mae"]
        baseline_mae = self.state.get("baseline_mae", {}).get("__global__")

        if baseline_mae is None:
            self.logger.info(
                "No existe baseline_mae previo — se fija el MAE actual (%.6f) como baseline inicial.",
                current_mae,
            )
            return False, "baseline_inicializado"

        dynamic_threshold = baseline_mae * (1.0 + self.config.drift_relative_threshold)
        hard_limit = self.config.drift_hard_mae_limit

        triggered_dynamic = current_mae > dynamic_threshold
        triggered_hard = hard_limit is not None and current_mae > hard_limit

        if triggered_dynamic or triggered_hard:
            limit_desc = (
                f"límite duro={hard_limit:.6f}" if triggered_hard
                else f"threshold dinámico={dynamic_threshold:.6f} (baseline={baseline_mae:.6f}, "
                     f"desviación permitida={self.config.drift_relative_threshold:.0%})"
            )
            return True, f"MAE actual={current_mae:.6f} superó {limit_desc}."

        return False, (
            f"Sin drift — MAE actual={current_mae:.6f} dentro del threshold dinámico="
            f"{dynamic_threshold:.6f} (baseline={baseline_mae:.6f})."
        )

    # ------------------------------------------------------------------ #
    # 3. INCREMENTAL TRAINING — el challenger
    # ------------------------------------------------------------------ #

    def build_finetune_dataset(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Dataset GLOBAL de fine-tuning: combina TODOS los tickers del
        universo (mismo esquema que el modelo en producción). Split
        CRONOLÓGICO (no aleatorio, por ser serie de tiempo):
            [ ---------- train ---------- ][ val ][ holdout Champion/Challenger ]
        """
        lookback = self.scalers["lookback"]
        macro_tickers = self.scalers["macro_tickers"]
        asset_to_id = self.scalers["asset_to_id"]

        X_parts, y_parts, asset_parts, date_parts = [], [], [], []

        for ticker, asset_id in asset_to_id.items():
            X_raw, y_raw, dates = self._prepare_ticker_dataset(
                ticker, macro_tickers, lookback, months=self.config.training_history_months
            )
            if len(X_raw) == 0:
                self.logger.warning("Sin ventanas utilizables para %s — se omite del fine-tuning.", ticker)
                continue

            X_scaled, y_scaled = self._scale_dataset(X_raw, y_raw, ticker)
            X_parts.append(X_scaled)
            y_parts.append(y_scaled)
            asset_parts.append(np.full(len(X_scaled), asset_id, dtype=np.int32))
            date_parts.append(dates)

        if not X_parts:
            raise RuntimeError("No se pudo construir el dataset de fine-tuning: 0 tickers con datos válidos.")

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        asset_ids = np.concatenate(asset_parts, axis=0).reshape(-1, 1)
        dates = pd.DatetimeIndex(np.concatenate([d.values for d in date_parts]))

        order = np.argsort(dates.values)
        X, y, asset_ids = X[order], y[order], asset_ids[order]

        n = len(X)
        n_tickers = len(asset_to_id)
        n_holdout = max(1, min(self.config.holdout_days * n_tickers, n // 5))
        n_val = max(1, min(self.config.validation_split_days * n_tickers, (n - n_holdout) // 5))

        if n - n_holdout - n_val <= 0:
            raise RuntimeError(
                f"Dataset insuficiente para separar train/val/holdout (n={n}, holdout={n_holdout}, "
                f"val={n_val}). Aumenta `training_history_months` en MLOpsConfig."
            )

        holdout_slice = slice(n - n_holdout, n)
        val_slice = slice(n - n_holdout - n_val, n - n_holdout)
        train_slice = slice(0, n - n_holdout - n_val)

        return {
            "train": (X[train_slice], y[train_slice], asset_ids[train_slice]),
            "val": (X[val_slice], y[val_slice], asset_ids[val_slice]),
            "holdout": (X[holdout_slice], y[holdout_slice], asset_ids[holdout_slice]),
        }

    def build_challenger(self, champion_raw):
        """
        Clona arquitectura Y pesos del Champion (`clone_model` + `set_weights`
        — objetos completamente independientes en memoria, nunca referencias
        compartidas) y repara el nodo de dropout congelado para hacerlo
        entrenable. LR ultra-bajo (`1e-5` default) para Fine-Tuning
        conservador que evite Catastrophic Forgetting de la estructura
        macro histórica aprendida en Colab.
        """
        challenger = keras.models.clone_model(champion_raw)
        challenger.set_weights(champion_raw.get_weights())
        challenger = self._make_trainable_graph(challenger)

        challenger.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.finetune_learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        return challenger

    def fine_tune(self, challenger, dataset: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]):
        """Fine-tuning incremental con Early Stopping obligatorio anti-overfitting."""
        X_train, y_train, asset_train = dataset["train"]
        X_val, y_val, asset_val = dataset["val"]

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
        )

        return challenger.fit(
            [X_train, asset_train],
            y_train,
            validation_data=([X_val, asset_val], y_val),
            epochs=self.config.finetune_max_epochs,
            batch_size=self.config.finetune_batch_size,
            shuffle=True,
            verbose=0,
            callbacks=[early_stopping],
        )

    # ------------------------------------------------------------------ #
    # 4. CHAMPION VS. CHALLENGER — A/B testing matemático
    # ------------------------------------------------------------------ #

    def evaluate_holdout(
        self, model, holdout_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, dict[str, float]]:
        X_hold, y_hold_scaled, asset_hold = holdout_data

        y_pred_scaled = np.asarray(model.predict([X_hold, asset_hold], verbose=0)).reshape(-1)
        y_true_real = self._inverse_transform_targets(y_hold_scaled, asset_hold)
        y_pred_real = self._inverse_transform_targets(y_pred_scaled, asset_hold)

        abs_err = np.abs(y_true_real - y_pred_real)
        metrics = {
            "mae": float(mean_absolute_error(y_true_real, y_pred_real)),
            "mse": float(mean_squared_error(y_true_real, y_pred_real)),
            "n_samples": int(len(y_true_real)),
        }
        return abs_err, metrics

    def compare_champion_challenger(
        self, champion_raw, challenger, holdout_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> dict[str, Any]:
        """
        REGLA DE NEGOCIO INQUEBRANTABLE: el Challenger solo gana si su MAE
        en Holdout es MENOR **y** esa diferencia es estadísticamente
        significativa — test t pareado (mismas observaciones Holdout para
        ambos modelos) sobre el error absoluto por muestra, en escala REAL
        de log-return (post inverse_transform, por ticker).
        """
        champion_err, champion_metrics = self.evaluate_holdout(champion_raw, holdout_data)
        challenger_err, challenger_metrics = self.evaluate_holdout(challenger, holdout_data)

        try:
            t_stat, p_value = stats.ttest_rel(challenger_err, champion_err)
            t_stat, p_value = float(t_stat), float(p_value)
        except Exception as exc:  # noqa: BLE001 — el test nunca debe tumbar la comparación
            self.logger.warning("Test t pareado falló (%s) — se usa solo comparación directa de MAE.", exc)
            t_stat, p_value = None, 1.0

        mae_improved = challenger_metrics["mae"] < champion_metrics["mae"] * (
            1.0 - self.config.promotion_min_relative_improvement
        )
        statistically_significant = p_value < self.config.promotion_p_value_threshold
        promote = bool(mae_improved and statistically_significant)

        return {
            "champion": champion_metrics,
            "challenger": challenger_metrics,
            "t_stat": t_stat,
            "p_value": p_value,
            "mae_improved": mae_improved,
            "statistically_significant": statistically_significant,
            "promote": promote,
        }

    def promote_challenger(self, challenger) -> None:
        """Overwrite atómico (`os.replace`) del `.keras` de producción, con backup previo."""
        self.config.backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        if self.config.model_path.exists():
            shutil.copy2(self.config.model_path, self.config.backup_dir / f"champion_{stamp}.keras")
        if self.config.scalers_path.exists():
            shutil.copy2(self.config.scalers_path, self.config.backup_dir / f"scalers_{stamp}.pkl")

        tmp_model_path = self.config.model_path.with_suffix(".tmp.keras")
        challenger.save(str(tmp_model_path))
        os.replace(tmp_model_path, self.config.model_path)

        # Los scalers no cambian en un fine-tuning incremental (se reutilizan
        # los mismos, fit solo en el train histórico original en Colab) — se
        # re-escriben tal cual para dejar el par modelo/scalers consistente
        # y con backup en el mismo timestamp de despliegue.
        tmp_scalers_path = self.config.scalers_path.with_suffix(".tmp.pkl")
        with open(tmp_scalers_path, "wb") as fh:
            pickle.dump(self.scalers, fh)
        os.replace(tmp_scalers_path, self.config.scalers_path)

        self.logger.info("PROMOCIÓN: Challenger sobrescribió al Champion en producción (backup en %s).", self.config.backup_dir)
        self.logger.info(
            "IMPORTANTE: quanti_engine._get_keras_model()/_get_scalers() cachean vía @lru_cache — "
            "reinicia el proceso del backend FastAPI para que el nuevo Champion entre en producción."
        )

    # ------------------------------------------------------------------ #
    # 5. MEMORY LEAK PREVENTION — macOS strict
    # ------------------------------------------------------------------ #

    def _release_memory(self, *objs: Any) -> None:
        for obj in objs:
            del obj
        gc.collect()
        try:
            keras.backend.clear_session()
        except Exception as exc:  # noqa: BLE001 — la limpieza nunca debe tumbar el ciclo
            self.logger.debug("keras.backend.clear_session() no disponible/falló (%s) — se ignora.", exc)
        gc.collect()

    # ------------------------------------------------------------------ #
    # Orquestador principal
    # ------------------------------------------------------------------ #

    def run_cycle(self, force_retrain: bool = False) -> dict[str, Any]:
        cycle_start = datetime.now(timezone.utc)
        self.logger.info("==================== CICLO MLOps INICIADO ====================")
        self.state = self._load_state()

        result: dict[str, Any] = {"status": "error", "reason": "ciclo no completado"}
        champion_raw = None
        challenger = None
        dataset = None

        try:
            self.scalers = self._load_scalers()
            champion_raw = self._load_champion_raw()

            drift_metrics = self.evaluate_champion_drift(champion_raw)
            drift_detected, drift_reason = self.detect_drift(drift_metrics)
            if force_retrain:
                drift_detected, drift_reason = True, f"Forzado vía --force-retrain (chequeo real: {drift_reason})"
            self.logger.info("Chequeo de drift: %s", drift_reason)

            global_metrics = drift_metrics.get("__global__")
            if global_metrics is not None:
                self.state.setdefault("baseline_mae", {}).setdefault("__global__", global_metrics["mae"])

            if not drift_detected:
                result = {"status": "no_drift", "reason": drift_reason, "drift_metrics": drift_metrics}
                return result

            self.logger.warning("DRIFT DETECTADO — %s. Se dispara el ciclo Challenger.", drift_reason)

            dataset = self.build_finetune_dataset()
            challenger = self.build_challenger(champion_raw)
            self.fine_tune(challenger, dataset)

            comparison = self.compare_champion_challenger(champion_raw, challenger, dataset["holdout"])
            self.logger.info(
                "Champion MAE: %.6f | Challenger MAE: %.6f | p-value: %s | Winner: %s",
                comparison["champion"]["mae"],
                comparison["challenger"]["mae"],
                f"{comparison['p_value']:.4f}" if comparison["p_value"] is not None else "N/A",
                "CHALLENGER" if comparison["promote"] else "CHAMPION",
            )

            if comparison["promote"]:
                self.promote_challenger(challenger)
                self.state.setdefault("baseline_mae", {})["__global__"] = comparison["challenger"]["mae"]
                result = {"status": "promoted", "comparison": comparison, "drift_metrics": drift_metrics}
            else:
                self.logger.warning(
                    "Degradación rechazada — Challenger NO superó estadísticamente al Champion "
                    "(mae_improved=%s, statistically_significant=%s). El Champion se mantiene en producción.",
                    comparison["mae_improved"], comparison["statistically_significant"],
                )
                result = {"status": "rejected", "comparison": comparison, "drift_metrics": drift_metrics}

        except Exception as exc:  # noqa: BLE001 — el demonio nunca debe morir sin loggear
            self.logger.exception("Fallo no controlado durante el ciclo MLOps: %s", exc)
            result = {"status": "error", "error": str(exc)}

        finally:
            self.state["last_run"] = cycle_start.isoformat()
            self.state.setdefault("history", []).append(
                {"timestamp": cycle_start.isoformat(), "status": result.get("status")}
            )
            self.state["history"] = self.state["history"][-50:]
            self._save_state()
            self._release_memory(champion_raw, challenger, dataset)
            self.logger.info(
                "==================== CICLO MLOps FINALIZADO (status=%s) ====================",
                result.get("status"),
            )

        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KodaQuant MLOps — reentrenamiento incremental disparado por drift (Champion vs. Challenger).",
    )
    parser.add_argument("--months", type=int, default=None, help="Meses de histórico para fine-tuning.")
    parser.add_argument("--holdout-days", type=int, default=None, help="Días de holdout Champion vs. Challenger.")
    parser.add_argument("--drift-threshold", type=float, default=None, help="Umbral relativo de drift (ej. 0.05 = 5%%).")
    parser.add_argument("--hard-mae-limit", type=float, default=None, help="Límite duro de MAE (escala log-return).")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate del fine-tuning del Challenger.")
    parser.add_argument("--force-retrain", action="store_true", help="Ignora el chequeo de drift y fuerza el ciclo Challenger.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    overrides: dict[str, Any] = {}
    if args.months is not None:
        overrides["training_history_months"] = args.months
    if args.holdout_days is not None:
        overrides["holdout_days"] = args.holdout_days
    if args.drift_threshold is not None:
        overrides["drift_relative_threshold"] = args.drift_threshold
    if args.hard_mae_limit is not None:
        overrides["drift_hard_mae_limit"] = args.hard_mae_limit
    if args.lr is not None:
        overrides["finetune_learning_rate"] = args.lr

    pipeline = KodaQuantMLOps(MLOpsConfig(**overrides))
    result = pipeline.run_cycle(force_retrain=args.force_retrain)

    sys.exit(0 if result.get("status") in {"no_drift", "promoted", "rejected"} else 1)


if __name__ == "__main__":
    main()