#!/usr/bin/env python3
"""
train_pipeline.py — KodaQuant V5 MLOps: Reentrenamiento Incremental por Régimen
=================================================================================
Demonio/cronjob de fine-tuning incremental con validación estricta
"Champion vs. Challenger", **exclusivo para la arquitectura V5** (dos
especialistas independientes por régimen — `equity_specialist` /
`crypto_specialist` — bajo `services/kodaquant_models/<regimen>/`).

DEPRECACIÓN EXPLÍCITA: esta reescritura abandona por completo el modelo
GLOBAL V3/V4 (`attention_bilstm_global.keras`, 14 features, sin
NEWS_SENTIMENT_SCORE) y su dependencia de `services/quanti_engine.py`. Ese
modelo y ese pipeline de features quedaron huérfanos desde que
`train_kodaquant_v5.py` introdujo el enrutamiento por régimen; este script
ya no los referencia en absoluto.

DISEÑO — DECISIONES CRÍTICAS
-----------------------------
1. CERO reimplementación paralela del feature engineering NI del proveedor
   de datos de mercado. `engineer_asset`, `TECH_COLS`, `N_FEATURES`,
   `REGIMES`, `MACRO_TICKERS`, `LOOKBACK`, `_fetch_all_symbols_flat` y la
   arquitectura/pérdida (`BahdanauAttention`, `DirectionalGaussianNLL`,
   `mu_mae_metric`, `directional_accuracy_metric`) se importan DIRECTAMENTE
   desde `train_kodaquant_v5.py` — el mismo módulo que ejecuta el
   full-retrain. `engineer_asset` a su vez invoca
   `data_pipeline.get_daily_news_sentiment` internamente y
   `_fetch_all_symbols_flat` ya resuelve Twelve Data/FRED/Stooq con
   normalización tz-naive diaria (`market_data.py`, única fuente de verdad
   de proveedores desde V11) — así que las 17 features (LOG_RETURN_1D +
   11 técnicos incl. NEWS_SENTIMENT_SCORE/ADX_14/STOCH_K_14 + 5 macro)
   llegan sin reimplementar nada, sin riesgo de "training/fine-tuning
   skew" y sin una segunda ruta de descarga (yfinance) desincronizada de la
   que ya usa `train_kodaquant_v5.py` en producción.

2. Los `feature_scalers` / `target_scalers` de cada especialista NUNCA se
   re-ajustan (`.fit`) aquí — solo `.transform()` / `.inverse_transform()`.
   El fine-tuning es incremental sobre el mismo espacio de escalado con el
   que `run_regime_pipeline()` entrenó el especialista. Si el esquema de
   features cambia, el retrain completo sigue siendo responsabilidad de
   `train_kodaquant_v5.py`, no de este script.

3. DOS modelos INDEPENDIENTES. El ciclo drift -> fine-tune -> promote corre
   de forma AISLADA por régimen: jamás se mezclan datasets, scalers,
   embeddings de asset_id ni estado `baseline_mae` entre `equity_specialist`
   y `crypto_specialist`. Cada régimen tiene su propio `mlops_state.json` y
   su propio directorio de backups bajo su carpeta en `kodaquant_models/`.

4. RUTEO DINÁMICO equity <-> crypto (Requerimiento 3). `_resolve_regime()`
   determina a qué especialista pertenece un ticker: primero contra el
   mapeo real ya entrenado (`REGIMES`), con fallback heurístico por sufijo
   para tickers que aún no pasaron por un full retrain. Un ticker fuera del
   `asset_to_id` ya entrenado del especialista NO puede fine-tunearse (el
   embedding no tiene fila para él) — se omite con un aviso explícito
   pidiendo un full retrain.

5. Ningún workaround de "dropout congelado" (`_make_trainable_graph` de la
   versión V3/V4). Ese parche existía únicamente en el *bridge* de
   inferencia de `quanti_engine.py` (MC-Dropout con `training=True`
   horneado para simulación de paths) — el artefacto crudo `model_v5.keras`
   que guarda `run_regime_pipeline()` NO tiene ese bridge, así que
   `Dropout` se comporta de forma estándar: activo en `.fit()`, apagado en
   `.evaluate()`/`.predict()`, sin reconstrucción de grafo necesaria.

NUEVO EN V14.1 — SINCRONIZACIÓN CON `train_kodaquant_v5.py` (V6+) Y
ELIMINACIÓN DE YFINANCE (auditoría de compatibilidad al recibir los
módulos reales `market_data.py`/`data_pipeline.py`, antes ausentes):
    a) `train_kodaquant_v5.py` migró, desde su V6, la cabeza de salida del
       modelo de un único escalar a un head dual `Dense(2)` = [mu,
       log_var], entrenado con `DirectionalGaussianNLL` (no
       `DirectionalHuberLoss`, que el propio módulo deja "sin uso, como
       referencia histórica"). Esta versión de `train_pipeline.py`
       compilaba el Challenger con `DirectionalHuberLoss` sobre un modelo
       de 2 columnas de salida (mismatch de forma silencioso: Huber
       elemento-a-elemento entre `y_true` (N,1) y `y_pred` (N,2) nunca
       levanta un error claro, corrompe el gradiente) y extraía
       predicciones con `.reshape(-1, 1)` sobre un array (N, 2) -- eso NO
       selecciona la columna `mu`, INTERCALA mu y log_var en una sola
       columna de longitud 2N. Fix: import de `DirectionalGaussianNLL`/
       `mu_mae_metric` (idénticos hiperparámetros que produce el
       especialista Champion) y `y_pred[:, 0:1]` explícito en cada punto
       donde se extraía `mu` de una predicción cruda.
    b) `_download_regime_history` llamaba `yf.download` directo -- la
       fuente que TODO el resto del sistema retiró explícitamente
       (`market_data.py`: "Reemplaza por completo yfinance ... bloqueos
       recurrentes de Yahoo Finance / errores Proxy CONNECT aborted en
       producción"; `data_pipeline.py`: "yfinance retirado por completo").
       Además hacía un `.ffill().dropna(how="all")` CIEGO sobre el frame
       combinado ANTES de pasarlo a `engineer_asset` -- exactamente el
       patrón que la auditoría V13 de `train_kodaquant_v5.py` identificó
       como la causa raíz de aniquilar ~2/7 de un dataset 24/7 (cripto)
       al tratar NaN de fin de semana en columnas macro igual que un NaN
       real del propio activo. Fix: reemplazado por
       `_fetch_all_symbols_flat` (importado de `train_kodaquant_v5.py`,
       misma normalización tz-naive + outer join que usa el full-retrain),
       SIN ffill/dropna adicional -- ese trabajo es responsabilidad
       exclusiva de `engineer_asset`, que ya lo hace bien.
    c) Universo cripto reexpresado en formato estándar `BASE/QUOTE`
       (`BTC/USD`, ver V12 de `train_kodaquant_v5.py`) -- el heurístico de
       ruteo dinámico (`_CRYPTO_SUFFIXES`) y el `--tickers` de CLI seguían
       documentados/testeados contra el viejo sufijo `-USD` de yfinance.
       Se añade `_normalize_ticker_symbol` (acepta ambas notaciones en el
       input del usuario, normaliza a `/` antes de rutear) para no romper
       silenciosamente el ruteo si alguien pasa `BTC-USD` por costumbre.

Requisitos: mismos que `train_kodaquant_v5.py` (`keras`, `tensorflow`,
`pandas`, `numpy`, `scikit-learn`, `nltk`/`requests` para `data_pipeline`,
`requests` para `market_data`) más `scipy` para el test estadístico
Champion vs. Challenger. Debe vivir en el mismo directorio que
`train_kodaquant_v5.py`, `market_data.py` y `data_pipeline.py`.

Uso:
    python train_pipeline.py                             # ambos regímenes, respeta drift
    python train_pipeline.py --force-retrain              # fuerza fine-tuning en ambos regímenes
    python train_pipeline.py --regimes crypto_specialist --months 12 --lr 5e-6
    python train_pipeline.py --tickers TSLA,BTC/USD        # rutea dinámicamente y corre solo esos especialistas
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

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from scipy import stats
except ImportError as exc:  # noqa: BLE001
    raise ImportError(
        "train_pipeline.py requiere scipy para el test estadístico pareado "
        "Champion vs. Challenger (ttest_rel). Instala con: pip install scipy"
    ) from exc

# ---------------------------------------------------------------------------
# Bootstrap de path — hace visible `train_kodaquant_v5.py` / `data_pipeline.py`
# sin importar desde qué cwd dispare el cron.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    from train_kodaquant_v5 import (  # noqa: E402
        BahdanauAttention,
        DirectionalGaussianNLL,
        GAMMA_MAX,
        GAUSSIAN_NLL_BETA,
        GRAD_CLIPNORM,
        LOG_VAR_BARRIER_LAMBDA,
        LOG_VAR_L2_LAMBDA,
        LOG_VAR_MAX,
        LOG_VAR_MIN,
        MACRO_TICKERS,
        MODELS_ROOT,
        N_FEATURES,
        REGIMES,
        TECH_COLS,
        _fetch_all_symbols_flat,
        directional_accuracy_metric,
        engineer_asset,
        mu_mae_metric,
    )
except ImportError as exc:  # noqa: BLE001
    raise ImportError(
        "No se pudo importar train_kodaquant_v5.py. train_pipeline.py debe "
        "vivir en el mismo directorio que train_kodaquant_v5.py, "
        "market_data.py y data_pipeline.py, con Keras 3 disponible ahí. "
        f"Detalle: {exc!r}"
    ) from exc

# Requerimiento 1 — integración 100% explícita con el pipeline NLP de
# sentimiento. No se llama directamente en este módulo (ya la invoca
# `engineer_asset` internamente), pero el import se deja EXPLÍCITO y a nivel
# de módulo para que un `data_pipeline.py` roto (nltk/VADER/Finnhub faltante,
# etc.) falle aquí, de inmediato, en vez de a mitad de una descarga
# multi-proveedor (Twelve Data/FRED/Stooq) de varios minutos dentro del
# ciclo de fine-tuning.
try:
    from data_pipeline import get_daily_news_sentiment  # noqa: F401,E402
except ImportError as exc:  # noqa: BLE001
    raise ImportError(
        "No se pudo importar data_pipeline.py (requerido por engineer_asset "
        f"para NEWS_SENTIMENT_SCORE). Detalle: {exc!r}"
    ) from exc


# ---------------------------------------------------------------------------
# Ruteo dinámico de régimen (Requerimiento 3)
# ---------------------------------------------------------------------------
TICKER_TO_REGIME: dict[str, str] = {
    ticker.upper(): regime_name
    for regime_name, cfg in REGIMES.items()
    for ticker in cfg["tickers"]
}
# V14.1 — universo cripto real vive en notación "BASE/QUOTE" (`BTC/USD`, ver
# V12 de train_kodaquant_v5.py: el guion `BTC-USD` era un artefacto propio
# del ticker de yfinance, ya retirado). `_CRYPTO_QUOTE_SUFFIXES` alimenta
# tanto el heurístico de fallback (activo nuevo, aún sin full retrain) como
# `_normalize_ticker_symbol` (abajo), que acepta la notación legacy con
# guion en el input del usuario/CLI y la reescribe a `/` antes de rutear.
_CRYPTO_QUOTE_SUFFIXES = ("USD", "USDT", "USDC", "BTC", "ETH")


def _normalize_ticker_symbol(ticker: str) -> str:
    """
    `BTC-USD`/`btc-usd` (notación legacy de yfinance, aún común en input
    humano/scripts viejos) -> `BTC/USD` (notación real del universo
    entrenado). No-op para cualquier ticker que ya venga en formato
    estándar (equity plano o cripto con `/`) o que no calce el patrón
    `BASE-QUOTE` de un par cripto reconocido.
    """
    t = ticker.strip().upper()
    if "-" in t and "/" not in t:
        base, _, quote = t.partition("-")
        if quote in _CRYPTO_QUOTE_SUFFIXES:
            return f"{base}/{quote}"
    return t


def _resolve_regime(ticker: str) -> str:
    """
    Decide a qué especialista V5 pertenece un ticker. Prioriza el universo
    REAL ya entrenado (`REGIMES`); si el ticker es nuevo (aún no pasó por un
    full retrain de `train_kodaquant_v5.py`), cae a una heurística de sufijo
    para pares cripto (`BTC/USD`, `SOL/USDT`, ...; también acepta el guion
    legacy vía `_normalize_ticker_symbol`).
    """
    ticker_norm = _normalize_ticker_symbol(ticker)
    if ticker_norm in TICKER_TO_REGIME:
        return TICKER_TO_REGIME[ticker_norm]
    if "/" in ticker_norm and ticker_norm.split("/")[-1] in _CRYPTO_QUOTE_SUFFIXES \
            and "crypto_specialist" in REGIMES:
        return "crypto_specialist"
    return "equity_specialist" if "equity_specialist" in REGIMES else next(iter(REGIMES))


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MLOpsConfig:
    """Parámetros del ciclo MLOps V5. Todos con default institucional razonable."""

    log_dir: Path = MODELS_ROOT / "logs"

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
    logger = logging.getLogger("kodaquant.mlops_v5")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(
            fmt="[%(asctime)s] - [%(levelname)s] - [%(module)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(log_dir / "mlops_pipeline_v5.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    return logger


# ---------------------------------------------------------------------------
# KodaQuantV5MLOps
# ---------------------------------------------------------------------------

class KodaQuantV5MLOps:
    """
    Orquestador del ciclo de vida de los especialistas V5: ingestión (OHLCV +
    NEWS_SENTIMENT_SCORE vía `data_pipeline`), detección de drift, fine-tuning
    incremental, comparación estadística Champion vs. Challenger y promoción
    atómica — todo AISLADO por régimen (`equity_specialist`/`crypto_specialist`).
    """

    def __init__(self, config: MLOpsConfig | None = None) -> None:
        self.config = config or MLOpsConfig()
        self.logger = _configure_logging(self.config.log_dir)
        np.random.seed(self.config.random_seed)

    # ------------------------------------------------------------------ #
    # Estado persistente (baseline de MAE entre corridas) — POR RÉGIMEN
    # ------------------------------------------------------------------ #

    def _state_path(self, regime_name: str) -> Path:
        return MODELS_ROOT / regime_name / "mlops_state.json"

    def _load_state(self, regime_name: str) -> dict[str, Any]:
        state_path = self._state_path(regime_name)
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                self.logger.warning(
                    "[%s] No se pudo leer %s (%s) — se reinicia el estado.",
                    regime_name, state_path, exc,
                )
        return {"baseline_mae": {}, "last_run": None, "history": []}

    def _save_state(self, regime_name: str, state: dict[str, Any]) -> None:
        state_path = self._state_path(regime_name)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = state_path.with_suffix(".tmp.json")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, state_path)

    # ------------------------------------------------------------------ #
    # Carga de artefactos de producción del especialista V5 — POR RÉGIMEN
    # ------------------------------------------------------------------ #

    def _load_scalers(self, regime_name: str) -> dict[str, Any]:
        scalers_path = MODELS_ROOT / regime_name / "scalers_dict.pkl"
        if not scalers_path.exists():
            raise FileNotFoundError(
                f"No se encontró scalers_dict.pkl para '{regime_name}' en {scalers_path}. "
                "Ejecuta primero train_kodaquant_v5.py (full retrain) para generar el especialista."
            )
        with open(scalers_path, "rb") as fh:
            scalers = pickle.load(fh)

        # Validación defensiva V5 (mismo espíritu que quanti_engine._get_scalers
        # en V3): si el bundle viene de un TECH_COLS distinto al vigente, el
        # feature_scaler.transform() desplazaría cada columna EN SILENCIO.
        saved_tech_cols = scalers.get("tech_cols")
        if saved_tech_cols is not None and list(saved_tech_cols) != TECH_COLS:
            raise ValueError(
                f"[{regime_name}] Desalineación de TECH_COLS: scalers_dict.pkl tiene "
                f"{list(saved_tech_cols)} pero train_kodaquant_v5.py define {TECH_COLS} "
                "vigente. Regenera el especialista con un full retrain antes de fine-tunear."
            )
        return scalers

    def _load_champion(self, regime_name: str):
        model_path = MODELS_ROOT / regime_name / "model_v5.keras"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo Champion V5 de '{regime_name}' en {model_path}. "
                "Ejecuta primero train_kodaquant_v5.py."
            )
        return keras.models.load_model(
            str(model_path),
            custom_objects={
                "BahdanauAttention": BahdanauAttention,
                "DirectionalGaussianNLL": DirectionalGaussianNLL,
                "mu_mae_metric": mu_mae_metric,
                "directional_accuracy_metric": directional_accuracy_metric,
            },
            compile=False,
        )

    # ------------------------------------------------------------------ #
    # 1. DATA INGESTION & FEATURE ENGINEERING — reuso 100% de V5
    # ------------------------------------------------------------------ #

    def _download_regime_history(
        self, tickers: list[str], macro_tickers: list[str], months: int
    ) -> pd.DataFrame:
        """
        Descarga OHLCV fresca (SIN cache local en parquet — a diferencia de
        `download_all` de train_kodaquant_v5.py, un ciclo incremental
        necesita el dato del día, nunca un artefacto potencialmente viejo)
        para todos los tickers del régimen + macro factores.

        V14.1 — reemplaza `yf.download` (yfinance, retirado en TODO el
        resto del sistema por bloqueos/inestabilidad recurrentes, ver
        `market_data.py`) por `_fetch_all_symbols_flat`, la MISMA función
        que usa `train_kodaquant_v5.py` para el full retrain: cascada
        Twelve Data -> FRED -> Stooq, con normalización de índice tz-naive
        diaria aplicada a CADA pieza ANTES del outer join (`market_data.py`
        + `_normalize_daily_index`, fix V13 — evita la duplicación
        tz-aware/tz-naive de entradas por día calendario que corrompía
        silenciosamente el dataset de `crypto_specialist`).

        Deliberadamente SIN ffill/dropna acá: ese trabajo es responsabilidad
        exclusiva de `engineer_asset` (ffill(limit=4) SOLO sobre columnas
        macro + dropna sobre el OHLCV propio del activo, fix V13) — un
        `.ffill().dropna()` ciego aplicado ACÁ, antes de `engineer_asset`,
        reintroduciría exactamente el bug que esa auditoría corrigió
        (aniquilar filas de fin de semana de un activo 24/7 al tratarlas
        igual que un NaN real).
        """
        period = f"{max(1, int(months))}mo"  # _period_to_sessions ya entiende el sufijo "mo"
        flat = _fetch_all_symbols_flat(tickers, macro_tickers, period)

        required = {f"{t}_{f}" for t in tickers for f in ("Close", "High", "Low", "Volume")}
        required |= {f"{m}_Close" for m in macro_tickers}
        missing = required - set(flat.columns)
        if missing:
            raise RuntimeError(
                f"Faltan columnas tras la descarga multi-proveedor (Twelve Data/FRED/Stooq): "
                f"{sorted(missing)} -- ver logs de _fetch_all_symbols_flat para el detalle "
                f"por símbolo (un ticker que falle en TODOS los proveedores se omite, no aborta)."
            )
        return flat

    def _build_windows_existing_scalers(
        self,
        df: pd.DataFrame,
        ticker: str,
        lookback: int,
        feature_scaler,
        target_scaler,
        asset_id: int,
        feature_cols: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Ventanas 3D `(samples, lookback, N_FEATURES)` — misma indexación
        EXACTA que `build_asset_dataset` de train_kodaquant_v5.py (ventana
        `[t-lookback:t]` -> target `log_returns[t-1]`, fecha `df.index[t]`),
        pero aplicando SOLO `.transform()` sobre los scalers ya fit por el
        último full retrain — jamás se re-ajustan aquí.
        """
        # V6: `engineer_asset()` ya no expone "PRICE" (nivel absoluto) — el
        # bookkeeping de precio crudo vive en RAW_CLOSE (excluida de
        # `feature_cols`/del tensor, ver train_kodaquant_v5.py). Usar la
        # columna vieja aquí desalinearía el target de log-return en
        # silencio (KeyError en el mejor caso, ya que "PRICE" no existe más).
        prices = df["RAW_CLOSE"].values
        features = df[feature_cols].values
        log_returns = np.diff(np.log(prices))
        features_scaled = feature_scaler.transform(features)

        n = len(df)
        X, y_raw, dates = [], [], []
        for t in range(lookback, n):
            X.append(features_scaled[t - lookback: t])
            y_raw.append(log_returns[t - 1])
            dates.append(df.index[t])

        if not X:
            empty = np.empty((0, lookback, len(feature_cols)), dtype=np.float32)
            return empty, np.empty((0, 1), dtype=np.float32), np.empty((0,), dtype=np.int32), pd.DatetimeIndex([])

        X = np.array(X, dtype=np.float32)
        assert X.shape[-1] == N_FEATURES, (
            f"[{ticker}] Desalineación de features: la ventana tiene {X.shape[-1]} "
            f"columnas pero N_FEATURES={N_FEATURES} (TECH_COLS={TECH_COLS})."
        )

        y_raw = np.array(y_raw, dtype=np.float32).reshape(-1, 1)
        y_scaled = target_scaler.transform(y_raw).astype(np.float32)
        asset_ids = np.full((len(X),), asset_id, dtype=np.int32)
        return X, y_scaled, asset_ids, pd.DatetimeIndex(dates)

    def _prepare_ticker_dataset(
        self,
        all_market_data: pd.DataFrame,
        ticker: str,
        macro_tickers: list[str],
        lookback: int,
        feature_scaler,
        target_scaler,
        asset_id: int,
    ):
        """Bit a bit `engineer_asset()` de train_kodaquant_v5.py (incl. NEWS_SENTIMENT_SCORE)."""
        df_asset = engineer_asset(all_market_data, ticker, macro_tickers)
        feature_cols = ["LOG_RETURN_1D"] + TECH_COLS + macro_tickers
        return self._build_windows_existing_scalers(
            df_asset, ticker, lookback, feature_scaler, target_scaler, asset_id, feature_cols
        )

    # ------------------------------------------------------------------ #
    # 2. STATISTICAL DRIFT DETECTION — el vigilante, aislado por régimen
    # ------------------------------------------------------------------ #

    def evaluate_champion_drift(
        self, regime_name: str, champion, scalers: dict[str, Any]
    ) -> dict[str, dict[str, float]]:
        """
        Evalúa el Champion del régimen sobre los últimos `holdout_days`
        reales de CADA ticker de SU universo (`scalers["asset_to_id"]`).
        Devuelve MAE/MSE en escala REAL de log-return por ticker, más un
        agregado `__global__` ponderado por número de muestras — SOLO de
        este régimen, nunca mezclado con el otro especialista.
        """
        lookback = scalers["lookback"]
        macro_tickers = scalers["macro_tickers"]
        asset_to_id = scalers["asset_to_id"]
        tickers = list(asset_to_id.keys())

        drift_check_months = max(3, self.config.training_history_months // 4)
        all_market_data = self._download_regime_history(tickers, macro_tickers, drift_check_months)

        results: dict[str, dict[str, float]] = {}
        pooled_abs_err: list[np.ndarray] = []

        for ticker in tickers:
            try:
                X, y_scaled, asset_ids, _dates = self._prepare_ticker_dataset(
                    all_market_data, ticker, macro_tickers, lookback,
                    scalers["feature_scalers"][ticker], scalers["target_scalers"][ticker],
                    asset_to_id[ticker],
                )
            except Exception as exc:  # noqa: BLE001 — un ticker caído no debe tumbar el chequeo del régimen
                self.logger.warning("[%s] Drift check omitido para %s: %r", regime_name, ticker, exc)
                continue

            if len(X) == 0:
                continue
            if len(X) < self.config.holdout_days:
                self.logger.warning(
                    "[%s] Drift check: %s solo tiene %d ventanas (< holdout_days=%d) — se usa todo lo disponible.",
                    regime_name, ticker, len(X), self.config.holdout_days,
                )

            X_recent = X[-self.config.holdout_days:]
            y_recent_scaled = y_scaled[-self.config.holdout_days:]
            asset_recent = asset_ids[-self.config.holdout_days:].reshape(-1, 1)

            y_pred_scaled = np.asarray(
                champion.predict([X_recent, asset_recent], verbose=0)
            )[:, 0:1]  # V14.1: head dual [mu, log_var] -- extrae SOLO mu; .reshape(-1,1) intercalaba ambas columnas
            target_scaler = scalers["target_scalers"][ticker]
            y_pred_real = target_scaler.inverse_transform(y_pred_scaled).reshape(-1)
            y_true_real = target_scaler.inverse_transform(y_recent_scaled).reshape(-1)

            mae = float(mean_absolute_error(y_true_real, y_pred_real))
            mse = float(mean_squared_error(y_true_real, y_pred_real))
            results[ticker] = {"mae": mae, "mse": mse, "n_samples": float(len(y_true_real))}
            pooled_abs_err.append(np.abs(y_true_real - y_pred_real))

        if pooled_abs_err:
            pooled = np.concatenate(pooled_abs_err)
            results["__global__"] = {
                "mae": float(np.mean(pooled)),
                "mse": float(np.mean(pooled ** 2)),
                "n_samples": float(pooled.size),
            }

        return results

    def detect_drift(
        self, state: dict[str, Any], drift_metrics: dict[str, dict[str, float]]
    ) -> tuple[bool, str]:
        """
        No reentrena por defecto. Solo dispara si el MAE global actual del
        régimen supera el threshold DINÁMICO (`baseline * (1 + threshold)`)
        o el límite duro opcional. Sin baseline previo, se inicializa y NO
        se dispara en la primera corrida de ese régimen.
        """
        global_metrics = drift_metrics.get("__global__")
        if not global_metrics:
            return False, "Sin datos suficientes de ningún ticker — se omite el ciclo."

        current_mae = global_metrics["mae"]
        baseline_mae = state.get("baseline_mae", {}).get("__global__")

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
    # 3. INCREMENTAL TRAINING — el challenger, aislado por régimen
    # ------------------------------------------------------------------ #

    def build_finetune_dataset(
        self, regime_name: str, scalers: dict[str, Any]
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Dataset de fine-tuning combinando TODOS los tickers de ESTE régimen
        (mismo universo con el que el especialista fue entrenado). Split
        CRONOLÓGICO (no aleatorio, por ser serie de tiempo):
            [ ---------- train ---------- ][ val ][ holdout Champion/Challenger ]
        """
        lookback = scalers["lookback"]
        macro_tickers = scalers["macro_tickers"]
        asset_to_id = scalers["asset_to_id"]
        tickers = list(asset_to_id.keys())

        all_market_data = self._download_regime_history(
            tickers, macro_tickers, self.config.training_history_months
        )

        X_parts, y_parts, asset_parts, date_parts = [], [], [], []
        for ticker in tickers:
            try:
                X, y_scaled, asset_ids, dates = self._prepare_ticker_dataset(
                    all_market_data, ticker, macro_tickers, lookback,
                    scalers["feature_scalers"][ticker], scalers["target_scalers"][ticker],
                    asset_to_id[ticker],
                )
            except Exception as exc:  # noqa: BLE001 — un ticker caído no debe tumbar el fine-tuning del régimen
                self.logger.warning(
                    "[%s] Fine-tuning: %s omitido -- %r (mismo criterio de tolerancia a fallos "
                    "que evaluate_champion_drift).", regime_name, ticker, exc,
                )
                continue
            if len(X) == 0:
                self.logger.warning(
                    "[%s] Sin ventanas utilizables para %s — se omite del fine-tuning.",
                    regime_name, ticker,
                )
                continue
            X_parts.append(X)
            y_parts.append(y_scaled)
            asset_parts.append(asset_ids.reshape(-1, 1))
            date_parts.append(dates)

        if not X_parts:
            raise RuntimeError(
                f"[{regime_name}] No se pudo construir el dataset de fine-tuning: "
                "0 tickers con datos válidos."
            )

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        asset_ids = np.concatenate(asset_parts, axis=0)
        dates = pd.DatetimeIndex(np.concatenate([d.values for d in date_parts]))

        assert X.shape[-1] == N_FEATURES, (
            f"[{regime_name}] Desalineación de features: X tiene {X.shape[-1]} columnas "
            f"pero N_FEATURES={N_FEATURES} (TECH_COLS={TECH_COLS})."
        )

        order = np.argsort(dates.values)
        X, y, asset_ids = X[order], y[order], asset_ids[order]

        n = len(X)
        n_tickers = len(tickers)
        n_holdout = max(1, min(self.config.holdout_days * n_tickers, n // 5))
        n_val = max(1, min(self.config.validation_split_days * n_tickers, (n - n_holdout) // 5))

        if n - n_holdout - n_val <= 0:
            raise RuntimeError(
                f"[{regime_name}] Dataset insuficiente para separar train/val/holdout "
                f"(n={n}, holdout={n_holdout}, val={n_val}). Aumenta training_history_months."
            )

        holdout_slice = slice(n - n_holdout, n)
        val_slice = slice(n - n_holdout - n_val, n - n_holdout)
        train_slice = slice(0, n - n_holdout - n_val)

        return {
            "train": (X[train_slice], y[train_slice], asset_ids[train_slice]),
            "val": (X[val_slice], y[val_slice], asset_ids[val_slice]),
            "holdout": (X[holdout_slice], y[holdout_slice], asset_ids[holdout_slice]),
        }

    def build_challenger(self, champion):
        """
        Clona arquitectura Y pesos del Champion (`clone_model` + `set_weights`
        — objetos completamente independientes en memoria) y recompila con
        LR ultra-bajo (`1e-5` default) para fine-tuning conservador. Usa la
        MISMA `DirectionalGaussianNLL` (head dual [mu, log_var], V6+ de
        train_kodaquant_v5.py — NO `DirectionalHuberLoss`, que ese módulo
        mantiene solo como referencia histórica sin uso) con IDÉNTICOS
        hiperparámetros anti-colapso de varianza (β-NLL, barrera de
        log_var) con los que el Champion fue entrenado, y `gamma` fijo en
        `GAMMA_MAX` (curriculum ya completo — no se re-arranca el warmup en
        un fine-tuning incremental).
        """
        challenger = keras.models.clone_model(champion)
        challenger.set_weights(champion.get_weights())

        gamma_variable = keras.Variable(
            GAMMA_MAX, trainable=False, dtype="float32", name="directional_gamma_finetune"
        )
        challenger.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=self.config.finetune_learning_rate,
                weight_decay=1e-4,
                clipnorm=GRAD_CLIPNORM,
            ),
            loss=DirectionalGaussianNLL(
                gamma_directional=gamma_variable,
                beta=GAUSSIAN_NLL_BETA,
                log_var_min=LOG_VAR_MIN, log_var_max=LOG_VAR_MAX,
                log_var_l2_lambda=LOG_VAR_L2_LAMBDA,
                log_var_barrier_lambda=LOG_VAR_BARRIER_LAMBDA,
            ),
            metrics=[mu_mae_metric, directional_accuracy_metric],
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
    # 4. CHAMPION VS. CHALLENGER — A/B testing matemático, por régimen
    # ------------------------------------------------------------------ #

    def evaluate_holdout(
        self, model, holdout_data: tuple[np.ndarray, np.ndarray, np.ndarray], scalers: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, float]]:
        X_hold, y_hold_scaled, asset_hold = holdout_data
        id_to_ticker = {v: k for k, v in scalers["asset_to_id"].items()}
        flat_ids = asset_hold.reshape(-1)

        y_pred_scaled = np.asarray(model.predict([X_hold, asset_hold], verbose=0))[:, 0:1]  # V14.1: solo mu

        y_true_real = np.empty(flat_ids.shape[0], dtype=np.float64)
        y_pred_real = np.empty(flat_ids.shape[0], dtype=np.float64)
        for asset_id in np.unique(flat_ids):
            ticker = id_to_ticker[int(asset_id)]
            mask = flat_ids == asset_id
            scaler = scalers["target_scalers"][ticker]
            y_true_real[mask] = scaler.inverse_transform(y_hold_scaled[mask].reshape(-1, 1)).reshape(-1)
            y_pred_real[mask] = scaler.inverse_transform(y_pred_scaled[mask].reshape(-1, 1)).reshape(-1)

        abs_err = np.abs(y_true_real - y_pred_real)
        metrics = {
            "mae": float(mean_absolute_error(y_true_real, y_pred_real)),
            "mse": float(mean_squared_error(y_true_real, y_pred_real)),
            "n_samples": int(len(y_true_real)),
        }
        return abs_err, metrics

    def compare_champion_challenger(
        self, champion, challenger,
        holdout_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        scalers: dict[str, Any],
    ) -> dict[str, Any]:
        """
        REGLA DE NEGOCIO INQUEBRANTABLE: el Challenger solo gana si su MAE
        en Holdout es MENOR **y** esa diferencia es estadísticamente
        significativa — test t pareado sobre el error absoluto por muestra,
        en escala REAL de log-return, dentro del universo de ESTE régimen.
        """
        champion_err, champion_metrics = self.evaluate_holdout(champion, holdout_data, scalers)
        challenger_err, challenger_metrics = self.evaluate_holdout(challenger, holdout_data, scalers)

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

    def promote_challenger(self, regime_name: str, challenger, scalers: dict[str, Any]) -> None:
        """Overwrite atómico (`os.replace`) del `.keras`/scalers del especialista, con backup previo."""
        regime_dir = MODELS_ROOT / regime_name
        backup_dir = regime_dir / "mlops_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        model_path = regime_dir / "model_v5.keras"
        scalers_path = regime_dir / "scalers_dict.pkl"

        if model_path.exists():
            shutil.copy2(model_path, backup_dir / f"model_v5_{stamp}.keras")
        if scalers_path.exists():
            shutil.copy2(scalers_path, backup_dir / f"scalers_dict_{stamp}.pkl")

        tmp_model_path = model_path.with_suffix(".tmp.keras")
        challenger.save(str(tmp_model_path))
        os.replace(tmp_model_path, model_path)

        # Los scalers no cambian en un fine-tuning incremental (se reutilizan
        # los mismos, fit solo en el train histórico del último full retrain)
        # — se re-escriben tal cual para dejar el par modelo/scalers
        # consistente y con backup en el mismo timestamp de despliegue.
        tmp_scalers_path = scalers_path.with_suffix(".tmp.pkl")
        with open(tmp_scalers_path, "wb") as fh:
            pickle.dump(scalers, fh)
        os.replace(tmp_scalers_path, scalers_path)

        self.logger.info(
            "[%s] PROMOCIÓN: Challenger sobrescribió al Champion en producción (backup en %s).",
            regime_name, backup_dir,
        )

    # ------------------------------------------------------------------ #
    # 5. MEMORY LEAK PREVENTION
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
    # Orquestador — UN régimen
    # ------------------------------------------------------------------ #

    def run_cycle_for_regime(self, regime_name: str, force_retrain: bool = False) -> dict[str, Any]:
        cycle_start = datetime.now(timezone.utc)
        self.logger.info("==== [%s] CICLO MLOps V5 INICIADO ====", regime_name)
        state = self._load_state(regime_name)

        result: dict[str, Any] = {"status": "error", "reason": "ciclo no completado"}
        champion = None
        challenger = None
        dataset = None

        try:
            scalers = self._load_scalers(regime_name)
            champion = self._load_champion(regime_name)

            drift_metrics = self.evaluate_champion_drift(regime_name, champion, scalers)
            drift_detected, drift_reason = self.detect_drift(state, drift_metrics)
            if force_retrain:
                drift_detected, drift_reason = True, f"Forzado vía --force-retrain (chequeo real: {drift_reason})"
            self.logger.info("[%s] Chequeo de drift: %s", regime_name, drift_reason)

            global_metrics = drift_metrics.get("__global__")
            if global_metrics is not None:
                state.setdefault("baseline_mae", {}).setdefault("__global__", global_metrics["mae"])

            if not drift_detected:
                result = {"status": "no_drift", "reason": drift_reason, "drift_metrics": drift_metrics}
                return result

            self.logger.warning("[%s] DRIFT DETECTADO — %s. Se dispara el ciclo Challenger.", regime_name, drift_reason)

            dataset = self.build_finetune_dataset(regime_name, scalers)
            challenger = self.build_challenger(champion)
            self.fine_tune(challenger, dataset)

            comparison = self.compare_champion_challenger(champion, challenger, dataset["holdout"], scalers)
            self.logger.info(
                "[%s] Champion MAE: %.6f | Challenger MAE: %.6f | p-value: %s | Winner: %s",
                regime_name,
                comparison["champion"]["mae"],
                comparison["challenger"]["mae"],
                f"{comparison['p_value']:.4f}" if comparison["p_value"] is not None else "N/A",
                "CHALLENGER" if comparison["promote"] else "CHAMPION",
            )

            if comparison["promote"]:
                self.promote_challenger(regime_name, challenger, scalers)
                state.setdefault("baseline_mae", {})["__global__"] = comparison["challenger"]["mae"]
                result = {"status": "promoted", "comparison": comparison, "drift_metrics": drift_metrics}
            else:
                self.logger.warning(
                    "[%s] Degradación rechazada — Challenger NO superó estadísticamente al Champion "
                    "(mae_improved=%s, statistically_significant=%s). El Champion se mantiene en producción.",
                    regime_name, comparison["mae_improved"], comparison["statistically_significant"],
                )
                result = {"status": "rejected", "comparison": comparison, "drift_metrics": drift_metrics}

        except Exception as exc:  # noqa: BLE001 — el demonio nunca debe morir sin loggear
            self.logger.exception("[%s] Fallo no controlado durante el ciclo MLOps: %s", regime_name, exc)
            result = {"status": "error", "error": str(exc)}

        finally:
            state["last_run"] = cycle_start.isoformat()
            state.setdefault("history", []).append(
                {"timestamp": cycle_start.isoformat(), "status": result.get("status")}
            )
            state["history"] = state["history"][-50:]
            self._save_state(regime_name, state)
            self._release_memory(champion, challenger, dataset)
            self.logger.info(
                "==== [%s] CICLO MLOps V5 FINALIZADO (status=%s) ====",
                regime_name, result.get("status"),
            )

        return result

    # ------------------------------------------------------------------ #
    # Orquestador — TODOS los regímenes solicitados + ruteo dinámico
    # ------------------------------------------------------------------ #

    def run_cycle(
        self,
        regimes: list[str] | None = None,
        tickers: list[str] | None = None,
        force_retrain: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """
        Si `tickers` viene dado, cada ticker se rutea dinámicamente a su
        especialista (`_resolve_regime`) y se valida contra el
        `asset_to_id` YA entrenado de ese régimen — un ticker nuevo no puede
        fine-tunearse (el embedding no tiene fila para él) y se omite con un
        aviso pidiendo un full retrain. El fine-tuning sigue operando sobre
        el universo COMPLETO del régimen resuelto (el embedding y el
        backbone son compartidos; no tiene sentido fine-tunear con una
        fracción del universo entrenado). Sin `tickers`, corre TODOS los
        regímenes de `REGIMES` (o el subconjunto de `regimes`).
        """
        if tickers:
            grouped: dict[str, list[str]] = {}
            for raw_ticker in tickers:
                ticker = _normalize_ticker_symbol(raw_ticker)
                if not ticker:
                    continue
                regime_name = _resolve_regime(ticker)
                try:
                    scalers = self._load_scalers(regime_name)
                except (FileNotFoundError, ValueError) as exc:
                    self.logger.error("No se puede rutear %s -> '%s': %s", ticker, regime_name, exc)
                    continue
                if ticker not in scalers["asset_to_id"]:
                    self.logger.warning(
                        "%s no existe en el universo entrenado de '%s' (asset_to_id=%s). El "
                        "fine-tuning incremental NO puede añadir activos nuevos al embedding "
                        "— ejecuta train_kodaquant_v5.py (full retrain) para incorporarlo. Se omite.",
                        ticker, regime_name, list(scalers["asset_to_id"].keys()),
                    )
                    continue
                grouped.setdefault(regime_name, []).append(ticker)

            if not grouped:
                self.logger.error("Ningún ticker solicitado es procesable — abortando ciclo.")
                return {}

            for regime_name, matched in grouped.items():
                self.logger.info("Ruteo dinámico: %s -> especialista '%s'.", matched, regime_name)
            target_regimes = list(grouped.keys())
        else:
            target_regimes = regimes or list(REGIMES.keys())

        summary: dict[str, dict[str, Any]] = {}
        for regime_name in target_regimes:
            if regime_name not in REGIMES:
                self.logger.warning(
                    "Régimen desconocido '%s' — se omite (no existe en REGIMES de train_kodaquant_v5.py).",
                    regime_name,
                )
                summary[regime_name] = {"status": "error", "error": f"régimen desconocido: {regime_name}"}
                continue
            summary[regime_name] = self.run_cycle_for_regime(regime_name, force_retrain=force_retrain)

        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KodaQuant V5 MLOps — reentrenamiento incremental por régimen, "
                     "disparado por drift (Champion vs. Challenger).",
    )
    parser.add_argument("--months", type=int, default=None, help="Meses de histórico para fine-tuning.")
    parser.add_argument("--holdout-days", type=int, default=None, help="Días de holdout Champion vs. Challenger.")
    parser.add_argument("--drift-threshold", type=float, default=None, help="Umbral relativo de drift (ej. 0.05 = 5%%).")
    parser.add_argument("--hard-mae-limit", type=float, default=None, help="Límite duro de MAE (escala log-return).")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate del fine-tuning del Challenger.")
    parser.add_argument("--force-retrain", action="store_true", help="Ignora el chequeo de drift y fuerza el ciclo Challenger.")
    parser.add_argument(
        "--regimes", type=str, default=None,
        help="Lista separada por comas de regímenes a procesar (default: todos en REGIMES). Ej: equity_specialist",
    )
    parser.add_argument(
        "--tickers", type=str, default=None,
        help="Lista separada por comas de tickers a rutear dinámicamente hacia su especialista. "
             "Ej: TSLA,BTC/USD (también acepta BTC-USD, se normaliza automáticamente).",
    )
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

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()] if args.regimes else None
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else None

    pipeline = KodaQuantV5MLOps(MLOpsConfig(**overrides))
    summary = pipeline.run_cycle(regimes=regimes, tickers=tickers, force_retrain=args.force_retrain)

    ok_statuses = {"no_drift", "promoted", "rejected"}
    success = bool(summary) and all(r.get("status") in ok_statuses for r in summary.values())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()