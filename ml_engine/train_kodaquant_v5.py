# %%
#!pip install yfinance nltk -q

"""
train_kodaquant_v5.py — KodaQuant V5: Enrutamiento de Modelos Especialistas
=============================================================================
Refactor arquitectónico respecto a V4 (modelo GLOBAL único, 10 activos
mezclados en el mismo espacio latente). DIAGNÓSTICO V4: forzar Cripto
(volatilidad extrema, 24/7) y Acciones (volatilidad tradicional, horario
cerrado) dentro del mismo embedding compartido corrompió la optimización de
pesos -> edge >50% en cripto, colapso a 34-38% en acciones.

FIX V5 — ENRUTAMIENTO DE MODELOS ESPECIALISTAS POR RÉGIMEN: en vez de UN
modelo global, este script entrena DOS ciclos de optimización completamente
independientes y secuenciales, uno por régimen:

    EQUITY  -> ['AAPL','MSFT','NVDA','TSLA','GOOGL','AMZN','META','SPY']
    CRYPTO  -> ['BTC-USD', 'ETH-USD']

Cada especialista es una instancia SEPARADA de la MISMA arquitectura
(invarianza topológica estricta, ver `build_model`): no se comparten pesos,
scalers, ni embedding de activos entre regímenes. Esto es simplemente dos
corridas de la Fase A-E de V4 con universos distintos; toda la matemática
(BiLSTM(64) -> LayerNorm -> MultiHeadAttention+Residual -> LayerNorm ->
BahdanauAttention -> Concat(context, asset_embed) -> Dense(32)+Residual ->
Dense(1), DirectionalHuberLoss con penalización de varianza, curriculum de
gamma sigmoide) se mantiene BIT A BIT idéntica a V4.

NUEVO EN V5:
    1) NEWS_SENTIMENT_SCORE (services/data_pipeline.py): titulares vía
       yfinance.Ticker().news / Finnhub, escalar VADER en [-1, 1],
       concatenado al tensor de features ANTES de escalar.
    2) Persistencia de nivel institucional: cada especialista se guarda en
       su propia carpeta bajo services/kodaquant_models/, con nombres de
       archivo ESTABLES (model_v5.keras, scalers_dict.pkl,
       evaluation_chart.png) — cada corrida SOBREESCRIBE el artefacto
       anterior, nunca genera duplicados versionados a mano.
    3) CERO bloqueos: `matplotlib` corre en backend "Agg" (sin display) y
       toda visualización usa `plt.savefig(...)`, jamás `plt.show()` — el
       script se ejecuta de punta a punta sin intervención humana en
       servidores headless / runners de CI.

Entorno objetivo: local o servidor headless, TensorFlow/Keras 3.x.

NUEVO EN V6 — FIX COLAPSO DE VARIANZA EN `crypto_specialist` (auditoría:
underfitting crítico atribuido a la caída de volumen de entrenamiento del
dataset GLOBAL V4 mezclado (~1M muestras, 10 activos en un solo embedding
compartido) al dataset por-régimen V5 (~100k muestras, `crypto_specialist`
aislado con solo 2 tickers). Bajo escasez de datos, `DirectionalGaussianNLL`
puede minimizar la pérdida INFLANDO `log_var` en vez de mejorar `mu` —el
gradiente sobre `mu` está ponderado por 1/var, así que a menos datos, menos
señal de corrección y "var→∞" se vuelve el óptimo local dominante (Seitzer
et al., "On the Pitfalls of Heteroscedastic Uncertainty Estimation with
Probabilistic Neural Networks", ICLR 2022, arXiv:2203.09168). Dos frentes,
CERO cambios de arquitectura (Fase B intacta, invarianza topológica
preservada) y CERO ruptura de compatibilidad con `services/quanti_engine.py`
en inferencia (el head `log_var` sigue siendo literalmente `log_var`, sin
ningún remapeo — `quanti_engine.py` sigue leyendo `y_pred[:, 1]` tal cual):

    4) `DirectionalGaussianNLL` — 3 mecanismos apilados (ver docstring de
       la clase): β-NLL (ataca la causa raíz del incentivo perverso),
       barrera cuadrática blanda + ridge L2 sobre `log_var` (amortiguación
       directa de su magnitud, gradiente nunca nulo a diferencia de un
       hard-clip) y clip numérico interno solo para el forward de la NLL
       (evita `exp(log_var)` -> inf/NaN mientras la barrera reeduca al
       optimizador).
    5) Oversampling por Magnitude-Warping por régimen
       (`_magnitude_warp_oversample`, Fase A/C, factor=4.0x en
       `crypto_specialist`): descartado el ruido i.i.d. por-paso (destruye
       la autocorrelación temporal intra-ventana, el mismo defecto que el
       bootstrap+jitter gaussiano de la iteración anterior de este
       mecanismo). En su lugar, curva de deformación de MAGNITUD por
       ventana y por canal (Um et al. 2017; Iwana & Uchida 2021): pocos
       nodos de control (`MAGNITUDE_WARP_KNOTS`) muestreados ~N(1,
       `MAGNITUDE_WARP_SIGMA`) e interpolados con spline cúbico sobre los
       `LOOKBACK` pasos -> curva suave de baja frecuencia que reescala la
       serie ALREDEDOR de su propia media local
       (`media_ventana + (X - media_ventana) · curva`, nunca el valor
       absoluto en [0,1] de MinMaxScaler — ver docstring de la función),
       preservando la forma/inercia temporal real del criptomercado y
       expandiendo la frontera de decisión sin memorizar duplicados
       exactos ni degenerar en ruido blanco. Aplicado ÚNICAMENTE sobre el
       split de train (cero fuga hacia test). `equity_specialist`
       (8 tickers, volumen suficiente) queda con `oversample_factor=1.0`
       -> no-op exacto, cero cambio de comportamiento respecto a V5.
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # CRÍTICO: backend no interactivo — CERO plt.show() en todo el módulo.

import keras
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from keras import layers
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# data_pipeline.py vive junto a este script (o en services/, ver sys.path
# más abajo) — Requerimiento 1: pipeline NLP de sentimiento de noticias.
sys.path.append(str(Path(__file__).resolve().parent))
from data_pipeline import get_daily_news_sentiment  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("kodaquant.train_v5")

# ============================================================
# CONFIG — PERSISTENCIA DE NIVEL INSTITUCIONAL (Requerimiento 3)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
DATA_DIR = BASE_DIR / "data_cache"
MODELS_ROOT = BASE_DIR.parent / "services" / "kodaquant_models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_ROOT.mkdir(parents=True, exist_ok=True)

# --- Enrutamiento de Regímenes (Requerimiento 2) ---------------------------
# V6 (Requerimiento 5): `oversample_factor` por régimen — 8 tickers equity vs.
# 2 tickers cripto es la MISMA razón (~4x) que separa el volumen efectivo de
# ambos especialistas; CRYPTO_OVERSAMPLE_FACTOR busca emparejar la
# representación por-activo entre regímenes, no un número arbitrario.
CRYPTO_OVERSAMPLE_FACTOR = 4.0

REGIMES: dict[str, dict] = {
    "equity_specialist": {
        "tickers": ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "SPY"],
        "oversample_factor": 1.0,  # 8 tickers, volumen ya suficiente -> no-op exacto (idéntico a V5)
    },
    "crypto_specialist": {
        "tickers": ["BTC-USD", "ETH-USD"],
        "oversample_factor": CRYPTO_OVERSAMPLE_FACTOR,
    },
}
MACRO_TICKERS = ["^GSPC", "^TNX", "^VIX", "GC=F", "DX-Y.NYB"]  # SPX, UST10Y, VIX, Oro, Dollar Index

PERIOD = "10y"
LOOKBACK = 60
TRAIN_RATIO = 0.8
SEED = 42
ASSET_EMBED_DIM = 8

SENTIMENT_LOOKBACK = 20         # ventana de la correlación móvil activo<->macro (V4)
SENTIMENT_MACRO_PROXY = "^VIX"  # factor macro de referencia; fallback: MACRO_TICKERS[0]

# TECH_COLS — ÚNICA fuente de verdad de qué técnicos entran al tensor.
# V6 — MIGRACIÓN A ESTACIONARIEDAD ESTRICTA (Reinicio Estructural): el input
# "PRICE" (nivel absoluto) se elimina del tensor y se reemplaza por
# LOG_RETURN_1D (ver `engineer_asset`) — ese es el cambio real que resuelve
# la ceguera ante ATH, no un parche de post-procesamiento. En paralelo, los
# 4 TECH_COLS que dependían del NIVEL de precio (no auto-acotados: ver
# auditoría SOFT_CLIP_MARGIN en quanti_engine.py) se re-expresan como razones
# adimensionales, invariantes a escala:
#   EMA_20        -> EMA20_DEV_PCT   = (Close - EMA20) / EMA20
#   MACD          -> MACD_PCT        = MACD / Close
#   MACD_SIGNAL   -> MACD_SIGNAL_PCT = MACD_SIGNAL / Close
#   ATR_14        -> ATR_PCT         = ATR_14 / Close            ("ATRP")
#   OBV           -> OBV_ROC_20      = OBV.pct_change(20), clip [-3, 3]
#     (OBV es SUMA ACUMULADA sin techo ni mean-reversion propia — ni
#     siquiera "retorno de OBV" la vuelve comparable entre activos, porque
#     su escala depende del volumen histórico TOTAL negociado; el fix real
#     es una tasa de cambio ACOTADA, no una re-escala de la magnitud cruda)
# RSI_14 / BB_WIDTH_20 (ya es el ratio 2σ/μ, adimensional por construcción,
# PESE al comentario previo en quanti_engine.py que lo trataba como "atado
# al nivel de precio") / ADX_14 / STOCH_K_14 / SENTIMENT_SCORE /
# NEWS_SENTIMENT_SCORE quedan IDÉNTICOS — son osciladores/ratios ya acotados
# que nunca extrapolan fuera de [0,1]/[-1,1] sin importar cuántos máximos
# históricos haga el activo.
OBV_ROC_LOOKBACK_DAYS = 20   # ventana del rate-of-change — DEBE calzar con OBV_ROC_LOOKBACK_DAYS en quanti_engine.py
OBV_ROC_CLIP = 3.0           # ±300% — techo defensivo ante denominadores (OBV_{t-20}) cercanos a cero

TECH_COLS = [
    "RSI_14", "EMA20_DEV_PCT", "MACD_PCT", "MACD_SIGNAL_PCT", "SENTIMENT_SCORE",
    "ATR_PCT", "BB_WIDTH_20", "OBV_ROC_20", "ADX_14", "STOCH_K_14",
    "NEWS_SENTIMENT_SCORE",
]
N_FEATURES = 1 + len(TECH_COLS) + len(MACRO_TICKERS)  # LOG_RETURN_1D + técnicos + macro (mismo conteo: 17)

# --- Hiperparámetros de entrenamiento (idénticos a V4 — invarianza matemática) ---
BATCH_SIZE = 64
EPOCHS = 100
VALIDATION_SPLIT = 0.1
HUBER_DELTA = 1.0

GAMMA_INITIAL = 0.0
GAMMA_MAX = 1.5
GAMMA_WARMUP_EPOCHS = 35
GAMMA_SCHEDULE = "sigmoid"
GAMMA_SIGMOID_STEEPNESS = 6.0   # curva sigmoide más suave -> transición de gamma menos abrupta

LR_INITIAL = 1e-3
GRAD_CLIPNORM = 1.0

# --- V10 — Topología por régimen (Arquitectura Híbrida Asimétrica) --------
# Sustituye a KODAQUANT_EXPERIMENTAL_MODE (V9.2, flag de entorno manual):
# ya no existe un perfil "experimental" aislado que dependa de una var de
# entorno -- `build_model` recibe `regime_name` y dicta su propia topología
# consultando este diccionario, DETERMINISTA tanto en corrida local como en
# el cron de CI/CD (entrenamiento_semanal.yml), sin ramas condicionales
# externas ni sufijos `_experimental` en disco. Perfiles validados
# empíricamente en pruebas de estrés (backtest comparativo):
#   equity_specialist (8 tickers, volumen alto) -> topología de mayor
#     capacidad: 52.8% de asertividad direccional. BiLSTM 64->128 duplica
#     el ancho de salida Bidirectional (128->256); MHA_HEADS 4->8 con
#     key_dim SIN cambiar duplica en paralelo la dimensión total de
#     atención -- mismo factor de escala, arquitectura dimensionalmente
#     coherente. Dropout/L2 NO escalan 1:1 con la capacidad (saturarían
#     dropout cerca de 1.0): +33% en dropout (0.45->0.60) y 3x en L2
#     (1e-5->3e-5) compensan ~4x de parámetros sin apagar la red. Incluye
#     el bloque residual profundo adicional Dense(64)+skip antes de la
#     cabeza final (ver `build_model`).
#   crypto_specialist (2 tickers, volumen bajo) -> topología ágil y
#     conservadora: 51.3% constante -- el perfil de mayor capacidad
#     SOBREAJUSTÓ en cripto (colapso a 49.9%) por exceso de parámetros
#     frente a solo 2 activos. CERO bloque residual extra.
REGIME_ARCHITECTURE: dict[str, dict] = {
    "equity_specialist": {
        "lstm_units": 128,
        "attention_units": 64,
        "mha_heads": 8,
        "mha_key_dim": 32,
        "dropout_rate": 0.60,
        "dense_l2_reg": 3e-5,
        "extra_residual_block": True,
    },
    "crypto_specialist": {
        "lstm_units": 64,
        "attention_units": 32,
        "mha_heads": 4,
        "mha_key_dim": 32,
        "dropout_rate": 0.45,
        "dense_l2_reg": 1e-5,
        "extra_residual_block": False,
    },
}

# V5.1: CosineDecayRestarts reemplaza a ReduceLROnPlateau — un schedule
# recalcula el LR por *step*, así que combinarlo con un callback que también
# reescribe optimizer.learning_rate por *época* produce ajustes en conflicto
# (el schedule pisa silenciosamente la reducción del callback). El schedule
# ya provee escape agresivo de mínimos locales por diseño (restart a LR
# pico cada ciclo); EarlyStopping sobre val_directional_accuracy_metric
# sigue siendo el mecanismo real que evita el colapso "cobarde" (V4) y se
# mantiene sin cambios.
COSINE_RESTART_EPOCH_PERIOD = 6    # ciclo inicial más corto -> primer restart más temprano -> más exploración
COSINE_T_MUL = 1.6                 # crecimiento más moderado -> más restarts totales durante el entrenamiento
COSINE_M_MUL = 0.94                # decaimiento de pico más lento -> LR de restart se sostiene alto más tiempo
COSINE_ALPHA = 1e-5                # piso de LR (fracción de LR_INITIAL) — evita colapso a 0
EARLY_STOPPING_PATIENCE = 12
EARLY_STOPPING_START_EPOCH = 15

VARIANCE_LAMBDA = 0.15
VARIANCE_CAP = 4.0

# --- V6 — Anti-colapso de varianza en DirectionalGaussianNLL (ver docstring
# de módulo/clase). Con GAUSSIAN_NLL_BETA=0.0, LOG_VAR_L2_LAMBDA=0.0 y
# LOG_VAR_BARRIER_LAMBDA=0.0 la pérdida es matemáticamente IDÉNTICA a la
# NLL Gaussiana V5 original (invarianza sobre equity_specialist si algún día
# se quisiera desactivar por régimen).
GAUSSIAN_NLL_BETA = 0.5          # β-NLL (Seitzer et al. 2022) — β≈0.5 recomendado por los autores como punto de partida
LOG_VAR_MIN = -6.0               # var floor numérico ≈ exp(-6) ≈ 0.0025 — evita división por var≈0 en la NLL
LOG_VAR_MAX = 3.0                # var techo numérico ≈ exp(3)  ≈ 20.1  — evita exp(log_var) -> inf/NaN en la NLL
LOG_VAR_L2_LAMBDA = 1e-3         # ridge suave hacia log_var≈0 (var≈1) sobre TODO el rango
LOG_VAR_BARRIER_LAMBDA = 5e-3    # barrera cuadrática blanda, CERO dentro de [LOG_VAR_MIN, LOG_VAR_MAX]

# --- V6 — Oversampling por Magnitude-Warping (Requerimiento 5, ver
# `_magnitude_warp_oversample`). MAGNITUDE_WARP_SIGMA=0.15 es deliberadamente
# moderado: mantiene las ventanas sintéticas dentro de una vecindad plausible
# de la ventana real que las origina (evita que `crypto_specialist` vea, en
# las primeras épocas, un régimen de entrada tan distorsionado que fuerce a
# `log_var` a saltar fuera de [LOG_VAR_MIN, LOG_VAR_MAX] de entrada — ver
# Punto Ciego #2 del mensaje de entrega). MAGNITUDE_WARP_KNOTS=4 mantiene la
# curva de deformación en baja frecuencia relativa a LOOKBACK=60 (ver Punto
# Ciego #2): suficientes nodos para no degenerar en un simple reescalado
# constante por ventana, pocos para no degenerar en ruido i.i.d. de alta
# frecuencia (el defecto original que este mecanismo reemplaza).
MAGNITUDE_WARP_SIGMA = 0.15
MAGNITUDE_WARP_KNOTS = 4

keras.utils.set_random_seed(SEED)


# ============================================================
# FASE A: ETL — DESCARGA UNIFICADA (OHLCV) + FEATURE ENGINEERING POR ACTIVO
# ============================================================
MAX_DOWNLOAD_RETRIES = 5
BACKOFF_BASE_SECONDS = 2.0
# Smart Cache Invalidation — antigüedad máxima tolerada del .parquet cacheado
# ANTES de forzarse la caché era válida indefinidamente mientras compilara el
# set de columnas requeridas (ver `download_all`), sin importar cuán vieja
# fuera su fecha más reciente -- exactamente la causa del "abismo temporal"
# auditado en NEWS_SENTIMENT_SCORE: OHLCV estancado en el pasado cruzado
# contra noticias descargadas HOY, ninguna fecha cae dentro de la tolerancia
# del merge_asof (`_NEWS_MERGE_TOLERANCE_DAYS` en data_pipeline.py).
_CACHE_MAX_AGE_HOURS = 48

# --- V9 (CI/CD — Purgado Total en Producción, ver entrenamiento_semanal.yml) ---
# El cron de GitHub Actions ('0 23 * * 5') corre este script en un runner
# EFÍMERO: sin estado persistente entre corridas salvo lo que el propio repo
# versiona. Ahí, la Smart Cache Invalidation de 48h (arriba) es la regla
# equivocada -- el riesgo no es un .parquet viejo dentro de una MISMA
# corrida, es una corrida NUEVA arrancando sobre un .parquet commiteado por
# error o sobreviviente de un checkout anterior, aún "fresco" según la
# regla de antigüedad pero potencialmente desincronizado del `merge_asof`
# de NEWS_SENTIMENT_SCORE (que siempre trae titulares de HOY). FORCE_FRESH_DATA
# anula la regla de 48h por completo para el pipeline de entrenamiento.
# Default True (bandera, no hardcode): en producción/CI SIEMPRE se quiere
# data 100% fresca; override explícito vía env var para iteración local
# (`KODAQUANT_FORCE_FRESH_DATA=0`) donde reutilizar la caché ahorra tiempo
# de desarrollo sin tocar el comportamiento en CI.
FORCE_FRESH_DATA = os.getenv("KODAQUANT_FORCE_FRESH_DATA", "1").strip().lower() not in ("0", "false", "no")


def purge_data_cache(data_dir: Path = DATA_DIR) -> None:
    """
    Purgado Total (V9): vacía programáticamente `data_dir` de cualquier
    .parquet cacheado. Se invoca EXACTAMENTE UNA VEZ por corrida de
    entrenamiento, en `__main__`, ANTES del bucle de regímenes -- nunca
    dentro de `download_all` (llamada una vez POR régimen): purgar ahí
    borraría, en la segunda llamada, el .parquet que el régimen anterior
    de la MISMA corrida acaba de escribir para consumo posterior (Requerimiento
    3: el .parquet debe sobrevivir intacto al final de la corrida).
    """
    removed = list(data_dir.glob("*.parquet"))
    for f in removed:
        f.unlink(missing_ok=True)
    if removed:
        logger.warning(
            "[purga CI/CD] FORCE_FRESH_DATA=True -> %d archivo(s) .parquet "
            "purgado(s) de %s (Data Drift prevention).", len(removed), data_dir,
        )
    else:
        logger.info("[purga CI/CD] FORCE_FRESH_DATA=True -> %s ya estaba vacío.", data_dir)


def _download_with_backoff(symbols: list[str], period: str,
                            max_retries: int = MAX_DOWNLOAD_RETRIES,
                            backoff_base: float = BACKOFF_BASE_SECONDS) -> pd.DataFrame:
    """Reintenta la descarga con backoff exponencial ante 429/timeouts transitorios."""
    import time

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(symbols, period=period, auto_adjust=True, progress=False)
            if data is None or data.empty:
                raise ValueError("yfinance devolvió un DataFrame vacío")
            return data
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_retries:
                break
            wait_s = backoff_base * (2 ** (attempt - 1))
            logger.warning("Descarga falló (intento %d/%d): %r — reintentando en %.0fs",
                           attempt, max_retries, exc, wait_s)
            time.sleep(wait_s)
    raise RuntimeError(f"No se pudo descargar {symbols} tras {max_retries} intentos") from last_exc


def _cache_is_stale(cached: pd.DataFrame, max_age_hours: int = _CACHE_MAX_AGE_HOURS) -> bool:
    """
    True si la fecha MÁS RECIENTE del índice cacheado tiene más de
    `max_age_hours` de antigüedad respecto a `datetime.now()` -- la caché
    debe descartarse y recargarse fresca desde la API (ver `download_all`).
    """
    if cached.empty:
        return True

    max_date = pd.Timestamp(cached.index.max())
    # Normaliza a tz-aware UTC ANTES de restar contra `datetime.now(utc)` --
    # el índice cacheado puede venir tz-naive (yfinance diario típico) o
    # tz-aware (localizado al exchange), según símbolo/versión; mismo
    # patrón defensivo que `get_daily_news_sentiment` en data_pipeline.py.
    max_date = max_date.tz_localize("UTC") if max_date.tzinfo is None else max_date.tz_convert("UTC")
    age = datetime.now(timezone.utc) - max_date.to_pydatetime()
    return age > timedelta(hours=max_age_hours)


def download_all(tickers: list[str], macro_tickers: list[str], period: str,
                  cache_tag: str, force_refresh: bool | None = None) -> pd.DataFrame:
    """
    Descarga OHLCV completo de `tickers` + Close de `macro_tickers`, con
    cache local en parquet + Smart Cache Invalidation: una caché con el set
    de columnas requerido YA NO se reutiliza a ciegas -- si su fecha más
    reciente tiene más de `_CACHE_MAX_AGE_HOURS` de antigüedad, se descarta,
    se advierte en el log, y se dispara una descarga fresca que sobrescribe
    el .parquet viejo (garantiza sincronía temporal con
    NEWS_SENTIMENT_SCORE, que siempre usa titulares de HOY).

    `force_refresh` (V9, CI/CD): None (default) delega al flag global
    `FORCE_FRESH_DATA` -- True/False lo anula puntualmente para esta
    llamada. Con force_refresh activo se ignora CUALQUIER .parquet
    cacheado (staleness irrelevante, ni siquiera se lee) y se va directo a
    la descarga fresca; el .parquet se sigue escribiendo al final
    igual que siempre.
    """
    effective_force_refresh = FORCE_FRESH_DATA if force_refresh is None else force_refresh
    all_symbols = list(dict.fromkeys(tickers + macro_tickers))
    fields = ["Open", "High", "Low", "Close", "Volume"]
    cache_path = DATA_DIR / f"all_ohlcv_{cache_tag}_{period}.parquet"

    required_cols = set()
    for t in tickers:
        required_cols.update(f"{t}_{f}" for f in fields)
    for m in macro_tickers:
        required_cols.add(f"{m}_Close")

    if effective_force_refresh:
        logger.info(
            "[cache local] FORCE_FRESH_DATA activo -> se ignora cualquier "
            "%s cacheado, descarga 100%% fresca.", cache_path.name,
        )
    elif cache_path.exists():
        logger.info("[cache local] leyendo %s", cache_path.name)
        cached = pd.read_parquet(cache_path)
        if required_cols.issubset(set(cached.columns)):
            if not _cache_is_stale(cached):
                return cached
            logger.warning(
                "[cache local] %s obsoleta -- fecha más reciente %s tiene más "
                "de %dh de antigüedad -> Caché obsoleta, forzando recarga.",
                cache_path.name, cached.index.max(), _CACHE_MAX_AGE_HOURS,
            )

    raw = _download_with_backoff(all_symbols, period)
    flat = pd.DataFrame(index=raw.index)
    available_fields = set(raw.columns.get_level_values(0))
    for field in fields:
        if field not in available_fields:
            continue
        sub = raw[field]
        for sym in all_symbols:
            if sym in sub.columns:
                flat[f"{sym}_{field}"] = sub[sym]

    flat.to_parquet(cache_path)
    logger.info("[cache local] guardado en %s (sincronizada, fecha más reciente %s)",
                cache_path.name, flat.index.max() if not flat.empty else "N/A")
    return flat


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_sentiment_score(asset_log_return: pd.Series, macro_log_return: pd.Series,
                             window: int = SENTIMENT_LOOKBACK) -> pd.Series:
    """Correlación de Pearson móvil activo<->macro (proxy de shock idiosincrático, V4)."""
    rho = asset_log_return.rolling(window=window, min_periods=window).corr(macro_log_return)
    return rho.fillna(0.0)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr.fillna(0.0)


def compute_bb_width(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    width = (2 * num_std * rolling_std) / rolling_mean.replace(0.0, np.nan)
    return width.fillna(0.0)


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    return obv.fillna(0.0)


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ADX (Wilder) — fuerza de tendencia direccional, independiente del signo."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0.0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0.0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx.fillna(0.0)


def compute_stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Oscilador estocástico %K — posición del cierre relativa al rango reciente [0, 100]."""
    lowest_low = low.rolling(window=period, min_periods=period).min()
    highest_high = high.rolling(window=period, min_periods=period).max()
    denom = (highest_high - lowest_low).replace(0.0, np.nan)
    stoch_k = 100 * (close - lowest_low) / denom
    return stoch_k.fillna(50.0)


def engineer_asset(all_data: pd.DataFrame, ticker: str, macro_tickers: list[str]) -> pd.DataFrame:
    """
    Frame de features de UN activo — V6, estacionariedad estricta. El precio
    NUNCA entra al tensor en nivel absoluto: `close` (variable local) se usa
    para DERIVAR razones/retornos, pero el único remanente de nivel absoluto
    que sobrevive en el DataFrame devuelto es `RAW_CLOSE` — una columna de
    bookkeeping EXCLUIDA de `feature_cols` en `build_asset_dataset` (nunca
    toca el `feature_scaler`/el tensor de entrada), usada solo para
    reconstruir `last_price`/el target de log-return. Los factores macro
    siguen transformándose a log-return ANTES del feature_scaler (V4,
    sin cambios).
    """
    close_col = f"{ticker}_Close"
    high_col = f"{ticker}_High"
    low_col = f"{ticker}_Low"
    vol_col = f"{ticker}_Volume"
    macro_close_cols = [f"{m}_Close" for m in macro_tickers]

    df = all_data[[close_col, high_col, low_col, vol_col] + macro_close_cols].copy()
    df = df.ffill().dropna()

    close, high, low, volume = df[close_col], df[high_col], df[low_col], df[vol_col]

    for macro_close_col in macro_close_cols:
        macro_price = df[macro_close_col]
        df[macro_close_col] = np.log(macro_price / macro_price.shift(1))

    # --- RAW_CLOSE: bookkeeping puro (last_price / target de log-return en
    # build_asset_dataset) — jamás forma parte de feature_cols/del tensor.
    df["RAW_CLOSE"] = close
    # --- LOG_RETURN_1D reemplaza a "PRICE" como ÚNICO input de nivel-precio:
    # estacionario por construcción, misma escala sin importar el precio
    # absoluto del activo (SPY a $600 o a $6 producen la MISMA distribución
    # de LOG_RETURN_1D) — esto es lo que elimina la ceguera ante ATH en la
    # raíz, no un parche sobre el output.
    df["LOG_RETURN_1D"] = np.log(close / close.shift(1))

    df["RSI_14"] = compute_rsi(close)  # oscilador ya acotado [0,100] — sin cambios

    ema_20 = close.ewm(span=20, adjust=False).mean()
    df["EMA20_DEV_PCT"] = (close - ema_20) / ema_20  # % de desviación vs. la propia media móvil

    macd_line, signal_line = compute_macd(close)
    df["MACD_PCT"] = macd_line / close                # normalizado por nivel de precio
    df["MACD_SIGNAL_PCT"] = signal_line / close

    asset_log_return = df["LOG_RETURN_1D"]
    proxy_ticker = SENTIMENT_MACRO_PROXY if SENTIMENT_MACRO_PROXY in macro_tickers else macro_tickers[0]
    proxy_log_return = df[f"{proxy_ticker}_Close"]
    df["SENTIMENT_SCORE"] = compute_sentiment_score(asset_log_return, proxy_log_return, SENTIMENT_LOOKBACK)

    atr = compute_atr(high, low, close)
    df["ATR_PCT"] = atr / close                        # "ATRP" — volatilidad absoluta normalizada por precio
    df["BB_WIDTH_20"] = compute_bb_width(close)         # YA es 2σ/μ — adimensional, sin cambios

    obv = compute_obv(close, volume)
    obv_roc = obv.pct_change(periods=OBV_ROC_LOOKBACK_DAYS)
    df["OBV_ROC_20"] = obv_roc.replace([np.inf, -np.inf], np.nan).clip(-OBV_ROC_CLIP, OBV_ROC_CLIP).fillna(0.0)

    df["ADX_14"] = compute_adx(high, low, close)                    # ya acotado [0,100] — sin cambios
    df["STOCH_K_14"] = compute_stochastic_k(high, low, close)       # ya acotado [0,100] — sin cambios

    # --- V5 (Requerimiento 1): NEWS_SENTIMENT_SCORE vía yfinance/Finnhub + FinBERT.
    # Se alinea temporalmente contra el índice YA depurado del activo, ANTES
    # de recortar filas por ffill/dropna final, igual que el resto de técnicos.
    df["NEWS_SENTIMENT_SCORE"] = get_daily_news_sentiment(ticker, df.index)

    df = df.ffill().dropna()

    feature_cols = ["LOG_RETURN_1D"] + TECH_COLS + macro_close_cols
    result = df[["RAW_CLOSE"] + feature_cols].copy()
    result.columns = ["RAW_CLOSE", "LOG_RETURN_1D"] + TECH_COLS + macro_tickers
    return result


def build_asset_dataset(df: pd.DataFrame, ticker: str, lookback: int, train_ratio: float,
                         asset_to_id: dict[str, int]):
    """Ventanas (X) + targets de log-return (y) para UN activo, con scalers propios por activo."""
    feature_cols = ["LOG_RETURN_1D"] + TECH_COLS + MACRO_TICKERS
    # RAW_CLOSE es bookkeeping puro (ver engineer_asset) — nunca entra a
    # `features`/al feature_scaler, solo se usa para derivar `last_price` y
    # el target de log-return, exactamente como antes usaba "PRICE".
    prices = df["RAW_CLOSE"].values
    features = df[feature_cols].values
    log_returns = np.diff(np.log(prices))

    n = len(df)
    split_idx = int(n * train_ratio)

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(features[:split_idx])
    features_scaled = feature_scaler.transform(features)

    X, y_raw, last_price, dates = [], [], [], []
    for t in range(lookback, n):
        X.append(features_scaled[t - lookback: t])
        y_raw.append(log_returns[t - 1])
        last_price.append(prices[t - 1])
        dates.append(df.index[t])

    X = np.array(X, dtype=np.float32)
    y_raw = np.array(y_raw, dtype=np.float32).reshape(-1, 1)
    last_price = np.array(last_price, dtype=np.float32)
    dates = pd.DatetimeIndex(dates)

    window_split = split_idx - lookback

    target_scaler = StandardScaler(with_mean=False)  # preserva el signo del retorno (V4)
    target_scaler.fit(y_raw[:window_split])
    y_scaled = target_scaler.transform(y_raw).astype(np.float32)

    asset_id = np.full((len(X),), asset_to_id[ticker], dtype=np.int32)

    train = dict(
        X=X[:window_split], y=y_scaled[:window_split], asset_id=asset_id[:window_split],
        last_price=last_price[:window_split], dates=dates[:window_split],
    )
    test = dict(
        X=X[window_split:], y=y_scaled[window_split:], asset_id=asset_id[window_split:],
        last_price=last_price[window_split:], dates=dates[window_split:],
    )
    return train, test, feature_scaler, target_scaler


def _magnitude_warp_oversample(X: np.ndarray, y: np.ndarray, asset_id: np.ndarray,
                                factor: float, sigma: float = MAGNITUDE_WARP_SIGMA,
                                n_knots: int = MAGNITUDE_WARP_KNOTS,
                                seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    V6 (Requerimiento 5) — balanceo de volumen en el Data Loader, SOLO sobre
    train (se invoca en `run_regime_pipeline` DESPUÉS del split train/test de
    `build_asset_dataset` -> cero fuga temporal hacia test).

    Deliberadamente NO es `np.tile`/`np.repeat` (duplicar filas EXACTAS no
    aporta gradiente nuevo) NI ruido i.i.d. por-paso (destruye la
    autocorrelación temporal intra-ventana que el propio BiLSTM/atención
    necesita para aprender dinámica, no solo estadística marginal). En su
    lugar: Magnitude-Warping (Um et al. 2017, "Data Augmentation of
    Wearable Sensor Data..."; Iwana & Uchida 2021, "An Empirical Survey of
    Data Augmentation for Time Series Classification with Neural
    Networks") — por cada ventana bootstrap-muestreada (CON reemplazo) y
    por CADA canal de feature de forma independiente, se generan
    `n_knots` nodos de control ~N(1, sigma) y se interpolan con spline
    cúbico sobre los `n_timesteps` pasos de la ventana, produciendo una
    curva de deformación suave de BAJA frecuencia (a diferencia del ruido
    por-paso que reemplaza).

    La curva NO multiplica el valor absoluto en [0,1] de MinMaxScaler
    (asimétrico e inseguro: un canal cerca de 1.0 se saldría de rango, uno
    cerca de 0.0 quedaría inerte pase lo que pase con la curva — ver Punto
    Ciego #1 del mensaje de entrega). En su lugar reescala la serie
    ALREDEDOR de su propia media local por ventana/canal:
        X_warp = media_ventana + (X - media_ventana) · curva
    — la forma temporal (inercia, autocorrelación) se preserva intacta;
    solo se expande/contrae la amplitud de la variación alrededor de esa
    media. `np.clip(..., 0, 1)` final es un backstop defensivo (mismo rol
    que en la iteración anterior), no el mecanismo primario de seguridad.

    `factor<=1.0` es un no-op exacto — usado por `equity_specialist`
    (8 tickers, volumen ya suficiente) para preservar el comportamiento V5.
    """
    if factor <= 1.0:
        return X, y, asset_id

    rng = np.random.default_rng(seed)
    n_original, n_timesteps, n_channels = X.shape
    n_extra = int(round(n_original * (factor - 1.0)))
    if n_extra <= 0:
        return X, y, asset_id

    idx = rng.integers(0, n_original, size=n_extra)
    windows = X[idx]  # (n_extra, n_timesteps, n_channels)

    # Curvas vectorizadas: un CubicSpline por (muestra, canal) evaluado en
    # un único call — `axis=0` interpola independientemente cada columna
    # de `knot_values` (n_extra * n_channels curvas) sobre los mismos
    # `knot_positions`, evitando un loop Python por muestra/canal.
    knot_positions = np.linspace(0, n_timesteps - 1, n_knots)
    knot_values = rng.normal(loc=1.0, scale=sigma, size=(n_knots, n_extra, n_channels))
    warp_curves = CubicSpline(knot_positions, knot_values, axis=0)(np.arange(n_timesteps))
    warp_curves = warp_curves.astype(np.float32)  # (n_timesteps, n_extra, n_channels)
    warp_curves = np.moveaxis(warp_curves, 0, 1)   # -> (n_extra, n_timesteps, n_channels)

    window_mean = windows.mean(axis=1, keepdims=True)  # (n_extra, 1, n_channels) — ancla LOCAL, no el nivel absoluto
    X_aug = np.clip(window_mean + (windows - window_mean) * warp_curves, 0.0, 1.0).astype(np.float32)

    X_bal = np.concatenate([X, X_aug], axis=0)
    y_bal = np.concatenate([y, y[idx]], axis=0)
    asset_id_bal = np.concatenate([asset_id, asset_id[idx]], axis=0)

    perm = rng.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm], asset_id_bal[perm]


# ============================================================
# FASE B: ARQUITECTURA — INVARIANZA TOPOLÓGICA ESTRICTA (idéntica a V4)
# ============================================================
@keras.saving.register_keras_serializable(package="quanti")
class BahdanauAttention(layers.Layer):
    """Self-attention aditiva (Bahdanau) — pooling final de contexto."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units, name="attn_W")
        self.U = layers.Dense(units, name="attn_U")
        self.V = layers.Dense(1, name="attn_V")

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.W.build((None, None, feature_dim))
        self.U.build((None, 1, feature_dim))
        self.V.build((None, None, self.units))
        super().build(input_shape)

    def call(self, hidden_states):
        query = keras.ops.mean(hidden_states, axis=1, keepdims=True)
        score = self.V(keras.ops.tanh(self.W(hidden_states) + self.U(query)))
        weights = keras.ops.softmax(score, axis=1)
        context = keras.ops.sum(weights * hidden_states, axis=1)
        return context, weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


@keras.saving.register_keras_serializable(package="quanti")
class DirectionalHuberLoss(keras.losses.Loss):
    """
    L(y, ŷ) = Huber_δ(y, ŷ) · exp(γ · 1[sign(y) ≠ sign(ŷ)]) - λ·min(Var(ŷ), cap)
    V5.1: penalización direccional pasa de lineal (1+γ·mismatch) a exponencial
    (e^(γ·mismatch)) — a γ=0 (arranque del curriculum) ambas formas coinciden
    en 1.0, pero a γ=GAMMA_MAX=1.5 el multiplicador exponencial (≈4.48x) es
    sustancialmente más agresivo que el lineal (2.5x) sin discontinuidad
    respecto al curriculum sigmoide ya existente.
    """

    def __init__(self, delta: float = 1.0, gamma=1.5,
                 variance_lambda: float = 0.0, variance_cap: float | None = None,
                 name: str = "directional_huber", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = delta
        self.gamma = gamma if isinstance(gamma, keras.Variable) else keras.Variable(
            gamma, trainable=False, dtype="float32", name=f"{name}_gamma"
        )
        self.variance_lambda = float(variance_lambda)
        self.variance_cap = None if variance_cap is None else float(variance_cap)

    def call(self, y_true, y_pred):
        y_true = keras.ops.cast(y_true, "float32")
        y_pred = keras.ops.cast(y_pred, "float32")

        error = y_true - y_pred
        abs_error = keras.ops.abs(error)
        quadratic = keras.ops.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        huber = 0.5 * keras.ops.square(quadratic) + self.delta * linear

        mismatch = keras.ops.cast(
            keras.ops.not_equal(keras.ops.sign(y_true), keras.ops.sign(y_pred)),
            dtype=huber.dtype,
        )
        penalty = keras.ops.exp(self.gamma * mismatch)
        directional_huber = keras.ops.mean(huber * penalty, axis=-1)

        if self.variance_lambda > 0.0:
            batch_variance = keras.ops.var(y_pred)
            if self.variance_cap is not None:
                batch_variance = keras.ops.minimum(batch_variance, self.variance_cap)
            directional_huber = directional_huber - self.variance_lambda * batch_variance

        return directional_huber

    def get_config(self):
        config = super().get_config()
        config.update({
            "delta": self.delta,
            "gamma": float(keras.ops.convert_to_numpy(self.gamma)),
            "variance_lambda": self.variance_lambda,
            "variance_cap": self.variance_cap,
        })
        return config

@keras.saving.register_keras_serializable(package="quanti")
class DirectionalGaussianNLL(keras.losses.Loss):
    """
    μ/logσ² dual-head: acopla magnitud a incertidumbre real del modelo en
    vez de premiar Var(ŷ) del batch sin condición (ver DirectionalHuberLoss,
    ahora sin uso, dejada arriba como referencia histórica).

    V6 — 3 mecanismos anti-colapso de varianza bajo escasez de datos
    (auditoría `crypto_specialist`: 2 tickers, dataset por-régimen ~10x más
    chico que el global mezclado V4). La NLL Gaussiana estándar permite al
    optimizador minimizar la pérdida INFLANDO `log_var` en vez de mejorar
    `mu`, porque el gradiente sobre `mu` está ponderado por 1/var: a menos
    datos, menos señal para corregir `mu`, y "var→∞" se vuelve el óptimo
    local dominante (Seitzer, Tavakoli, Antic, Martius — "On the Pitfalls of
    Heteroscedastic Uncertainty Estimation with Probabilistic Neural
    Networks", ICLR 2022, arXiv:2203.09168).

        1) β-NLL (paper cit., Sec. 3): cada muestra se pondera por
           stop_gradient(var_clip)^beta. beta=0 == NLL estándar (el modo
           patológico); beta=1 == MSE puro (ignora la incertidumbre). El
           paper reporta beta≈0.5 como punto de partida robusto en la
           mayoría de los dominios evaluados — interpola entre ambos
           extremos sin tocar arquitectura ni forward pass, solo reescala
           el gradiente por muestra. stop_gradient es OBLIGATORIO: si el
           propio factor de ponderación retropropagara a través de `var`,
           recrearía el mismo incentivo perverso que intenta corregir.
        2) Clip numérico interno (`LOG_VAR_MIN`/`LOG_VAR_MAX`) — SOLO para
           calcular `var` dentro de la NLL de este `call()`, evitando
           `exp(log_var) -> inf` (y por ende NaN vía TerminateOnNaN) en los
           primeros steps, mientras el mecanismo (3) reeduca al head.
        3) Amortiguación directa sobre `log_var` CRUDO (sin clip): ridge L2
           hacia log_var≈0 + barrera cuadrática blanda fuera de
           [LOG_VAR_MIN, LOG_VAR_MAX] (CERO penalización dentro del rango,
           gradiente que CRECE cuadráticamente cuanto más lejos se excede
           -a diferencia de un hard-clip, cuyo gradiente es EXACTAMENTE 0
           fuera de rango y congelaría al optimizador justo donde más
           necesita corregirse).

    COMPATIBILIDAD DE INFERENCIA (crítico): a propósito, el head `log_var`
    NUNCA se remapea/reparametriza (nada de tanh-squashing sobre el output)
    — sigue siendo literalmente log_var, igual que en V5. `services/
    quanti_engine.py` (`log_var_scaled_batch = y_pred_np[:, 1:2]`) sigue
    funcionando SIN NINGÚN cambio; los 3 mecanismos de arriba solo dan forma
    al GRADIENTE de entrenamiento, no a la semántica del output guardado en
    `model_v5.keras`.

    Con beta=0.0, log_var_l2_lambda=0.0 y log_var_barrier_lambda=0.0, esta
    clase es matemáticamente IDÉNTICA a la NLL Gaussiana V5 original.
    """

    def __init__(self, gamma_directional=1.5, beta: float = GAUSSIAN_NLL_BETA,
                 log_var_min: float = LOG_VAR_MIN, log_var_max: float = LOG_VAR_MAX,
                 log_var_l2_lambda: float = LOG_VAR_L2_LAMBDA,
                 log_var_barrier_lambda: float = LOG_VAR_BARRIER_LAMBDA,
                 name: str = "directional_gaussian_nll", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma_directional if isinstance(gamma_directional, keras.Variable) else keras.Variable(
            gamma_directional, trainable=False, dtype="float32", name=f"{name}_gamma"
        )
        self.beta = float(beta)
        self.log_var_min = float(log_var_min)
        self.log_var_max = float(log_var_max)
        self.log_var_l2_lambda = float(log_var_l2_lambda)
        self.log_var_barrier_lambda = float(log_var_barrier_lambda)

    def call(self, y_true, y_pred):  # y_pred: (batch, 2) = [mu, log_var] — SIN remapeo, ver docstring
        y_true = keras.ops.cast(y_true, "float32")
        y_true = keras.ops.reshape(y_true, (-1,))
        mu, log_var = y_pred[:, 0], y_pred[:, 1]

        # --- (2) clip NUMÉRICO interno, solo para var/nll de este call() ---
        log_var_safe = keras.ops.clip(log_var, self.log_var_min, self.log_var_max)
        var_safe = keras.ops.exp(log_var_safe)

        nll = 0.5 * (log_var_safe + keras.ops.square(y_true - mu) / var_safe)

        # --- (1) β-NLL — ver docstring; stop_gradient es obligatorio ---
        if self.beta > 0.0:
            nll = nll * keras.ops.stop_gradient(keras.ops.power(var_safe, self.beta))

        mismatch = keras.ops.cast(keras.ops.not_equal(keras.ops.sign(y_true), keras.ops.sign(mu)), nll.dtype)
        directional_nll = keras.ops.mean(nll * keras.ops.exp(self.gamma * mismatch))

        # --- (3) amortiguación sobre log_var CRUDO (sin clip) ---
        if self.log_var_l2_lambda > 0.0:
            directional_nll = directional_nll + self.log_var_l2_lambda * keras.ops.mean(keras.ops.square(log_var))
        if self.log_var_barrier_lambda > 0.0:
            excess_hi = keras.ops.relu(log_var - self.log_var_max)
            excess_lo = keras.ops.relu(self.log_var_min - log_var)
            barrier = keras.ops.mean(keras.ops.square(excess_hi) + keras.ops.square(excess_lo))
            directional_nll = directional_nll + self.log_var_barrier_lambda * barrier

        return directional_nll

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma_directional": float(keras.ops.convert_to_numpy(self.gamma)),
            "beta": self.beta,
            "log_var_min": self.log_var_min,
            "log_var_max": self.log_var_max,
            "log_var_l2_lambda": self.log_var_l2_lambda,
            "log_var_barrier_lambda": self.log_var_barrier_lambda,
        })
        return config

@keras.saving.register_keras_serializable(package="quanti")
def directional_accuracy_metric(y_true, y_pred):
    y_true = keras.ops.reshape(keras.ops.cast(y_true, "float32"), (-1,))
    mu = keras.ops.cast(y_pred, "float32")[:, 0]
    match = keras.ops.cast(keras.ops.equal(keras.ops.sign(y_true), keras.ops.sign(mu)), dtype="float32")
    return keras.ops.mean(match)


@keras.saving.register_keras_serializable(package="quanti")
def mu_mae_metric(y_true, y_pred):
    # reemplaza "mae" en compile — "mae" plano rompe con y_pred (batch,2)
    y_true = keras.ops.reshape(keras.ops.cast(y_true, "float32"), (-1,))
    mu = keras.ops.cast(y_pred, "float32")[:, 0]
    return keras.ops.mean(keras.ops.abs(y_true - mu))


class DynamicGammaCallback(keras.callbacks.Callback):
    """Curriculum learning: rampa sigmoide/lineal de gamma de 0.0 a GAMMA_MAX (idéntico a V4)."""

    def __init__(self, gamma_variable, gamma_max: float = GAMMA_MAX,
                 warmup_epochs: int = GAMMA_WARMUP_EPOCHS,
                 schedule: str = GAMMA_SCHEDULE,
                 sigmoid_steepness: float = GAMMA_SIGMOID_STEEPNESS,
                 verbose: bool = True):
        super().__init__()
        self.gamma_variable = gamma_variable
        self.gamma_max = gamma_max
        self.warmup_epochs = max(1, warmup_epochs)
        self.schedule = schedule
        self.sigmoid_steepness = sigmoid_steepness
        self.verbose = verbose
        k = self.sigmoid_steepness
        self._sig_lo = 1.0 / (1.0 + np.exp(k * 0.5))
        self._sig_hi = 1.0 / (1.0 + np.exp(-k * 0.5))

    def _progress_to_fraction(self, progress: float) -> float:
        progress = min(max(progress, 0.0), 1.0)
        if self.schedule == "linear":
            return progress
        k = self.sigmoid_steepness
        raw = 1.0 / (1.0 + np.exp(-k * (progress - 0.5)))
        return (raw - self._sig_lo) / (self._sig_hi - self._sig_lo)

    def on_epoch_begin(self, epoch, logs=None):
        progress = epoch / self.warmup_epochs
        new_gamma = float(self.gamma_max * self._progress_to_fraction(progress))
        self.gamma_variable.assign(new_gamma)
        if self.verbose and (epoch % 5 == 0 or epoch == self.warmup_epochs):
            tag = "  <- curriculum completo" if epoch >= self.warmup_epochs else ""
            logger.info("[DynamicGamma] epoch=%3d  gamma=%.4f%s", epoch, new_gamma, tag)


def build_model(n_timesteps: int, n_features: int, n_assets: int,
                 steps_per_epoch: int, regime_name: str,
                 embed_dim: int = ASSET_EMBED_DIM,
                 huber_delta: float = HUBER_DELTA,
                 gamma_initial: float = GAMMA_INITIAL,
                 variance_lambda: float = VARIANCE_LAMBDA,
                 variance_cap: float = VARIANCE_CAP,
                 gaussian_nll_beta: float = GAUSSIAN_NLL_BETA,
                 log_var_min: float = LOG_VAR_MIN,
                 log_var_max: float = LOG_VAR_MAX,
                 log_var_l2_lambda: float = LOG_VAR_L2_LAMBDA,
                 log_var_barrier_lambda: float = LOG_VAR_BARRIER_LAMBDA,
                 model_name: str | None = None) -> tuple:
    """
    Topología Híbrida Asimétrica: la firma ya NO recibe hiperparámetros de
    arquitectura sueltos -- `regime_name` indexa `REGIME_ARCHITECTURE` (ver
    diccionario, arriba) y la función dicta su propia topología
    internamente. Grafo común a ambos especialistas: Input ->
    BiLSTM(lstm_units) -> LayerNorm -> MultiHeadAttention+Residual ->
    LayerNorm -> BahdanauAttention -> Concat(context, asset_embed) ->
    [bloque residual extra si `extra_residual_block=True`] ->
    Dense(32)+Residual -> Dense(1). Solo cambian, por régimen: n_features
    (por NEWS_SENTIMENT_SCORE), n_assets (universo del régimen) y el
    perfil asignado en `REGIME_ARCHITECTURE[regime_name]`.
    """
    arch = REGIME_ARCHITECTURE[regime_name]
    lstm_units = arch["lstm_units"]
    attention_units = arch["attention_units"]
    mha_heads = arch["mha_heads"]
    mha_key_dim = arch["mha_key_dim"]
    dropout_rate = arch["dropout_rate"]
    dense_l2_reg = arch["dense_l2_reg"]
    extra_residual_block = arch["extra_residual_block"]
    if model_name is None:
        model_name = f"KodaQuant_{regime_name}_V5"

    seq_input = keras.Input(shape=(n_timesteps, n_features), name="input_sequence")
    asset_input = keras.Input(shape=(1,), dtype="int32", name="input_asset_id")

    l2 = keras.regularizers.l2(dense_l2_reg)  # V5.1: definido temprano — reutilizado en BiLSTM/MHA/Dense

    bilstm = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate * 0.3,
                    kernel_regularizer=l2, recurrent_regularizer=l2),
        name=f"bilstm_{lstm_units}",
    )(seq_input)
    bilstm = layers.LayerNormalization(name=f"ln_post_bilstm_{lstm_units}")(bilstm)

    mha_out = layers.MultiHeadAttention(
        num_heads=mha_heads, key_dim=mha_key_dim, kernel_regularizer=l2,
        name="multi_head_self_attention",
    )(query=bilstm, value=bilstm, key=bilstm)
    mha_out = layers.Dropout(dropout_rate * 0.5, name="dropout_post_mha")(mha_out)
    attn_block = layers.Add(name="mha_residual_add")([bilstm, mha_out])
    attn_block = layers.LayerNormalization(name="ln_post_mha_residual")(attn_block)

    context_vector, attention_weights = BahdanauAttention(
        attention_units, name="bahdanau_attention"
    )(attn_block)

    asset_embed = layers.Embedding(n_assets, embed_dim, name="asset_embedding")(asset_input)
    asset_embed = layers.Flatten(name="asset_embedding_flat")(asset_embed)

    context_vector = layers.Dropout(dropout_rate * 0.5, name="dropout_post_bahdanau")(context_vector)
    fused = layers.Concatenate(name="fusion_attn_asset")([context_vector, asset_embed])
    fused = layers.Dropout(dropout_rate, name="dropout_regularizer")(fused)

    # --- Bloque residual profundo adicional (perfil equity_specialist) —
    # no-op estructural cuando extra_residual_block=False (crypto_specialist:
    # el grafo queda BIT A BIT idéntico, sin este bloque).
    if extra_residual_block:
        wide_hidden = layers.Dense(64, activation="relu", name="extra_residual_dense",
                                    kernel_regularizer=l2)(fused)
        wide_projection = layers.Dense(64, name="extra_residual_skip",
                                        kernel_regularizer=l2)(fused)
        fused = layers.Add(name="extra_residual_add")([wide_hidden, wide_projection])
        fused = layers.LayerNormalization(name="ln_post_extra_residual")(fused)
        fused = layers.Dropout(dropout_rate * 0.5, name="dropout_post_extra_residual")(fused)

    dense_hidden = layers.Dense(32, activation="relu", name="post_fusion_dense",
                                 kernel_regularizer=l2)(fused)
    fused_projection = layers.Dense(32, name="fused_projection_skip",
                                     kernel_regularizer=l2)(fused)
    dense_block = layers.Add(name="dense_residual_add")([dense_hidden, fused_projection])
    dense_block = layers.LayerNormalization(name="ln_post_dense_residual")(dense_block)

    output = layers.Dense(2, name="return_head", kernel_regularizer=l2)(dense_block)

    model = keras.Model(inputs=[seq_input, asset_input], outputs=output, name=model_name)

    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LR_INITIAL,
        first_decay_steps=max(1, steps_per_epoch * COSINE_RESTART_EPOCH_PERIOD),
        t_mul=COSINE_T_MUL,
        m_mul=COSINE_M_MUL,
        alpha=COSINE_ALPHA,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=GRAD_CLIPNORM
    )

    gamma_variable = keras.Variable(
        gamma_initial, trainable=False, dtype="float32", name="directional_gamma"
    )
    model.compile(
        optimizer=optimizer,
        loss=DirectionalGaussianNLL(
            gamma_directional=gamma_variable,
            beta=gaussian_nll_beta,
            log_var_min=log_var_min, log_var_max=log_var_max,
            log_var_l2_lambda=log_var_l2_lambda,
            log_var_barrier_lambda=log_var_barrier_lambda,
        ),
        metrics=[mu_mae_metric, directional_accuracy_metric],
    )
    return model, gamma_variable


# ============================================================
# FASE C: ENTRENAMIENTO (idéntico a V4)
# ============================================================
def train_model(model: keras.Model, X_train, asset_id_train, y_train,
                 checkpoint_path: Path,
                 epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
                 validation_split: float = VALIDATION_SPLIT,
                 gamma_variable=None,
                 gamma_max: float = GAMMA_MAX,
                 gamma_warmup_epochs: int = GAMMA_WARMUP_EPOCHS,
                 gamma_schedule: str = GAMMA_SCHEDULE,
                 gamma_sigmoid_steepness: float = GAMMA_SIGMOID_STEEPNESS):
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1
        ),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(
            monitor="val_directional_accuracy_metric", mode="max",
            patience=EARLY_STOPPING_PATIENCE, start_from_epoch=EARLY_STOPPING_START_EPOCH,
            restore_best_weights=True, verbose=1,
        ),
    ]
    if gamma_variable is not None:
        callbacks.append(DynamicGammaCallback(
            gamma_variable=gamma_variable, gamma_max=gamma_max,
            warmup_epochs=gamma_warmup_epochs, schedule=gamma_schedule,
            sigmoid_steepness=gamma_sigmoid_steepness,
        ))
    history = model.fit(
        [X_train, asset_id_train], y_train,
        validation_split=validation_split,
        epochs=epochs, batch_size=batch_size, shuffle=True,
        callbacks=callbacks, verbose=1,
    )
    best_model = keras.models.load_model(checkpoint_path)
    return history, best_model


# ============================================================
# FASE D: EVALUACIÓN — RECONSTRUCCIÓN DE PRECIOS + DIRECTIONAL ACCURACY
# ============================================================
def evaluate_asset(model: keras.Model, test: dict, ticker: str, target_scaler: StandardScaler) -> dict:
    asset_id_col = test["asset_id"].reshape(-1, 1)
    y_pred_scaled = model.predict([test["X"], asset_id_col], verbose=0)
    mu_scaled = y_pred_scaled[:, 0:1]
    r_hat = target_scaler.inverse_transform(mu_scaled).flatten()
    r_true = target_scaler.inverse_transform(test["y"]).flatten()

    price_pred = test["last_price"] * np.exp(r_hat)
    price_true = test["last_price"] * np.exp(r_true)

    mae_price = mean_absolute_error(price_true, price_pred)
    rmse_price = np.sqrt(mean_squared_error(price_true, price_pred))
    directional_acc = float(np.mean(np.sign(r_hat) == np.sign(r_true)))

    return {
        "ticker": ticker, "mae_price": mae_price, "rmse_price": rmse_price,
        "directional_accuracy": directional_acc,
        "price_true": price_true, "price_pred": price_pred, "dates": test["dates"],
    }


FEATURE_NAMES = ["LOG_RETURN_1D"] + TECH_COLS + MACRO_TICKERS  # orden exacto de los 17 canales de entrada


def compute_feature_attribution(model: keras.Model, X_test: np.ndarray, asset_id_test: np.ndarray,
                                  feature_names: list[str] = FEATURE_NAMES) -> pd.Series:
    """
    Atribución por gradiente |∂ŷ/∂x| promediada sobre timesteps y muestras. NO son los pesos
    de BahdanauAttention (esos ponderan PASOS TEMPORALES, no canales de feature) — esta es la
    métrica correcta para cuantificar cuánto responde la salida a NEWS_SENTIMENT_SCORE frente
    a las otras 16 variables del tensor.
    """
    import tensorflow as tf

    x_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    asset_tensor = tf.convert_to_tensor(asset_id_test.reshape(-1, 1), dtype=tf.int32)

    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        preds = model([x_tensor, asset_tensor], training=False)
    grads = tape.gradient(preds, x_tensor)

    mean_abs_grad = keras.ops.convert_to_numpy(keras.ops.mean(keras.ops.abs(grads), axis=(0, 1)))
    return pd.Series(mean_abs_grad, index=feature_names, name="mean_abs_gradient").sort_values(ascending=False)


def extract_temporal_attention(model: keras.Model, X_test: np.ndarray, asset_id_test: np.ndarray) -> np.ndarray:
    """Pesos reales de BahdanauAttention por PASO TEMPORAL (diagnóstico complementario, no por feature)."""
    attention_extractor = keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("bahdanau_attention").output[1],
        name="attention_extractor",
    )
    weights = attention_extractor.predict([X_test, asset_id_test.reshape(-1, 1)], verbose=0)
    return weights.squeeze(-1)


def plot_regime_evaluation(results: dict[str, dict], regime_name: str, output_path: Path) -> None:
    """
    Grilla no-bloqueante (Requerimiento 3): un panel Real-vs-Predicho por
    ticker del régimen, todo en UN solo `evaluation_chart.png`. JAMÁS
    `plt.show()` — el archivo se renderiza directo a disco vía `savefig`
    para permitir ejecución 100% headless.
    """
    tickers = list(results.keys())
    n = len(tickers)
    n_cols = 2 if n > 1 else 1
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows), squeeze=False)

    for idx, ticker in enumerate(tickers):
        ax = axes[idx // n_cols][idx % n_cols]
        r = results[ticker]
        ax.plot(r["dates"], r["price_true"], label="Real", color="#1f77b4", linewidth=1.6)
        ax.plot(r["dates"], r["price_pred"], label="Predicho (reconstruido)",
                 color="#d62728", linewidth=1.3, linestyle="--")
        ax.set_title(f"{ticker} — DirAcc={r['directional_accuracy'] * 100:.1f}%  "
                      f"MAE=${r['mae_price']:.2f}")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Oculta ejes sobrantes si el número de tickers es impar.
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(f"KodaQuant V5 — {regime_name} — Predicho vs. Real (Test)", fontsize=14)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)   # NUNCA plt.show() — Requerimiento 3 (cero bloqueos).
    plt.close(fig)
    logger.info("      Gráfico de evaluación -> %s", output_path)


# ============================================================
# ORQUESTACIÓN POR RÉGIMEN (Requerimiento 2: ciclos independientes y secuenciales)
# ============================================================
def run_regime_pipeline(regime_name: str, regime_cfg: dict) -> dict:
    """Ejecuta las Fases A-E completas para UN especialista y persiste sus artefactos."""
    tickers = regime_cfg["tickers"]
    oversample_factor = float(regime_cfg.get("oversample_factor", 1.0))
    arch = REGIME_ARCHITECTURE[regime_name]

    # Rutas de producción estándar — sin sufijos `_experimental`: cada
    # régimen escribe DIRECTO en la carpeta que lee `quanti_engine.py` en
    # inferencia, idéntico en corrida local y en el cron de CI/CD.
    output_dir = MODELS_ROOT / regime_name
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_to_id = {t: i for i, t in enumerate(tickers)}

    logger.info("=" * 78)
    logger.info("REGIMEN: %s  |  Universo: %s  |  BiLSTM=%du  heads=%d  "
                "residual_extra=%s  dropout=%.2f  L2=%.0e",
                regime_name, tickers, arch["lstm_units"], arch["mha_heads"],
                arch["extra_residual_block"], arch["dropout_rate"], arch["dense_l2_reg"])
    logger.info("=" * 78)

    logger.info("[1/5] Descargando %d activos (OHLCV) + %d factores macro (%s)...",
                len(tickers), len(MACRO_TICKERS), PERIOD)
    all_market_data = download_all(tickers, MACRO_TICKERS, PERIOD, cache_tag=regime_name)

    logger.info("[2/5] Ingeniería de features y construcción de dataset por activo...")
    feature_scalers, target_scalers = {}, {}
    X_train_parts, y_train_parts, asset_id_train_parts = [], [], []
    test_sets = {}

    for ticker in tickers:
        df_asset = engineer_asset(all_market_data, ticker, MACRO_TICKERS)
        train, test, f_scaler, t_scaler = build_asset_dataset(
            df_asset, ticker, LOOKBACK, TRAIN_RATIO, asset_to_id
        )
        feature_scalers[ticker] = f_scaler
        target_scalers[ticker] = t_scaler
        test_sets[ticker] = test

        X_train_parts.append(train["X"])
        y_train_parts.append(train["y"])
        asset_id_train_parts.append(train["asset_id"])
        logger.info("      %-10s train=%5d  test=%5d", ticker, len(train["X"]), len(test["X"]))

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    asset_id_train_flat = np.concatenate(asset_id_train_parts, axis=0)
    logger.info("      Dataset %s combinado -> X_train %s", regime_name, X_train.shape)

    assert X_train.shape[-1] == N_FEATURES, (
        f"Desalineación de features: X_train tiene {X_train.shape[-1]} columnas "
        f"pero N_FEATURES={N_FEATURES} (TECH_COLS={TECH_COLS})."
    )

    # V6 (Requerimiento 5) — balanceo de volumen SOLO sobre train (ver
    # `_magnitude_warp_oversample`); no-op exacto si oversample_factor<=1.0.
    n_before = len(X_train)
    X_train, y_train, asset_id_train_flat = _magnitude_warp_oversample(
        X_train, y_train, asset_id_train_flat, factor=oversample_factor,
    )
    if oversample_factor > 1.0:
        logger.info("      Oversampling Magnitude-Warping (factor=%.1fx, sigma=%.2f, knots=%d): X_train %d -> %d muestras",
                    oversample_factor, MAGNITUDE_WARP_SIGMA, MAGNITUDE_WARP_KNOTS, n_before, len(X_train))
    asset_id_train = asset_id_train_flat.reshape(-1, 1)

    logger.info("[3/5] Construyendo y entrenando el modelo especialista '%s'...", regime_name)
    n_fit_samples = int(len(X_train) * (1 - VALIDATION_SPLIT))
    steps_per_epoch = max(1, n_fit_samples // BATCH_SIZE)
    model, gamma_variable = build_model(
        n_timesteps=LOOKBACK, n_features=N_FEATURES, n_assets=len(tickers),
        steps_per_epoch=steps_per_epoch, regime_name=regime_name,
        huber_delta=HUBER_DELTA, gamma_initial=GAMMA_INITIAL,
        variance_lambda=VARIANCE_LAMBDA, variance_cap=VARIANCE_CAP,
        gaussian_nll_beta=GAUSSIAN_NLL_BETA,
        log_var_min=LOG_VAR_MIN, log_var_max=LOG_VAR_MAX,
        log_var_l2_lambda=LOG_VAR_L2_LAMBDA, log_var_barrier_lambda=LOG_VAR_BARRIER_LAMBDA,
    )
    model.summary(print_fn=logger.info)
    logger.info("      Curriculum gamma: %.1f -> %.1f en %d épocas (schedule=%s)",
               GAMMA_INITIAL, GAMMA_MAX, GAMMA_WARMUP_EPOCHS, GAMMA_SCHEDULE)
    logger.info("      LR CosineDecayRestarts: init=%.1e primer_ciclo=%d steps (%d épocas) t_mul=%.1f m_mul=%.2f",
               LR_INITIAL, steps_per_epoch * COSINE_RESTART_EPOCH_PERIOD,
               COSINE_RESTART_EPOCH_PERIOD, COSINE_T_MUL, COSINE_M_MUL)

    checkpoint_path = output_dir / "model_v5_best.keras"
    history, model = train_model(
        model, X_train, asset_id_train, y_train,
        checkpoint_path=checkpoint_path,
        epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
        gamma_variable=gamma_variable, gamma_max=GAMMA_MAX,
        gamma_warmup_epochs=GAMMA_WARMUP_EPOCHS, gamma_schedule=GAMMA_SCHEDULE,
        gamma_sigmoid_steepness=GAMMA_SIGMOID_STEEPNESS,
    )

    logger.info("[4/5] Evaluación por activo (reconstrucción de precio + directional accuracy)...")
    results = {}
    for ticker in tickers:
        results[ticker] = evaluate_asset(model, test_sets[ticker], ticker, target_scalers[ticker])
        r = results[ticker]
        logger.info("      %-10s MAE=$%.2f  RMSE=$%.2f  DirAcc=%.1f%%",
                    ticker, r["mae_price"], r["rmse_price"], r["directional_accuracy"] * 100)

    X_test_all = np.concatenate([test_sets[t]["X"] for t in tickers], axis=0)
    asset_id_test_all = np.concatenate([test_sets[t]["asset_id"] for t in tickers], axis=0)
    attribution = compute_feature_attribution(model, X_test_all, asset_id_test_all)
    logger.info("      Atribución por gradiente (|∂ŷ/∂x|) — top 5 de 17 variables:")
    for feat, val in attribution.head(5).items():
        marker = "  <- NEWS_SENTIMENT_SCORE" if feat == "NEWS_SENTIMENT_SCORE" else ""
        logger.info("        %-22s %.6f%s", feat, val, marker)
    news_rank = int(attribution.index.get_loc("NEWS_SENTIMENT_SCORE")) + 1
    logger.info("      NEWS_SENTIMENT_SCORE -> rank %d/%d, atribución=%.6f",
                news_rank, len(attribution), attribution["NEWS_SENTIMENT_SCORE"])

    chart_path = output_dir / "evaluation_chart.png"
    plot_regime_evaluation(results, regime_name, chart_path)

    logger.info("[5/5] Exportando modelo '%s' + scalers -> %s ...", regime_name, output_dir)
    model_path = output_dir / "model_v5.keras"
    scalers_path = output_dir / "scalers_dict.pkl"

    model.save(model_path, overwrite=True)
    scalers_payload = {
        "regime_name": regime_name,
        "feature_scalers": feature_scalers,
        "target_scalers": target_scalers,
        "asset_to_id": asset_to_id,
        "lookback": LOOKBACK,
        "macro_tickers": MACRO_TICKERS,
        "tickers": tickers,
        "tech_cols": TECH_COLS,
    }
    with open(scalers_path, "wb") as f:
        pickle.dump(scalers_payload, f)

    checkpoint_path.unlink(missing_ok=True)  # artefacto intermedio; model_v5.keras es la fuente final

    logger.info("      Modelo  -> %s", model_path)
    logger.info("      Scalers -> %s", scalers_path)
    logger.info("      Chart   -> %s", chart_path)

    return {"model_path": model_path, "scalers_path": scalers_path, "results": results}


# ============================================================
# ORQUESTACIÓN GLOBAL — dos ciclos independientes y secuenciales
# ============================================================
if __name__ == "__main__":
    # V9 (CI/CD) — UNA vez por corrida, antes de tocar cualquier régimen
    # (ver purge_data_cache: purgar dentro del bucle borraría el .parquet
    # que el régimen anterior de esta misma corrida acaba de escribir).
    if FORCE_FRESH_DATA:
        purge_data_cache()

    summary = {}
    for regime_name, regime_cfg in REGIMES.items():
        summary[regime_name] = run_regime_pipeline(regime_name, regime_cfg)

    logger.info("=" * 78)
    logger.info("ENTRENAMIENTO V5 COMPLETO — %d especialistas persistidos en %s",
               len(summary), MODELS_ROOT)
    for regime_name, info in summary.items():
        avg_dir_acc = np.mean([r["directional_accuracy"] for r in info["results"].values()])
        logger.info("  %-18s edge direccional promedio = %.1f%%",
                    regime_name, avg_dir_acc * 100)
    logger.info("=" * 78)
# %%