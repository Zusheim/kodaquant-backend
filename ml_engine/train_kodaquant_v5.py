# %%
# NOTA 2026-09 -- limpieza: la línea original acá (`#!pip install yfinance nltk
# -q`) era dead-reference doble: `nltk` no se importa en ningún punto de este
# archivo (confirmado con pyflakes) y `yfinance` vive en `requirements.txt`
# (ver auditoría de dependencias) -- el punto de entrada real de instalación
# es `pip install -r requirements.txt` en la raíz del proyecto, no esta línea.

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

NUEVO EN V23.5 (Directiva Head Quant -- BTC/equity rezagados vs. ETH/SPY):
    6) Calibración de Triple Barrier DESACOPLADA por activo (ver
       `calibrate_crypto_triple_barrier`): cada ticker cripto busca su
       propio pt_mult/sl_mult/vol_lookback/max_holding_days de forma
       INDEPENDIENTE (ya no maximin pooled) -- ETH ya no cede óptimo por
       compromiso con BTC. ROMPE el esquema de `scalers_dict.pkl`:
       `tb_params` pasa de dict plano a dict-de-dicts `{ticker: {...}}`.
       `services/quanti_engine.py` (`_forecast_asset_triple_barrier`) NO
       está en este archivo y DEBE actualizarse a mano para leer
       `tb_params[ticker]` en vez de `tb_params` plano, o la inferencia
       usará barreras equivocadas sin error visible.
    7) [ROLLBACK EN V23.6 -- ver nota abajo] `PUELL_MULTIPLE_ZSCORE` --
       feature BTC-únicamente (ver TECH_COLS/`engineer_asset`), ciclo de
       rentabilidad de minero (Puell, 2019). Cero llamada de red (a
       diferencia de ONCHAIN_ACTIVITY_ROC): la emisión diaria de BTC es un
       calendario determinístico conocido (halvings), se deriva 100% de
       RAW_CLOSE ya descargado.

ROLLBACK QUIRÚRGICO V23.6 (Directiva Head Quant -- solo equity_specialist):
    V23.5 introdujo `EQUITY_VOL_SAMPLE_WEIGHT_*`/`_compute_equity_vol_
    sample_weights` (punto 7 original de este changelog, ver historial de
    versiones): sample_weight por ticker proporcional a volatilidad
    realizada, con el objetivo de dejar de castigar a TSLA/META por ser
    más ruidosos que SPY/AAPL. Resultado real medido: el modelo aprendió a
    perseguir el RUIDO de los outliers volátiles en vez de la señal
    direccional -- Directional Accuracy colapsó (NVDA 52.x%->47.3%, META
    48.3%). Diagnóstico: ponderar por volatilidad realizada en un target
    de RETORNO (a diferencia del sample_weight por PnL/uniqueness de
    `crypto_specialist`, que pondera APUESTAS discretas de Triple Barrier,
    no un continuo) amplifica exactamente las muestras con peor relación
    señal/ruido del propio universo.
    PURGADO por completo de este archivo: `EQUITY_VOL_SAMPLE_WEIGHT_ENABLED`,
    `EQUITY_VOL_SAMPLE_WEIGHT_KAPPA`, `_compute_equity_vol_sample_weights`,
    y toda la rama de ensamblado de `sample_weight` para
    target_mode="regression" en `run_regime_pipeline` -- `equity_specialist`
    vuelve a entrenar con gradiente PLANO/UNIFORME (cada día pesa 1.0,
    sample_weight=None), bit a bit idéntico a V6/pre-V23.5. Reflejado en
    paralelo en `walk_forward_eval.py` y `pnl_backtest.py` (ambos
    importaban `tkv5.EQUITY_VOL_SAMPLE_WEIGHT_*` -- ver sus propios
    changelogs). `crypto_specialist` (Triple Barrier asimétrica por activo,
    Puell Multiple, Fractional Differencing) NO se toca -- ver puntos 6 y 7
    de este changelog, siguen intactos.
"""

from __future__ import annotations

import logging
import os

# V23.2 -- DETERMINISMO REAL: `keras.utils.set_random_seed(SEED)` (más abajo)
# fija numpy/python/TF RNG, pero TF en CPU sigue siendo no-determinístico por
# default (reducciones paralelas en Eigen no son orden-estable entre runs) --
# esto DEBE fijarse ANTES de `import keras`/`import tensorflow`, cualquier
# import tardío ya no tiene efecto. Confirmado empíricamente: dos corridas
# con `dist_real` IDÉNTICO (mismas etiquetas Triple Barrier, mismo seed=42)
# dieron AccAccionable distinta (BTC 42.2%->46.0%, ETH 51.8%->46.1%) -- la
# única variable no fijada era el orden de reducción flotante intra-CPU.
# Costo: fuerza single-thread en TF (más lento) -- dataset de este script es
# pequeño (miles de filas x <=8 tickers), el trade-off vale la pena para
# poder comparar corridas de forma válida. Override con
# `KODAQUANT_FAST_NONDETERMINISTIC=1` si se necesita velocidad y no
# comparabilidad exacta (p.ej. una corrida final ya calibrada).
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("PYTHONHASHSEED", "42")
_FORCE_DETERMINISTIC_CPU = os.environ.get("KODAQUANT_FAST_NONDETERMINISTIC", "0") != "1"

import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
# FIX 2026-08-31 (auditoría rescate) -- `Optional` se usaba en 4 firmas de
# función (_fetch_fear_greed_index/_fetch_onchain_tx_volume_usd/
# _fetch_binance_funding_history/_fetch_onchain_activity, bloque V24 Alpha
# Cripto) sin estar importado. `from __future__ import annotations` (arriba)
# lo dejaba LATENTE -- las anotaciones quedan como string sin evaluar salvo
# que algo llame typing.get_type_hints()/inspect.signature con eval_str, así
# que nunca explotó en esta corrida -- pero cualquier tooling que sí
# introspeccione (pydantic, algunos linters) lo hubiera roto con NameError
# real. `timedelta` (mismo import) se retira: cero uso real en todo el
# archivo (confirmado con pyflakes).
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # CRÍTICO: backend no interactivo — CERO plt.show() en todo el módulo.

import keras
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from keras import layers
from scipy.interpolate import CubicSpline
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- RESOLUCIÓN ABSOLUTA DE IMPORTS (Directiva 2 — fin al Import Hell) ----
# Este script vive en `ml_engine/`, junto a `data_pipeline.py` y a
# `kodaquant_core.py` (Directiva 1: SINGLE SOURCE OF TRUTH, sin copias
# duplicadas). Sube desde este archivo hasta encontrar la raíz del proyecto
# -- el primer ancestro que contiene TANTO `ml_engine/` como `services/`
# como subdirectorios -- y la inserta en `sys.path[0]`. A partir de ahí,
# `ml_engine` y `services` son paquetes válidos sin importar el cwd desde
# el que se invoque `python ml_engine/train_kodaquant_v5.py`. Reemplaza por
# completo el patrón try/except dual + copias duplicadas de
# `kodaquant_core.py`/`data_pipeline.py` que causaba
# `ModuleNotFoundError`/`ImportError` al invocar desde la raíz del repo.
def _bootstrap_project_root() -> Path:
    here = Path(__file__).resolve()
    for ancestor in (here.parent, *here.parents):
        if (ancestor / "ml_engine").is_dir() and (ancestor / "services").is_dir():
            return ancestor
    # Fallback determinista: este archivo vive en ml_engine/, por lo que su
    # abuelo es la raíz del proyecto por construcción, incluso si `services/`
    # todavía no existe en el entorno actual (ej. checkout parcial/CI).
    return here.parent.parent


_PROJECT_ROOT = _bootstrap_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# V23.5 -- FIX RAÍZ de los avisos "FredClient sin api_key" / titulares
# escasos que caen a Finnhub sin key: `python-dotenv` YA está pineado en
# requirements.txt pero `load_dotenv()` NUNCA se invocaba en ningún archivo
# del proyecto (verificado: cero ocurrencias en train_kodaquant_v5.py,
# data_pipeline.py, market_data.py, quanti_engine.py, prediccion.py) --
# FRED_API_KEY/FINNHUB_API_KEY se leen de `os.environ` a nivel de módulo
# (línea ~680 acá, línea 223 de data_pipeline.py, línea 331 de
# market_data.py) ANTES de que nada tuviera oportunidad de cargar un
# `.env`. Si tu `.env` vive en la raíz del repo (junto a `ml_engine/` y
# `services/`) con `FRED_API_KEY=...`/`FINNHUB_API_KEY=...`, esto ya lo
# resuelve solo. Si NUNCA generaste esas keys, esto no las inventa --
# FRED: https://fred.stlouisfed.org/docs/api/api_key.html (gratis,
# instantáneo) -- Finnhub: https://finnhub.io/register (gratis, tier
# suficiente para /company-news y /news?category=crypto). DEBE ejecutarse
# ANTES de `from ml_engine.data_pipeline import ...` (línea de abajo) y
# antes de `_fred_client = FredClient(...)` (más abajo en este archivo) --
# ambos leen el env var a nivel de módulo, en el orden de import/ejecución.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

# Requerimiento 1: pipeline NLP de sentimiento de noticias. Import absoluto
# y directo -- ya no depende de que `ml_engine/` esté implícitamente en
# sys.path por ser el directorio del script.
from ml_engine.data_pipeline import get_daily_news_sentiment  # noqa: E402

# Núcleo compartido train/inferencia -- SINGLE SOURCE OF TRUTH en
# `ml_engine/kodaquant_core.py` (Directiva 1). Import directo y absoluto,
# sin fallback dual: con la raíz del proyecto ya en sys.path, esta ruta
# resuelve siempre, sin importar el cwd de invocación.
from ml_engine.kodaquant_core import (  # noqa: E402
    CacheManager, FredClient, normalize_daily_index, project_macro_frame,
)

# V28 -- alpha gratuito. SINGLE SOURCE OF TRUTH train/serve (copia idéntica en
# services/free_alpha.py). TECH_COLS hereda V28_COLS desde acá: no puede desincronizarse.
from ml_engine.free_alpha import (  # noqa: E402
    V28_COLS, V28_CRYPTO_COLS, V28_EQUITY_COLS,
    build_crypto_flow_features, compute_equity_open_features,
)

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

# ============================================================
# V22 — TRIPLE BARRIER META-LABELING (crypto_specialist ÚNICAMENTE)
# ============================================================
# Directiva del Head Quant: el edge direccional puro (DirAcc ±1 al cierre)
# está estadísticamente agotado en crypto_specialist (49-50% constante,
# ver docstring de módulo V6/V19 arriba) tras agotar fuentes gratuitas
# (OHLCV, Funding Rates, Fear&Greed, multi-seed). Cambio de paradigma: en
# vez de predecir la DIRECCIÓN de mañana, se predice si el precio de HOY
# tocará una barrera de toma de ganancia o de stop-loss (ancho ESCALADO a
# la volatilidad realizada del momento, cero parámetro fijo en unidades
# de precio) ANTES de que expire una ventana de `max_holding_days`, o si
# ninguna de las dos se toca dentro de ese horizonte -- Triple Barrier
# Method (López de Prado, "Advances in Financial Machine Learning", cap.
# 3: barrera superior/inferior dinámicas + barrera vertical de tiempo,
# clasificación del primer toque). `equity_specialist` NO se toca --
# `target_mode="regression"` (ver REGIMES abajo) preserva EXACTAMENTE el
# pipeline DirectionalGaussianNLL ya validado (Sharpe 1.52, operando en
# paper trading vía `services/live_trade.py`) -- ver `build_model`/
# `build_asset_dataset`/`train_model`, cada uno bifurca en
# `target_mode` con el branch de regresión BIT A BIT idéntico a V6.
TRIPLE_BARRIER_PT_MULT = 2.0          # barrera superior = close_t * exp(+pt_mult * vol_t)
TRIPLE_BARRIER_SL_MULT = 2.0          # barrera inferior = close_t * exp(-sl_mult * vol_t) -- simétrica por defecto
TRIPLE_BARRIER_MAX_HOLDING_DAYS = 10  # barrera vertical -- horizonte máximo de la apuesta (~2 semanas hábiles)
TRIPLE_BARRIER_VOL_LOOKBACK = 20      # ventana de vol realizada que escala pt/sl -- mismo rol que VOL_LOOKBACK_DAYS en live_trade.py/pnl_backtest.py
TRIPLE_BARRIER_MIN_VOL = 1e-6         # floor defensivo -- evita barreras degeneradas si vol_t≈0

# Orden FIJO de clases -- índice 1 (TIMEOUT) es intencionalmente el centro:
# permite tratar "SL" y "TP" como los dos extremos direccionales de un
# espectro, análogo al signo ±1 del régimen de regresión, con 0=TIMEOUT
# como la clase de "sin convicción" (meta-labeling: no vale la pena operar).
TB_CLASS_NAMES = ["SL", "TIMEOUT", "TP"]
TB_LABEL_TO_CLASS_IDX = {-1: 0, 0: 1, 1: 2}
TB_CLASS_IDX_TO_LABEL = {v: k for k, v in TB_LABEL_TO_CLASS_IDX.items()}
TB_TIMEOUT_CLASS_IDX = TB_LABEL_TO_CLASS_IDX[0]  # = 1

# CategoricalFocalCrossentropy (Lin et al. 2017, "Focal Loss for Dense
# Object Detection") -- built-in Keras 3 (`keras.losses.
# CategoricalFocalCrossentropy`, validado contra la doc oficial antes de
# usarlo acá). `gamma` reduce el peso de gradiente de ejemplos "fáciles"
# (softmax ya confiado en la clase correcta) y concentra el aprendizaje en
# los difíciles; `alpha` (lista por-clase, ver `run_regime_pipeline`) hace
# el trabajo de balanceo de clases -- TIMEOUT casi siempre domina en
# volumen frente a SL/TP. Directiva 3 (Risk-Adjusted Loss) se implementa
# APARTE, vía `sample_weight` en `train_model` (magnitud del retorno
# realmente logrado por cada apuesta, ver `build_asset_dataset`) --
# separación deliberada de responsabilidades: `alpha` corrige DESBALANCE
# DE CLASES, `sample_weight` corrige IMPORTANCIA POR PnL. Combinarlos en
# un único factor sería estadísticamente correcto también, pero mezclaría
# dos fuentes de sesgo distintas en un solo número sin poder auditar cada
# una por separado.
TRIPLE_BARRIER_FOCAL_GAMMA = 2.0
TRIPLE_BARRIER_LABEL_SMOOTHING = 0.05

# ============================================================
# V23 — CALIBRACIÓN DINÁMICA DE BARRERAS (crypto_specialist)
# ============================================================
# DIAGNÓSTICO V22: Acc=33.2% (BTC) / 38.7% (ETH) en un problema de 3 clases
# ES adivinar al azar (33.3%) -- confirma que pt_mult=sl_mult=2.0 FIJO
# (mismo valor, literatura genérica, jamás calibrado contra la distribución
# real de clases) produce, o (a) TIMEOUT dominando >80% de las muestras
# (barreras demasiado anchas para vol_lookback=20/max_holding=10 en cripto)
# o (b) barreras demasiado ANGOSTAS relativo al ruido intradía-a-diario no
# capturado por cierres (ver docstring de compute_triple_barrier_labels) --
# en cualquiera de los dos extremos, la CategoricalFocalCrossentropy con
# `alpha` de rebalanceo extremo empuja al softmax hacia una salida ~uniforme
# (ver quantmemo.com/concepts/triple-barrier-labeling: "Barriers too tight
# relative to noise produce near-random labels"). Fix: grid search 100%
# gratuito (cero API extra, solo los mismos closes ya descargados) que
# calibra pt_mult/sl_mult ASIMÉTRICOS (paperswithbacktest.com/course/
# triple-barrier-method: "Setting asymmetric barriers... is common") +
# vol_lookback/max_holding_days, ÚNICAMENTE sobre el split de TRAIN de cada
# ticker cripto (cero fuga hacia test -- ver `calibrate_crypto_triple_barrier`),
# maximizando el balance de clases accionables (SL/TP) sin dejar que
# TIMEOUT degenere el problema a una clase única.
CRYPTO_CALIBRATE_BARRIERS = True
CRYPTO_BARRIER_PT_SL_GRID = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5)   # asimétrico: pt != sl permitido
CRYPTO_BARRIER_VOL_LOOKBACK_GRID = (10, 15, 20, 25)
CRYPTO_BARRIER_MAX_HOLDING_GRID = (5, 7, 10, 14)
CRYPTO_BARRIER_TIMEOUT_FRAC_BAND = (0.15, 0.65)  # fuera de esta banda, combo descalificado (degenerado)


def _calibrate_single_asset_triple_barrier(ticker: str, train_slice: pd.DataFrame) -> dict:
    """
    V23.5 -- Grid search de barreras dinámicas para UN ticker, TRAIN-ONLY
    (cero fuga hacia test). Mismas 3 etapas que la calibración V23/V23.1-2
    (pt_mult x sl_mult asimétrico -> vol_lookback x max_holding_days ->
    refinamiento fino de pt/sl), pero el score ya NO es maximin sobre
    varios tickers -- es `min(P(SL), P(TP))` DEL PROPIO TICKER. Extraída
    como función de un solo activo para que `calibrate_crypto_triple_barrier`
    la corra independiente por ticker (ver Directiva 1, V23.5): BTC ya no
    comparte barreras con ETH, así que deja de heredar un compromiso
    optimizado para el activo ajeno.
    """
    def _score(pt_mult: float, sl_mult: float, vol_lookback: int, max_holding_days: int) -> tuple[float, dict]:
        tb = compute_triple_barrier_labels(
            train_slice, pt_mult=pt_mult, sl_mult=sl_mult,
            max_holding_days=max_holding_days, vol_lookback=vol_lookback,
        )
        labels = tb["TB_LABEL"].dropna().to_numpy()
        if labels.size == 0:
            return -np.inf, {}
        sl_n = int((labels == -1).sum())
        to_n = int((labels == 0).sum())
        tp_n = int((labels == 1).sum())
        total = labels.size
        p_sl, p_to, p_tp = sl_n / total, to_n / total, tp_n / total
        lo, hi = CRYPTO_BARRIER_TIMEOUT_FRAC_BAND
        stats = {"p_sl": p_sl, "p_timeout": p_to, "p_tp": p_tp, "n": total}
        if not (lo <= p_to <= hi):
            return -np.inf, stats
        return min(p_sl, p_tp), stats

    logger.info("      [Calibración V23.5] %s -- Etapa 1/3 -- grid pt_mult x sl_mult (asimétrico)...", ticker)
    best_score, best_pt, best_sl, best_stats = -np.inf, TRIPLE_BARRIER_PT_MULT, TRIPLE_BARRIER_SL_MULT, {}
    for pt in CRYPTO_BARRIER_PT_SL_GRID:
        for sl in CRYPTO_BARRIER_PT_SL_GRID:
            score, stats = _score(pt, sl, TRIPLE_BARRIER_VOL_LOOKBACK, TRIPLE_BARRIER_MAX_HOLDING_DAYS)
            if score > best_score:
                best_score, best_pt, best_sl, best_stats = score, pt, sl, stats
    logger.info("      [Calibración V23.5] %s -- Mejor pt_mult=%.2f sl_mult=%.2f -> %s (score=%.4f)",
                ticker, best_pt, best_sl, best_stats, best_score)

    logger.info("      [Calibración V23.5] %s -- Etapa 2/3 -- grid vol_lookback x max_holding_days...", ticker)
    best_vl, best_mh = TRIPLE_BARRIER_VOL_LOOKBACK, TRIPLE_BARRIER_MAX_HOLDING_DAYS
    for vl in CRYPTO_BARRIER_VOL_LOOKBACK_GRID:
        for mh in CRYPTO_BARRIER_MAX_HOLDING_GRID:
            score, stats = _score(best_pt, best_sl, vl, mh)
            if score > best_score:
                best_score, best_vl, best_mh, best_stats = score, vl, mh, stats
    logger.info("      [Calibración V23.5] %s -- Etapa 2/3 FIN pt_mult=%.2f sl_mult=%.2f vol_lookback=%d "
                "max_holding_days=%d -> %s (score=%.4f)",
                ticker, best_pt, best_sl, best_vl, best_mh, best_stats, best_score)

    logger.info("      [Calibración V23.5] %s -- Etapa 3/3 -- refinamiento fino pt_mult x sl_mult (paso 0.1)...", ticker)
    fine_grid = [round(best_pt + d, 2) for d in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3) if best_pt + d > 0]
    fine_grid_sl = [round(best_sl + d, 2) for d in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3) if best_sl + d > 0]
    for pt in fine_grid:
        for sl in fine_grid_sl:
            score, stats = _score(pt, sl, best_vl, best_mh)
            if score > best_score:
                best_score, best_pt, best_sl, best_stats = score, pt, sl, stats
    logger.info("      [Calibración V23.5] %s -- FINAL pt_mult=%.2f sl_mult=%.2f vol_lookback=%d "
                "max_holding_days=%d -> %s (score=%.4f)",
                ticker, best_pt, best_sl, best_vl, best_mh, best_stats, best_score)

    if best_score == -np.inf:
        logger.warning("      [Calibración V23.5] %s -- NINGÚN combo cumplió la banda de TIMEOUT -- "
                        "se conservan los defaults de literatura (pt=sl=2.0, vl=20, mh=10).", ticker)
        return {
            "pt_mult": TRIPLE_BARRIER_PT_MULT, "sl_mult": TRIPLE_BARRIER_SL_MULT,
            "max_holding_days": TRIPLE_BARRIER_MAX_HOLDING_DAYS, "vol_lookback": TRIPLE_BARRIER_VOL_LOOKBACK,
        }
    return {"pt_mult": best_pt, "sl_mult": best_sl, "max_holding_days": best_mh, "vol_lookback": best_vl}


def calibrate_crypto_triple_barrier(df_assets: dict[str, pd.DataFrame], tickers: list[str],
                                     train_ratio: float | None = None) -> dict[str, dict]:
    """
    V23.5 (Directiva 1, Head Quant) -- DESACOPLADA por activo: cada ticker
    cripto corre su propio grid search TRAIN-ONLY vía
    `_calibrate_single_asset_triple_barrier`, maximizando SU PROPIO
    `min(P(SL), P(TP))`, sin compromiso con ningún otro ticker del régimen.

    V23/V23.1 corrían esto pooled/maximin sobre TODOS los tickers cripto a
    la vez porque `tb_params` era un único dict compartido en producción
    (mismo esquema que `quanti_engine.py` esperaba). Diagnóstico de esta
    sesión: eso significa que ETH (ya en 54.5% AccAccionable) seguía
    cediendo parte de su óptimo para no dejar a BTC en un combo degenerado
    -- BTC en 47.9% confirma que el compromiso pooled TODAVÍA no le alcanza
    a BTC, y ETH no tiene por qué seguir pagando ese costo.

    ROMPE DE ESQUEMA (documentado en el docstring de módulo, V23.5 punto 6):
    devuelve `{ticker: {pt_mult, sl_mult, vol_lookback, max_holding_days}}`
    en vez de un único dict plano. `run_regime_pipeline`/`scalers_dict.pkl`
    ya están actualizados acá para el nuevo esquema -- `quanti_engine.py`
    (fuera de este archivo) todavía NO, y debe indexar `tb_params[ticker]`
    en inferencia o quedará leyendo un dict con llaves de ticker donde
    esperaba `pt_mult` directo (KeyError inmediato y visible -- no un
    silent-fail, pero sí un breaking change de producción real).
    """
    train_ratio = TRAIN_RATIO if train_ratio is None else train_ratio  # TRAIN_RATIO se define más abajo en el módulo -- default NO puede resolverse en la firma
    tb_params_by_ticker: dict[str, dict] = {}
    for t in tickers:
        df = df_assets[t]
        split_idx = int(len(df) * train_ratio)
        train_slice = df.iloc[:split_idx]
        tb_params_by_ticker[t] = _calibrate_single_asset_triple_barrier(t, train_slice)
    return tb_params_by_ticker


# ============================================================
# V23.5 SAMPLE WEIGHTING DINÁMICO POR VOLATILIDAD (equity_specialist)
# -- ROLLBACK QUIRÚRGICO EN V23.6, ver changelog del módulo --
# ============================================================
# `EQUITY_VOL_SAMPLE_WEIGHT_ENABLED`/`EQUITY_VOL_SAMPLE_WEIGHT_KAPPA`/
# `_compute_equity_vol_sample_weights` vivían acá. Purgados por completo:
# ponderar por volatilidad realizada le enseñó al modelo a perseguir ruido
# de outliers volátiles en vez de señal direccional (DirAcc NVDA
# 52.x%->47.3%, META 48.3%). `equity_specialist` ya NO calcula ni pasa
# ningún sample_weight -- gradiente plano/uniforme, bit a bit idéntico a
# V6/pre-V23.5. `crypto_specialist` no se ve afectado por este rollback:
# su propio sample_weight (magnitud de PnL + uniqueness, ver
# `build_asset_dataset`/Directiva 3 más abajo) es un mecanismo
# completamente distinto y sigue intacto.


def _average_uniqueness(entry_idx: np.ndarray, bars_to_exit: np.ndarray, n: int) -> np.ndarray:
    """
    V23 -- Uniqueness promedio (López de Prado, AFML cap. 4) para etiquetas
    Triple Barrier CON SOLAPAMIENTO: cada apuesta `s` ocupa el tramo
    `[s, s+bars_to_exit[s]]`; con `max_holding_days` >= 5-14 y entradas
    diarias consecutivas, tramos vecinos se solapan masivamente -- el
    gradiente ve la misma "sorpresa" de mercado una y otra vez, contada
    como si fueran muestras i.i.d. independientes, lo cual puede degradar
    la señal efectiva de entrenamiento tanto como una barrera mal calibrada
    (ver quantmemo.com/concepts/triple-barrier-labeling, "Class imbalance...
    weight samples by their uniqueness"). Concurrencia `c[t]` = número de
    tramos activos en la barra `t`; uniqueness de la muestra `s` = promedio
    de `1/c[t]` sobre su propio tramo -- una apuesta que solapa con muchas
    otras pesa menos, una época de mercado "propia" (poco solapamiento)
    pesa más. Se multiplica (no reemplaza) al peso de magnitud/PnL ya
    existente (Directiva 3) en `build_asset_dataset`.
    """
    concurrency = np.zeros(n, dtype=np.float64)
    exit_idx = np.minimum(entry_idx + bars_to_exit.astype(np.int64), n - 1)
    for s, e in zip(entry_idx, exit_idx):
        concurrency[s:e + 1] += 1.0
    concurrency = np.maximum(concurrency, 1.0)
    uniqueness = np.array([float(np.mean(1.0 / concurrency[s:e + 1])) for s, e in zip(entry_idx, exit_idx)])
    return uniqueness

REGIMES: dict[str, dict] = {
    "equity_specialist": {
        "tickers": ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "SPY"],
        "oversample_factor": 1.0,  # 8 tickers, volumen ya suficiente -> no-op exacto (idéntico a V5)
        "target_mode": "regression",  # V22 — dataset/arquitectura intactos. V23.5 había añadido sample_weight por volatilidad; ROLLBACK en V23.6 (ver changelog del módulo) -- vuelve a ser BIT A BIT idéntico a V6/pre-V23.5, sample_weight=None. Sharpe 1.52 mide este mismo pipeline plano.
    },
    "crypto_specialist": {
        # V19: vuelta a 2 tickers a pedido explícito -- BTC-USD/ETH-USD
        # solos midieron 46.8%/48.1% en la corrida de 5 activos (V18,
        # consistente con las dos mediciones previas de solo-2: 47-50%).
        # Ampliar a 5 NO los movió a ellos -- lo que cambió fue el
        # agregado (arrastrado por XRP/BCH, más nuevos y volátiles).
        # Volver a 2 no resuelve por sí solo que BTC/ETH crucen 50%: el
        # número individual de cada uno sigue siendo el mismo problema de
        # fondo, y con n=8 el IC 95% vuelve a ser demasiado ancho para
        # distinguir "sin edge" de "no hay muestra suficiente" (ver
        # walk_forward_eval.py). Documentado, no oculto.
        "tickers": ["BTC-USD", "ETH-USD"],
        # V27 -- lever no probada hasta hoy: BTC-USD tiene histórico en
        # yfinance desde 2014 (~12 años), PERIOD="10y" global le recorta
        # ~2 años sin que nadie lo haya medido. "max" = todo lo disponible
        # por ticker (ETH, con historia desde 2017, no gana nada extra --
        # esto es específicamente una apuesta por BTC). Override SOLO acá,
        # equity_specialist sigue usando el PERIOD global sin tocar.
        "period": "max",
        "oversample_factor": CRYPTO_OVERSAMPLE_FACTOR,
        # V22 — CAMBIO DE PARADIGMA (Directiva Head Quant): DirAcc ±1 al
        # cierre está estadísticamente agotado acá (49-50% constante, ver
        # docstring de módulo V6). Se reemplaza por Triple Barrier Method
        # (López de Prado) -- clasificación 3-way (SL/TIMEOUT/TP) en vez
        # de regresión de retorno, ver `compute_triple_barrier_labels`,
        # `build_model` y `build_asset_dataset`. `equity_specialist`
        # arriba NO se toca -- target_mode="regression" es el default y
        # preserva el pipeline V5/V6 bit a bit.
        "target_mode": "triple_barrier",
        "pt_mult": TRIPLE_BARRIER_PT_MULT,
        "sl_mult": TRIPLE_BARRIER_SL_MULT,
        "max_holding_days": TRIPLE_BARRIER_MAX_HOLDING_DAYS,
        "vol_lookback": TRIPLE_BARRIER_VOL_LOOKBACK,
        # V23.4 -- log de la corrida V23.3 mostró "Restoring model weights
        # from the end of the best epoch: 91" con EPOCHS=100: el modelo
        # TODAVÍA estaba mejorando cerca del techo artificial del cap, no
        # convergido/estancado (si estuviera estancado, `patience=12`
        # habría cortado bien antes de la época 88). Se sube el presupuesto
        # SOLO para cripto -- CERO cambio de arquitectura/features/seed,
        # `EarlyStopping(patience=12)` sigue siendo el que decide cuándo
        # parar de verdad; esto solo le saca el techo artificial.
        "epochs": 180,
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

# V24 — ALPHA CRIPTO (Directiva Head Quant, ver bloque de funciones antes de
# `engineer_asset`): FUNDING_RATE_ZSCORE/ONCHAIN_ACTIVITY_ROC/FRAC_DIFF_LOG_CLOSE
# neutras (0.0) para equity_specialist -- MISMO patrón que FEAR_GREED_INDEX
# arriba, cero impacto real de información en ese régimen. V23.5 agrega
# PUELL_MULTIPLE_ZSCORE con el mismo patrón, pero BTC-únicamente incluso
# DENTRO de crypto_specialist -- 0.0 neutro también para ETH-USD (ver
# docstring del bloque de constantes `BTC_HALVING_SCHEDULE`).
TECH_COLS = [
    "RSI_14", "EMA20_DEV_PCT", "MACD_PCT", "MACD_SIGNAL_PCT", "SENTIMENT_SCORE",
    "ATR_PCT", "BB_WIDTH_20", "OBV_ROC_20", "ADX_14", "STOCH_K_14",
    "NEWS_SENTIMENT_SCORE", "FEAR_GREED_INDEX",
    "FUNDING_RATE_ZSCORE", "ONCHAIN_ACTIVITY_ROC", "FRAC_DIFF_LOG_CLOSE",
    "PUELL_MULTIPLE_ZSCORE",  # V23.5 (Directiva 3) -- BTC únicamente, 0.0 neutro para ETH/equity
    "PARKINSON_VOL_ZSCORE",   # AUDITORÍA RESCATE 2026-08 -- AMBOS regímenes, proxy libre de IV (ver engineer_asset)
    "VOL_OF_VOL_ZSCORE",      # AUDITORÍA RESCATE 2026-08 -- cripto únicamente, 0.0 neutro en equity
    "FUNDING_RATE_MOMENTUM_ZSCORE",  # V26 (Directiva Head Quant) -- velocidad del crowding, cripto únicamente
    "AMIHUD_ILLIQUIDITY_ZSCORE",     # V26 (Directiva Head Quant) -- liquidez, 100% OHLCV, cripto únicamente
    "GARMAN_KLASS_VOL_ZSCORE",       # V27 -- microestructura OHLC, AMBOS regímenes
    "FIB_GAP_Z_20", "FIB_GAP_Z_50",  # V27 -- gap firmado a Fibonacci sobre swings rolling, AMBOS regímenes
    "OFI_PROXY_ZSCORE",              # V27 -- order flow imbalance vía CLV*volumen, AMBOS regímenes
    "RETURN_ENTROPY_ZSCORE",         # V27 -- entropía de Shannon normalizada, AMBOS regímenes
] + list(V28_COLS)  # V28 -- 4 equity + 4 cripto (0.0 neutro en el régimen opuesto), orden = free_alpha.V28_COLS
# V28 -- 25 base + 8 = 33 TECH_COLS -> N_FEATURES = 1 + 33 + 5 = 39 (38 si el gate quita NEWS).
if 1 + len(TECH_COLS) + len(MACRO_TICKERS) != 39:
    raise ImportError(
        f"[V28] N_FEATURES={1 + len(TECH_COLS) + len(MACRO_TICKERS)} != 39 esperado "
        "-- TECH_COLS/V28_COLS/MACRO_TICKERS desalineados.")
# V27 -- NEWS_SENTIMENT_SCORE puede ser REMOVIDA in-place por
# `run_regime_pipeline` si no hay backend de sentimiento (ver
# `_apply_news_sentiment_gate`). `N_FEATURES` se recalcula ahí mismo.
N_FEATURES = 1 + len(TECH_COLS) + len(MACRO_TICKERS)  # LOG_RETURN_1D + técnicos + macro -- SIEMPRE derivado de
# len(TECH_COLS)/len(MACRO_TICKERS), NUNCA hardcodear este número en un comentario/docstring en otro archivo
# (esa es exactamente la causa raíz auditada en services/quanti_engine.py -- ver notas de la auditoría 2026-08).


def _apply_news_sentiment_gate() -> bool:
    """Excluye NEWS_SENTIMENT_SCORE del tensor si no hay backend de sentimiento.

    Se ejecuta UNA vez, ANTES de construir cualquier dataset. Muta TECH_COLS
    in-place (no rebind: `engineer_asset`/`build_asset_dataset` leen el
    global en tiempo de llamada) y recalcula N_FEATURES. La lista resultante
    viaja en scalers_dict.pkl['tech_cols'] -> quanti_engine.py la adopta
    desde ahí, nunca por probe propio: un probe en inferencia podría dar un
    resultado DISTINTO al del entrenamiento y desplazar cada columna dentro
    de feature_scaler.transform() en silencio.
    """
    global N_FEATURES
    if os.getenv("KODAQUANT_FORCE_NEWS_SENTIMENT", "").strip() == "1":
        return True
    try:
        from ml_engine.data_pipeline import news_sentiment_available
        ok = bool(news_sentiment_available())
    except Exception as exc:  # noqa: BLE001
        logger.warning("[NEWS GATE] probe falló (%s: %s) -> se asume NO disponible.",
                       type(exc).__name__, exc)
        ok = False
    if ok:
        logger.info("[NEWS GATE] backend de sentimiento ACTIVO -> NEWS_SENTIMENT_SCORE incluida.")
        return True
    if "NEWS_SENTIMENT_SCORE" in TECH_COLS:
        TECH_COLS.remove("NEWS_SENTIMENT_SCORE")
        N_FEATURES = 1 + len(TECH_COLS) + len(MACRO_TICKERS)
        logger.warning(
            "[NEWS GATE] sin backend de sentimiento -> NEWS_SENTIMENT_SCORE EXCLUIDA "
            "del tensor (N_FEATURES=%d). Inyectar 0.0 constante ajustaría el "
            "feature_scaler sobre una constante y entrenaría un slot muerto. "
            "Los artefactos de esta corrida NO son intercambiables con los de "
            "una corrida CON sentimiento.", N_FEATURES)
    return False


def get_tech_cols(regime_name: str) -> list[str]:
    """Devuelve TECH_COLS para el régimen dado.

    AUDITORÍA RESCATE 2026-08 -- restaurado tras `AttributeError:
    module 'ml_engine.train_kodaquant_v5' has no attribute 'get_tech_cols'`
    real en `walk_forward_eval.py` (línea ~220) y `pnl_backtest.py` (línea
    ~104) -- AMBOS son callers reales de esta función, ninguno la tenía
    definida en este módulo. `TECH_COLS` es hoy una lista GLOBAL compartida
    por los dos regímenes -- las columnas cripto-únicamente
    (`FUNDING_RATE_ZSCORE`, `ONCHAIN_ACTIVITY_ROC`, `FRAC_DIFF_LOG_CLOSE`,
    `PUELL_MULTIPLE_ZSCORE`) ya son 0.0 neutro para equity DENTRO de esa
    misma lista (ver `engineer_asset`), así que `regime_name` se acepta
    solo por compatibilidad de firma con ambos callers y NO cambia el
    resultado -- devolver un subconjunto distinto por régimen rompería
    `N_FEATURES`/la forma del tensor de entrada frente al modelo ya
    entrenado de cada régimen (equity_specialist incluido).
    """
    return list(TECH_COLS)


def get_n_features(regime_name: str) -> int:
    """Devuelve N_FEATURES para el régimen dado.

    AUDITORÍA RESCATE 2026-08 -- mismo bug/mismo fix que `get_tech_cols`
    (ver docstring ahí): `AttributeError: ... has no attribute
    'get_n_features'` real en `walk_forward_eval.py` (línea ~339) y
    `pnl_backtest.py` (línea ~210), confirmado en corrida real del usuario
    -- función nunca definida en este módulo. `N_FEATURES` es un escalar
    GLOBAL (`1 + len(TECH_COLS) + len(MACRO_TICKERS)`) idéntico para los
    dos regímenes -- `regime_name` se acepta solo por compatibilidad de
    firma, no cambia el resultado.
    """
    return N_FEATURES

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
        "tcn_filters": 64,          # V15 — rama TCN paralela, ver build_model
        # V27 — GLU de entrada: proyección compuerta por feature ANTES de
        # BiLSTM/TCN. Con 29 features (6 nuevas de V27) la red ya no puede
        # asumir que todo slot aporta; la compuerta sigmoide aprende a
        # anular los irrelevantes por activo en vez de diluirlos en el L2.
        "glu_units": 32,
    },
    "crypto_specialist": {
        # V28 (Directiva Head Quant) -- reducción de capacidad, NO agregado
        # de señal. Motivo: overfitting confirmado EN VIVO en la corrida sin
        # fuga (2026-09-01): train actionable_accuracy 50%->74% en 25
        # épocas mientras val se mantuvo plano en 41-50%, y el test final
        # terminó por debajo del baseline trivial en ambos activos (BTC
        # 37.1% vs 43.7%, ETH 39.9% vs 45.1%) pese a que `dist_pred` NO
        # colapsó a una clase (proporciones cercanas a `dist_real` en las
        # 3 categorías) -- descarta alpha mal calibrado, apunta a un
        # modelo con demasiada capacidad relativa al dato (145K params
        # para ~4400 ventanas crudas antes de oversampling). Bajado a
        # ~1/4 de params (lstm_units 64->32, tcn_filters 32->16,
        # attention/mha 32->16, heads 4->2), dropout 0.45->0.55, L2
        # 1e-5->5e-5. CAMBIO AISLADO a propósito -- TECH_COLS/features sin
        # tocar, para poder atribuir cualquier diferencia de resultado
        # específicamente a la capacidad, no a una mezcla de variables.
        # HONESTIDAD: dado el patrón de 4 mediciones independientes previas
        # (V19 y anteriores: 46.8-50%; esta corrida sin fuga: 37-40%),
        # esto puede no ser suficiente -- el techo real puede ser de datos
        # (edge inexistente a esta granularidad/universo), no de capacidad.
        "lstm_units": 32,
        "attention_units": 16,
        "mha_heads": 2,
        "mha_key_dim": 16,
        "dropout_rate": 0.55,
        "dense_l2_reg": 5e-5,
        "extra_residual_block": False,
        "tcn_filters": 16,          # V15 — rama TCN paralela, ver build_model
        # V27 — CUELLO DE BOTELLA, no capacidad extra: 29 -> 16. Coherente
        # con el hallazgo V28 (train 74% / val 41% = exceso de capacidad
        # relativa al dato). Subir este número acá revierte ese fix.
        "glu_units": 16,
    },
}

# V15 — Directiva 4 (evolución matemática direccional). Rama TCN (Temporal
# Convolutional Network, Bai/Kolter/Koltun 2018) PARALELA a la rama
# BiLSTM+MultiHeadAttention+BahdanauAttention existente (que YA logra
# 52.8%/51.3% de DirAcc validado, ver docstring de `REGIME_ARCHITECTURE`
# arriba) -- no la reemplaza, la complementa: convoluciones causales
# dilatadas capturan patrones locales/multi-escala de forma NO
# recurrente (sin el sesgo de recencia del último estado oculto de un
# LSTM), robustas a ruido de alta frecuencia por el propio ensanchamiento
# del campo receptivo con la dilatación. Fusionada por concatenación con
# el contexto de Bahdanau antes de `asset_embed` -- aditiva y reversible:
# `KODAQUANT_ENABLE_TCN=0` colapsa el grafo a la topología V10 anterior,
# bit a bit, para poder comparar A/B contra la arquitectura ya validada
# antes de promoverla a producción.
ENABLE_TCN_BRANCH = os.getenv("KODAQUANT_ENABLE_TCN", "1").strip().lower() not in ("0", "false", "no")
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = (1, 2, 4, 8)   # campo receptivo ~ kernel*sum(dilations) > LOOKBACK=60 con margen

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
# AUDITORÍA RESCATE 2026-08 — patience=12/start_epoch=15 cortaba el
# entrenamiento ANTES de que TCN+BiLSTM+MHA convergiera de verdad bajo
# --epochs-per-fold=30 (default de iteración de walk_forward_eval.py/
# pnl_backtest.py, ver docstrings de esos módulos) -- subido al piso
# mínimo pedido (15-20) para dar margen real de convergencia tanto en el
# full-retrain (EPOCHS=100/180) como en cualquier fold recortado.
EARLY_STOPPING_PATIENCE = 18
EARLY_STOPPING_START_EPOCH = 20

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
if _FORCE_DETERMINISTIC_CPU:
    # V23.2 -- pinea el pool de hilos de TF a 1 ANTES de cualquier op/build de
    # modelo: las reducciones paralelas (suma/softmax/etc.) en Eigen no son
    # orden-estable entre hilos, así que aunque numpy/python/TF RNG estén
    # fijados por `set_random_seed`, el resultado numérico de cada forward/
    # backward pass puede variar corrida a corrida. Con 1 hilo, la reducción
    # es secuencial y reproducible bit-a-bit para el mismo seed+datos.
    import tensorflow as _tf
    _tf.config.threading.set_intra_op_parallelism_threads(1)
    _tf.config.threading.set_inter_op_parallelism_threads(1)


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


# --- CONSOLIDACIÓN DEL CORE (Directiva 1) -----------------------------------
# `_ohlcv_cache` reemplaza el parquet ad-hoc anterior: MISMA clase
# (`kodaquant_core.CacheManager`) que usaría el lado de inferencia para su
# propio caché en `services/`, cada uno con su `cache_dir` independiente.
_ohlcv_cache = CacheManager(DATA_DIR, default_max_age_hours=_CACHE_MAX_AGE_HOURS)
# `_fred_client` reemplaza el acceso exclusivo-a-inferencia que tenía FRED
# en `market_data.py` -- ahora el entrenamiento resuelve ^TNX con la MISMA
# fuente primaria (serie oficial DGS10) que ve el modelo en producción,
# vía FredClient. Solo ^TNX tiene serie FRED equivalente 1:1 (rendimiento,
# no precio de instrumento) -- el resto de MACRO_TICKERS (^GSPC/^VIX/GC=F/
# DX-Y.NYB) se mantiene en yfinance, ver `_fetch_all_symbols_flat`.
_fred_client = FredClient(api_key=os.getenv("FRED_API_KEY", ""))
# AUDITORÍA 2026-09 (Directiva Head Quant -- diagnóstico degradación equity+crypto
# simultánea) -- root cause más probable de que AMBOS especialistas (arquitecturas
# y universos de tickers completamente independientes) colapsen a la vez: lo ÚNICO
# que comparten es `MACRO_TICKERS` vía `engineer_asset`/`TECH_COLS`, y de los 5
# tickers macro, 4 de 5 seguían 100% en yfinance pese a que el propio log de
# `_fetch_all_symbols_flat` ("Diagnóstico cobertura", más abajo) documenta
# cobertura real ~0-20% en corridas pasadas para 'DX-Y.NYB' y advierte migrar a
# Twelve Data "igual que ya se hizo en services/market_data.py" -- esa migración
# nunca se replicó acá. Ampliado a series FRED reales (gratis, sin límite de
# volumen relevante para este pipeline) donde existe equivalente 1:1 real:
#   ^VIX  -> VIXCLS (CBOE Volatility Index, serie oficial)
#   ^GSPC -> SP500   (S&P 500, serie oficial FRED -- OJO: histórico FRED solo
#            arranca ~2013, más corto que los ~10y de yfinance; con
#            tail_days=3700 la ventana de entrenamiento típica queda cubierta,
#            pero un LOOKBACK/backtest que pida fechas anteriores a 2013 cae
#            en NaN y aterriza en el fallback yfinance de abajo igual)
# GC=F / DX-Y.NYB NO tienen serie FRED 1:1 (oro/USD index no son series que FRED
# publique de fuente primaria) -- quedan en el fallback Stooq/yfinance de abajo.
_FRED_MACRO_SERIES_MAP: dict[str, str] = {"^TNX": "DGS10", "^VIX": "VIXCLS", "^GSPC": "SP500"}

# Fallback Stooq (CSV público, sin API key, sin límite de rate documentado para
# este volumen) para los macro tickers SIN serie FRED -- gratis, igual que el
# resto del pipeline (Directiva "100% gratuito"). NO VERIFICADO EN VIVO: este
# entorno de ejecución no tiene salida de red a stooq.com (dominio fuera del
# allowlist del sandbox), así que los símbolos de abajo son de memoria/alta
# confianza pero NO confirmados contra la respuesta real del endpoint --
# `_fetch_macro_stooq_fallback` está escrito para fallar en silencio (log +
# devuelve None) ante CUALQUIER respuesta inesperada, así que un símbolo mal
# mapeado aquí degrada exactamente al comportamiento ACTUAL (yfinance) sin
# romper nada -- CERO riesgo de regresión, solo upside si el símbolo es
# correcto. Verificar manualmente antes de asumir que esta capa está activa.
_STOOQ_MACRO_SYMBOL_MAP: dict[str, str] = {
    "GC=F": "gc.f",        # oro, futuro continuo COMEX -- proxy, no instrumento idéntico
    "DX-Y.NYB": "dx.f",    # ICE Dollar Index futuro continuo -- proxy del índice NYBOT real
}
STOOQ_FALLBACK_ENABLED = os.getenv("KODAQUANT_ENABLE_STOOQ_FALLBACK", "1").strip().lower() not in ("0", "false", "no")


def _fetch_macro_stooq_fallback(ticker: str) -> Optional[pd.Series]:
    """Serie de cierre diario vía CSV público de Stooq (sin key). Devuelve
    `None` (nunca levanta) ante cualquier fallo -- red, símbolo inexistente,
    CSV vacío/malformado -- para que el caller siga cayendo a yfinance sin
    interrumpir `_fetch_all_symbols_flat`. Ver nota de honestidad junto a
    `_STOOQ_MACRO_SYMBOL_MAP` arriba: mapeo de símbolos no verificado en vivo
    en este entorno."""
    symbol = _STOOQ_MACRO_SYMBOL_MAP.get(ticker)
    if not symbol or not STOOQ_FALLBACK_ENABLED:
        return None
    try:
        resp = requests.get(
            "https://stooq.com/q/d/l/",
            params={"s": symbol, "i": "d"},
            timeout=10,
        )
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        if df.empty or "Close" not in df.columns or "Date" not in df.columns:
            logger.warning("[Stooq fallback] '%s' (%s) respuesta vacía/inesperada -- cae a yfinance.", ticker, symbol)
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        series = df.set_index("Date")["Close"].astype(float).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        if series.empty:
            return None
        return series
    except Exception as exc:  # noqa: BLE001 -- fallback best-effort, jamás debe tumbar el pipeline
        logger.warning("[Stooq fallback] '%s' (%s) falló (%r) -- cae a yfinance.", ticker, symbol, exc)
        return None


def purge_data_cache(data_dir: Path = DATA_DIR) -> None:
    """
    Purgado Total (V9): vacía `data_dir` de cualquier caché (.json/.parquet)
    vía `CacheManager.purge_all` -- se invoca EXACTAMENTE UNA VEZ por
    corrida de entrenamiento, en `__main__`, ANTES del bucle de regímenes
    -- nunca dentro de `_fetch_all_symbols_flat` (llamada una vez POR
    régimen): purgar ahí borraría, en la segunda llamada, el artefacto que
    el régimen anterior de la MISMA corrida acaba de escribir para consumo
    posterior (Requerimiento 3: el caché debe sobrevivir intacto al final
    de la corrida).
    """
    cache = _ohlcv_cache if data_dir == DATA_DIR else CacheManager(data_dir, default_max_age_hours=_CACHE_MAX_AGE_HOURS)
    removed = cache.purge_all()
    if removed:
        logger.warning("[purga CI/CD] FORCE_FRESH_DATA=True -> %d archivo(s) purgado(s) de %s (Data Drift prevention).",
                        removed, data_dir)
    else:
        logger.info("[purga CI/CD] FORCE_FRESH_DATA=True -> %s ya estaba vacío.", data_dir)


def _download_with_backoff(symbols: list[str], period: str,
                            max_retries: int = MAX_DOWNLOAD_RETRIES,
                            backoff_base: float = BACKOFF_BASE_SECONDS,
                            min_coverage: float = 0.5) -> pd.DataFrame:
    """Reintenta la descarga con backoff exponencial ante 429/timeouts transitorios,
    Y ANTE FALLOS PARCIALES -- yfinance a veces devuelve un batch "no vacío" en
    general pero con UN símbolo puntual ~100% NaN (rate-limit silencioso por-símbolo,
    típico al encadenar descargas rápidas -- causa real observada: '^TNX_Close' con
    0.7% de cobertura en una corrida de crypto_specialist lanzada 12s después de
    equity_specialist). Antes, esa caída parcial pasaba el chequeo de "no vacío",
    quedaba en caché, y solo se detectaba 700+ líneas más abajo cuando
    `engineer_asset` colapsaba el dataset a 0 filas. Ahora se valida cobertura
    por-símbolo ACÁ, en el mismo bucle de reintento -- Equities/cripto: SIEMPRE
    yfinance (Directiva 3)."""
    import time

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(symbols, period=period, auto_adjust=True, progress=False)
            if data is None or data.empty:
                raise ValueError("yfinance devolvió un DataFrame vacío")

            close = (data["Close"] if isinstance(data.columns, pd.MultiIndex)
                     else data[["Close"]].rename(columns={"Close": symbols[0]}))
            missing = [s for s in symbols if s not in close.columns]
            starved = [s for s in symbols if s in close.columns and close[s].notna().mean() < min_coverage]
            if missing or starved:
                raise ValueError(
                    f"cobertura insuficiente tras la descarga -- ausentes={missing} "
                    f"con <{min_coverage:.0%} de datos reales={starved}"
                )
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


def _fetch_all_symbols_flat(tickers: list[str], macro_tickers: list[str], period: str,
                             cache_tag: str, force_refresh: bool | None = None,
                             macro_period: str | None = None) -> pd.DataFrame:
    """
    Reemplazo canónico de `download_all` (alias más abajo, compatibilidad
    hacia atrás con `train_pipeline.py`). Devuelve OHLCV completo de
    `tickers` (`{ticker}_{Open,High,Low,Close,Volume}`) + Close de
    `macro_tickers` (`{macro}_Close`), SIN ffill/dropna -- esa
    responsabilidad sigue siendo exclusiva de `engineer_asset`.

    Directiva 1 (caché): unificada vía `CacheManager` (`_ohlcv_cache`,
    antes parquet ad-hoc acá / JSON ad-hoc en inferencia), mismo
    `_CACHE_MAX_AGE_HOURS`/`FORCE_FRESH_DATA` de siempre.

    Directiva 2 (fuga de datos / alineación 24/7 vs 5/7): cada serie macro
    se proyecta contra el calendario REAL del universo de `tickers` de
    este régimen vía `kodaquant_core.project_macro_frame` (upsample a
    calendario diario corrido + ffill ANTES del merge) -- misma lógica,
    ya no un `.ffill()` ciego sobre el frame combinado dentro de
    `engineer_asset` cruzando dedos con el orden de columnas.

    Directiva 3 (alineación train/inferencia): ^TNX se resuelve vía
    `FredClient`/serie oficial DGS10 -- LA MISMA fuente primaria que
    `market_data.py` ya usa en inferencia para ese mismo factor -- con
    fallback automático a yfinance si `FRED_API_KEY` no está configurada
    o la request falla puntualmente. El resto de `macro_tickers` y TODO
    el universo de `tickers` (equities + cripto) permanece en yfinance,
    concurrente, sin cambios.
    """
    effective_force_refresh = FORCE_FRESH_DATA if force_refresh is None else force_refresh
    cache_key = f"ohlcv_{cache_tag}_{period}"
    fields = ["Open", "High", "Low", "Close", "Volume"]
    required_cols = {f"{t}_{f}" for t in tickers for f in fields} | {f"{m}_Close" for m in macro_tickers}

    if effective_force_refresh:
        logger.info("[CacheManager] FORCE_FRESH_DATA activo -> se ignora cualquier %s cacheado, descarga 100%% fresca.", cache_key)
    else:
        cached = _ohlcv_cache.get_dataframe(cache_key, required_columns=required_cols)
        if cached is not None:
            logger.info("[CacheManager] leyendo %s (fresco).", cache_key)
            return cached

    # --- Precio equities/cripto: yfinance, concurrente (Directiva 3) ------
    raw = _download_with_backoff(tickers, period)
    flat = pd.DataFrame(index=raw.index)
    available_fields = set(raw.columns.get_level_values(0)) if isinstance(raw.columns, pd.MultiIndex) else set(raw.columns)
    for field in fields:
        if field not in available_fields:
            continue
        sub = raw[field] if isinstance(raw.columns, pd.MultiIndex) else raw[[field]].rename(columns={field: tickers[0]})
        for sym in tickers:
            if sym in sub.columns:
                flat[f"{sym}_{field}"] = sub[sym]
    flat.index = normalize_daily_index(pd.DatetimeIndex(flat.index))
    flat = flat[~flat.index.duplicated(keep="last")].sort_index()

    # --- Factores macro: FRED (^TNX) -> yfinance (resto + fallback) -------
    macro_closes: dict[str, pd.Series] = {}
    macro_via_yf: list[str] = []
    for m in macro_tickers:
        fred_series_id = _FRED_MACRO_SERIES_MAP.get(m)
        # V20: tail_days=520 (~1.4 años) era el bug real detrás del
        # "19.9%/0.7% de cobertura" que veníamos viendo en '^TNX_Close' en
        # TODAS las corridas -- incluso con FRED_API_KEY correctamente
        # configurada. No era Yahoo Finance fallando: era esta llamada
        # pidiéndole a FRED muchísimos menos días de los que cubre
        # PERIOD="10y" para el resto del pipeline (equities/otros macro
        # vía yfinance). 520/~2600 días de trading ≈ 20% -- coincide exacto
        # con el número de cobertura que se repitió en cada corrida.
        # 3700 días cubre 10y + margen (fines de semana/feriados incluidos
        # en el conteo corrido de FRED).
        series = _fred_client.daily_series(fred_series_id, tail_days=3700) if fred_series_id else None
        if series is not None and not series.empty:
            macro_closes[m] = series
            continue
        stooq_series = _fetch_macro_stooq_fallback(m)
        if stooq_series is not None and not stooq_series.empty:
            logger.info("[Stooq fallback] '%s' resuelto vía Stooq (sin FRED equivalente).", m)
            macro_closes[m] = stooq_series
        else:
            macro_via_yf.append(m)

    if macro_via_yf:
        # V27 -- FIX: `period="max"` (BTC-USD, ver REGIMES.crypto_specialist)
        # propagado sin querer al fallback yfinance de macro rompió GC=F
        # ("cobertura insuficiente" -- ese ticker no tiene una serie "max"
        # sana en Yahoo). Los factores macro SIEMPRE usan `macro_period`
        # (default: `period`, o sea CERO cambio para cualquier caller que
        # no lo pase explícito -- equity_specialist y todo lo de antes de
        # hoy quedan bit a bit igual). Solo los `tickers` (precio) usan el
        # `period` que le pasó el caller.
        macro_raw = _download_with_backoff(macro_via_yf, macro_period or period)
        if isinstance(macro_raw.columns, pd.MultiIndex):
            macro_available = set(macro_raw.columns.get_level_values(0))
            if "Close" in macro_available:
                macro_close_block = macro_raw["Close"]
                for m in macro_via_yf:
                    if m in macro_close_block.columns:
                        macro_closes[m] = macro_close_block[m].dropna()
        elif "Close" in macro_raw.columns and len(macro_via_yf) == 1:
            macro_closes[macro_via_yf[0]] = macro_raw["Close"].dropna()

    macro_frame = project_macro_frame(macro_closes, flat.index) if macro_closes else pd.DataFrame(index=flat.index)
    for m in macro_tickers:
        flat[f"{m}_Close"] = macro_frame[m] if m in macro_frame.columns else np.nan

    # --- Diagnóstico de cobertura (NO altera el pipeline, solo visibilidad) --
    # V16: un macro ticker con cobertura ~0% (símbolo roto/deslistado en
    # Yahoo Finance -- "DX-Y.NYB" es un caso conocido de esto) hace que
    # `engineer_asset` -- que exige TODAS las columnas macro no-nulas antes
    # de aceptar una fila, sin cambios de política -- termine con 0 filas
    # para el activo: un `ValueError` de sklearn ~700 líneas más abajo, sin
    # ninguna pista de la causa real. Este bloque deja explícito en el log,
    # ACÁ mismo, qué columna es la culpable, antes de que el pipeline siga.
    for col in sorted(required_cols):
        if col not in flat.columns:
            logger.warning("[Diagnóstico cobertura] '%s' NO llegó a descargarse (columna ausente).", col)
            continue
        coverage_pct = 100.0 * flat[col].notna().mean() if len(flat) else 0.0
        if coverage_pct < 50.0:
            logger.warning(
                "[Diagnóstico cobertura] '%s': solo %.1f%% de filas con dato real "
                "en todo el rango descargado -- candidato directo a colapsar "
                "`engineer_asset` (exige TODAS las columnas no-nulas por fila). "
                "Si es un macro ticker, lo más probable es que Yahoo Finance haya "
                "dejado de servirlo con este período/granularidad -- considerar "
                "migrarlo a Twelve Data, igual que ya se hizo en "
                "services/market_data.py para ^GSPC/^VIX/GC=F/DX-Y.NYB.",
                col, coverage_pct,
            )

    _ohlcv_cache.set_dataframe(cache_key, flat)
    logger.info("[CacheManager] %s guardado (sincronizado, fecha más reciente %s).",
                cache_key, flat.index.max() if not flat.empty else "N/A")
    return flat


# Alias de compatibilidad hacia atrás: `download_all` era el nombre previo
# a la consolidación del core; `train_pipeline.py` (MLOps incremental) y
# cualquier caller externo que aún lo referencie siguen funcionando sin
# ningún cambio de firma.
download_all = _fetch_all_symbols_flat


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


def _fetch_fear_greed_index(tail_days: int) -> Optional[pd.Series]:
    """Serie diaria del Crypto Fear & Greed Index (alternative.me, endpoint
    público, SIN key) -- 0=miedo extremo, 100=codicia extrema, 50=neutro.
    A diferencia de FUNDING_RATE (revertido, derivado de precio/leverage --
    correlacionado con lo que ya veían RSI/ATR/BB_WIDTH), esto es un
    agregado de PSICOLOGÍA de mercado (volatilidad, momentum, redes
    sociales, dominancia BTC) -- categoría de señal distinta, no probada
    antes. Historia disponible desde feb-2018 -- fechas anteriores quedan
    en 50.0 (neutro) por default en `engineer_asset`, nunca NaN."""
    try:
        resp = requests.get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"}, timeout=10.0)
        resp.raise_for_status()
        records = resp.json().get("data", [])
    except Exception as exc:  # noqa: BLE001 -- proveedor opcional, jamás tumba el pipeline
        logger.debug("Fear & Greed Index falló: %r", exc)
        return None
    if not records:
        return None
    fg = pd.DataFrame(records)
    fg["date"] = pd.to_datetime(pd.to_numeric(fg["timestamp"]), unit="s").dt.normalize()
    fg["value"] = pd.to_numeric(fg["value"], errors="coerce")
    return fg.groupby("date")["value"].mean()


# ---------------------------------------------------------------------------
# V24 — ALPHA CRIPTO: DERIVADOS + ON-CHAIN + TRANSFORMACIÓN MATEMÁTICA
# ---------------------------------------------------------------------------
# FUNDING_RATE crudo ya fue probado y REVERTIDO en V17 (correlacionaba con lo
# que RSI_14/ATR_PCT/BB_WIDTH_20 ya capturan -- todos derivados de precio/
# leverage, misma categoría de información). Las tres adiciones de abajo
# apuntan a categorías DISTINTAS, no probadas antes en este pipeline:
#
#   1) FUNDING_RATE_ZSCORE -- no es el NIVEL del funding (eso fracasó), es su
#      EXTREMIDAD relativa a su propia distribución móvil: proxy de estrés de
#      posicionamiento/crowding en derivados (funding en el percentil 99 ha
#      precedido reversiones en ciclos previos) -- señal de POSICIONAMIENTO,
#      no de precio.
#   2) ONCHAIN_ACTIVITY_ROC -- actividad de RED, no de PRECIO: tasa de cambio
#      de transacciones on-chain confirmadas. BTC ÚNICAMENTE (ver docstring
#      de `_fetch_onchain_activity` -- no existe equivalente gratuito de
#      historial largo verificado para ETH; se deja en 0.0 neutro, NO se
#      inventa el dato).
#   3) FRAC_DIFF_LOG_CLOSE -- la transformación matemática pedida explícita-
#      mente (López de Prado, AFML cap. 5). Mantiene memoria de largo plazo
#      del precio -- a diferencia de LOG_RETURN_1D (diferenciación entera
#      d=1, borra toda memoria) -- mientras logra estacionariedad. El orden
#      `d` se calibra por ADF SOLO sobre el prefijo de train (ver
#      `find_min_ffd_d`) -- nunca sobre la serie completa, para no filtrar
#      información de test hacia la selección del hiperparámetro, el mismo
#      estándar que ya aplica `feature_scaler.fit(features[:split_idx])` en
#      `build_asset_dataset`.
#
# WAVELETS -- evaluado y DESCARTADO deliberadamente en este pase: una DWT
# estándar (PyWavelets `dwt`/`wavedec`) sobre una ventana fija usa muestras a
# AMBOS lados de cada coeficiente según el modo de borde (`symmetric`/
# `periodic`) -- aplicarla sobre la serie completa antes de trocear en
# ventanas de `LOOKBACK` inyecta fuga de información futura DENTRO de cada
# ventana de entrenamiento, exactamente la clase de bug que los docstrings de
# este archivo (`build_asset_dataset`, `align_macro_to_calendar`) llevan
# 20+ versiones evitando. Una variante causal real (MODWT streaming,
# recalculada por timestep) es viable pero cambia el costo computacional de
# O(n) a O(n·LOOKBACK) SOLO para esa columna -- no se implementa acá sin
# benchmarking explícito primero. FRAC_DIFF_LOG_CLOSE (causal por
# construcción) cubre el mismo objetivo (memoria + estacionariedad) sin ese
# riesgo.
try:
    from statsmodels.tsa.stattools import adfuller
except (ImportError, TypeError):  # TypeError = pandas 3.x deprecate_kwarg compat issue (see venv patch RONDA 8)
    adfuller = None
    logger.warning(
        "statsmodels no instalado o incompatible (`pip install statsmodels` + ver RONDA 8 en requirements.txt) "
        "-- FRAC_DIFF_LOG_CLOSE usará FRAC_DIFF_DEFAULT_D=0.40 fijo en vez de calibrar `d` vía ADF por activo."
    )


BINANCE_FAPI_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_FUNDING_SYMBOL_MAP = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}
FUNDING_ZSCORE_WINDOW_DAYS = 90     # ~1 trimestre de funding 8h-agregado-a-diario
FUNDING_ZSCORE_MIN_PERIODS = 20
FUNDING_ZSCORE_CLIP = 5.0
# AUDITORÍA RESCATE 2026-08 -- RESTRUCTURACIÓN ASIMÉTRICA (Directiva Head
# Quant, "funding rates asimétricos"): el z-score simétrico de V24 trata
# funding extremo POSITIVO (longs pagando -- crowding alcista, apalancado
# contra el sesgo estructural de oferta creciente del activo) igual que
# funding extremo NEGATIVO (shorts pagando -- crowding bajista, a contra-
# tendencia de ese mismo sesgo estructural). Son regímenes de crowding
# cualitativamente distintos, no reflejos especulares uno del otro -- se
# separan las dos colas y se les aplica una ganancia potencia (`gamma`)
# INDEPENDIENTE antes de re-unirlas, en vez de una única curva simétrica.
# `_POS`>1.0 amplifica la cola de crowding-largo; `_NEG`<1.0 amortigua la de
# crowding-corto -- ambas calibrables por separado si la evidencia empírica
# de una corrida real sugiere lo contrario, sin tocar la ventana/mínimo de
# períodos (que siguen definiendo la ESCALA, no la FORMA, de la señal).
FUNDING_ZSCORE_ASYM_POS_GAMMA = 1.35
FUNDING_ZSCORE_ASYM_NEG_GAMMA = 0.85

# V26 (Directiva Head Quant) -- FUNDING_RATE_MOMENTUM_ZSCORE: ver
# `compute_funding_rate_momentum_zscore`. Ventana de diff corta (1 semana)
# a propósito -- es una señal de VELOCIDAD, una ventana larga la
# convertiría de vuelta en una medida de nivel, redundante con
# FUNDING_RATE_ZSCORE. La ventana/mínimo de períodos del z-score que
# normaliza esa velocidad sí se mantiene igual de larga (90d) que el resto
# de *_ZSCORE del módulo -- eso define la ESCALA de "velocidad extrema",
# no la velocidad en sí.
FUNDING_MOMENTUM_LOOKBACK_DAYS = 7
FUNDING_MOMENTUM_ZSCORE_WINDOW_DAYS = 90
FUNDING_MOMENTUM_ZSCORE_MIN_PERIODS = 20
FUNDING_MOMENTUM_ZSCORE_CLIP = 5.0

BLOCKCHAIN_INFO_CHART_URL = "https://api.blockchain.info/charts/{chart}"
ONCHAIN_CHART_NAME = "n-transactions"   # transacciones confirmadas/día -- proxy de demanda real de red, BTC únicamente
ONCHAIN_ROC_LOOKBACK_DAYS = 7
ONCHAIN_ROC_CLIP = 3.0

FRAC_DIFF_D_GRID = tuple(round(float(d), 2) for d in np.arange(0.10, 1.01, 0.10))
FRAC_DIFF_ADF_PVALUE_MAX = 0.05
FRAC_DIFF_THRESH = 1e-4
FRAC_DIFF_DEFAULT_D = 0.40   # fallback sin statsmodels -- valor de referencia AFML para series financieras

# AUDITORÍA RESCATE 2026-08 -- FIX DE DESALINEACIÓN TRAIN/SERVE REAL: `d` se
# calibra acá sobre el prefijo de train de PERIOD="10y" (~8 años), pero
# `quanti_engine.py::_find_min_ffd_d` RECALIBRABA de forma INDEPENDIENTE en
# cada request, sobre una ventana de solo ~2 años (`DEFAULT_HISTORY_SESSIONS`,
# ver services/market_data.py) -- el ADF-óptimo sobre una ventana ~4x más
# corta no tiene por qué coincidir con el `d` real de entrenamiento, así que
# FRAC_DIFF_LOG_CLOSE llegaba a inferencia con una transformación matemática
# DISTINTA a la que vio el modelo en entrenamiento para ese mismo canal --
# sin lanzar ningún error (ambos "funcionan", solo producen números
# distintos). Este diccionario captura el `d` REAL usado en entrenamiento,
# por ticker, para persistirlo en scalers_dict.pkl (ver run_regime_pipeline)
# -- quanti_engine.py lo lee y lo usa DIRECTO en vez de recalibrar, con
# fallback automático a su comportamiento previo si el bundle es viejo (no
# tiene la clave). Afecta EXCLUSIVAMENTE a FRAC_DIFF_LOG_CLOSE en tickers
# cripto -- esa columna ya es 0.0 neutro por diseño para equity_specialist
# (ver bloque `if ticker in REGIMES["crypto_specialist"]["tickers"]` en
# `engineer_asset`), así que este fix no puede, ni en principio, alterar un
# solo número de la inferencia de equity_specialist.
_LAST_FFD_D_BY_TICKER: dict[str, float] = {}

# ---------------------------------------------------------------------------
# V23.5 (Directiva 3, Head Quant) — PUELL_MULTIPLE_ZSCORE, BTC ÚNICAMENTE
# ---------------------------------------------------------------------------
# Puell Multiple (David Puell, 2019) = emisión diaria de BTC valuada en USD
# / su propio promedio móvil de 365 días -- señal de OFERTA estructural
# (cuánto BTC nuevo deben vender los mineros para cubrir costos operativos),
# categoría DISTINTA a las 3 de V24 arriba (posicionamiento en derivados /
# actividad de red / memoria de precio). Documentado en vivo por
# lookintobitcoin.com y bitcoinmagazinepro.com como indicador de ciclo.
#
# CERO llamada de red -- a diferencia de ONCHAIN_ACTIVITY_ROC (depende de
# blockchain.info, con fallback silencioso a 0.0 si el endpoint cae), la
# emisión diaria de BTC es un CALENDARIO DETERMINÍSTICO conocido de
# antemano (reward/bloque se reduce a la mitad cada halving) -- se deriva
# 100% de `RAW_CLOSE` ya descargado, cero dependencia externa nueva, cero
# riesgo de degradar a neutro por un API caído.
#
# BTC ÚNICAMENTE: el denominador (emisión Proof-of-Work) no tiene análogo
# en ETH-USD, que es Proof-of-Stake sin emisión por minería desde The
# Merge (2022) -- no se inventa un "Puell" sintético para ETH; queda en
# 0.0 neutro para ETH-USD/equity, MISMO patrón que ONCHAIN_ACTIVITY_ROC.
#
# `BTC_BLOCKS_PER_DAY=144` es la aproximación estándar de la literatura
# (10 min/bloque objetivo, dificultad autoajustada) -- no se descarga el
# conteo real de bloques/día por ticker (evita otra dependencia de red
# para una corrección de segundo orden). El halving de 2028 es una
# ESTIMACIÓN (fecha exacta depende del ritmo real de minado); de
# correrse unas semanas, el único efecto es una discontinuidad leve en
# `daily_issuance_btc` cerca de esa fecha, amortiguada por el propio
# z-score móvil (ver `compute_puell_multiple_zscore`).
BTC_HALVING_SCHEDULE: tuple[tuple[datetime, float], ...] = (
    (datetime(2009, 1, 3, tzinfo=timezone.utc), 50.0),
    (datetime(2012, 11, 28, tzinfo=timezone.utc), 25.0),
    (datetime(2016, 7, 9, tzinfo=timezone.utc), 12.5),
    (datetime(2020, 5, 11, tzinfo=timezone.utc), 6.25),
    (datetime(2024, 4, 20, tzinfo=timezone.utc), 3.125),
    (datetime(2028, 4, 1, tzinfo=timezone.utc), 1.5625),  # estimado -- ver nota arriba
)
BTC_BLOCKS_PER_DAY = 144.0
PUELL_MA_WINDOW_DAYS = 365
PUELL_MA_MIN_PERIODS = 30
PUELL_ZSCORE_WINDOW_DAYS = 365
PUELL_ZSCORE_MIN_PERIODS = 60
PUELL_ZSCORE_CLIP = 5.0


def _btc_block_reward_schedule(dates: pd.DatetimeIndex) -> np.ndarray:
    """Reward BTC/bloque vigente en cada fecha de `dates`, según
    `BTC_HALVING_SCHEDULE` -- calendario determinístico, cero fetch."""
    idx = pd.DatetimeIndex(dates)
    idx_utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    schedule_dates = pd.DatetimeIndex([d for d, _ in BTC_HALVING_SCHEDULE])
    schedule_rewards = np.array([r for _, r in BTC_HALVING_SCHEDULE])
    pos = schedule_dates.searchsorted(idx_utc, side="right") - 1
    pos = np.clip(pos, 0, len(schedule_rewards) - 1)
    return schedule_rewards[pos]


def _btc_halving_epoch_cum_supply() -> np.ndarray:
    """Suministro acumulado de BTC al INICIO de cada época de `BTC_HALVING_SCHEDULE`
    -- precomputado una sola vez, cero llamada de red (mismo insumo determinístico
    que ya usa `compute_puell_multiple_zscore`)."""
    dates = [d for d, _ in BTC_HALVING_SCHEDULE]
    rewards = [r for _, r in BTC_HALVING_SCHEDULE]
    cum = [0.0]
    for i in range(1, len(dates)):
        days = (dates[i] - dates[i - 1]).total_seconds() / 86400.0
        cum.append(cum[-1] + days * BTC_BLOCKS_PER_DAY * rewards[i - 1])
    return np.array(cum)


_BTC_EPOCH_CUM_SUPPLY_AT_START = _btc_halving_epoch_cum_supply()


def _btc_cumulative_supply(dates: pd.DatetimeIndex) -> np.ndarray:
    """Suministro circulante APROXIMADO de BTC en cada fecha de `dates` --
    integra `BTC_HALVING_SCHEDULE`/`BTC_BLOCKS_PER_DAY` desde el genesis
    (2009-01-03), cero llamada de red. Denominador de mercado (`market_cap`)
    para `compute_nvt_ratio_zscore` -- MISMA filosofía "cero fetch" que ya
    usa `_btc_block_reward_schedule`/Puell Multiple arriba: la aproximación
    (reward/bloque constante dentro de cada época, 144 bloques/día) es
    idéntica a la que ya se acepta ahí, ningún nuevo supuesto."""
    idx = pd.DatetimeIndex(dates)
    idx_utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    schedule_dates = pd.DatetimeIndex([d for d, _ in BTC_HALVING_SCHEDULE])
    schedule_rewards = np.array([r for _, r in BTC_HALVING_SCHEDULE])
    pos = np.clip(schedule_dates.searchsorted(idx_utc, side="right") - 1, 0, len(schedule_rewards) - 1)
    epoch_start = schedule_dates[pos]
    days_since_epoch_start = (idx_utc - epoch_start).total_seconds() / 86400.0
    return _BTC_EPOCH_CUM_SUPPLY_AT_START[pos] + days_since_epoch_start * BTC_BLOCKS_PER_DAY * schedule_rewards[pos]


# ---------------------------------------------------------------------------
# AUDITORÍA RESCATE 2026-08 — RESTRUCTURACIÓN NVT RATIO (Directiva Head Quant,
# reemplaza la definición de ONCHAIN_ACTIVITY_ROC, MISMO nombre/slot de
# TECH_COLS -- cero cambio de N_FEATURES/shape, cero impacto en
# equity_specialist, que ya recibe 0.0 neutro en esta columna sin importar
# su fórmula interna)
# ---------------------------------------------------------------------------
# V24 media esta columna como tasa de cambio CRUDA de conteo de
# transacciones -- actividad de red sin contexto de VALUACIÓN. NVT Ratio
# (Willy Woo/Chris Burniske) = valor de red (market cap) / valor
# transaccionado on-chain diario -- categoría de señal distinta a las 5 ya
# presentes (posicionamiento en derivados / ciclo de oferta del minero /
# memoria de precio / vol realizada / vol-de-vol): es un proxy de
# SOBRE-valuación relativa a USO REAL de la red, análogo a un P/E ratio.
# `market_cap` usa `_btc_cumulative_supply` (determinístico, cero fetch,
# arriba); `tx_volume_usd` viene de blockchain.info (mismo endpoint que
# `_fetch_onchain_activity`, chart distinto -- `_fetch_onchain_tx_volume_usd`
# abajo). Z-score de log(NVT) -- no el nivel crudo (no acotado, y su escala
# de mercado crece con la capitalización total) -- mismo criterio de
# "extremidad relativa" que el resto del bloque V23.5/V24.
ONCHAIN_TX_VALUE_CHART_NAME = "estimated-transaction-volume-usd"
NVT_ZSCORE_WINDOW_DAYS = 365
NVT_ZSCORE_MIN_PERIODS = 60
NVT_ZSCORE_CLIP = 5.0


def _fetch_onchain_tx_volume_usd() -> Optional[pd.Series]:
    """Volumen diario de transacciones on-chain BTC valuado en USD --
    numerador de utilidad real de NVT Ratio. Mismo endpoint/patrón que
    `_fetch_onchain_activity` (blockchain.info /charts, público, sin key),
    chart distinto."""
    return _fetch_onchain_activity(chart=ONCHAIN_TX_VALUE_CHART_NAME)


def compute_nvt_ratio_zscore(close: pd.Series, tx_volume_usd: pd.Series,
                               window: int = NVT_ZSCORE_WINDOW_DAYS,
                               min_periods: int = NVT_ZSCORE_MIN_PERIODS) -> pd.Series:
    """Z-score móvil (rolling, SOLO pasado -- cero fuga) de log(NVT Ratio).
    BTC ÚNICAMENTE (ver `engineer_asset`) -- mismo motivo que Puell Multiple:
    el "valor de red" (`market_cap`) vía suministro determinístico solo
    tiene forma cerrada para una cadena Proof-of-Work con calendario de
    emisión conocido."""
    market_cap = close * _btc_cumulative_supply(close.index)
    nvt = market_cap / tx_volume_usd.replace(0.0, np.nan)
    log_nvt = np.log(nvt.clip(lower=1e-6))
    mean = log_nvt.rolling(window, min_periods=min_periods).mean()
    std = log_nvt.rolling(window, min_periods=min_periods).std()
    z = (log_nvt - mean) / std.replace(0.0, np.nan)
    return z.clip(-NVT_ZSCORE_CLIP, NVT_ZSCORE_CLIP).fillna(0.0)


def compute_puell_multiple_zscore(close: pd.Series, window: int = PUELL_ZSCORE_WINDOW_DAYS,
                                   min_periods: int = PUELL_ZSCORE_MIN_PERIODS) -> pd.Series:
    """
    Z-score móvil (rolling, SOLO pasado -- cero fuga) de log(Puell
    Multiple) -- mismo criterio de "extremidad relativa" que
    `compute_funding_rate_zscore`, no el nivel crudo (no acotado por
    construcción, y su escala decae en cada halving -- ver docstring del
    bloque de constantes arriba).
    """
    reward_per_block = _btc_block_reward_schedule(close.index)
    daily_issuance_usd = reward_per_block * BTC_BLOCKS_PER_DAY * close.to_numpy()
    issuance = pd.Series(daily_issuance_usd, index=close.index)
    issuance_ma = issuance.rolling(PUELL_MA_WINDOW_DAYS, min_periods=PUELL_MA_MIN_PERIODS).mean()
    puell = issuance / issuance_ma.replace(0.0, np.nan)
    log_puell = np.log(puell.clip(lower=1e-6))
    mean = log_puell.rolling(window, min_periods=min_periods).mean()
    std = log_puell.rolling(window, min_periods=min_periods).std()
    z = (log_puell - mean) / std.replace(0.0, np.nan)
    return z.clip(-PUELL_ZSCORE_CLIP, PUELL_ZSCORE_CLIP).fillna(0.0)


def _fetch_binance_funding_history(ticker: str) -> Optional[pd.Series]:
    """
    Historial diario (promedio de los ~3 pagos de 8h) de funding rate de
    perpetuos Binance -- endpoint REST público, SIN API key. Pagina hacia
    adelante en bloques de 1000 registros (límite del endpoint) desde el
    lanzamiento del contrato hasta hoy.
    """
    symbol = BINANCE_FUNDING_SYMBOL_MAP.get(ticker)
    if symbol is None:
        return None
    all_records: list[dict] = []
    cursor_ms = int(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    for _ in range(50):  # techo defensivo -- ~50k registros, muy por encima de cualquier historial real
        if cursor_ms >= now_ms:
            break
        try:
            resp = requests.get(
                BINANCE_FAPI_FUNDING_URL,
                params={"symbol": symbol, "startTime": cursor_ms, "limit": 1000},
                timeout=15,
            )
            resp.raise_for_status()
            batch = resp.json()
        except Exception as exc:  # noqa: BLE001 -- red/HTTP transitorio, proveedor opcional
            logger.warning("Binance fundingRate falló para %s: %r", symbol, exc)
            break
        if not batch:
            break
        all_records.extend(batch)
        last_time = int(batch[-1]["fundingTime"])
        if last_time <= cursor_ms:
            break
        cursor_ms = last_time + 1
        if len(batch) < 1000:
            break
        time.sleep(0.15)  # cortesía de rate-limit -- público, sin key, sin límite documentado estricto
    if not all_records:
        return None
    raw = pd.DataFrame(all_records)
    raw["date"] = pd.to_datetime(pd.to_numeric(raw["fundingTime"]), unit="ms").dt.normalize()
    raw["fundingRate"] = pd.to_numeric(raw["fundingRate"], errors="coerce")
    return raw.groupby("date")["fundingRate"].mean().sort_index()


def compute_funding_rate_zscore(daily_funding: pd.Series, window: int = FUNDING_ZSCORE_WINDOW_DAYS,
                                  min_periods: int = FUNDING_ZSCORE_MIN_PERIODS) -> pd.Series:
    """Z-score móvil ASIMÉTRICO (`rolling` estándar para media/std, SOLO
    pasado -- cero fuga) -- extremidad del funding relativa a su propio
    régimen reciente, no su nivel absoluto. V25 (auditoría rescate 2026-08):
    el z-score crudo se separa en cola positiva/negativa y cada una recibe
    su propia ganancia potencia (`FUNDING_ZSCORE_ASYM_*_GAMMA`) antes de
    reunirse y recortarse -- ver docstring del bloque de constantes arriba.
    En cualquier punto solo una de las dos colas es no-nula (`pos`/`neg` son
    mutuamente excluyentes por construcción de `.clip`), así que la unión
    preserva el signo original sin introducir ninguna discontinuidad en 0."""
    mean = daily_funding.rolling(window, min_periods=min_periods).mean()
    std = daily_funding.rolling(window, min_periods=min_periods).std()
    z = (daily_funding - mean) / std.replace(0.0, np.nan)
    pos = z.clip(lower=0.0)
    neg_abs = (-z.clip(upper=0.0))
    z_asym = pos.pow(FUNDING_ZSCORE_ASYM_POS_GAMMA) - neg_abs.pow(FUNDING_ZSCORE_ASYM_NEG_GAMMA)
    return z_asym.clip(-FUNDING_ZSCORE_CLIP, FUNDING_ZSCORE_CLIP).fillna(0.0)


def compute_funding_rate_momentum_zscore(daily_funding: pd.Series, lookback: int = FUNDING_MOMENTUM_LOOKBACK_DAYS,
                                          window: int = FUNDING_MOMENTUM_ZSCORE_WINDOW_DAYS,
                                          min_periods: int = FUNDING_MOMENTUM_ZSCORE_MIN_PERIODS) -> pd.Series:
    """V26 (Directiva Head Quant) -- Z-score móvil (rolling, SOLO pasado) de
    la VARIACIÓN del funding rate sobre `lookback` días (diff, no nivel) --
    si el crowding en derivados se está ACELERANDO o DESHACIENDO. Señal
    complementaria y categóricamente distinta a `compute_funding_rate_zscore`
    (que mide el NIVEL relativo, no su velocidad): dos activos con el MISMO
    FUNDING_RATE_ZSCORE pueden estar en fases opuestas -- crowding recién
    formado (momentum aún subiendo) vs. crowding ya en reversión (momentum
    cayendo pese a nivel todavía alto) -- información que un z-score de
    nivel puro no puede distinguir. Reutiliza el MISMO `daily_funding` ya
    fetcheado/cacheado por `_fetch_binance_funding_history` en
    `engineer_asset` -- cero llamada de red adicional."""
    momentum = daily_funding.diff(lookback)
    mean = momentum.rolling(window, min_periods=min_periods).mean()
    std = momentum.rolling(window, min_periods=min_periods).std()
    z = (momentum - mean) / std.replace(0.0, np.nan)
    return z.clip(-FUNDING_MOMENTUM_ZSCORE_CLIP, FUNDING_MOMENTUM_ZSCORE_CLIP).fillna(0.0)


def _fetch_onchain_activity(chart: str = ONCHAIN_CHART_NAME) -> Optional[pd.Series]:
    """
    Serie diaria de actividad on-chain BTC vía blockchain.info `/charts` --
    endpoint público, SIN API key, historial completo (`timespan=all`) en una
    sola llamada. BTC ÚNICAMENTE: no existe un equivalente gratuito con
    historial largo verificado para ETH (Etherscan expone gas/supply EN VIVO
    con key gratuita, pero sus series históricas diarias de varios años viven
    detrás de su tier de pago) -- no se inventa ese dato para ETH-USD, ver
    `engineer_asset`: queda en 0.0 neutro, mismo patrón que FEAR_GREED_INDEX
    para tickers no-cripto.
    """
    try:
        resp = requests.get(
            BLOCKCHAIN_INFO_CHART_URL.format(chart=chart),
            params={"timespan": "all", "format": "json", "sampled": "false"},
            timeout=20,
        )
        resp.raise_for_status()
        values = resp.json().get("values", [])
    except Exception as exc:  # noqa: BLE001 -- proveedor opcional, jamás tumba el pipeline
        logger.debug("blockchain.info/charts/%s falló: %r", chart, exc)
        return None
    if not values:
        return None
    idx = pd.to_datetime([v["x"] for v in values], unit="s").normalize()
    series = pd.Series([v["y"] for v in values], index=idx, dtype=float)
    return series.groupby(level=0).mean().sort_index()


def _ffd_weights(d: float, thresh: float) -> np.ndarray:
    """Pesos de Fixed-Width Window Fractional Differencing (López de Prado,
    AFML cap. 5, ec. 5.1-5.2). `w[0]=1` (peso del valor actual); decae vía la
    recursión `w_k = -w_{k-1}*(d-k+1)/k` y se trunca en `thresh` (peso
    despreciable) -- ventana FINITA y CAUSAL por construcción: cada salida
    usa solo `values[t], values[t-1], ..., values[t-width+1]`."""
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.array(w, dtype=np.float64)


def frac_diff_ffd(log_price: pd.Series, d: float, thresh: float = FRAC_DIFF_THRESH) -> pd.Series:
    """Serie fraccionalmente diferenciada de `log_price` -- causal, sin fuga:
    la salida en `t` depende solo de `log_price[t], ..., log_price[t-width+1]`.
    Los primeros `width-1` puntos quedan NaN (warm-up, misma convención que
    RSI_14/ADX_14/etc. -- se recortan en el `dropna()` final de `engineer_asset`)."""
    w = _ffd_weights(d, thresh)
    width = len(w)
    values = log_price.to_numpy(dtype=np.float64)
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i in range(width - 1, len(values)):
        window = values[i - width + 1: i + 1][::-1]
        out[i] = float(np.dot(w, window))
    return pd.Series(out, index=log_price.index)


def find_min_ffd_d(log_price: pd.Series, train_ratio: float = TRAIN_RATIO) -> float:
    """
    Busca el `d` MÍNIMO (mayor memoria conservada) que logra estacionariedad
    (ADF p-value < FRAC_DIFF_ADF_PVALUE_MAX) -- grid search AFML estándar.
    CRÍTICO: el ADF corre SOLO sobre el PREFIJO de train (`train_ratio` de
    `log_price`, la misma fracción que separa train/test en
    `build_asset_dataset`) -- calibrar `d` contra la serie COMPLETA filtraría
    información de test hacia la selección del hiperparámetro, la misma fuga
    que `feature_scaler.fit(features[:split_idx])` ya evita para el resto
    del tensor.
    """
    if adfuller is None:
        return FRAC_DIFF_DEFAULT_D
    prefix_n = max(int(len(log_price) * train_ratio), 100)
    train_prefix = log_price.iloc[:prefix_n]
    for d in FRAC_DIFF_D_GRID:
        diffed = frac_diff_ffd(train_prefix, d).dropna()
        if len(diffed) < 100:
            continue
        try:
            pval = adfuller(diffed.to_numpy(), maxlag=1, regression="c", autolag=None)[1]
        except Exception as exc:  # noqa: BLE001 -- ADF numéricamente inestable en algún d puntual -- se prueba el siguiente
            logger.debug("ADF falló para d=%.2f: %r", d, exc)
            continue
        if pval < FRAC_DIFF_ADF_PVALUE_MAX:
            return float(d)
    return 1.0  # ningún d<1 logró estacionariedad -- d=1.0 (diferenciación entera clásica) como último recurso


# ---------------------------------------------------------------------------
# AUDITORÍA RESCATE 2026-08 — ALPHA NUEVO (Directiva 2: baseline >53%)
# ---------------------------------------------------------------------------
# PARKINSON_VOL_ZSCORE (equity + cripto, AMBOS regímenes -- sin neutralizar):
# proxy de volatilidad implícita 100% GRATUITO y derivado exclusivamente de
# OHLCV ya descargado (High/Low), sin requerir superficie de opciones ni
# ningún dato de pago. Parkinson (1980), "The Extreme Value Method for
# Estimating the Variance of the Rate of Return", J. Business 53(1) --
# estimador de rango intradía ~5x más eficiente que la volatilidad
# close-to-close bajo movimiento browniano geométrico, ampliamente citado
# como sustituto libre de IV a nivel de activo individual cuando no hay
# acceso a datos de opciones. No requiere Open (no descargado por este
# pipeline de features) a diferencia de Garman-Klass/Yang-Zhang -- por eso
# se elige Parkinson y no un estimador OHLC completo.
# VOL_OF_VOL_ZSCORE (cripto ÚNICAMENTE, 0.0 neutro en equity -- mismo patrón
# que FEAR_GREED_INDEX/FUNDING_RATE_ZSCORE/etc.): volatilidad-de-la-
# volatilidad -- inestabilidad del propio régimen de vol, categoría de señal
# DISTINTA al nivel de vol (que ya captura PARKINSON_VOL_ZSCORE/ATR_PCT/
# BB_WIDTH_20). Los clusters de volatilidad de BTC/ETH son estructuralmente
# más violentos que los de equities tradicionales (ver literatura de
# realized-volatility de criptoactivos, p.ej. Baur & Dimpfl 2018), lo que
# hace de este ratio una señal de régimen específicamente informativa en
# cripto -- se calcula 100% de OHLCV, cero llamada de red nueva.
PARKINSON_VOL_LOOKBACK_DAYS = 20
PARKINSON_VOL_ZSCORE_WINDOW_DAYS = 60
PARKINSON_VOL_ZSCORE_MIN_PERIODS = 20
PARKINSON_VOL_ZSCORE_CLIP = 5.0

VOL_OF_VOL_LOOKBACK_DAYS = 20
VOL_OF_VOL_ZSCORE_WINDOW_DAYS = 60
VOL_OF_VOL_ZSCORE_MIN_PERIODS = 20
VOL_OF_VOL_ZSCORE_CLIP = 5.0

# V26 (Directiva Head Quant) -- AMIHUD_ILLIQUIDITY_ZSCORE: proxy de liquidez
# de mercado (Amihud, 2002, "Illiquidity and stock returns") -- |retorno
# diario| / volumen-en-USD del mismo día, impacto de precio por dólar
# transaccionado. Categoría de señal DISTINTA a la volatilidad (PARKINSON_*/
# ATR_PCT/BB_WIDTH_20/VOL_OF_VOL_*): un activo puede moverse mucho con
# volumen profundo detrás (alta vol, baja iliquidez) o moverse lo mismo con
# poco volumen (alta vol, ALTA iliquidez) -- son regímenes de riesgo
# distintos para el tamaño de posición/slippage real, no la misma
# información redundante. 100% derivado de OHLCV YA descargado (cero fetch
# nuevo, cero proveedor externo adicional) -- mismo criterio de costo-cero
# que PARKINSON_VOL_ZSCORE/VOL_OF_VOL_ZSCORE arriba.
AMIHUD_ZSCORE_WINDOW_DAYS = 90
AMIHUD_ZSCORE_MIN_PERIODS = 20
AMIHUD_ZSCORE_CLIP = 5.0


def compute_parkinson_vol(high: pd.Series, low: pd.Series,
                           window: int = PARKINSON_VOL_LOOKBACK_DAYS) -> pd.Series:
    """
    Estimador de volatilidad realizada de Parkinson (rolling, SOLO pasado --
    cero fuga): sigma_t = sqrt( mean_{ventana}( ln(H/L)^2 ) / (4*ln2) ).
    Requiere únicamente High/Low del propio activo -- ya presentes en
    `df` (ver `engineer_asset`).
    """
    log_hl = np.log(high / low)
    variance = (log_hl ** 2).rolling(window=window, min_periods=window).mean() / (4.0 * np.log(2.0))
    return np.sqrt(variance.clip(lower=0.0))


def compute_parkinson_vol_zscore(parkinson_vol: pd.Series,
                                  window: int = PARKINSON_VOL_ZSCORE_WINDOW_DAYS,
                                  min_periods: int = PARKINSON_VOL_ZSCORE_MIN_PERIODS) -> pd.Series:
    """Z-score móvil (rolling, SOLO pasado) de la vol de Parkinson -- extremidad
    relativa al propio régimen reciente de volatilidad, mismo criterio que
    `compute_puell_multiple_zscore`/`compute_funding_rate_zscore`."""
    mean = parkinson_vol.rolling(window, min_periods=min_periods).mean()
    std = parkinson_vol.rolling(window, min_periods=min_periods).std()
    z = (parkinson_vol - mean) / std.replace(0.0, np.nan)
    return z.clip(-PARKINSON_VOL_ZSCORE_CLIP, PARKINSON_VOL_ZSCORE_CLIP).fillna(0.0)


def compute_vol_of_vol_zscore(parkinson_vol: pd.Series,
                               lookback: int = VOL_OF_VOL_LOOKBACK_DAYS,
                               window: int = VOL_OF_VOL_ZSCORE_WINDOW_DAYS,
                               min_periods: int = VOL_OF_VOL_ZSCORE_MIN_PERIODS) -> pd.Series:
    """Z-score móvil (rolling, SOLO pasado) de la volatilidad DE la
    volatilidad de Parkinson (std rolling de `parkinson_vol` sobre
    `lookback` días) -- inestabilidad del régimen de vol, no su nivel."""
    vol_of_vol = parkinson_vol.rolling(lookback, min_periods=lookback).std()
    mean = vol_of_vol.rolling(window, min_periods=min_periods).mean()
    std = vol_of_vol.rolling(window, min_periods=min_periods).std()
    z = (vol_of_vol - mean) / std.replace(0.0, np.nan)
    return z.clip(-VOL_OF_VOL_ZSCORE_CLIP, VOL_OF_VOL_ZSCORE_CLIP).fillna(0.0)


def compute_amihud_illiquidity_zscore(log_return: pd.Series, close: pd.Series, volume: pd.Series,
                                       window: int = AMIHUD_ZSCORE_WINDOW_DAYS,
                                       min_periods: int = AMIHUD_ZSCORE_MIN_PERIODS) -> pd.Series:
    """V26 (Directiva Head Quant) -- Z-score móvil (rolling, SOLO pasado) del
    ratio de iliquidez de Amihud: |LOG_RETURN_1D| / (close * volume) del
    mismo día -- extremidad relativa al propio régimen reciente de liquidez
    del activo, mismo criterio de "extremidad relativa, no nivel absoluto"
    que el resto de *_ZSCORE de este módulo. Valor alto -> el mercado se
    movió mucho con poco volumen detrás (impacto/iliquidez elevados,
    régimen frágil ante órdenes grandes); valor bajo -> mercado profundo."""
    dollar_volume = (close * volume).replace(0.0, np.nan)
    amihud = log_return.abs() / dollar_volume
    mean = amihud.rolling(window, min_periods=min_periods).mean()
    std = amihud.rolling(window, min_periods=min_periods).std()
    z = (amihud - mean) / std.replace(0.0, np.nan)
    return z.clip(-AMIHUD_ZSCORE_CLIP, AMIHUD_ZSCORE_CLIP).fillna(0.0)


# V27 -- ¿el término ln(C/O) de Garman-Klass participó del entrenamiento?
# `engineer_asset` lo setea según exista `{ticker}_Open` en el feed; se
# persiste en scalers_dict.pkl y quanti_engine.py lo OBEDECE en vez de
# decidirlo por su cuenta: train y serve NO tienen el mismo layout OHLCV
# (fetch_feature_ohlcv devuelve High/Low/Volume, sin Open), y recalcular la
# rama en inferencia produciría un valor numéricamente distinto contra un
# feature_scaler ajustado con el otro -- desync silencioso, sin excepción.
_GK_USED_OPEN: bool = False

# V28 -- por ticker: ¿las features overnight/intraday se entrenaron CON Open real?
# Se persiste en scalers_dict.pkl["v28_open_by_ticker"]; quanti_engine lo obedece.
_V28_OPEN_BY_TICKER: dict[str, bool] = {}

GK_VOL_LOOKBACK_DAYS = 20
GK_ZSCORE_WINDOW_DAYS = 60
GK_ZSCORE_MIN_PERIODS = 20
GK_ZSCORE_CLIP = 5.0

FIB_WINDOWS = (20, 50)
FIB_RATIOS = (0.382, 0.500, 0.618, 1.272)
FIB_ZSCORE_WINDOW_DAYS = 120
FIB_ZSCORE_MIN_PERIODS = 30
FIB_ZSCORE_CLIP = 5.0

OFI_LOOKBACK_DAYS = 10
OFI_ZSCORE_WINDOW_DAYS = 90
OFI_ZSCORE_MIN_PERIODS = 30
OFI_ZSCORE_CLIP = 5.0

ENTROPY_WINDOW_DAYS = 30
ENTROPY_SIGMA_LOOKBACK = 60
ENTROPY_DEADZONE_SIGMA = 0.25
ENTROPY_ZSCORE_WINDOW_DAYS = 120
ENTROPY_ZSCORE_MIN_PERIODS = 30
ENTROPY_ZSCORE_CLIP = 5.0


def _rolling_zscore(s: pd.Series, window: int, min_periods: int, clip: float) -> pd.Series:
    """Z-score móvil trailing (cero lookahead) -- mismo criterio que el resto
    de los *_ZSCORE de este módulo."""
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    z = (s - mean) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).clip(-clip, clip).fillna(0.0)


def compute_garman_klass_vol(open_: Optional[pd.Series], high: pd.Series, low: pd.Series,
                             close: pd.Series, window: int = GK_VOL_LOOKBACK_DAYS) -> pd.Series:
    """sigma^2 = 0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2, media rolling.
    Sin `open_` degrada al término H/L (== Parkinson salvo constante)."""
    var = 0.5 * np.log(high / low) ** 2
    if open_ is not None:
        var = var - (2.0 * np.log(2.0) - 1.0) * np.log(close / open_) ** 2
    var = var.replace([np.inf, -np.inf], np.nan)
    return np.sqrt(var.rolling(window, min_periods=window).mean().clip(lower=0.0))


def compute_fib_gap_zscore(high: pd.Series, low: pd.Series, close: pd.Series,
                           window: int) -> pd.Series:
    """Distancia FIRMADA al nivel de Fibonacci más cercano, z-scoreada.

    pos = (C - swing_low)/(swing_high - swing_low). Swings por `.rolling()`
    TRAILING -- una ventana `center=True` filtraría el futuro. El signo se
    preserva: rebotar desde debajo del 0.618 y perderlo desde arriba son
    estados direccionalmente opuestos; |gap| los colapsaría.
    """
    swing_high = high.rolling(window, min_periods=window).max()
    swing_low = low.rolling(window, min_periods=window).min()
    rng = (swing_high - swing_low).replace(0.0, np.nan)
    pos = (close - swing_low) / rng

    gaps = np.stack([(pos - r).to_numpy(dtype=float) for r in FIB_RATIOS], axis=1)
    abs_gaps = np.abs(gaps)
    all_nan = np.isnan(abs_gaps).all(axis=1)
    nearest = gaps[np.arange(gaps.shape[0]),
                   np.where(np.isnan(abs_gaps), np.inf, abs_gaps).argmin(axis=1)]
    nearest[all_nan] = np.nan
    return _rolling_zscore(pd.Series(nearest, index=close.index),
                           FIB_ZSCORE_WINDOW_DAYS, FIB_ZSCORE_MIN_PERIODS, FIB_ZSCORE_CLIP)


def compute_ofi_proxy_zscore(high: pd.Series, low: pd.Series, close: pd.Series,
                             volume: pd.Series, lookback: int = OFI_LOOKBACK_DAYS) -> pd.Series:
    """Order Flow Imbalance proxy desde OHLCV puro.

    CLV = ((C-L) - (H-C)) / (H-L) in [-1,1] -- dónde cerró el precio DENTRO
    del rango de la barra, proxy de qué lado del libro absorbió. Ponderado
    por volumen y normalizado por el volumen de la ventana: imbalance neto
    adimensional, comparable entre activos y entre regímenes de volumen.
    """
    rng = (high - low).replace(0.0, np.nan)
    clv = ((close - low) - (high - close)) / rng
    signed_vol = (clv * volume).replace([np.inf, -np.inf], np.nan)
    num = signed_vol.rolling(lookback, min_periods=lookback).sum()
    den = volume.rolling(lookback, min_periods=lookback).sum().replace(0.0, np.nan)
    return _rolling_zscore((num / den), OFI_ZSCORE_WINDOW_DAYS,
                           OFI_ZSCORE_MIN_PERIODS, OFI_ZSCORE_CLIP)


def compute_return_entropy_zscore(log_return: pd.Series,
                                  window: int = ENTROPY_WINDOW_DAYS) -> pd.Series:
    """Entropía de Shannon normalizada de la serie simbolizada de retornos.

    Simbolización en 3 estados (bajista / zona muerta / alcista) con umbral
    +-ENTROPY_DEADZONE_SIGMA * sigma_rolling -- el umbral se adapta al
    régimen de volatilidad del propio activo en vez de ser un absoluto que
    envía todo a la zona muerta en calma y nada en pánico.
    H = -sum p_i ln p_i / ln 3 in [0,1]: 1 = mercado máximamente aleatorio
    (sin estructura explotable), valores bajos = persistencia direccional.
    """
    sigma = log_return.rolling(ENTROPY_SIGMA_LOOKBACK,
                               min_periods=ENTROPY_SIGMA_LOOKBACK).std()
    thr = ENTROPY_DEADZONE_SIGMA * sigma
    up = (log_return > thr).astype(float)
    down = (log_return < -thr).astype(float)
    flat = 1.0 - up - down

    h = None
    for state in (up, down, flat):
        p = state.rolling(window, min_periods=window).mean()
        term = -(p * np.log(p.where(p > 0.0)))
        h = term if h is None else h.add(term, fill_value=0.0)
    h = (h / np.log(3.0)).where(sigma.notna())
    return _rolling_zscore(h, ENTROPY_ZSCORE_WINDOW_DAYS,
                           ENTROPY_ZSCORE_MIN_PERIODS, ENTROPY_ZSCORE_CLIP)


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

    V15 (consolidación del core): `all_data` ya llega con los macro
    tickers pre-alineados al calendario real del régimen (24/7 cripto o
    5/7 equity) por `_fetch_all_symbols_flat`/`kodaquant_core.
    project_macro_frame` -- el `.ffill().dropna()` de abajo ya NO es la
    única defensa contra el hueco de fin de semana/feriado (esa lógica
    vive ahora en una única fuente de verdad, compartida con inferencia),
    solo limpia remanentes menores (ej. el burn-in inicial de indicadores).
    """
    close_col = f"{ticker}_Close"
    high_col = f"{ticker}_High"
    low_col = f"{ticker}_Low"
    vol_col = f"{ticker}_Volume"
    macro_close_cols = [f"{m}_Close" for m in macro_tickers]

    df = all_data[[close_col, high_col, low_col, vol_col] + macro_close_cols].copy()
    rows_before = len(df)
    df = df.ffill().dropna()

    if df.empty:
        # Diagnóstico específico (V16): reemplaza el `ValueError` críptico de
        # sklearn en `build_asset_dataset` (~700 líneas más abajo, "Found
        # array with 0 sample(s)") por un mensaje que apunta directo a la
        # columna responsable -- causa más común: un macro ticker sin UN
        # SOLO dato real en todo `all_data` (símbolo roto/deslistado en
        # Yahoo Finance), que el `dropna()` de arriba (exige TODAS las
        # columnas no-nulas por fila, sin cambios de política) vacía por
        # completo. Cero impacto en el pipeline cuando `df` no queda vacío.
        raw = all_data[[close_col, high_col, low_col, vol_col] + macro_close_cols]
        fully_nan_cols = [c for c in raw.columns if raw[c].isna().all()]
        raise ValueError(
            f"'{ticker}': el DataFrame de features quedó VACÍO tras "
            f"ffill().dropna() ({rows_before} fila(s) antes -> 0 después). "
            + (f"Columna(s) 100% NaN en TODO el historial descargado: "
               f"{fully_nan_cols} -- revisá esa fuente (ver también el "
               "'[Diagnóstico cobertura]' logueado por _fetch_all_symbols_flat "
               "más arriba en esta misma corrida)."
               if fully_nan_cols else
               "Ninguna columna individual está 100% NaN -- es un problema de "
               "SUPERPOSICIÓN de fechas entre columnas (ej. un macro ticker con "
               "historial mucho más corto que el resto, sin fechas en común "
               "con el activo objetivo).")
        )

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

# --- V21: FEAR_GREED_INDEX (cripto) -- ver `_fetch_fear_greed_index` arriba
    # de esta función. Señal de PSICOLOGÍA de mercado, categoría distinta a
    # FUNDING_RATE (revertido en V17). Para tickers NO-cripto la columna
    # queda en 50.0 (neutro) constante -- CERO impacto en equity_specialist.
    if ticker in REGIMES["crypto_specialist"]["tickers"]:
        cache_key = "fear_greed_index"
        cached_fg = _ohlcv_cache.get_dataframe(cache_key)
        fg_series = cached_fg.iloc[:, 0] if cached_fg is not None and not cached_fg.empty else None
        if fg_series is None:
            fg_series = _fetch_fear_greed_index(tail_days=3700)
            if fg_series is not None and not fg_series.empty:
                _ohlcv_cache.set_dataframe(cache_key, fg_series.to_frame("fg"))
        if fg_series is not None and not fg_series.empty:
            fg_series.index = normalize_daily_index(pd.DatetimeIndex(fg_series.index))
            df["FEAR_GREED_INDEX"] = fg_series.reindex(df.index).ffill().fillna(50.0)
        else:
            df["FEAR_GREED_INDEX"] = 50.0
    else:
        df["FEAR_GREED_INDEX"] = 50.0

    # --- V24: FUNDING_RATE_ZSCORE (derivados -- extremidad de posicionamiento) ---
    if ticker in REGIMES["crypto_specialist"]["tickers"]:
        funding_cache_key = f"funding_rate_{ticker}"
        cached_funding = _ohlcv_cache.get_dataframe(funding_cache_key)
        funding_series = cached_funding.iloc[:, 0] if cached_funding is not None and not cached_funding.empty else None
        if funding_series is None:
            funding_series = _fetch_binance_funding_history(ticker)
            if funding_series is not None and not funding_series.empty:
                _ohlcv_cache.set_dataframe(funding_cache_key, funding_series.to_frame("funding"))
        if funding_series is not None and not funding_series.empty:
            funding_series.index = normalize_daily_index(pd.DatetimeIndex(funding_series.index))
            funding_z_full = compute_funding_rate_zscore(funding_series)
            df["FUNDING_RATE_ZSCORE"] = funding_z_full.reindex(df.index).ffill().fillna(0.0)
            # V26 -- FUNDING_RATE_MOMENTUM_ZSCORE: reutiliza el MISMO
            # funding_series ya fetcheado/cacheado arriba (cero llamada de
            # red adicional) -- ver docstring de
            # compute_funding_rate_momentum_zscore.
            funding_mom_z_full = compute_funding_rate_momentum_zscore(funding_series)
            df["FUNDING_RATE_MOMENTUM_ZSCORE"] = funding_mom_z_full.reindex(df.index).ffill().fillna(0.0)
        else:
            df["FUNDING_RATE_ZSCORE"] = 0.0
            df["FUNDING_RATE_MOMENTUM_ZSCORE"] = 0.0
    else:
        df["FUNDING_RATE_ZSCORE"] = 0.0
        df["FUNDING_RATE_MOMENTUM_ZSCORE"] = 0.0

    # --- V25 (reestructuración Alpha Cripto, auditoría rescate 2026-08):
    # ONCHAIN_ACTIVITY_ROC ahora es NVT Ratio Z-score (Woo/Burniske) --
    # MISMO nombre de columna/slot de TECH_COLS (BTC únicamente, 0.0 neutro
    # en ETH/equity, CERO cambio de N_FEATURES/shape), pero categoría de
    # señal distinta: valuación de red relativa a USO real, no solo
    # actividad bruta de transacciones. V24 (tx-count ROC crudo) se conserva
    # como fallback si el chart de volumen USD no responde -- nunca peor
    # que la señal V24 previa, nunca silenciosamente 0.0 si CUALQUIERA de
    # los dos providers de blockchain.info está arriba.
    if ticker == "BTC-USD":
        txval_cache_key = "onchain_tx_value_usd_btc"
        cached_txval = _ohlcv_cache.get_dataframe(txval_cache_key)
        tx_value_series = cached_txval.iloc[:, 0] if cached_txval is not None and not cached_txval.empty else None
        if tx_value_series is None:
            tx_value_series = _fetch_onchain_tx_volume_usd()
            if tx_value_series is not None and not tx_value_series.empty:
                _ohlcv_cache.set_dataframe(txval_cache_key, tx_value_series.to_frame("tx_value_usd"))
        if tx_value_series is not None and not tx_value_series.empty:
            tx_value_series.index = normalize_daily_index(pd.DatetimeIndex(tx_value_series.index))
            nvt_z_full = compute_nvt_ratio_zscore(close, tx_value_series.reindex(df.index).ffill())
            df["ONCHAIN_ACTIVITY_ROC"] = nvt_z_full.reindex(df.index).fillna(0.0)
        else:
            # Fallback V24 -- tx-count ROC crudo, si el chart de volumen USD no responde.
            onchain_cache_key = "onchain_activity_btc"
            cached_onchain = _ohlcv_cache.get_dataframe(onchain_cache_key)
            onchain_series = cached_onchain.iloc[:, 0] if cached_onchain is not None and not cached_onchain.empty else None
            if onchain_series is None:
                onchain_series = _fetch_onchain_activity()
                if onchain_series is not None and not onchain_series.empty:
                    _ohlcv_cache.set_dataframe(onchain_cache_key, onchain_series.to_frame("onchain"))
            if onchain_series is not None and not onchain_series.empty:
                onchain_series.index = normalize_daily_index(pd.DatetimeIndex(onchain_series.index))
                onchain_roc_full = onchain_series.pct_change(periods=ONCHAIN_ROC_LOOKBACK_DAYS)
                onchain_roc_full = onchain_roc_full.replace([np.inf, -np.inf], np.nan).clip(-ONCHAIN_ROC_CLIP, ONCHAIN_ROC_CLIP)
                df["ONCHAIN_ACTIVITY_ROC"] = onchain_roc_full.reindex(df.index).ffill().fillna(0.0)
            else:
                df["ONCHAIN_ACTIVITY_ROC"] = 0.0
    else:
        df["ONCHAIN_ACTIVITY_ROC"] = 0.0

    # --- V23.5 (Directiva 3): PUELL_MULTIPLE_ZSCORE -- ciclo de rentabilidad
    # de minero, BTC ÚNICAMENTE, cero fetch (ver bloque de constantes
    # `BTC_HALVING_SCHEDULE`/`compute_puell_multiple_zscore` más arriba). ---
    if ticker == "BTC-USD":
        df["PUELL_MULTIPLE_ZSCORE"] = compute_puell_multiple_zscore(close)
    else:
        df["PUELL_MULTIPLE_ZSCORE"] = 0.0

    # --- V24: FRAC_DIFF_LOG_CLOSE (memoria + estacionariedad, López de Prado) ---
    if ticker in REGIMES["crypto_specialist"]["tickers"]:
        raw_log_price = np.log(close)
        ffd_d = find_min_ffd_d(raw_log_price)
        df["FRAC_DIFF_LOG_CLOSE"] = frac_diff_ffd(raw_log_price, ffd_d)
        _LAST_FFD_D_BY_TICKER[ticker] = float(ffd_d)  # ver run_regime_pipeline -- persistido en scalers_dict.pkl
        logger.info("[FRAC_DIFF] %s -> d=%.2f (memoria preservada, calibrado por ADF sobre prefijo de train).",
                    ticker, ffd_d)
    else:
        df["FRAC_DIFF_LOG_CLOSE"] = 0.0

    # --- AUDITORÍA RESCATE 2026-08 (Directiva 2): PARKINSON_VOL_ZSCORE -----
    # AMBOS regímenes -- proxy libre de volatilidad implícita, ver docstring
    # arriba de `compute_parkinson_vol`. Requiere High/Low, ya en `df`.
    parkinson_vol = compute_parkinson_vol(high, low)
    df["PARKINSON_VOL_ZSCORE"] = compute_parkinson_vol_zscore(parkinson_vol)

    # --- AUDITORÍA RESCATE 2026-08 (Directiva 2): VOL_OF_VOL_ZSCORE ---------
    # Cripto ÚNICAMENTE -- 0.0 neutro en equity, mismo patrón que
    # FEAR_GREED_INDEX/FUNDING_RATE_ZSCORE/ONCHAIN_ACTIVITY_ROC arriba.
    if ticker in REGIMES["crypto_specialist"]["tickers"]:
        df["VOL_OF_VOL_ZSCORE"] = compute_vol_of_vol_zscore(parkinson_vol)
    else:
        df["VOL_OF_VOL_ZSCORE"] = 0.0

    # --- V26 (Directiva Head Quant): AMIHUD_ILLIQUIDITY_ZSCORE --------------
    # Cripto ÚNICAMENTE -- 0.0 neutro en equity, mismo patrón que
    # VOL_OF_VOL_ZSCORE/FUNDING_RATE_ZSCORE arriba. 100% OHLCV ya en scope
    # (LOG_RETURN_1D/close/volume) -- cero fetch nuevo.
    if ticker in REGIMES["crypto_specialist"]["tickers"]:
        df["AMIHUD_ILLIQUIDITY_ZSCORE"] = compute_amihud_illiquidity_zscore(df["LOG_RETURN_1D"], close, volume)
    else:
        df["AMIHUD_ILLIQUIDITY_ZSCORE"] = 0.0

    # --- V27: microestructura OHLC + geometría fractal + flujo + caos ------
    # AMBOS regímenes, 100% OHLCV ya en scope -- cero fetch nuevo.
    global _GK_USED_OPEN
    _open_col = close_col.replace("Close", "Open")
    _open = df[_open_col] if _open_col in df.columns else None
    _GK_USED_OPEN = _open is not None
    if _open is None:
        logger.warning("[GK] '%s': columna Open ausente -> GARMAN_KLASS_VOL_ZSCORE "
                       "degrada al término H/L (Parkinson salvo constante).", ticker)
    df["GARMAN_KLASS_VOL_ZSCORE"] = _rolling_zscore(
        compute_garman_klass_vol(_open, high, low, close),
        GK_ZSCORE_WINDOW_DAYS, GK_ZSCORE_MIN_PERIODS, GK_ZSCORE_CLIP)

    for _w in FIB_WINDOWS:
        df[f"FIB_GAP_Z_{_w}"] = compute_fib_gap_zscore(high, low, close, _w)

    df["OFI_PROXY_ZSCORE"] = compute_ofi_proxy_zscore(high, low, close, volume)
    df["RETURN_ENTROPY_ZSCORE"] = compute_return_entropy_zscore(df["LOG_RETURN_1D"])

    # --- V28: free_alpha.py (una sola implementación, train == serve) ------
    # Cripto -> 4 cols de flujo; equity -> 4 cols overnight/intraday/rel-strength.
    # El bloque del régimen opuesto queda en 0.0 (mismo patrón que FUNDING_RATE en equity).
    if ticker in REGIMES["crypto_specialist"]["tickers"]:
        for _c in V28_EQUITY_COLS:
            df[_c] = 0.0
        _flow = build_crypto_flow_features(
            ticker, df.index, cache=None if FORCE_FRESH_DATA else _ohlcv_cache)
        if len(_flow) != len(df):
            raise RuntimeError(f"[V28] '{ticker}': flujo cripto len={len(_flow)} != len(df)={len(df)}.")
        for _c in V28_CRYPTO_COLS:
            df[_c] = _flow[_c].to_numpy(dtype=float)
        _V28_OPEN_BY_TICKER[ticker] = False
    else:
        for _c in V28_CRYPTO_COLS:
            df[_c] = 0.0
        _v28_open = None
        _v28_open_col = f"{ticker}_Open"
        if _v28_open_col in all_data.columns:
            _v28_open = all_data[_v28_open_col].reindex(df.index)
            if _v28_open.notna().mean() < 0.9:
                logger.warning("[V28] '%s': Open con <90%% de cobertura -> overnight/intraday = 0.0.", ticker)
                _v28_open = None
        _V28_OPEN_BY_TICKER[ticker] = _v28_open is not None
        # `^GSPC_Close` YA es log-return aquí (loop macro de arriba).
        _spx_col = "^GSPC_Close"
        _eq = compute_equity_open_features(
            _v28_open, close, df[_spx_col] if _spx_col in df.columns else None, ticker)
        for _c in V28_EQUITY_COLS:
            df[_c] = _eq[_c]

    df = df.ffill().dropna()

    feature_cols = ["LOG_RETURN_1D"] + TECH_COLS + macro_close_cols
    result = df[["RAW_CLOSE"] + feature_cols].copy()
    result.columns = ["RAW_CLOSE", "LOG_RETURN_1D"] + TECH_COLS + macro_tickers
    return result


def compute_triple_barrier_labels(df_asset: pd.DataFrame, pt_mult: float, sl_mult: float,
                                    max_holding_days: int, vol_lookback: int,
                                    min_vol: float = TRIPLE_BARRIER_MIN_VOL) -> pd.DataFrame:
    """
    Triple Barrier Method (López de Prado, cap. 3) sobre cierres diarios.
    Para cada día de entrada `s` (precio hipotético de entrada = close_s):

        barrera_superior_s = close_s * exp(+pt_mult * vol_s)
        barrera_inferior_s = close_s * exp(-sl_mult * vol_s)

    con `vol_s` = desvío estándar de LOG_RETURN_1D en una ventana móvil de
    `vol_lookback` días TERMINADA en `s` (SOLO pasado -- cero fuga hacia
    el futuro en el ANCHO de la barrera; la ETIQUETA en sí es
    necesariamente forward-looking, como cualquier target supervisado).

    Recorre los cierres de s+1 .. s+max_holding_days (barrera vertical) y
    determina cuál barrera se toca PRIMERO:
        +1 ("TP")      -- close cruza la barrera superior primero
        -1 ("SL")      -- close cruza la barrera inferior primero
         0 ("TIMEOUT") -- ninguna barrera se toca -- barrera vertical,
                          "sin convicción" en el espíritu de meta-labeling

    LIMITACIÓN CONOCIDA (documentada, no oculta): con cierre diario (sin
    OHLC intradía) un único valor de cierre por día no puede cruzar ambas
    barreras a la vez, así que no hay ambigüedad de ORDEN intra-día -- pero
    tampoco hay forma de saber si, DENTRO de un día dado, el máximo/mínimo
    intradía tocó una barrera que el cierre no refleja. El primer-toque
    intradía real requeriría barras horarias/tick -- fuera del alcance
    100%-gratuito de este pipeline (Restricción de Datos, directiva Head
    Quant). Sesgo esperado: subestima levemente la frecuencia de toques
    (barreras algo más "difíciles" de activar que con datos intradía).

    Devuelve un DataFrame alineado a `df_asset.index` con:
        TB_LABEL            -1/0/+1 (NaN en warm-up de `vol_lookback` o en
                             la cola sin suficiente futuro para resolver
                             `max_holding_days` -- se filtra aguas abajo)
        TB_REALIZED_RETURN  log-return REAL logrado al momento del exit
                             (barrera tocada o vertical) -- análogo de
                             `r_true` en el régimen de regresión; es lo
                             que usa `pnl_backtest.py` como P&L de la
                             apuesta y `sample_weight` (ver
                             `build_asset_dataset`) como su magnitud.
        TB_BARS_TO_EXIT     días hábiles hasta el exit (diagnóstico).
    """
    close = df_asset["RAW_CLOSE"].to_numpy(dtype=np.float64)
    log_ret_1d = df_asset["LOG_RETURN_1D"].to_numpy(dtype=np.float64)
    n = len(close)

    vol = (
        pd.Series(log_ret_1d, index=df_asset.index)
        .rolling(vol_lookback, min_periods=vol_lookback)
        .std()
        .to_numpy(dtype=np.float64)
    )
    vol = np.maximum(vol, min_vol)

    labels = np.full(n, np.nan, dtype=np.float64)
    realized_return = np.full(n, np.nan, dtype=np.float64)
    bars_to_exit = np.full(n, np.nan, dtype=np.float64)

    for s in range(n):
        if np.isnan(vol[s]) or s + max_holding_days >= n:
            continue  # vol insuficiente (warm-up) o cola sin futuro suficiente -- NaN, se filtra en build_asset_dataset

        entry_price = close[s]
        upper = entry_price * np.exp(pt_mult * vol[s])
        lower = entry_price * np.exp(-sl_mult * vol[s])

        exit_label = 0.0
        exit_ret = float(np.log(close[s + max_holding_days] / entry_price))  # default: barrera vertical
        exit_bars = float(max_holding_days)
        for h in range(1, max_holding_days + 1):
            price_h = close[s + h]
            if price_h >= upper:
                exit_label, exit_ret, exit_bars = 1.0, float(np.log(price_h / entry_price)), float(h)
                break
            if price_h <= lower:
                exit_label, exit_ret, exit_bars = -1.0, float(np.log(price_h / entry_price)), float(h)
                break

        labels[s] = exit_label
        realized_return[s] = exit_ret
        bars_to_exit[s] = exit_bars

    return pd.DataFrame(
        {"TB_LABEL": labels, "TB_REALIZED_RETURN": realized_return, "TB_BARS_TO_EXIT": bars_to_exit},
        index=df_asset.index,
    )


def purged_fold_bounds(n: int, k: int, embargo_frac: float = 0.02) -> list[tuple[int, int, int]]:
    """
    V27 -- Purged K-Fold (un solo path walk-forward por fold, NO la
    combinatoria completa C(n_groups, n_test_groups) -- a ese costo, en este
    hardware, un CPCV de 6-eligiendo-2 sobre equity_specialist (~150s/época)
    son horas por régimen. Un Purged K-Fold de k bloques contiguos da la
    misma auditoría de varianza OOS que preocupa (¿el 52.6% depende del
    split que elegiste?) a 1/C(n,r) del costo).

    Cada fold define un bloque de test contiguo `[test_start, test_end)`
    sobre las `n` filas del DataFrame de UN activo, YA en orden temporal.
    Devuelve `(test_start, test_end, embargo_end)` por fold -- el caller
    (Fase 2, próximo paso) es responsable de purgar del train cualquier
    fila cuyo horizonte de etiqueta caiga en `[test_start, embargo_end)`,
    exactamente la misma lógica de `embargo_violation` que ya existe en
    `build_asset_dataset` para el split único -- acá generalizada a k
    bloques en vez de 1.

    No toca `build_asset_dataset` ni ningún pipeline de training existente
    -- es aritmética de índices pura, verificable sola.
    """
    if k < 2:
        raise ValueError(f"purged_fold_bounds: k debe ser >= 2 (recibido {k})")
    bounds = np.linspace(0, n, k + 1, dtype=int)
    embargo = max(1, int(n * embargo_frac))
    folds = []
    for i in range(k):
        test_start, test_end = int(bounds[i]), int(bounds[i + 1])
        if test_end <= test_start:
            continue
        embargo_end = min(n, test_end + embargo)
        folds.append((test_start, test_end, embargo_end))
    return folds


def build_asset_dataset(df: pd.DataFrame, ticker: str, lookback: int, train_ratio: float,
                         asset_to_id: dict[str, int], target_mode: str = "regression",
                         tb_params: dict | None = None, exclude_timeout_from_train: bool = False):
    """
    Ventanas (X) + targets para UN activo, con scalers propios por activo.

    `target_mode="regression"` (default, `equity_specialist`): CERO
    cambios respecto al pipeline V5/V6 original -- target = log-return
    continuo, `StandardScaler(with_mean=False)`.

    `target_mode="triple_barrier"` (V22, `crypto_specialist`): target =
    clase 3-way (SL/TIMEOUT/TP) de `compute_triple_barrier_labels`,
    one-hot. La ventana de features sigue terminando en `t-1` (mismo
    contrato que el régimen de regresión: predicción usa SOLO información
    hasta `t-1`); la "entrada" hipotética de la apuesta Triple Barrier es
    `close[t-1]` -- mismo índice que `last_price[t-1]` ya usaba el régimen
    de regresión, cero drift de convención entre ambos modos. Filas cuya
    etiqueta cae en la cola sin `max_holding_days` de futuro suficiente
    (`TB_LABEL` = NaN) se SALTEAN por completo (no se agregan a X/y) --
    reduce el tamaño de test en, como mucho, `max_holding_days` filas.
    """
    feature_cols = ["LOG_RETURN_1D"] + TECH_COLS + MACRO_TICKERS
    # RAW_CLOSE es bookkeeping puro (ver engineer_asset) — nunca entra a
    # `features`/al feature_scaler, solo se usa para derivar `last_price`,
    # el target de log-return (regresión) y el precio de entrada Triple
    # Barrier (clasificación).
    prices = df["RAW_CLOSE"].values
    features = df[feature_cols].values
    log_returns = np.diff(np.log(prices))

    n = len(df)
    split_idx = int(n * train_ratio)

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(features[:split_idx])
    features_scaled = feature_scaler.transform(features)

    tb_label_arr = tb_return_arr = None
    if target_mode == "triple_barrier":
        tb_params = tb_params or {}
        tb = compute_triple_barrier_labels(
            df,
            pt_mult=tb_params.get("pt_mult", TRIPLE_BARRIER_PT_MULT),
            sl_mult=tb_params.get("sl_mult", TRIPLE_BARRIER_SL_MULT),
            max_holding_days=tb_params.get("max_holding_days", TRIPLE_BARRIER_MAX_HOLDING_DAYS),
            vol_lookback=tb_params.get("vol_lookback", TRIPLE_BARRIER_VOL_LOOKBACK),
        )
        tb_label_arr = tb["TB_LABEL"].to_numpy()
        tb_return_arr = tb["TB_REALIZED_RETURN"].to_numpy()
        tb_bars_arr = tb["TB_BARS_TO_EXIT"].to_numpy()

    X, y_raw, last_price, dates = [], [], [], []
    tb_class_idx_list, tb_ret_list, tb_entry_idx_list, tb_bars_list = [], [], [], []
    for t in range(lookback, n):
        entry_idx = t - 1
        if target_mode == "triple_barrier" and np.isnan(tb_label_arr[entry_idx]):
            continue  # cola sin max_holding_days de futuro (o warm-up de vol) -- se saltea la fila entera
        X.append(features_scaled[t - lookback: t])
        last_price.append(prices[entry_idx])
        dates.append(df.index[t])
        if target_mode == "triple_barrier":
            tb_class_idx_list.append(TB_LABEL_TO_CLASS_IDX[int(tb_label_arr[entry_idx])])
            tb_ret_list.append(tb_return_arr[entry_idx])
            tb_entry_idx_list.append(entry_idx)
            tb_bars_list.append(tb_bars_arr[entry_idx])
        else:
            y_raw.append(log_returns[entry_idx])

    X = np.array(X, dtype=np.float32)
    last_price = np.array(last_price, dtype=np.float32)
    dates = pd.DatetimeIndex(dates)
    asset_id = np.full((len(X),), asset_to_id[ticker], dtype=np.int32)

    # `window_split` cuenta POSICIONES del loop -- idéntico al cómputo
    # original. Con target_mode="triple_barrier" se saltean filas de la
    # COLA (después de split_idx en la inmensa mayoría de los casos, ver
    # docstring); el clip defensivo cubre el caso límite de un fold/test
    # tan angosto que la cola sin-futuro invada el propio split.
    window_split = max(0, min(split_idx - lookback, len(X)))

    if target_mode == "triple_barrier":
        tb_class_idx = np.array(tb_class_idx_list, dtype=np.int32)
        tb_ret = np.array(tb_ret_list, dtype=np.float32)
        y_onehot = np.eye(3, dtype=np.float32)[tb_class_idx]

        # --- sample_weight -- Directiva 3 (Risk-Adjusted Loss) ------------
        # Pondera cada muestra por la MAGNITUD del retorno real logrado en
        # esa apuesta (normalizada por la mediana de |retorno| en TRAIN YA
        # PURGADO -- ver bloque de embargo justo abajo --, clip [0.25x, 4x]
        # para que un outlier puntual no domine el gradiente de una época
        # entera). El modelo paga más caro equivocarse en un día de
        # colapso/despegue (+5%, ver directiva) que en uno aburrido --
        # exactamente lo pedido, separado del balanceo de CLASE (ese lo
        # resuelve `alpha` en CategoricalFocalCrossentropy, ver
        # `build_model`/`run_regime_pipeline`). `entry_idx_arr`/`bars_arr`
        # (V23, uniqueness -- ver `_average_uniqueness`) se calculan ACÁ
        # porque el embargo de abajo los necesita antes de poder derivar
        # `train_keep`/`test_keep`.
        entry_idx_arr = np.array(tb_entry_idx_list, dtype=np.int64)
        bars_arr = np.array(tb_bars_list, dtype=np.float64)

        # --- EMBARGO/PURGE (López de Prado cap. 7 -- el mismo texto ya citado en
        # el docstring de esta función) -- una apuesta de TRAIN cuyo exit
        # (entry_idx + bars_to_exit) cae EN o DESPUÉS de `window_split` tiene su
        # propia etiqueta (TB_LABEL/TB_REALIZED_RETURN) determinada observando
        # precios que ya pertenecen a `test` -- fuga real, aunque acotada (a lo
        # sumo `max_holding_days` filas por activo), nunca antes purgada en este
        # pipeline. Se DESCARTA la fila completa de train (no se trunca el
        # retorno) -- `test` no se toca, sus propios exits miran hacia adelante
        # dentro de datos reales/futuros, sin cruzar hacia train.
        is_train_row = entry_idx_arr < window_split
        embargo_violation = is_train_row & ((entry_idx_arr + bars_arr) >= window_split)
        train_keep = is_train_row & ~embargo_violation
        test_keep = ~is_train_row
        if embargo_violation.any():
            logger.info(
                "[Embargo TB] %d fila(s) de train purgadas por fuga de horizonte hacia test "
                "(López de Prado cap. 7).", int(embargo_violation.sum()),
            )

        uniqueness = _average_uniqueness(entry_idx_arr, bars_arr, n)
        uniqueness_train_mean = float(np.mean(uniqueness[train_keep])) if train_keep.any() else 1.0
        uniqueness_norm = uniqueness / max(uniqueness_train_mean, 1e-8)

        # `train_mag_median` (arriba) ya se calculó sobre `tb_ret[:window_split]`
        # -- con el embargo activo, esas pocas filas fronterizas purgadas no
        # deberían inflar la mediana de magnitud; recalculada acá sobre la
        # máscara real de train para mantener ambos pesos (pnl/uniqueness)
        # consistentes con el mismo conjunto de filas que finalmente se entrena.
        train_mag_purged = np.abs(tb_ret[train_keep])
        positive_mag_purged = train_mag_purged[train_mag_purged > 0]
        train_mag_median_purged = float(np.median(positive_mag_purged)) if positive_mag_purged.size else 1.0
        pnl_weight = np.abs(tb_ret) / max(train_mag_median_purged, 1e-8)

        sample_weight = np.clip(pnl_weight * uniqueness_norm, 0.25, 4.0).astype(np.float32)

        # V27 -- opt-in (default False = CERO cambio de comportamiento en
        # cualquier caller existente). Evidencia de 3 experimentos
        # independientes (bias-cal bias=0.00 óptimo, meta-labeling no
        # generalizó, recall TIMEOUT std=0% en 4 folds) apunta a que
        # TIMEOUT no tiene firma aprendible con las features actuales --
        # esto deja de pedírselo al modelo EN TRAIN. `test` NUNCA se toca:
        # sigue evaluando las 3 clases reales, comparable contra cada
        # corrida anterior de esta sesión sin ningún ajuste.
        train_keep_final = train_keep
        if exclude_timeout_from_train:
            train_keep_final = train_keep & (tb_class_idx != TB_TIMEOUT_CLASS_IDX)
            logger.info("      [2-CLASE] '%s': %d/%d filas de train excluidas (TIMEOUT).",
                       ticker, int((train_keep & ~train_keep_final).sum()), int(train_keep.sum()))

        train = dict(
            X=X[train_keep_final], y=y_onehot[train_keep_final], asset_id=asset_id[train_keep_final],
            last_price=last_price[train_keep_final], dates=dates[train_keep_final],
            sample_weight=sample_weight[train_keep_final], y_return=tb_ret[train_keep_final],
            y_class_idx=tb_class_idx[train_keep_final],
        )
        test = dict(
            X=X[test_keep], y=y_onehot[test_keep], asset_id=asset_id[test_keep],
            last_price=last_price[test_keep], dates=dates[test_keep],
            sample_weight=sample_weight[test_keep], y_return=tb_ret[test_keep],
            y_class_idx=tb_class_idx[test_keep],
        )
        return train, test, feature_scaler, None

    # --- target_mode == "regression" (default) — CERO cambios respecto al
    # pipeline V5/V6 original: misma construcción, mismo StandardScaler.
    y_raw = np.array(y_raw, dtype=np.float32).reshape(-1, 1)

    target_scaler = StandardScaler(with_mean=False)  # preserva el signo del retorno (V4)
    target_scaler.fit(y_raw[:window_split])
    y_scaled = target_scaler.transform(y_raw).astype(np.float32)

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
                                seed: int = SEED,
                                sample_weight: np.ndarray | None = None):
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

    V22: `sample_weight` es OPCIONAL (None para `equity_specialist`/
    regresión; ver Directiva 3 para `crypto_specialist`/triple_barrier).
    Cuando se provee, se resamplea con los MISMOS índices `idx`/`perm` que
    X/y/asset_id -- resamplearlo por separado rompería la correspondencia
    fila-a-fila entre una ventana aumentada y su peso de muestra real.
    Devuelve SIEMPRE una tupla de 4 elementos; el cuarto es `None` si la
    entrada lo fue.
    """
    if factor <= 1.0:
        return X, y, asset_id, sample_weight

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
    sw_bal = None if sample_weight is None else np.concatenate([sample_weight, sample_weight[idx]], axis=0)

    perm = rng.permutation(len(X_bal))
    sw_out = None if sw_bal is None else sw_bal[perm]
    return X_bal[perm], y_bal[perm], asset_id_bal[perm], sw_out


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

    ROLLBACK V23.6 (Head Quant, ver changelog del módulo): V23.5 había
    hecho que `call()` NO redujera a escalar internamente (para que
    Keras pudiera aplicar un `sample_weight` externo por volatilidad de
    equity antes de la reducción). Ese sample_weight fue purgado por
    completo (empeoró Directional Accuracy en vez de mejorarla) y
    `equity_specialist` (única rama que usa esta loss) nunca vuelve a
    pasar `sample_weight` -- `call()` vuelve a reducir con
    `keras.ops.mean(...)` acá adentro, EXACTAMENTE como en V5/V6. Cero
    mecanismo residual atado a un sample_weight que ya no existe.
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
        per_sample_loss = nll * keras.ops.exp(self.gamma * mismatch)  # (batch,)

        # --- (3) amortiguación sobre log_var CRUDO (sin clip) -- término
        # PER-SAMPLE sumado antes de reducir (mean(a_i + c) == mean(a_i) + c).
        if self.log_var_l2_lambda > 0.0:
            per_sample_loss = per_sample_loss + self.log_var_l2_lambda * keras.ops.square(log_var)
        if self.log_var_barrier_lambda > 0.0:
            excess_hi = keras.ops.relu(log_var - self.log_var_max)
            excess_lo = keras.ops.relu(self.log_var_min - log_var)
            per_sample_loss = per_sample_loss + self.log_var_barrier_lambda * (
                keras.ops.square(excess_hi) + keras.ops.square(excess_lo)
            )

        # ROLLBACK V23.6 -- reducción interna a escalar restaurada (idéntico
        # a V5/V6): `equity_specialist` nunca vuelve a pasar `sample_weight`
        # a esta loss, así que no hay razón para dejar la reducción a cargo
        # de Keras (eso solo importaba para el sample_weight ya purgado).
        return keras.ops.mean(per_sample_loss)

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


# ============================================================
# V22 — MÉTRICAS DE CLASIFICACIÓN (target_mode="triple_barrier", ver
# build_model/REGIMES["crypto_specialist"])
# ============================================================
@keras.saving.register_keras_serializable(package="quanti")
class ActionableAccuracy(keras.metrics.Metric):
    """
    Accuracy restringida a apuestas ACCIONABLES (clase VERDADERA !=
    TIMEOUT) -- el análogo, en el espíritu de meta-labeling, de "cuando el
    mercado SÍ tocó una barrera, ¿el modelo acertó cuál?". Deliberadamente
    IGNORA la clase TIMEOUT (mayoritaria en volumen -- ver distribución de
    clases logueada en `run_regime_pipeline`): un modelo que colapsa a
    predecir siempre TIMEOUT tendría accuracy 3-way alta pero
    `actionable_accuracy` = 0%, exactamente el colapso "cobarde" que
    `EarlyStopping` monitorea acá (ver `train_model`) en vez de la accuracy
    plana -- mismo rol que `val_directional_accuracy_metric` cumple para
    el régimen de regresión.

    Implementada CON ESTADO (no como función simple, a diferencia de
    `directional_accuracy_metric`) porque el promedio debe calcularse
    SOLO sobre el subconjunto accionable de cada batch, un conteo propio
    independiente del `sample_weight` de la pérdida (magnitud/PnL, ver
    `build_asset_dataset`) -- por eso `sample_weight` se recibe pero NO se
    usa: esta métrica pondera por igual cada apuesta accionable,
    deliberadamente distinto del criterio de la pérdida.
    """

    def __init__(self, timeout_class_idx: int = TB_TIMEOUT_CLASS_IDX,
                 name: str = "actionable_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.timeout_class_idx = timeout_class_idx
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):  # noqa: ARG002 -- ver docstring
        y_true = keras.ops.cast(y_true, "float32")
        true_idx = keras.ops.argmax(y_true, axis=-1)
        pred_idx = keras.ops.argmax(y_pred, axis=-1)
        is_actionable = keras.ops.cast(keras.ops.not_equal(true_idx, self.timeout_class_idx), "float32")
        is_correct = keras.ops.cast(keras.ops.equal(true_idx, pred_idx), "float32")
        self.correct.assign_add(keras.ops.sum(is_correct * is_actionable))
        self.total.assign_add(keras.ops.sum(is_actionable))

    def result(self):
        return self.correct / keras.ops.maximum(self.total, 1e-8)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"timeout_class_idx": self.timeout_class_idx})
        return config


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


def _build_tcn_branch(seq_input, filters: int, l2reg, dropout_rate: float,
                       kernel_size: int = TCN_KERNEL_SIZE, dilations=TCN_DILATIONS,
                       name_prefix: str = "tcn"):
    """
    Bloque TCN (V15, Directiva 4): pila de convoluciones causales 1D
    dilatadas con conexión residual por bloque (patrón estándar Bai et al.
    2018), seguida de GlobalAveragePooling1D para colapsar a un vector de
    contexto del mismo rol que `BahdanauAttention` -- ambos se concatenan
    aguas abajo en `build_model`. `padding="causal"` es obligatorio: sin
    causalidad el filtro vería pasos FUTUROS de la ventana, fuga de
    información hacia el propio target que se intenta predecir.
    """
    x = seq_input
    for i, dilation in enumerate(dilations):
        residual = x
        conv = layers.Conv1D(
            filters, kernel_size, padding="causal", dilation_rate=dilation,
            activation="relu", kernel_regularizer=l2reg, name=f"{name_prefix}_conv_{i}_d{dilation}",
        )(x)
        conv = layers.Dropout(dropout_rate * 0.3, name=f"{name_prefix}_dropout_{i}")(conv)
        if residual.shape[-1] != filters:
            residual = layers.Conv1D(filters, 1, padding="same", name=f"{name_prefix}_proj_{i}")(residual)
        x = layers.Add(name=f"{name_prefix}_residual_add_{i}")([residual, conv])
        x = layers.LayerNormalization(name=f"{name_prefix}_ln_{i}")(x)
    return layers.GlobalAveragePooling1D(name=f"{name_prefix}_global_pool")(x)


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
                 model_name: str | None = None,
                 target_mode: str = "regression",
                 class_alpha: list[float] | None = None,
                 focal_gamma: float = TRIPLE_BARRIER_FOCAL_GAMMA,
                 label_smoothing: float = TRIPLE_BARRIER_LABEL_SMOOTHING) -> tuple:
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

    V22 — `target_mode` bifurca ÚNICAMENTE la cabeza de salida final y el
    `compile()` (loss/métricas); el tronco común (BiLSTM->MHA->Bahdanau
    [->TCN]->asset_embed->Dense residual) es IDÉNTICO en ambos modos --
    misma invarianza topológica que ya regía entre regímenes.
    `target_mode="regression"` (default) es BIT A BIT idéntico a V6: cabeza
    `Dense(2)` (mu, log_var) + `DirectionalGaussianNLL`. `target_mode=
    "triple_barrier"` reemplaza la cabeza por `Dense(3, softmax)` (SL/
    TIMEOUT/TP, ver `TB_CLASS_NAMES`) + `CategoricalFocalCrossentropy`
    (Lin et al. 2017, built-in Keras 3) con `alpha` por-clase (balanceo,
    ver `run_regime_pipeline`) -- el curriculum de `gamma` direccional
    (`DynamicGammaCallback`) es exclusivo del régimen de regresión; acá
    `gamma_variable` vuelve `None`, sin aristas huérfanas en `train_model`.
    """
    arch = REGIME_ARCHITECTURE[regime_name]
    lstm_units = arch["lstm_units"]
    attention_units = arch["attention_units"]
    mha_heads = arch["mha_heads"]
    mha_key_dim = arch["mha_key_dim"]
    dropout_rate = arch["dropout_rate"]
    dense_l2_reg = arch["dense_l2_reg"]
    extra_residual_block = arch["extra_residual_block"]
    tcn_filters = arch.get("tcn_filters", lstm_units // 2)
    if model_name is None:
        model_name = f"KodaQuant_{regime_name}_V5"

    seq_input = keras.Input(shape=(n_timesteps, n_features), name="input_sequence")
    asset_input = keras.Input(shape=(1,), dtype="int32", name="input_asset_id")

    l2 = keras.regularizers.l2(dense_l2_reg)  # V5.1: definido temprano — reutilizado en BiLSTM/MHA/Dense

    # --- V27: Gated Linear Unit sobre la secuencia CRUDA -------------------
    # value * sigmoid(gate), ambas proyecciones desde el mismo input. La
    # rama value usa Swish (suave, no satura ni mata gradiente como ReLU en
    # z-scores centrados en 0, que es lo que son TODAS las features nuevas).
    # LayerNormalization post-gate: la multiplicación elemento a elemento
    # contrae la escala y sin renormalizar el gradiente se desvanece al
    # atravesar el BiLSTM. `glu_units=None`/ausente -> grafo IDÉNTICO al
    # anterior (aditivo y reversible, mismo criterio que ENABLE_TCN_BRANCH).
    glu_units = arch.get("glu_units")
    if glu_units:
        _value = layers.Dense(glu_units, activation="swish", kernel_regularizer=l2,
                              name="glu_input_value")(seq_input)
        _gate = layers.Dense(glu_units, activation="sigmoid", kernel_regularizer=l2,
                             name="glu_input_gate")(seq_input)
        gated_seq = layers.Multiply(name="glu_input_mul")([_value, _gate])
        gated_seq = layers.LayerNormalization(name="glu_input_ln")(gated_seq)
        gated_seq = layers.SpatialDropout1D(dropout_rate * 0.25,
                                            name="glu_input_spatial_dropout")(gated_seq)
    else:
        gated_seq = seq_input

    bilstm = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate * 0.3,
                    kernel_regularizer=l2, recurrent_regularizer=l2),
        name=f"bilstm_{lstm_units}",
    )(gated_seq)
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

    # --- V15 (Directiva 4) — rama TCN paralela, fusionada por concat -----
    # Opera sobre `seq_input` CRUDO (no sobre `attn_block`): busca patrones
    # temporales complementarios a los que ya extrae BiLSTM+MHA+Bahdanau,
    # no refinar la misma representación dos veces. `ENABLE_TCN_BRANCH=0`
    # deja el grafo IDÉNTICO a la topología V10 (invarianza reversible).
    if ENABLE_TCN_BRANCH:
        # V27 -- TCN consume la MISMA secuencia compuerta que el BiLSTM.
        # Alimentarla con `seq_input` crudo dejaría a las dos ramas viendo
        # espacios de entrada distintos y volvería no interpretable cualquier
        # atribución por gradiente entre ellas.
        tcn_context = _build_tcn_branch(gated_seq, tcn_filters, l2, dropout_rate)
        tcn_context = layers.Dropout(dropout_rate * 0.5, name="dropout_post_tcn")(tcn_context)
        fusion_inputs = [context_vector, tcn_context, asset_embed]
    else:
        fusion_inputs = [context_vector, asset_embed]

    fused = layers.Concatenate(name="fusion_attn_asset")(fusion_inputs)
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

    # --- V22 — cabeza de salida bifurcada por target_mode ------------------
    if target_mode == "triple_barrier":
        output = layers.Dense(3, activation="softmax", name="triple_barrier_head",
                               kernel_regularizer=l2)(dense_block)
    else:
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

    if target_mode == "triple_barrier":
        # Curriculum de gamma direccional es exclusivo de DirectionalGaussianNLL
        # (régimen de regresión) -- acá no hay "mismatch de signo" que penalizar
        # progresivamente, el desbalance de clases ya lo resuelve `alpha`.
        gamma_variable = None
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.CategoricalFocalCrossentropy(
                alpha=class_alpha if class_alpha is not None else 0.25,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
            ),
            metrics=[keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
                     ActionableAccuracy(name="actionable_accuracy")],
        )
    else:
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
                 validation_data=None,
                 gamma_variable=None,
                 gamma_max: float = GAMMA_MAX,
                 gamma_warmup_epochs: int = GAMMA_WARMUP_EPOCHS,
                 gamma_schedule: str = GAMMA_SCHEDULE,
                 gamma_sigmoid_steepness: float = GAMMA_SIGMOID_STEEPNESS,
                 sample_weight: np.ndarray | None = None,
                 monitor_metric: str = "val_directional_accuracy_metric",
                 checkpoint_monitor: str | None = None):
    """
    V22: `sample_weight` (None por default -- `equity_specialist` no pasa
    nada acá, comportamiento IDÉNTICO a V6) permite el peso por-muestra de
    Directiva 3 (magnitud de PnL, ver `build_asset_dataset`) vía el
    argumento nativo de Keras en `model.fit`. `monitor_metric` generaliza
    el `EarlyStopping`: `crypto_specialist`/triple_barrier monitorea
    `val_actionable_accuracy` (ver `ActionableAccuracy`) en vez de
    `val_directional_accuracy_metric` (que no existe en ese régimen, la
    cabeza ya no predice signo continuo) -- default sin cambios para
    preservar el comportamiento exacto de `equity_specialist`.

    V23.2 -- FIX BUG REAL (root cause del techo ~46% en crypto_specialist):
    `ModelCheckpoint` monitoreaba `val_loss` HARDCODEADO mientras
    `EarlyStopping` monitoreaba `monitor_metric` (`val_actionable_accuracy`
    en cripto) con `restore_best_weights=True`. Esto significa que las
    pesos EN MEMORIA al terminar `model.fit()` ya eran los de mejor
    actionable_accuracy (gracias a `restore_best_weights`) -- pero
    `keras.models.load_model(checkpoint_path)` los DESCARTABA y recargaba
    del disco el checkpoint de mejor `val_loss`, que en un problema de 3
    clases con `alpha` de rebalanceo NO necesariamente coincide con el de
    mejor actionable_accuracy (val_loss pondera las 3 clases incl. TIMEOUT,
    actionable_accuracy solo mide SL-vs-TP condicionado a señal). El modelo
    evaluado NUNCA fue el mejor que `EarlyStopping` encontró.
    `checkpoint_monitor=None` (default) preserva EXACTO el comportamiento
    de `equity_specialist` (`val_loss`/mode="min", cero cambio). Cuando se
    pasa `checkpoint_monitor=monitor_metric` (ver `run_regime_pipeline`,
    SOLO en la rama `triple_barrier`), `ModelCheckpoint` persigue la MISMA
    métrica que `EarlyStopping` -- ambos callbacks alineados.

    V27 -- `validation_data` (None por default -- `equity_specialist` no lo
    pasa, comportamiento IDÉNTICO: sigue usando `validation_split` nativo
    de Keras sobre `X_train` tal cual). Cuando se provee (SOLO
    `crypto_specialist`/triple_barrier, ver `run_regime_pipeline`), Keras
    usa ese validation set EXPLÍCITO en vez de cortar el último
    `validation_split` de `X_train` -- necesario porque `X_train` en ese
    régimen pasó por `_magnitude_warp_oversample` (bootstrap + permutación
    global), y cortar el último 10% de un array ya mezclado con copias
    aumentadas de sí mismo filtra información train->"validación" (ver
    docstring extenso en `run_regime_pipeline`, carve por-activo).
    """
    ckpt_monitor = checkpoint_monitor if checkpoint_monitor is not None else "val_loss"
    ckpt_mode = "min" if ckpt_monitor == "val_loss" else "max"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor=ckpt_monitor, mode=ckpt_mode, save_best_only=True, verbose=1
        ),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(
            monitor=monitor_metric, mode="max",
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
    # V27 -- `validation_data` explícito reemplaza `validation_split` por
    # completo cuando se provee (Keras no acepta ambos a la vez con
    # semántica bien definida) -- ver docstring arriba.
    fit_kwargs = (
        dict(validation_data=validation_data) if validation_data is not None
        else dict(validation_split=validation_split)
    )
    history = model.fit(
        [X_train, asset_id_train], y_train,
        epochs=epochs, batch_size=batch_size, shuffle=True,
        sample_weight=sample_weight,
        callbacks=callbacks, verbose=1,
        **fit_kwargs,
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


def evaluate_asset_triple_barrier(model: keras.Model, test: dict, ticker: str) -> dict:
    """
    Evaluación V22 -- `target_mode="triple_barrier"`. NO reconstruye
    precio (la cabeza ya no predice retorno continuo, ver `build_model`):
    reporta accuracy de clasificación 3-way (SL/TIMEOUT/TP) más DOS
    métricas con sentido de trading real --

        actionable_accuracy -- de los días en que el mercado SÍ tocó una
            barrera (clase verdadera != TIMEOUT), ¿cuántos acertó el
            modelo en clase? (ver `ActionableAccuracy`, mismo criterio
            usado como monitor de `EarlyStopping` en `train_model`).
        signal_precision -- de las señales que el modelo SÍ dispara
            (predicción != TIMEOUT), ¿cuántas son correctas? El análogo
            de "precision" en meta-labeling -- la métrica que de verdad
            importa para decidir si vale la pena OPERAR la señal (ver
            `pnl_backtest.py`, que reusa `y_class_idx`/`realized_return`
            de este mismo dict).
    """
    asset_id_col = test["asset_id"].reshape(-1, 1)
    y_pred_proba = model.predict([test["X"], asset_id_col], verbose=0)
    y_true_idx = np.argmax(test["y"], axis=-1)
    y_pred_idx = np.argmax(y_pred_proba, axis=-1)

    accuracy = float(np.mean(y_true_idx == y_pred_idx))

    actionable_mask = y_true_idx != TB_TIMEOUT_CLASS_IDX
    n_actionable = int(actionable_mask.sum())
    actionable_accuracy = (
        float(np.mean(y_pred_idx[actionable_mask] == y_true_idx[actionable_mask]))
        if n_actionable > 0 else float("nan")
    )

    signal_mask = y_pred_idx != TB_TIMEOUT_CLASS_IDX
    n_signals = int(signal_mask.sum())
    signal_precision = (
        float(np.mean(y_pred_idx[signal_mask] == y_true_idx[signal_mask]))
        if n_signals > 0 else float("nan")
    )

    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=[0, 1, 2])
    class_dist_true = {TB_CLASS_NAMES[i]: int((y_true_idx == i).sum()) for i in range(3)}
    class_dist_pred = {TB_CLASS_NAMES[i]: int((y_pred_idx == i).sum()) for i in range(3)}

    return {
        "ticker": ticker, "n_test": len(y_true_idx), "accuracy": accuracy,
        "actionable_accuracy": actionable_accuracy, "n_actionable": n_actionable,
        "signal_precision": signal_precision, "n_signals": n_signals,
        "confusion_matrix": cm, "class_dist_true": class_dist_true, "class_dist_pred": class_dist_pred,
        "y_true_idx": y_true_idx, "y_pred_idx": y_pred_idx, "y_pred_proba": y_pred_proba,
        "dates": test["dates"], "realized_return": test.get("y_return"),
    }


def train_meta_labeler(model: keras.Model, X_val: np.ndarray, asset_id_val: np.ndarray,
                        y_val: np.ndarray):
    """
    V27 -- Fase 2 capa 3 (meta-labeling, López de Prado cap. 3). Clasificador
    binario LIVIANO (RandomForest, ya en requirements.txt -- cero dependencia
    nueva) que aprende a distinguir cuándo confiar en la señal del modelo
    primario, en vez de mejorar su accuracy direccional cruda.

    Entrena EXCLUSIVAMENTE sobre `X_val`/`y_val` -- el slice crudo,
    cronológico, NUNCA usado para actualizar pesos del modelo primario (ver
    `validation_data_bundle` en `run_regime_pipeline`, separado ANTES del
    oversampling). Entrenarlo sobre train_fit sería circular: el primario ya
    memorizó esas muestras, el meta-labeler aprendería a confiar en TODO.

    Feature del meta-labeler = las 3 probabilidades softmax del primario
    (cuán segura está cada predicción) -- NO las features OHLCV crudas, para
    evitar que el meta-labeler reaprenda el mismo problema que el primario
    ya falló en resolver. Target = 1 si el primario acertó Y fue accionable
    (misma definición que `actionable_accuracy`), 0 si erró O el mercado
    hizo TIMEOUT (no había nada que acertar).

    Devuelve (None, None) si `X_val` es demasiado chico o si el primario
    nunca dispara señales accionables en ese slice -- el caller debe
    manejar ese caso sin filtrar nada (fail-open, no fail-silent).
    """
    from sklearn.ensemble import RandomForestClassifier

    y_pred_proba_val = model.predict([X_val, asset_id_val.reshape(-1, 1)], verbose=0)
    y_true_idx_val = np.argmax(y_val, axis=-1)
    y_pred_idx_val = np.argmax(y_pred_proba_val, axis=-1)

    meta_target = ((y_pred_idx_val == y_true_idx_val) & (y_true_idx_val != TB_TIMEOUT_CLASS_IDX)).astype(int)

    if len(meta_target) < 30 or meta_target.sum() == 0 or meta_target.sum() == len(meta_target):
        logger.warning("[META-LABEL] X_val insuficiente o degenerado (n=%d, positivos=%d) -- "
                       "meta-labeler NO entrenado, todas las señales primarias pasan sin filtrar.",
                       len(meta_target), int(meta_target.sum()))
        return None, None

    meta_clf = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    meta_clf.fit(y_pred_proba_val, meta_target)

    meta_pred_val = meta_clf.predict(y_pred_proba_val)
    val_precision_before = float(np.mean(y_pred_idx_val[y_pred_idx_val != TB_TIMEOUT_CLASS_IDX] ==
                                          y_true_idx_val[y_pred_idx_val != TB_TIMEOUT_CLASS_IDX])) \
        if (y_pred_idx_val != TB_TIMEOUT_CLASS_IDX).any() else float("nan")
    trusted_mask_val = (meta_pred_val == 1) & (y_pred_idx_val != TB_TIMEOUT_CLASS_IDX)
    val_precision_after = float(np.mean(y_true_idx_val[trusted_mask_val] == y_pred_idx_val[trusted_mask_val])) \
        if trusted_mask_val.any() else float("nan")
    logger.info("      [META-LABEL] fit sobre X_val (n=%d, positivos=%d). "
               "PrecisiónSeñal val: sin_filtro=%.1f%% -> con_filtro=%.1f%% (cobertura=%.1f%%)",
               len(meta_target), int(meta_target.sum()), val_precision_before * 100,
               val_precision_after * 100, 100 * trusted_mask_val.sum() / max(1, (y_pred_idx_val != TB_TIMEOUT_CLASS_IDX).sum()))
    return meta_clf, {"val_precision_before": val_precision_before, "val_precision_after": val_precision_after}


def evaluate_meta_labeler(meta_clf, y_pred_proba: np.ndarray, y_pred_idx: np.ndarray,
                           y_true_idx: np.ndarray) -> dict | None:
    """
    V27 -- aplica el meta-labeler ya entrenado sobre el TEST set (nunca visto
    ni por el primario ni por el meta-labeler). `meta_clf=None` (ver
    `train_meta_labeler`) -> devuelve None, el caller reporta solo las
    métricas del primario sin filtro -- fail-open.
    """
    if meta_clf is None:
        return None
    meta_pred = meta_clf.predict(y_pred_proba)
    signal_mask = y_pred_idx != TB_TIMEOUT_CLASS_IDX
    trusted_mask = (meta_pred == 1) & signal_mask
    n_trusted = int(trusted_mask.sum())
    n_signals = int(signal_mask.sum())
    return {
        "n_signals_raw": n_signals, "n_signals_trusted": n_trusted,
        "coverage": (n_trusted / n_signals) if n_signals else float("nan"),
        "signal_precision_filtered": (
            float(np.mean(y_true_idx[trusted_mask] == y_pred_idx[trusted_mask])) if n_trusted else float("nan")
        ),
    }


def compute_feature_attribution(model: keras.Model, X_test: np.ndarray, asset_id_test: np.ndarray,
                                  feature_names: list[str] | None = None) -> pd.Series:
    """
    Atribución por gradiente |∂ŷ/∂x| promediada sobre timesteps y muestras. NO son los pesos
    de BahdanauAttention (esos ponderan PASOS TEMPORALES, no canales de feature) — esta es la
    métrica correcta para cuantificar cuánto responde la salida a NEWS_SENTIMENT_SCORE frente
    a las otras variables del tensor.

    V27 -- FIX (ValueError: Length of values (30) != length of index (31)):
    el default anterior (`FEATURE_NAMES` de módulo) se construía en IMPORT
    TIME, capturando TECH_COLS ANTES de que `_apply_news_sentiment_gate()`
    lo mutara en `run_regime_pipeline`. La concatenación `[...] + TECH_COLS
    + [...]` crea una lista NUEVA, no una vista -- quedaba congelada en 31
    elementos aunque TECH_COLS mutara a 24. Ahora se recalcula acá, en el
    momento de la llamada (post-gate), leyendo TECH_COLS YA MUTADO.
    """
    if feature_names is None:
        feature_names = ["LOG_RETURN_1D"] + TECH_COLS + MACRO_TICKERS
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


def plot_triple_barrier_evaluation(results: dict[str, dict], regime_name: str, output_path: Path) -> None:
    """
    Grilla no-bloqueante V22 (mismo contrato que `plot_regime_evaluation`:
    JAMÁS `plt.show()`) para `target_mode="triple_barrier"` -- no hay
    precio que reconstruir, así que el panel por activo es matriz de
    confusión (heatmap manual, sin sumar `sklearn.metrics.
    ConfusionMatrixDisplay` como dependencia extra) + distribución de
    clases real vs. predicha.
    """
    tickers = list(results.keys())
    n = len(tickers)
    fig, axes = plt.subplots(n, 2, figsize=(11, 5 * n), squeeze=False)

    for idx, ticker in enumerate(tickers):
        r = results[ticker]
        ax_cm, ax_dist = axes[idx][0], axes[idx][1]

        cm = r["confusion_matrix"]
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax_cm.set_xticks(range(3))
        ax_cm.set_xticklabels(TB_CLASS_NAMES)
        ax_cm.set_yticks(range(3))
        ax_cm.set_yticklabels(TB_CLASS_NAMES)
        ax_cm.set_xlabel("Predicho")
        ax_cm.set_ylabel("Real")
        for i in range(3):
            for j in range(3):
                ax_cm.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                           color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
        ax_cm.set_title(f"{ticker} — Matriz de confusión (Acc={r['accuracy'] * 100:.1f}%, "
                        f"AccAccionable={r['actionable_accuracy'] * 100:.1f}%)")

        x = np.arange(3)
        width = 0.35
        true_counts = [r["class_dist_true"][c] for c in TB_CLASS_NAMES]
        pred_counts = [r["class_dist_pred"][c] for c in TB_CLASS_NAMES]
        ax_dist.bar(x - width / 2, true_counts, width, label="Real", color="#1f77b4")
        ax_dist.bar(x + width / 2, pred_counts, width, label="Predicho", color="#d62728")
        ax_dist.set_xticks(x)
        ax_dist.set_xticklabels(TB_CLASS_NAMES)
        ax_dist.set_title(f"{ticker} — Distribución de clases (n_test={r['n_test']}, "
                          f"PrecisiónSeñal={r['signal_precision'] * 100:.1f}%)")
        ax_dist.legend(fontsize=8)
        ax_dist.grid(alpha=0.3, axis="y")

    fig.suptitle(f"KodaQuant V22 — {regime_name} — Triple Barrier (Test)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)   # NUNCA plt.show() — mismo contrato que plot_regime_evaluation.
    plt.close(fig)
    logger.info("      Gráfico de evaluación -> %s", output_path)


# ============================================================
# AUDITORÍA PURGED WALK-FORWARD (Fase 2, capa 1 -- ver purged_fold_bounds)
# ============================================================
def calibrate_class_bias(model: keras.Model, X_val: np.ndarray, asset_id_val: np.ndarray,
                          y_val: np.ndarray, class_idx: int,
                          bias_grid: tuple[float, ...] = tuple(round(b, 2) for b in np.arange(0.0, 0.32, 0.02))):
    """
    V27 -- ajuste de UMBRAL post-hoc, cero reentrenamiento. Busca, sobre
    `X_val` (cola cronológica separada de train, NUNCA vista en pesos), el
    bias aditivo a `y_pred_proba[:, class_idx]` que maximiza el RECALL
    MACRO (promedio simple de recall por clase, sin ponderar por soporte --
    a propósito: es la única métrica que castiga por igual un recall=0%
    en la clase minoritaria, que es EXACTAMENTE el síntoma que se está
    corrigiendo). El bias elegido se aplica de forma IDÉNTICA en test --
    la calibración es el ÚNICO uso permitido de X_val; ni una fila de test
    participa de esta búsqueda.
    """
    from sklearn.metrics import recall_score

    y_pred_proba_val = model.predict([X_val, asset_id_val.reshape(-1, 1)], verbose=0)
    y_true_idx_val = np.argmax(y_val, axis=-1)

    best_bias, best_macro_recall = 0.0, -1.0
    for bias in bias_grid:
        adjusted = y_pred_proba_val.copy()
        adjusted[:, class_idx] += bias
        y_pred_idx_adj = np.argmax(adjusted, axis=-1)
        macro_recall = recall_score(y_true_idx_val, y_pred_idx_adj, labels=[0, 1, 2],
                                    average="macro", zero_division=0)
        if macro_recall > best_macro_recall:
            best_bias, best_macro_recall = bias, macro_recall

    baseline_pred = np.argmax(y_pred_proba_val, axis=-1)
    baseline_macro_recall = recall_score(y_true_idx_val, baseline_pred, labels=[0, 1, 2],
                                         average="macro", zero_division=0)
    logger.info("      [BIAS-CAL] clase=%s  bias=%.2f  recall_macro val: sin_bias=%.1f%% -> con_bias=%.1f%%",
               TB_CLASS_NAMES[class_idx], best_bias, baseline_macro_recall * 100, best_macro_recall * 100)
    return best_bias


def run_cpcv_audit(ticker: str, k: int = 5, embargo_frac: float = 0.02,
                    epochs_per_fold: int = 20, exclude_timeout_from_train: bool = False) -> pd.DataFrame:
    """
    V27 -- re-entrena el MISMO build_model/build_asset_dataset k veces,
    cada vez con más historia de train y un bloque de test más adelante en
    el tiempo (walk-forward, prefijo creciente -- reutiliza `build_asset_
    dataset` SIN modificarlo, pasándole distintos `train_ratio` sobre
    slices `df.iloc[:test_end]`), purgando+embargando la fuga de horizonte
    con el mismo mecanismo que ya usa el split único de producción. Da una
    DISTRIBUCIÓN de score OOS en vez de un solo número.

    Un solo path por fold (walk-forward), NO CPCV combinatorio completo --
    ver costo real en el docstring de `purged_fold_bounds`. `epochs_per_
    fold` deliberadamente bajo (20 vs 100/180 de producción): esto audita
    VARIANZA entre folds, no reemplaza el modelo servido. NUNCA escribe a
    `services/kodaquant_models/` -- checkpoints de fold se borran al vuelo.
    """
    regime_name = next((r for r, cfg in REGIMES.items() if ticker in cfg["tickers"]), None)
    if regime_name is None:
        raise ValueError(f"[CPCV] '{ticker}' no pertenece a ningún régimen de REGIMES.")
    target_mode = REGIMES[regime_name]["target_mode"]

    logger.info("[CPCV] Descargando %s (régimen=%s)...", ticker, regime_name)
    all_market_data = download_all([ticker], MACRO_TICKERS, REGIMES[regime_name].get("period", PERIOD),
                                   cache_tag=f"cpcv_{ticker}", macro_period=PERIOD)
    df = engineer_asset(all_market_data, ticker, MACRO_TICKERS)

    tb_params_ticker = None
    if target_mode == "triple_barrier" and CRYPTO_CALIBRATE_BARRIERS:
        tb_params_ticker = calibrate_crypto_triple_barrier({ticker: df}, [ticker], train_ratio=TRAIN_RATIO)[ticker]

    folds = purged_fold_bounds(len(df), k, embargo_frac)
    asset_to_id_audit = {ticker: 0}
    monitor_metric = "val_actionable_accuracy" if target_mode == "triple_barrier" else "val_directional_accuracy_metric"
    rows = []

    for i, (test_start, test_end, _embargo_end) in enumerate(folds):
        if test_start < LOOKBACK + BATCH_SIZE:
            logger.warning("[CPCV] fold %d/%d: train insuficiente (test_start=%d) -- salteado.", i + 1, k, test_start)
            continue
        df_fold = df.iloc[:test_end]
        train_ratio_fold = test_start / len(df_fold)

        train, test, feature_scaler, target_scaler = build_asset_dataset(
            df_fold, ticker, LOOKBACK, train_ratio_fold, asset_to_id_audit,
            target_mode=target_mode, tb_params=tb_params_ticker,
            exclude_timeout_from_train=exclude_timeout_from_train,
        )
        if len(train["X"]) == 0 or len(test["X"]) == 0:
            logger.warning("[CPCV] fold %d/%d: train o test vacío tras purga -- salteado.", i + 1, k)
            continue

        class_alpha = None
        if target_mode == "triple_barrier":
            train_class_idx = np.argmax(train["y"], axis=-1)
            class_counts = np.maximum(np.bincount(train_class_idx, minlength=3).astype(np.float64), 1.0)
            class_alpha = np.clip(len(train_class_idx) / (3.0 * class_counts), 0.25, 4.0).tolist()

        # V27 -- cola de validación tal cual `run_regime_pipeline` (crudo,
        # cronológicamente posterior a todo lo que entra a train_fit, cero
        # oversampling) -- ÚNICO propósito: calibrar el bias de umbral sin
        # tocar test.
        # V27 -- FIX: el intento anterior entrenaba sobre X_fit_fold (10%
        # menos datos que train completo) -- con un dataset ya chico eso
        # bastó para desestabilizar la convergencia (colapso degenerado
        # distinto en cada corrida, no comparable). Se entrena SIEMPRE
        # sobre train COMPLETO (idéntico a toda corrida previa, incluida la
        # que dio recall TIMEOUT=0%/std=0% que estamos diagnosticando). La
        # cola de calibración se recorta ADEMÁS, no EN VEZ de -- queda "en
        # muestra" (el modelo ya vio esas filas al entrenar) a propósito:
        # es un límite superior optimista de lo que el bias puede lograr.
        # Si ni ahí ayuda en test, no vale la pena perseguirlo más.
        n_val_fold = max(1, int(round(len(train["X"]) * VALIDATION_SPLIT))) if target_mode == "triple_barrier" else 0
        n_fit_fold = len(train["X"]) - n_val_fold
        X_val_fold = train["X"][n_fit_fold:]
        y_val_fold = train["y"][n_fit_fold:]
        asset_id_val_fold = train["asset_id"][n_fit_fold:]

        steps_per_epoch = max(1, len(train["X"]) // BATCH_SIZE)
        model, gamma_variable = build_model(
            n_timesteps=LOOKBACK, n_features=N_FEATURES, n_assets=1,
            steps_per_epoch=steps_per_epoch, regime_name=regime_name,
            target_mode=target_mode, class_alpha=class_alpha,
            focal_gamma=TRIPLE_BARRIER_FOCAL_GAMMA, label_smoothing=TRIPLE_BARRIER_LABEL_SMOOTHING,
        )
        fold_ckpt = DATA_DIR / f"_cpcv_audit_{ticker}_fold{i}.keras"  # descartable -- nunca lo lee quanti_engine.py
        _, model = train_model(
            model, train["X"], train["asset_id"].reshape(-1, 1), train["y"],
            checkpoint_path=fold_ckpt, epochs=epochs_per_fold, batch_size=BATCH_SIZE,
            gamma_variable=gamma_variable, gamma_max=GAMMA_MAX,
            gamma_warmup_epochs=min(GAMMA_WARMUP_EPOCHS, max(1, epochs_per_fold // 2)),
            gamma_schedule=GAMMA_SCHEDULE, gamma_sigmoid_steepness=GAMMA_SIGMOID_STEEPNESS,
            sample_weight=train.get("sample_weight"), monitor_metric=monitor_metric,
        )
        fold_ckpt.unlink(missing_ok=True)

        timeout_bias = 0.0
        if target_mode == "triple_barrier" and len(X_val_fold) >= 20:
            timeout_bias = calibrate_class_bias(model, X_val_fold, asset_id_val_fold, y_val_fold,
                                                class_idx=TB_TIMEOUT_CLASS_IDX)

        if target_mode == "triple_barrier":
            r = evaluate_asset_triple_barrier(model, test, ticker)
            score = r["actionable_accuracy"]
            # V27 -- por-clase, NO solo el agregado. AccAccionable puede
            # esconder un sesgo sistemático (ver BTC 2026-09-20: 62% recall
            # en SL, 43% en TP -- el 53.2% agregado no lo delataba) o un
            # colapso de una clase entera (ver ETH: TIMEOUT casi sin
            # predecir pese a alpha=1.914). `zero_division=0` porque un
            # fold puede no predecir ninguna muestra de una clase -- eso es
            # justamente lo que queremos poder VER, no que tire excepción.
            from sklearn.metrics import precision_recall_fscore_support
            _prec, _rec, _f1, _sup = precision_recall_fscore_support(
                r["y_true_idx"], r["y_pred_idx"], labels=[0, 1, 2], zero_division=0)
            fold_per_class = {
                f"{cls}_{metric}": val
                for cls, prec, rec, sup in zip(TB_CLASS_NAMES, _prec, _rec, _sup)
                for metric, val in [("precision", prec), ("recall", rec), ("support", int(sup))]
            }
            logger.info("      [CPCV] fold %d/%d por-clase -- SL: prec=%.0f%% rec=%.0f%% (n=%d)  "
                       "TIMEOUT: prec=%.0f%% rec=%.0f%% (n=%d)  TP: prec=%.0f%% rec=%.0f%% (n=%d)",
                       i + 1, k, _prec[0] * 100, _rec[0] * 100, int(_sup[0]),
                       _prec[1] * 100, _rec[1] * 100, int(_sup[1]),
                       _prec[2] * 100, _rec[2] * 100, int(_sup[2]))
            # V27 -- diagnóstico: ¿TIMEOUT nunca gana el argmax porque está
            # CERCA (problema de umbral, corregible sin reentrenar) o porque
            # su probabilidad está MUERTA (problema de capacidad/features,
            # necesita cambios reales)? Media de probabilidad ASIGNADA,
            # no de veces que ganó -- eso es lo que la matriz de confusión
            # no puede mostrar.
            _mean_proba = r["y_pred_proba"].mean(axis=0)
            logger.info("      [CPCV] fold %d/%d probabilidad media asignada -- SL=%.2f  TIMEOUT=%.2f  TP=%.2f",
                       i + 1, k, _mean_proba[0], _mean_proba[1], _mean_proba[2])

            if timeout_bias > 0.0:
                # V27 -- efecto del bias sobre TEST (nunca usado en
                # calibrate_class_bias) -- comparación honesta, no la
                # ganancia inflada que ya vimos en val con el meta-labeler.
                _adj_proba = r["y_pred_proba"].copy()
                _adj_proba[:, TB_TIMEOUT_CLASS_IDX] += timeout_bias
                y_pred_idx_bias = np.argmax(_adj_proba, axis=-1)
                actionable_mask_bias = r["y_true_idx"] != TB_TIMEOUT_CLASS_IDX
                actionable_acc_bias = (
                    float(np.mean(y_pred_idx_bias[actionable_mask_bias] == r["y_true_idx"][actionable_mask_bias]))
                    if actionable_mask_bias.any() else float("nan")
                )
                _prec_b, _rec_b, _, _ = precision_recall_fscore_support(
                    r["y_true_idx"], y_pred_idx_bias, labels=[0, 1, 2], zero_division=0)
                logger.info("      [CPCV] fold %d/%d CON bias=%.2f -- AccAccionable: %.1f%% -> %.1f%%  "
                           "recall TIMEOUT: %.0f%% -> %.0f%%  recall SL: %.0f%% -> %.0f%%  recall TP: %.0f%% -> %.0f%%",
                           i + 1, k, timeout_bias, score * 100, actionable_acc_bias * 100,
                           _rec[1] * 100, _rec_b[1] * 100, _rec[0] * 100, _rec_b[0] * 100,
                           _rec[2] * 100, _rec_b[2] * 100)
        else:
            r = evaluate_asset(model, test, ticker, target_scaler)
            score = r["directional_accuracy"]
            fold_per_class = {}

        logger.info("[CPCV] fold %d/%d  test=[%d:%d)  n_train=%d  n_test=%d  score=%.1f%%",
                    i + 1, k, test_start, test_end, len(train["X"]), len(test["X"]), score * 100)
        rows.append({"fold": i, "test_start": test_start, "test_end": test_end,
                     "n_train": len(train["X"]), "n_test": len(test["X"]), "score": score,
                     **fold_per_class})

    result = pd.DataFrame(rows)
    if len(result) >= 2:
        logger.info("[CPCV] %s -- %d/%d folds válidos. score: media=%.1f%%  std=%.1f%%  min=%.1f%%  max=%.1f%%",
                    ticker, len(result), k, result["score"].mean() * 100, result["score"].std() * 100,
                    result["score"].min() * 100, result["score"].max() * 100)
        # V27 -- ¿el sesgo por clase es CONSISTENTE entre folds (señal real
        # de que el modelo tiene un problema estructural) o varía fold a
        # fold (ruido de muestra, no vale la pena perseguirlo)? Un recall
        # de TIMEOUT bajo Y con poca varianza entre folds = el modelo
        # estructuralmente no aprende a abstenerse, no es mala suerte de
        # este test set puntual.
        if "TIMEOUT_recall" in result.columns:
            for cls in TB_CLASS_NAMES:
                rec_col = f"{cls}_recall"
                if rec_col in result.columns and result[rec_col].notna().any():
                    logger.info("      [CPCV] %s -- recall %-8s entre folds: media=%.0f%%  std=%.0f%%  min=%.0f%%  max=%.0f%%",
                               ticker, cls, result[rec_col].mean() * 100, result[rec_col].std() * 100,
                               result[rec_col].min() * 100, result[rec_col].max() * 100)
    else:
        logger.warning("[CPCV] %s -- menos de 2 folds válidos, sin distribución que reportar.", ticker)
    return result


# ============================================================
# ORQUESTACIÓN POR RÉGIMEN (Requerimiento 2: ciclos independientes y secuenciales)
# ============================================================
def run_regime_pipeline(regime_name: str, regime_cfg: dict) -> dict:
    _apply_news_sentiment_gate()
    """Ejecuta las Fases A-E completas para UN especialista y persiste sus artefactos."""
    tickers = regime_cfg["tickers"]
    oversample_factor = float(regime_cfg.get("oversample_factor", 1.0))
    arch = REGIME_ARCHITECTURE[regime_name]

    # V22 — target_mode bifurca el pipeline aguas abajo (dataset/modelo/
    # loss/evaluación/persistencia); default "regression" preserva la
    # ARQUITECTURA de equity_specialist igual a V6. V23.5 había roto la
    # paridad bit a bit con un sample_weight por volatilidad -- ROLLBACK en
    # V23.6 (ver changelog del módulo): equity_specialist vuelve a ser bit a
    # bit idéntico a V6/pre-V23.5, gradiente plano/uniforme.
    target_mode = regime_cfg.get("target_mode", "regression")
    tb_params_by_ticker: dict[str, dict] | None = None
    if target_mode == "triple_barrier":
        # V23.5 -- default POR TICKER (mismos valores de regime_cfg para
        # todos, fallback si CRYPTO_CALIBRATE_BARRIERS=False) -- ver
        # docstring de `calibrate_crypto_triple_barrier` para el esquema.
        _default_tb = {
            "pt_mult": regime_cfg.get("pt_mult", TRIPLE_BARRIER_PT_MULT),
            "sl_mult": regime_cfg.get("sl_mult", TRIPLE_BARRIER_SL_MULT),
            "max_holding_days": regime_cfg.get("max_holding_days", TRIPLE_BARRIER_MAX_HOLDING_DAYS),
            "vol_lookback": regime_cfg.get("vol_lookback", TRIPLE_BARRIER_VOL_LOOKBACK),
        }
        tb_params_by_ticker = {t: dict(_default_tb) for t in regime_cfg["tickers"]}

    # Rutas de producción estándar — sin sufijos `_experimental`: cada
    # régimen escribe DIRECTO en la carpeta que lee `quanti_engine.py` en
    # inferencia, idéntico en corrida local y en el cron de CI/CD.
    output_dir = MODELS_ROOT / regime_name
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_to_id = {t: i for i, t in enumerate(tickers)}

    logger.info("=" * 78)
    logger.info("REGIMEN: %s  |  target_mode=%s  |  Universo: %s  |  BiLSTM=%du  heads=%d  "
                "residual_extra=%s  dropout=%.2f  L2=%.0e",
                regime_name, target_mode, tickers, arch["lstm_units"], arch["mha_heads"],
                arch["extra_residual_block"], arch["dropout_rate"], arch["dense_l2_reg"])
    logger.info("=" * 78)

    logger.info("[1/5] Descargando %d activos (OHLCV) + %d factores macro (%s)...",
                len(tickers), len(MACRO_TICKERS), PERIOD)
    all_market_data = download_all(tickers, MACRO_TICKERS, regime_cfg.get("period", PERIOD), cache_tag=regime_name,
                                   macro_period=PERIOD)

    # V23 -- Ingeniería de features UNA vez por ticker, reutilizada tanto
    # por la calibración dinámica de barreras (train-only) como por
    # `build_asset_dataset` abajo -- cero recómputo redundante.
    df_assets = {t: engineer_asset(all_market_data, t, MACRO_TICKERS) for t in tickers}

    if target_mode == "triple_barrier" and CRYPTO_CALIBRATE_BARRIERS:
        logger.info("[1.5/5] Calibración dinámica Triple Barrier (V23.5, train-only, POR ACTIVO %s)...", tickers)
        tb_params_by_ticker = calibrate_crypto_triple_barrier(df_assets, tickers, train_ratio=TRAIN_RATIO)

    if target_mode == "triple_barrier":
        for t in tickers:
            tb_t = tb_params_by_ticker[t]
            logger.info("      Triple Barrier [%s]: pt_mult=%.2f  sl_mult=%.2f  max_holding_days=%d  vol_lookback=%d",
                        t, tb_t["pt_mult"], tb_t["sl_mult"], tb_t["max_holding_days"], tb_t["vol_lookback"])

    # ROLLBACK V23.6: el bloque de sample_weight por volatilidad de equity
    # (V23.5, Directiva 2) vivía acá -- purgado por completo, ver changelog
    # del módulo. `equity_specialist` ya no calcula ni loguea ningún peso.
    logger.info("[2/5] Construcción de dataset por activo...")
    feature_scalers, target_scalers = {}, {}
    X_train_parts, y_train_parts, asset_id_train_parts, sample_weight_train_parts = [], [], [], []
    # V27 (fix REAL de leakage train/validación -- ver docstring extenso más
    # abajo, junto al carve) -- SOLO se usan para target_mode="triple_barrier";
    # quedan vacíos y sin ningún efecto para "regression" (equity_specialist,
    # comportamiento 100% intacto).
    X_val_parts, y_val_parts, asset_id_val_parts, sample_weight_val_parts = [], [], [], []
    test_sets = {}

    for ticker in tickers:
        df_asset = df_assets[ticker]
        tb_params_ticker = tb_params_by_ticker[ticker] if tb_params_by_ticker is not None else None
        train, test, f_scaler, t_scaler = build_asset_dataset(
            df_asset, ticker, LOOKBACK, TRAIN_RATIO, asset_to_id,
            target_mode=target_mode, tb_params=tb_params_ticker,
        )
        feature_scalers[ticker] = f_scaler
        target_scalers[ticker] = t_scaler  # None por-ticker si target_mode="triple_barrier"
        test_sets[ticker] = test

        if target_mode == "triple_barrier":
            # V27 (fix REAL de leakage train/validación) -- carve CRONOLÓGICO
            # (últimas `VALIDATION_SPLIT` filas de ESTE activo, en el orden
            # temporal que `build_asset_dataset` ya garantiza) ANTES de que
            # `_magnitude_warp_oversample` toque nada. Root cause del síntoma
            # real observado (val_actionable_accuracy llegando a 87% durante
            # el entrenamiento, PrecisiónSeñal real cayendo a 44% en el test
            # final -- por debajo del baseline trivial): `_magnitude_warp_oversample`
            # genera copias bootstrap+magnitude-warp de ventanas de TRAIN y
            # las PERMUTA junto con los originales en un único array; el
            # `validation_split` nativo de Keras (`model.fit(..., validation_
            # split=0.1)`, ver `train_model`) toma el último 10% de ESE array
            # ya mezclado como "validación" -- una ventana original y su
            # copia magnitude-warped (misma etiqueta, misma ventana subyacente
            # con la amplitud apenas deformada) podían caer una en el 90% de
            # entrenamiento y la otra en el 10% de "validación". El modelo no
            # generalizaba: memorizaba la firma de la ventana original y
            # reconocía a su hermana deformada -- de ahí una val_accuracy
            # fantasiosa para un problema de dirección diaria de cripto, que
            # jamás se sostuvo contra el test set genuinamente nunca-visto.
            # Fix: la validación se separa ACÁ, de datos 100% crudos (sin
            # ninguna copia aumentada, cronológicamente POSTERIORES a todo lo
            # que entra a `_magnitude_warp_oversample`) -- ni una ventana de
            # validación tiene, ni puede tener, una copia hermana en train.
            n_val = max(1, int(round(len(train["X"]) * VALIDATION_SPLIT)))
            n_fit = len(train["X"]) - n_val
            X_val_parts.append(train["X"][n_fit:])
            y_val_parts.append(train["y"][n_fit:])
            asset_id_val_parts.append(train["asset_id"][n_fit:])
            sample_weight_val_parts.append(train["sample_weight"][n_fit:])
            train = dict(
                X=train["X"][:n_fit], y=train["y"][:n_fit], asset_id=train["asset_id"][:n_fit],
                last_price=train["last_price"][:n_fit], dates=train["dates"][:n_fit],
                sample_weight=train["sample_weight"][:n_fit], y_return=train["y_return"][:n_fit],
                y_class_idx=train["y_class_idx"][:n_fit],
            )

        X_train_parts.append(train["X"])
        y_train_parts.append(train["y"])
        asset_id_train_parts.append(train["asset_id"])
        if target_mode == "triple_barrier":
            sample_weight_train_parts.append(train["sample_weight"])
        # ROLLBACK V23.6: la rama "regression" que aquí ensamblaba el
        # sample_weight por volatilidad de equity fue purgada -- ver
        # changelog del módulo. equity_specialist ya no acumula pesos.
        if target_mode == "triple_barrier":
            logger.info("      %-10s train_fit=%5d  train_val=%5d  test=%5d",
                        ticker, len(train["X"]), n_val, len(test["X"]))
        else:
            logger.info("      %-10s train=%5d  test=%5d", ticker, len(train["X"]), len(test["X"]))

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    asset_id_train_flat = np.concatenate(asset_id_train_parts, axis=0)
    # ROLLBACK V23.6: target_mode="regression" (equity_specialist) ya NUNCA
    # usa sample_weight -- gradiente plano/uniforme, sample_weight=None,
    # bit a bit idéntico a V6/pre-V23.5. Solo triple_barrier (cripto,
    # Directiva 3: magnitud de PnL + uniqueness) sigue pesando muestras.
    _uses_sample_weight = target_mode == "triple_barrier"
    sample_weight_train_flat = (
        np.concatenate(sample_weight_train_parts, axis=0) if _uses_sample_weight else None
    )
    logger.info("      Dataset %s combinado -> X_train %s", regime_name, X_train.shape)

    # V27 (fix REAL de leakage train/validación, ver docstring del carve
    # por-activo arriba) -- validation_data EXPLÍCITO, ensamblado de las
    # colas crudas por-activo, NUNCA tocado por `_magnitude_warp_oversample`.
    # `None` para target_mode="regression" -- equity_specialist sigue usando
    # el `validation_split` nativo de Keras exactamente como antes (ver
    # `train_model`), cero cambio de comportamiento ahí.
    validation_data_bundle = None
    if target_mode == "triple_barrier":
        X_val = np.concatenate(X_val_parts, axis=0)
        y_val = np.concatenate(y_val_parts, axis=0)
        asset_id_val_flat = np.concatenate(asset_id_val_parts, axis=0)
        sample_weight_val_flat = np.concatenate(sample_weight_val_parts, axis=0)
        asset_id_val = asset_id_val_flat.reshape(-1, 1)
        validation_data_bundle = ([X_val, asset_id_val], y_val, sample_weight_val_flat)
        logger.info("      Validación %s (cronológica, cruda, pre-oversampling) -> X_val %s",
                    regime_name, X_val.shape)

    assert X_train.shape[-1] == N_FEATURES, (
        f"Desalineación de features: X_train tiene {X_train.shape[-1]} columnas "
        f"pero N_FEATURES={N_FEATURES} (TECH_COLS={TECH_COLS})."
    )

    # V6 (Requerimiento 5) — balanceo de volumen SOLO sobre train (ver
    # `_magnitude_warp_oversample`); no-op exacto si oversample_factor<=1.0.
    # V22: `sample_weight_train_flat` viaja SINCRONIZADO con X/y/asset_id
    # (mismos índices de bootstrap/permutación) -- ver docstring de la función.
    n_before = len(X_train)
    X_train, y_train, asset_id_train_flat, sample_weight_train_flat = _magnitude_warp_oversample(
        X_train, y_train, asset_id_train_flat, factor=oversample_factor,
        sample_weight=sample_weight_train_flat,
    )
    if oversample_factor > 1.0:
        logger.info("      Oversampling Magnitude-Warping (factor=%.1fx, sigma=%.2f, knots=%d): X_train %d -> %d muestras",
                    oversample_factor, MAGNITUDE_WARP_SIGMA, MAGNITUDE_WARP_KNOTS, n_before, len(X_train))
    asset_id_train = asset_id_train_flat.reshape(-1, 1)

    # V22 — `class_alpha` (CategoricalFocalCrossentropy, ver build_model):
    # balanceo de clases fiteado sobre la distribución REAL de train
    # post-oversampling (fórmula de sklearn.utils.compute_class_weight,
    # 'balanced': n_samples / (n_classes * count_c) -- ver docstring de
    # TRIPLE_BARRIER_FOCAL_GAMMA arriba, deliberadamente separado del
    # sample_weight de magnitud/PnL).
    class_alpha = None
    if target_mode == "triple_barrier":
        train_class_idx = np.argmax(y_train, axis=-1)
        class_counts = np.maximum(np.bincount(train_class_idx, minlength=3).astype(np.float64), 1.0)
        class_alpha = np.clip(len(train_class_idx) / (3.0 * class_counts), 0.25, 4.0).tolist()
        logger.info("      Triple Barrier -- distribución de clases en train: SL=%d TIMEOUT=%d TP=%d -> alpha=%s",
                    int(class_counts[0]), int(class_counts[1]), int(class_counts[2]),
                    [f"{a:.3f}" for a in class_alpha])

    logger.info("[3/5] Construyendo y entrenando el modelo especialista '%s'...", regime_name)
    # V27 -- con `validation_data_bundle` explícito (triple_barrier), TODO
    # `X_train` post-oversampling es fit real -- ya no hay `validation_split`
    # interno de Keras restándole una porción (esa porción ahora es
    # `X_val`, separada ANTES del oversampling, ver arriba). `equity_
    # specialist` (regression) conserva la fórmula original intacta.
    n_fit_samples = len(X_train) if validation_data_bundle is not None else int(len(X_train) * (1 - VALIDATION_SPLIT))
    steps_per_epoch = max(1, n_fit_samples // BATCH_SIZE)
    model, gamma_variable = build_model(
        n_timesteps=LOOKBACK, n_features=N_FEATURES, n_assets=len(tickers),
        steps_per_epoch=steps_per_epoch, regime_name=regime_name,
        huber_delta=HUBER_DELTA, gamma_initial=GAMMA_INITIAL,
        variance_lambda=VARIANCE_LAMBDA, variance_cap=VARIANCE_CAP,
        gaussian_nll_beta=GAUSSIAN_NLL_BETA,
        log_var_min=LOG_VAR_MIN, log_var_max=LOG_VAR_MAX,
        log_var_l2_lambda=LOG_VAR_L2_LAMBDA, log_var_barrier_lambda=LOG_VAR_BARRIER_LAMBDA,
        target_mode=target_mode, class_alpha=class_alpha,
        focal_gamma=TRIPLE_BARRIER_FOCAL_GAMMA, label_smoothing=TRIPLE_BARRIER_LABEL_SMOOTHING,
    )
    model.summary(print_fn=logger.info)
    if target_mode == "triple_barrier":
        logger.info("      CategoricalFocalCrossentropy: gamma=%.1f  label_smoothing=%.2f  alpha=%s",
                   TRIPLE_BARRIER_FOCAL_GAMMA, TRIPLE_BARRIER_LABEL_SMOOTHING,
                   [f"{a:.3f}" for a in class_alpha])
    else:
        logger.info("      Curriculum gamma: %.1f -> %.1f en %d épocas (schedule=%s)",
                   GAMMA_INITIAL, GAMMA_MAX, GAMMA_WARMUP_EPOCHS, GAMMA_SCHEDULE)
    logger.info("      LR CosineDecayRestarts: init=%.1e primer_ciclo=%d steps (%d épocas) t_mul=%.1f m_mul=%.2f",
               LR_INITIAL, steps_per_epoch * COSINE_RESTART_EPOCH_PERIOD,
               COSINE_RESTART_EPOCH_PERIOD, COSINE_T_MUL, COSINE_M_MUL)

    checkpoint_path = output_dir / "model_v5_best.keras"
    _monitor_metric_val = ("val_actionable_accuracy" if target_mode == "triple_barrier"
                            else "val_directional_accuracy_metric")
    history, model = train_model(
        model, X_train, asset_id_train, y_train,
        checkpoint_path=checkpoint_path,
        epochs=regime_cfg.get("epochs", EPOCHS), batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
        validation_data=validation_data_bundle,
        gamma_variable=gamma_variable, gamma_max=GAMMA_MAX,
        gamma_warmup_epochs=GAMMA_WARMUP_EPOCHS, gamma_schedule=GAMMA_SCHEDULE,
        gamma_sigmoid_steepness=GAMMA_SIGMOID_STEEPNESS,
        sample_weight=sample_weight_train_flat,
        monitor_metric=_monitor_metric_val,
        # placeholder-anchor -- NO TOCAR el resto de este call, ver abajo
        # V23.4 -- REVERTIDO para equity_specialist (evidencia real, corrida
        # V23.3: alinear ModelCheckpoint a val_directional_accuracy_metric
        # empeoró 5 de 8 tickers -- AAPL 53.5->52.3, MSFT 52.3->51.3,
        # NVDA 52.7->52.1, GOOGL 52.7->51.7 -- vs solo 3 mejoraron. Causa:
        # a diferencia de cripto (donde val_loss pondera 3 clases + alpha,
        # objetivo CASI desconectado de actionable_accuracy), en la
        # regresión de equity val_loss (Gaussian NLL direccional) YA está
        # fuertemente correlacionado con DirAcc -- alinear el checkpoint a
        # una métrica discreta y RUIDOSA (accuracy binaria sobre un
        # val_split de solo 10%) generalizó peor que el val_loss suave que
        # ya traía V6. Mismo bug, pero el fix solo ayuda donde el
        # desacople objetivo/métrica es real (cripto); en equity el
        # checkpoint original YA era el correcto. Mantenido SOLO en
        # cripto, donde BTC AccAccionable 46.0->49.3% y ETH 46.1->46.5%
        # confirmaron la mejora.
        checkpoint_monitor=(_monitor_metric_val if target_mode == "triple_barrier" else None),
    )

    # V27 -- Fase 2 capa 3 (meta-labeling): entrena EXCLUSIVAMENTE si este
    # régimen usa triple_barrier (equity/regression no tiene noción de
    # "señal accionable" a filtrar). `meta_clf=None` es un resultado válido
    # (fail-open, ver docstring de train_meta_labeler) -- el resto del
    # pipeline sigue funcionando exactamente igual que hoy si eso pasa.
    meta_clf = None
    if target_mode == "triple_barrier":
        meta_clf, _meta_val_stats = train_meta_labeler(model, X_val, asset_id_val, y_val)

    if target_mode == "triple_barrier":
        logger.info("[4/5] Evaluación por activo (clasificación Triple Barrier SL/TIMEOUT/TP)...")
        results = {}
        for ticker in tickers:
            results[ticker] = evaluate_asset_triple_barrier(model, test_sets[ticker], ticker)
            r = results[ticker]
            # V27 (Directiva Head Quant) -- baseline trivial = predecir SIEMPRE
            # la clase mayoritaria REAL (SL/TP/TIMEOUT, la que sea) en este test
            # set. Si Acc del modelo queda POR DEBAJO de este número, el modelo
            # no superó ni la apuesta más tonta posible -- el problema no es
            # "falta de features"/"aprender más", es calibración/objetivo roto
            # (síntoma típico: alpha de clase mal ajustado, o early stopping
            # restaurando un checkpoint subóptimo). dist_pred (antes solo iba
            # al PNG, invisible en el log de texto) expone si el modelo
            # colapsó a predecir casi siempre UNA sola clase.
            majority_true_count = max(r["class_dist_true"].values())
            baseline_acc = majority_true_count / r["n_test"] if r["n_test"] else float("nan")
            logger.info("      %-10s Acc=%.1f%% (baseline trivial=%.1f%%)  AccAccionable=%.1f%%  "
                        "PrecisiónSeñal=%.1f%% (n_señales=%d)  dist_real=%s  dist_pred=%s",
                        ticker, r["accuracy"] * 100, baseline_acc * 100, r["actionable_accuracy"] * 100,
                        r["signal_precision"] * 100, r["n_signals"], r["class_dist_true"], r["class_dist_pred"])
            meta_r = evaluate_meta_labeler(meta_clf, r["y_pred_proba"], r["y_pred_idx"], r["y_true_idx"])
            if meta_r is not None:
                logger.info("      %-10s [META-LABEL] cobertura=%.1f%% (%d/%d señales) "
                           "PrecisiónSeñal filtrada=%.1f%% (vs %.1f%% sin filtro)",
                           ticker, meta_r["coverage"] * 100, meta_r["n_signals_trusted"],
                           meta_r["n_signals_raw"], meta_r["signal_precision_filtered"] * 100,
                           r["signal_precision"] * 100)
                results[ticker]["meta_label"] = meta_r
            if r["accuracy"] < baseline_acc:
                logger.warning(
                    "      %-10s Acc del modelo (%.1f%%) POR DEBAJO del baseline trivial "
                    "(predecir siempre '%s', %.1f%%) -- señal de calibración/objetivo roto, "
                    "NO de falta de features. Antes de agregar señal nueva: (1) revisar la "
                    "matriz de confusión completa en evaluation_chart.png (¿colapsó a una "
                    "sola clase?); (2) confirmar que `alpha` de CategoricalFocalCrossentropy "
                    "(calibrado sobre train) sigue representando la distribución real de "
                    "ESTE test; (3) confirmar que el checkpoint restaurado por early stopping "
                    "es el óptimo según la métrica de monitoreo real (%s), no un óptimo local "
                    "de una época temprana con pocas señales.",
                    ticker, r["accuracy"] * 100,
                    max(r["class_dist_true"], key=r["class_dist_true"].get), baseline_acc * 100,
                    _monitor_metric_val,
                )
        # Atribución por gradiente NO aplica acá (ver evaluate_asset_triple_barrier/
        # compute_feature_attribution): la suma de un softmax es constante=1, el
        # gradiente de `sum(preds)` que usa tf.GradientTape para un target no-escalar
        # sería ~0 en todo el tensor -- deliberadamente omitido, no un olvido.
        chart_path = output_dir / "evaluation_chart.png"
        plot_triple_barrier_evaluation(results, regime_name, chart_path)
    else:
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
        logger.info("      Atribución por gradiente (|∂ŷ/∂x|) — top 5 de %d variables:", len(attribution))
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
        "gk_uses_open": _GK_USED_OPEN,   # V27 -- ver _GK_USED_OPEN / quanti_engine._compute_garman_klass_vol
        "v28_cols": list(V28_COLS),      # V28 -- auditoría de bundle
        "v28_open_by_ticker": {t: bool(_V28_OPEN_BY_TICKER.get(t, False)) for t in tickers},  # V28 -- paridad Open train/serve
        "target_mode": target_mode,
    }
    if target_mode == "triple_barrier":
        # V23.5 -- esquema NUEVO: dict-de-dicts {ticker: {pt_mult, ...}},
        # ver docstring de módulo (punto 6) y de `calibrate_crypto_triple_
        # barrier` -- `quanti_engine.py` debe leer `tb_params[ticker]`.
        scalers_payload["tb_params"] = tb_params_by_ticker
        scalers_payload["tb_class_names"] = TB_CLASS_NAMES

    # AUDITORÍA RESCATE 2026-08 -- fix de alineación train/serve de
    # FRAC_DIFF_LOG_CLOSE (ver docstring de `_LAST_FFD_D_BY_TICKER` arriba):
    # persiste el `d` REAL calibrado en ESTA corrida para cada ticker de
    # ESTE régimen. `quanti_engine.py` lo usa directo en vez de recalibrar
    # sobre su ventana corta de inferencia; bundles viejos sin esta clave
    # (`.get("ffd_d_by_ticker", {})` en el lado de lectura) preservan el
    # comportamiento previo sin romperse.
    scalers_payload["ffd_d_by_ticker"] = {t: _LAST_FFD_D_BY_TICKER[t] for t in tickers if t in _LAST_FFD_D_BY_TICKER}
    with open(scalers_path, "wb") as f:
        pickle.dump(scalers_payload, f)

    if target_mode == "triple_barrier" and meta_clf is not None:
        # V27 -- artefacto SEPARADO del scalers_dict.pkl principal: falta de
        # este archivo debe degradar a "sin filtro" en quanti_engine.py, no
        # romper la carga del modelo entero. Wiring a producción es el
        # PRÓXIMO paso (no en este parche) -- por ahora esto queda
        # persistido y auditado con `--cpcv-audit`, sin tocar el motor
        # servido.
        meta_labeler_path = output_dir / "meta_labeler.pkl"
        with open(meta_labeler_path, "wb") as f:
            pickle.dump(meta_clf, f)
        logger.info("      Meta-labeler exportado -> %s", meta_labeler_path)

    checkpoint_path.unlink(missing_ok=True)  # artefacto intermedio; model_v5.keras es la fuente final

    logger.info("      Modelo  -> %s", model_path)
    logger.info("      Scalers -> %s", scalers_path)
    logger.info("      Chart   -> %s", chart_path)

    return {"model_path": model_path, "scalers_path": scalers_path, "results": results, "target_mode": target_mode}


# ============================================================
# ORQUESTACIÓN GLOBAL — dos ciclos independientes y secuenciales
# ============================================================
if __name__ == "__main__":
    # AUDITORÍA RESCATE 2026-08 -- este bucle iteraba SIEMPRE los dos
    # regímenes de `REGIMES`, sin forma de acotarlo -- correr
    # `python train_kodaquant_v5.py` para "solo retrain de crypto" (rescate
    # post-incidente, iteración rápida de alpha cripto) reentrenaba TAMBIÉN
    # `equity_specialist`, violando el congelamiento estricto declarado
    # sobre ese régimen (52.7%/56.3% DirAcc fuera de muestra). `--regime`
    # (mismo nombre/patrón que ya usan `walk_forward_eval.py`/
    # `pnl_backtest.py`) permite acotar la corrida a UN régimen sin tocar el
    # otro; default="all" preserva EXACTO el comportamiento previo (cron
    # semanal `entrenamiento_semanal.yml` sigue invocando este script sin
    # argumentos y sigue reentrenando ambos).
    import argparse
    _cli_parser = argparse.ArgumentParser(description="Entrenamiento KodaQuant V5 -- todos los regímenes o uno solo.")
    _cli_parser.add_argument("--regime", default="all", choices=["all", *REGIMES.keys()],
                              help="Régimen a (re)entrenar. 'all' (default) = comportamiento histórico, ambos regímenes.")
    _cli_parser.add_argument("--cpcv-audit", default=None, metavar="TICKER",
                              help="Auditoría Purged Walk-Forward de UN ticker (ver run_cpcv_audit). No entrena ni sobreescribe ningún artefacto de producción.")
    _cli_parser.add_argument("--cpcv-k", type=int, default=5, help="Folds para --cpcv-audit.")
    _cli_parser.add_argument("--cpcv-epochs", type=int, default=20, help="Épocas por fold para --cpcv-audit.")
    _cli_parser.add_argument("--cpcv-exclude-timeout", action="store_true",
                              help="Excluye filas TIMEOUT del train de cada fold (ver build_asset_dataset). Test intacto.")
    _cli_args = _cli_parser.parse_args()
    if _cli_args.cpcv_audit:
        run_cpcv_audit(_cli_args.cpcv_audit, k=_cli_args.cpcv_k, epochs_per_fold=_cli_args.cpcv_epochs,
                       exclude_timeout_from_train=_cli_args.cpcv_exclude_timeout)
        raise SystemExit(0)
    _regimes_to_run = dict(REGIMES) if _cli_args.regime == "all" else {_cli_args.regime: REGIMES[_cli_args.regime]}
    if _cli_args.regime != "all":
        logger.warning("[--regime] Corrida ACOTADA a '%s' -- el resto de los regímenes (%s) NO se tocan "
                        "(ni datos, ni modelo, ni scalers).", _cli_args.regime,
                        ", ".join(r for r in REGIMES if r != _cli_args.regime))

    # V9 (CI/CD) — UNA vez por corrida, antes de tocar cualquier régimen
    # (ver purge_data_cache: purgar dentro del bucle borraría el .parquet
    # que el régimen anterior de esta misma corrida acaba de escribir).
    if FORCE_FRESH_DATA:
        purge_data_cache()

    summary = {}
    for regime_name, regime_cfg in _regimes_to_run.items():
        summary[regime_name] = run_regime_pipeline(regime_name, regime_cfg)

    logger.info("=" * 78)
    logger.info("ENTRENAMIENTO V5 COMPLETO — %d especialistas persistidos en %s",
               len(summary), MODELS_ROOT)
    for regime_name, info in summary.items():
        if info.get("target_mode") == "triple_barrier":
            avg_actionable = np.mean([r["actionable_accuracy"] for r in info["results"].values()])
            avg_signal_prec = np.mean([r["signal_precision"] for r in info["results"].values()])
            logger.info("  %-18s AccAccionable promedio=%.1f%%  PrecisiónSeñal promedio=%.1f%% (Triple Barrier)",
                        regime_name, avg_actionable * 100, avg_signal_prec * 100)
        else:
            avg_dir_acc = np.mean([r["directional_accuracy"] for r in info["results"].values()])
            logger.info("  %-18s edge direccional promedio = %.1f%%",
                        regime_name, avg_dir_acc * 100)
    logger.info("=" * 78)
# %%