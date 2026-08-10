# services/quanti_engine.py
"""
Quanti AI Engine — KodaQuant Terminal
======================================
Inferencia matemática 100% Keras 3 / TensorFlow NATIVO — carga in-process
del especialista V5 correspondiente al régimen del ticker consultado
(`kodaquant_models/equity_specialist/model_v5.keras` o
`.../crypto_specialist/model_v5.keras`, CERO ONNX Runtime en este proceso)
+ voz generativa vía Groq LPU en la nube (`llama-3.3-70b-versatile`, SDK
oficial `groq`, cliente `AsyncGroq`), con Circuit Breaker a fallback
matemático si la API de Groq no responde. Soporte de inferencia local
(llama.cpp/Metal) retirado permanentemente — ver GROQ_API_KEY más abajo.

Requisitos locales (Mac): `pip install keras tensorflow scikit-learn pandas
yfinance httpx transformers torch` — sin re-entrenar nada, sin Colab. Cada
especialista V5 (modelo + `scalers_dict.pkl`) vive en su propia carpeta bajo
`services/kodaquant_models/<regimen>/`, ver `REGIME_TICKERS`/`_regime_for_ticker`.

Pipeline de inferencia (replica EXACTA del notebook de entrenamiento):
    scalers.pkl (pickle)
        -> feature_scalers[ticker]  (MinMaxScaler, fit en train)
        -> target_scalers[ticker]   (StandardScaler sobre log-returns)
        -> asset_to_id[ticker]      (embedding categórico)
    yfinance (live) -> engineer_asset() -> ventana (lookback, n_features)
        -> feature_scaler.transform() -> tensor (1, lookback, n_features)
    keras.Model(..., training=True) -> log-return escalado
        -> target_scaler.inverse_transform() -> log-return real (r_hat)
        -> P_hat = P_prev * exp(r_hat)   (reconstrucción autoregresiva)

Motor predictivo AUTOREGRESIVO (sliding window). En cada paso del bucle:
(1) la ventana histórica reciente entra al modelo Keras, (2) el modelo
predice el retorno logarítmico de T+1 — con `Dropout(0.4)` MANTENIDO VIVO
vía `training=True` (MC Dropout real, ver `_forecast_asset`) —, (3) ese
log-return se perturba con ruido gaussiano calibrado a la volatilidad
histórica REAL del activo (`VOLATILITY_INJECTION_*`, ver `_forecast_asset`)
cuando la inyección está activa, (4) el precio resultante se anexa al
final de la ventana y el dato más viejo se descarta, (5) se repite hasta
completar `steps`. Con la inyección activa, el path devuelto es señal del
modelo + ruido calibrado con volatilidad real — NO una inferencia 100%
determinista de la red en cada punto; `variance_source` en cada forecast
lo declara explícitamente. Ver `_forecast_asset`.

La IA generativa (Quanti) SOLO narra los números ya calculados por este
motor (modelo + volatilidad inyectada); tiene prohibido recalcularlos o
inventar cifras adicionales (ver REGLA INQUEBRANTABLE en
_build_system_prompt).

Contrato dinámico con el Command Center (frontend):
    generate_quanti_strategy() acepta analysis_mode / target_asset
    (selección de activo), forecast_horizon_days (horizonte de
    proyección) y risk_profile / risk_score (perfil de riesgo). El
    `risk_score` (0-100) es ahora la ÚNICA fuente de verdad matemática
    del split Plan A / Plan B — ver `_resolve_risk_split`. `risk_profile`
    queda como etiqueta narrativa/contextual, no participa en el cálculo.
"""

# --- PARCHE macOS Intel: bug de build en el wheel de curl_cffi ---------
# `_wrapper.abi3.so` referencia `_SCDynamicStoreCopyProxies` sin linkear
# SystemConfiguration.framework (confirmado vía `otool -L`, sin esa
# dependencia listada). Se precarga el framework en el namespace plano
# del proceso ANTES de que `yfinance` importe `curl_cffi` internamente,
# satisfaciendo el símbolo sin recompilar el wheel roto.
import ctypes as _ctypes
try:
    _ctypes.CDLL("/System/Library/Frameworks/SystemConfiguration.framework/SystemConfiguration")
except OSError:
    pass  # Linux en prod: el framework no existe ni hace falta
# -------------------------------------------------------------------------

import asyncio
import json
import math
import os
import pickle
import re
import hashlib
import threading
import time
import traceback
import logging
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator

logger = logging.getLogger("kodaquant.quanti_engine")
# FIX REAL (el motivo de que ningún logger.info se viera en terminal):
# ni este módulo ni main.py llamaban a logging.basicConfig() — el root
# logger de Python queda en WARNING por defecto, así que TODO logger.info()
# (el tuyo y el mío) se silenciaba en el "handler de último recurso" sin
# excepción ni rastro. print() sí aparecía porque ignora por completo el
# nivel de logging — de ahí que vieras std(cursor_prices) pero nunca la
# auditoría dual-head. Idempotente: si algo más arriba en el stack (uvicorn,
# gunicorn) ya configuró el root logger, basicConfig() es un no-op y no lo
# pisa.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger.setLevel(logging.INFO)

# --- PARCHE HuggingFace tokenizers: deadlock en el fork de Uvicorn --------
# `transformers` (cargado perezosamente por `_get_finbert_pipeline` en
# services/data_pipeline.py) trae un tokenizer Rust que paraleliza con su
# propio pool de hilos. Bajo un worker de Uvicorn que hace fork DESPUÉS de
# que ese pool ya se inicializó en el proceso padre (o bajo llamadas
# concurrentes vía ThreadPoolExecutor, mismo patrón que
# `_finbert_init_lock` en data_pipeline.py), el pool nativo puede quedar en
# un estado inconsistente tras el fork y colgar el proceso. Debe fijarse
# ANTES de cualquier import que pueda cargar `tokenizers` transitivamente —
# por eso vive aquí (tras `import os`, antes de `from services.data_pipeline
# import get_daily_news_sentiment` más abajo) y no dentro de data_pipeline.py.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# -------------------------------------------------------------------------

import numpy as np
import pandas as pd
import requests as _requests
import yfinance as yf

# FIX: el radar dispara ~10-15 tickers en paralelo (ThreadPoolExecutor), y
# cada uno hace 2 llamadas yf.download() (_build_features) + 1 yf.Ticker()
# (sentiment) -- 20-30+ requests concurrentes a query2.finance.yahoo.com.
# El pool por-host default de urllib3 es 10 conexiones: al superarlo se
# descartaban conexiones ("Connection pool is full, discarding connection")
# y Yahoo empezaba a devolver respuestas vacías/JSON inválido bajo esa
# carga, tumbando el radar entero. Una sesión compartida con pool más
# grande, pasada a TODAS las llamadas yfinance, resuelve ambos síntomas.
_YF_SESSION = _requests.Session()
_yf_adapter = _requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
_YF_SESSION.mount("https://", _yf_adapter)
_YF_SESSION.mount("http://", _yf_adapter)

# FIX #2: el pool más grande no alcanzó -- Yahoo sigue devolviendo cuerpo
# vacío ("Expecting value: line 1 column 1") cuando ~15 tickers disparan
# yf.download/yf.Ticker EN SIMULTÁNEO (rate-limit del lado de Yahoo, no de
# nuestro pool). _YF_CONCURRENCY acota cuántas llamadas a Yahoo corren
# REALMENTE al mismo tiempo (el resto espera, en vez de dispararse todas
# juntas); el retry con backoff absorbe los rechazos transitorios que
# igual ocurran dentro de ese límite.
_YF_CONCURRENCY = threading.BoundedSemaphore(4)


def _yf_call_with_retry(fn, *, attempts: int = 3, backoff_s: float = 0.8):
    """Ejecuta `fn` (una llamada yfinance) bajo el semáforo de concurrencia,
    con reintentos ante respuesta vacía/error transitorio de Yahoo."""
    with _YF_CONCURRENCY:
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                result = fn()
                if result is None or (hasattr(result, "empty") and result.empty):
                    raise ValueError("yfinance devolvió una respuesta vacía")
                return result
            except Exception as exc:  # noqa: BLE001 -- reintentamos cualquier fallo transitorio
                last_exc = exc
                if attempt < attempts - 1:
                    time.sleep(backoff_s * (attempt + 1))
        raise last_exc
from groq import (
    AsyncGroq,
    APIConnectionError as GroqAPIConnectionError,
    APIStatusError as GroqAPIStatusError,
    AuthenticationError as GroqAuthenticationError,
    RateLimitError as GroqRateLimitError,
)

# ---------------------------------------------------------------------------
# Alias de tipo SOLO para el type checker (Pylance). El símbolo `keras` de
# abajo es una variable de runtime reasignada dentro de un try/except (puede
# apuntar a `keras` standalone o a `tensorflow.keras` según el entorno) —
# Pylance no puede tratar una variable así como un módulo real del que
# extraer `.Model` en una anotación de tipo (`reportInvalidTypeForm`). Este
# bloque, evaluado únicamente por el type checker (TYPE_CHECKING es False en
# runtime), le da a Pylance un import ESTÁTICO real desde el que resolver
# `KerasModel`, totalmente desacoplado de la lógica dinámica de carga.
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    from keras import Model as KerasModel
else:
    KerasModel = Any

# ---------------------------------------------------------------------------
# Import de Keras 3 A PRUEBA DE BALAS (TF 2.16+ / Python 3.12 / venv macOS).
#
# `from keras import layers` puede reventar con
# "ImportError: cannot import name 'layers' from 'keras' (unknown location)"
# cuando el símbolo `keras` resuelto por Python es un namespace package roto
# (0 bytes / sin __init__.py real) en vez del paquete instalado — un
# `import keras` desnudo NO lanza error en ese caso, solo falla al pedirle
# un atributo. Por eso accedemos vía `keras.layers` (atributo), no vía
# `from keras import layers` (import con nombre), y con try/except real en
# vez de asumir que el primer camino siempre funciona.
#
# Fallback: `tensorflow.keras` (shim oficial hacia el mismo Keras 3 en
# TF>=2.16). Si AMBOS caminos fallan, es un entorno con tensorflow/keras
# incompletos o desalineados — fallamos ruidosamente con el fix exacto en
# vez de dejar un ImportError críptico más abajo en el archivo.
_keras_primary_error: Exception | None = None
keras = None
layers = None

try:
    # Debe fijarse ANTES del primer `import keras` — Keras 3 resuelve el
    # backend al importarse. El notebook de entrenamiento corrió sobre
    # backend TensorFlow (Colab estándar); fijamos el mismo backend
    # explícitamente en vez de asumir el default del entorno.
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silencia el ruido de logs de TF en consola

    import keras as _keras_standalone
    _ = _keras_standalone.layers  # fuerza la resolución real del atributo, no solo el import del nombre
    keras = _keras_standalone
    layers = _keras_standalone.layers
except Exception as exc:  # noqa: BLE001 — namespace package roto no siempre lanza ImportError "limpio"
    _keras_primary_error = exc

if keras is None or layers is None:
    try:
        import tensorflow as tf
        _ = tf.keras.layers  # idem: confirma el atributo antes de darlo por bueno
        keras = tf.keras
        layers = tf.keras.layers
    except Exception as exc2:  # noqa: BLE001
        raise ImportError(
            "No se pudo cargar Keras 3 en este entorno (ni `keras` standalone "
            "ni el shim `tensorflow.keras`). En tu venv (macOS / Python 3.12) "
            "reinstala ambos paquetes EMPAREJADOS y en ese orden:\n"
            "    pip uninstall -y keras tensorflow tf-keras tensorflow-macos tensorflow-metal\n"
            "    pip install --upgrade 'tensorflow>=2.16,<3' 'keras>=3,<4'\n"
            f"Fallo #1 (import keras / keras.layers): {_keras_primary_error!r}\n"
            f"Fallo #2 (import tensorflow / tensorflow.keras): {exc2!r}"
        ) from exc2

# ---------------------------------------------------------------------------
# CAUSA RAÍZ DEL SILENT HANG — Metal/GPU despachado desde un hilo que no es
# el principal. `_build_mc_dropout_bridge` fue auditado nodo por nodo contra
# la topología real de `equity_specialist` (MHA 8h/128u + bloque residual
# extra) reconstruyendo el grafo en un entorno de control: el recorrido
# `node.arguments.fill_in()` termina limpio, sin recursión sin cota y sin
# excepción — NO es la causa. La causa real: en este proceso, absolutamente
# TODA inferencia Keras (la verificación de `_verify_stochastic_variance` Y
# cada paso del bucle MC Dropout de `_forecast_asset`) corre exclusivamente
# dentro de `loop.run_in_executor(None, _forecast_asset, ...)` — un hilo de
# `ThreadPoolExecutor`, nunca el hilo principal (ver `_build_investment_plans`
# más abajo). Con `tensorflow-metal` instalado (macOS, ver bloque de arriba),
# TensorFlow detecta y usa el GPU Metal por defecto; despachar el PRIMER
# kernel Metal de un modelo desde un hilo secundario es un deadlock conocido
# y documentado del plugin `tensorflow-metal` (el command queue de Metal no
# es re-entrante de forma segura entre hilos para ese primer dispatch) — no
# lanza excepción, solo cuelga el proceso para siempre, exactamente el
# síntoma reportado: "[ARTIFACT LOAD]" se imprime (CPU, antes de tocar el
# grafo) y después silencio total. Como hay UN solo dispositivo GPU
# compartido por proceso, un solo hilo atascado ahí bloquea cualquier otra
# inferencia Keras subsiguiente — de ahí que "cuelgue el servidor" entero
# para datos en vivo y predicciones, no solo esa request puntual.
#
# FIX — CERO ambigüedad, CERO dependencia del hilo que dispare la primera
# llamada: se fuerza CPU-only ANTES de que exista un solo `keras.Input` o
# `model.load()` en el proceso (acá, inmediatamente tras resolver el import
# de Keras/TensorFlow, antes de que cualquier otro código del módulo pueda
# tocar un tensor). `set_visible_devices([], "GPU")` debe correr antes de
# que TensorFlow inicialice su lista de dispositivos — si corriera más tarde
# (ej. dentro de `_get_keras_model`), ya sería tarde para una llamada
# concurrente que arrancó antes. Modelo objetivo (128 unidades, batch de
# Monte Carlo ~100 filas): CPU resuelve un forward pass en milisegundos —
# cero costo real de rendimiento a cambio de eliminar la clase ENTERA de
# deadlocks de GPU/threading, nativo (sin librerías nuevas), agnóstico a la
# topología de cualquiera de los dos especialistas.
try:
    import tensorflow as _tf_device_guard
    _tf_device_guard.config.set_visible_devices([], "GPU")
    logging.getLogger("kodaquant.quanti_engine").info(
        "[DEVICE GUARD] GPU (Metal/CUDA) deshabilitada — inferencia Keras fijada a CPU "
        "(mitigación de hang de threading en tensorflow-metal; ver docstring de este bloque)."
    )
except Exception as _device_guard_exc:  # noqa: BLE001 — nunca tumbar el arranque por esto
    logging.getLogger("kodaquant.quanti_engine").warning(
        "[DEVICE GUARD] No se pudo forzar CPU-only en TensorFlow (%r) — si el proceso "
        "corre con tensorflow-metal y cuelga tras '[ARTIFACT LOAD]', esta es la causa.",
        _device_guard_exc,
    )
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Configuración — rutas y endpoints
# ---------------------------------------------------------------------------

# Resolución de rutas ANCLADA al propio archivo (no al cwd del proceso que
# arranca el servidor) — `attention_bilstm_global.keras` y `scalers.pkl`
# viven en la MISMA carpeta que este módulo (ej. backend/services/).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# V5: persistencia segregada por régimen (ver train_kodaquant_v5.py,
# Requerimiento 3) — reemplaza el modelo global único V4.
MODELS_ROOT = Path(_THIS_DIR) / "kodaquant_models"
REGIME_TICKERS: dict[str, tuple[str, ...]] = {
    "equity_specialist": ("AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "SPY"),
    "crypto_specialist": ("BTC-USD", "ETH-USD"),
}


def _regime_for_ticker(ticker: str) -> str:
    """Resuelve a qué especialista V5 pertenece un ticker. Falla ruidosamente si no está en ningún régimen conocido."""
    normalized = ticker.strip().upper()
    for regime_name, tickers in REGIME_TICKERS.items():
        if normalized in tickers:
            return regime_name
    raise ValueError(
        f"'{ticker}' no pertenece a ningún régimen V5 conocido "
        f"({list(REGIME_TICKERS)}) — no hay .keras/scalers_dict.pkl que lo cubra."
    )


def _model_path(regime: str) -> str:
    return str(MODELS_ROOT / regime / "model_v5.keras")


def _scalers_path(regime: str) -> Path:
    return MODELS_ROOT / regime / "scalers_dict.pkl"


# data_pipeline.py vive junto a este módulo en services/ — provee
# get_daily_news_sentiment() (FinBERT + Finnhub/yfinance.news), la MISMA
# fuente de NEWS_SENTIMENT_SCORE que entrenó los especialistas V5.
from services.data_pipeline import get_daily_news_sentiment  # noqa: E402

# Fallback defensivo únicamente. El valor real de cada request viaja en
# `forecastHorizonDays` desde el Command Center y pasa por
# `_resolve_horizon_days()` antes de tocar el motor Keras.
DEFAULT_FORECAST_HORIZON_DAYS = 5
PLAN_A_TICKER = "SPY"  # Reserva — activo ancla de baja volatilidad

# --- V6: TECH_COLS — fuente de verdad ÚNICA del orden de columnas técnicas
# del tensor de 17 features (LOG_RETURN_1D + 11 técnicos + 5 macro). DEBE
# coincidir EXACTO (mismo orden Y misma definición matemática) con TECH_COLS
# en train_kodaquant_v5.py — el feature_scaler cargado desde
# scalers_dict.pkl fue ajustado con ESTE orden/definición, así que
# desalinearlo acá desplazaría cada columna en silencio dentro de
# feature_scaler.transform(). Se valida contra scalers["tech_cols"] en
# `_get_scalers()` (ver más abajo), para fallar ruidosamente antes de la
# primera inferencia en vez de producir un forecast matemáticamente
# incorrecto sin ninguna excepción visible.
#
# MIGRACIÓN A ESTACIONARIEDAD ESTRICTA (Reinicio Estructural): "PRICE"
# (nivel absoluto) se eliminó del tensor — ver LOG_RETURN_1D más abajo en
# `_fetch_feature_window`. EMA_20/MACD/MACD_SIGNAL/ATR_14/OBV (los 5
# técnicos NO auto-acotados, ver auditoría SOFT_CLIP_MARGIN abajo) se
# re-expresan como razones adimensionales invariantes a escala —
# EMA20_DEV_PCT, MACD_PCT, MACD_SIGNAL_PCT, ATR_PCT, OBV_ROC_20. RSI_14 /
# BB_WIDTH_20 / ADX_14 / STOCH_K_14 / SENTIMENT_SCORE / NEWS_SENTIMENT_SCORE
# quedan idénticos (ya acotados/adimensionales por construcción).
TECH_COLS = ["RSI_14", "EMA20_DEV_PCT", "MACD_PCT", "MACD_SIGNAL_PCT", "SENTIMENT_SCORE",
             "ATR_PCT", "BB_WIDTH_20", "OBV_ROC_20", "ADX_14", "STOCH_K_14",
             "NEWS_SENTIMENT_SCORE"]
OBV_ROC_LOOKBACK_DAYS = 20   # DEBE calzar con OBV_ROC_LOOKBACK_DAYS en train_kodaquant_v5.py
OBV_ROC_CLIP = 3.0           # ±300% — DEBE calzar con OBV_ROC_CLIP en train_kodaquant_v5.py

# --- Inyección de Volatilidad Estocástica (post inverse_transform) --------
# El log-return que sale del modelo (`r_hat`, ya en escala real) se
# perturba, EN CADA PASO del bucle autoregresivo, con ruido gaussiano
# `N(0, (VOLATILITY_INJECTION_FACTOR * sigma_hist)^2)`, donde `sigma_hist`
# es la desviación estándar de los log-returns REALES de los últimos
# `VOLATILITY_LOOKBACK_DAYS` (la volatilidad realizada del propio activo,
# no un número inventado). Esto ocurre DESPUÉS de `inverse_transform` — el
# ruido nunca toca el tensor de entrada del modelo (`scaled_window`), así
# que no hay riesgo de romper shapes ni el pipeline de escalado.
# NOTA DE HONESTIDAD: esto hace que el path proyectado sea, en parte, una
# simulación estocástica calibrada con volatilidad histórica real — no una
# inferencia 100% determinista de la red en cada punto. `variance_source`
# lo refleja explícitamente para que el frontend/LLM no lo presenten como
# "predicción pura del modelo" sin matices.
VOLATILITY_INJECTION_ENABLED = True
VOLATILITY_INJECTION_FACTOR = 0.85  # escala de sigma_hist inyectada por paso
VOLATILITY_LOOKBACK_DAYS = 63  # ~3 meses bursátiles para estimar sigma_hist

# --- Cinturón de seguridad contra el "tobogán" (colapso direccional) ------
# La fuga autorregresiva documentada arriba (RSI/EMA/MACD/ATR/BB/OBV/ADX/
# STOCH sintéticos alejándose del rango de entrenamiento paso a paso) tiene
# un clip de ENTRADA (`np.clip(scaled_windows, 0.0, 1.0)`, ver bucle más
# abajo) que evita que el tensor escalado se vaya a infinito — pero NO
# evita que, una vez el tensor queda pegado en 0.0/1.0 (una región fuera
# de distribución que el modelo nunca vio en train), la RED responda con
# un retorno sesgado sistemáticamente en una sola dirección en cada paso
# restante. Con ruido gaussiano genuino de por medio (`noise_batch` más
# abajo) eso normalmente se disimula como textura día-a-día real; pero si
# el sesgo del modelo (`r_hat_model_batch`) es mucho más grande que ese
# ruido, la señal domina y el path se ve casi como una línea recta — un
# retorno diario sostenido fuera de cualquier rango estadísticamente
# plausible para el activo (ej. SPY perdiendo ~15% en 5 sesiones).
#
# MODEL_SIGNAL_CLIP_SIGMAS acota la SEÑAL del modelo — no el ruido, no el
# precio final — a un múltiplo de `sigma_hist` (la volatilidad diaria REAL
# del propio activo, ya calculada más abajo). Un movimiento diario de 5
# desviaciones estándar ya es un evento de cola extremo (p < 1e-6 bajo
# normalidad); nada que el modelo prediga más allá de eso es señal
# genuina, es el artefacto de fuga descrito. El ruido de volatilidad
# inyectada actúa DESPUÉS de este clip y sigue sin tocarse, así que el
# ancho real del cono P5-P95 y la textura diaria del `sample_path` no se
# ven artificialmente comprimidos — solo se corta la cola sesgada del
# propio modelo, no la varianza real de la simulación.
MODEL_SIGNAL_CLIP_SIGMAS = 5

# --- Soft-clip del tensor escalado (reemplaza el hard np.clip de entrada) --
# DIAGNÓSTICO (auditoría step T+5): `np.clip(scaled_windows, 0.0, 1.0)` corta
# la fuga autorregresiva (evita extrapolación sin límite del MinMaxScaler),
# pero PINEA cualquier valor fuera de rango exactamente en 0.0 o 1.0. Cuando
# la fracción de features pineados sube a 40-50% en los últimos pasos
# (indicadores no acotados por diseño — EMA_20/ATR_14/OBV/BB_WIDTH/MACD,
# todos derivados de `cursor_prices` y por tanto atados al nivel de precio,
# a diferencia de RSI_14/STOCH_K_14 que son osciladores naturalmente
# acotados a [0,100]), trayectorias con divergencias MUY distintas en
# espacio real terminan produciendo el MISMO tensor de entrada (idéntico
# 0.0/1.0 repetido) -> la red ya no puede diferenciarlas -> `r_hat_model_batch`
# converge al mismo valor para todas ellas. La varianza de PRECIO sigue viva
# (la inyecta `noise_batch`, ver más abajo, que actúa después y en espacio
# real), pero la textura que aporta la propia red (su respuesta direccional
# a "qué tan sobrecomprado/sobrevendido" está cada trayectoria) se aplana:
# eso es la pérdida de textura reportada, no un fallo del Monte Carlo en sí.
#
# FIX: soft-clip en vez de hard-clip. Dentro de [0, 1] es la función
# IDENTIDAD exacta (cero cambio de comportamiento para el caso normal, donde
# la inmensa mayoría de los features cae dentro del rango de entrenamiento).
# Fuera de [0, 1], en vez de pinear, comprime el excedente con tanh hacia una
# banda estrecha (1 ± SOFT_CLIP_MARGIN) que el modelo NUNCA vio en train,
# pero de forma estrictamente monótona e inyectiva: dos valores distintos
# fuera de rango YA NO colapsan al mismo escalar, así que la red conserva
# sensibilidad direccional incluso en la cola. La función es C¹-continua en
# los empalmes x=0 y x=1 (la derivada de tanh((x-1)/m) en x=1 vale 1/m·m=1,
# igual que la pendiente de la identidad), así que no hay discontinuidad de
# pendiente que el modelo pueda "ver" como un artefacto numérico.
SOFT_CLIP_MARGIN = 0.05  # banda de overshoot permitida: salida ∈ (-0.05, 1.05)


def _soft_clip_unit_interval(x: np.ndarray, margin: float = SOFT_CLIP_MARGIN) -> np.ndarray:
    """
    Soft-clip vectorizado a [0, 1] con overflow comprimido por tanh en vez de
    pineado duro. Identidad estricta dentro de [0, 1]; fuera de rango,
    monótono e inyectivo (nunca dos entradas distintas colapsan al mismo
    valor de salida), a diferencia de `np.clip`. Ver bloque SOFT_CLIP_MARGIN
    arriba para el razonamiento completo.
    """
    below = x < 0.0
    above = x > 1.0
    out = x
    out = np.where(below, margin * np.tanh(x / margin), out)
    out = np.where(above, 1.0 + margin * np.tanh((x - 1.0) / margin), out)
    return out.astype(x.dtype, copy=False)


# --- Anchor-pull + decay de señal (estabilizador anti "Autoregressive Drift") ---
# SÍNTOMA POST-SOFT-CLIP: al dejar de pinear en 0.0/1.0, el tensor de entrada
# ya no oculta la deriva progresiva de los indicadores sintéticos — lo cual
# arregló la textura, pero destapó un problema DISTINTO y preexistente: nada
# en el bucle empujaba esos indicadores (EMA_20/EMA_fast/EMA_slow/
# MACD_SIGNAL/ATR_14/OBV/BB_WIDTH_20 — los TECH_COLS no auto-acotados por su
# propia fórmula; RSI/STOCH_K/ADX/SENTIMENT SÍ lo están, ver auditoría en el
# bloque SOFT_CLIP_MARGIN) de vuelta hacia el último estado REAL conocido. Si
# el modelo responde a "técnicos cada vez peores" con retornos cada vez más
# negativos (patrón de momentum perfectamente legítimo que puede haber
# aprendido de datos reales), el ciclo se retroalimenta: precio baja ->
# EMA/ATR empeoran -> modelo predice una baja aún mayor -> se repite. Con
# `cursor_prices = cursor_prices * np.exp(r_hat_batch)` (compounding
# multiplicativo correcto, NO es un bug — así se reconstruye precio desde
# log-retornos) ese sesgo por-paso se acumula geométricamente: un sesgo
# COMPARTIDO por las 100 trayectorias (no ruido i.i.d. real) explica que el
# síntoma reportado no sea solo P5 cayendo (cola pesimista esperable) sino
# P95 cayendo también — la MEDIA del batch se está desplazando, no solo su
# dispersión.
#
# CALIBRACIÓN — GAP ENCONTRADO (auditoría T+5, colapso a -14% con P95
# desplomándose PESE a ANCHOR_RETENTION/SIGNAL_DECAY_BASE ya activos): el
# bloque de soft-clip arriba ya identificaba a OBV y BB_WIDTH_20 como
# indicadores atados al nivel de precio y no auto-acotados — pero el
# anchor-pull nunca se aplicaba a `obv_arr`/`bb_width_arr` en el bucle, solo
# a EMA/MACD_SIGNAL/ATR. Es la brecha más plausible para un sesgo COMPARTIDO
# entre las 100 trayectorias tan temprano como T+5: `obv_arr` es una SUMA
# ACUMULADA (`obv_arr = obv_arr + sign(Δprecio) * volume_proxy`, ver bucle
# más abajo) sin ningún término de reversión propio — a diferencia de una
# EWM (que ya se amortigua sola, solo más lento), una racha de
# `direction_arr` con el mismo signo en la mayoría de las trayectorias (la
# consecuencia esperable de una señal direccional compartida y aún poco
# decaída en los primeros pasos) empuja a `obv_arr` fuera de rango de forma
# monótona y ACELERADA, sin nada que la traiga de vuelta. Ese OBV fuera de
# rango reingresa al tensor de la iteración siguiente (soft-clipeado a la
# banda de saturación), reforzando la misma dirección otra vez — exactamente
# la retroalimentación que este bloque existe para cortar, colándose por la
# única puerta que había quedado sin anchor-pull. FIX: `obv_arr` y
# `bb_width_arr` ahora se anclan igual que EMA/MACD_SIGNAL/ATR (mismo
# `ANCHOR_RETENTION`, ver aplicación en el bucle más abajo).
#
# DOS estabilizadores independientes, complementarios, NINGUNO toca el ruido
# gaussiano i.i.d. (`noise_batch`, más abajo) que le da al cono su ancho
# P5-P95 genuino — tocar eso reintroduciría el problema original de textura:
#
# 1) ANCHOR_RETENTION — "spring-back" hacia el último valor REAL observado.
#    Cada paso, el estado recursivo de esos 5 indicadores conserva una
#    fracción `ANCHOR_RETENTION` de su desviación respecto al ancla real y
#    PIERDE el resto — no se toca la fórmula EWM/recursiva en sí (sigue
#    siendo la misma matemática honesta, cero cifras inventadas), solo se
#    amortigua cuánto puede alejarse del último dato real confirmado en un
#    horizonte de apenas `steps` sesiones sin nueva confirmación de mercado.
#    Es una media-reversión de Ornstein-Uhlenbeck discreta de libro de texto,
#    simétrica (no sesga dirección), sobre la DESVIACIÓN, no sobre el precio.
#
# 2) SIGNAL_DECAY_BASE — atenúa la SEÑAL DIRECCIONAL del modelo
#    (`r_hat_model_batch`, el componente determinista/compartido) con
#    `SIGNAL_DECAY_BASE ** step_i`: paso 1 al 100% de confianza, cada paso
#    subsiguiente un poco menos, reflejando que cada sesión adicional aleja
#    más al modelo de su distribución de entrenamiento real. El ruido
#    (`noise_batch`) NO se decae — la dispersión sigue creciendo con
#    sqrt(steps) como un random walk genuino; solo se descuenta cuánto pesa
#    la CONVICCIÓN DIRECCIONAL compartida del modelo en el precio final,
#    exactamente la fuente del arrastre sistemático que tira de P5 Y P95 a
#    la vez. El cinturón MODEL_SIGNAL_CLIP_SIGMAS se conserva intacto como
#    techo absoluto de última instancia (defensa en profundidad).
ANCHOR_RETENTION = 0.90   # fracción de la desviación vs. ancla real que sobrevive cada paso
SIGNAL_DECAY_BASE = 0.65  # atenúa más rápido en horizontes cortos (5 pasos); revalidar con backtest direccional V6 antes de fijarlo en prod
DIRECTIONAL_ACCURACY_HOLDOUT = 0.53  # punto medio DA 51-55% — actualizar tras cada reentrenamiento
EDGE_SHRINKAGE_FACTOR = max(0.0, 2 * (DIRECTIONAL_ACCURACY_HOLDOUT - 0.5))  # ≈0.06: la magnitud solo merece la confianza que el edge real (sobre 50/50) le gana al azar



def _anchor_pull(state: np.ndarray, anchor: float, retention: float = ANCHOR_RETENTION) -> np.ndarray:
    """
    Media-reversión discreta hacia `anchor` (el último valor REAL observado,
    congelado antes del bucle): conserva `retention` de la desviación actual
    y descarta el resto. Identidad cuando `state == anchor`. No fabrica
    ningún dato — solo amortigua qué tan lejos puede derivar el ESTADO
    recursivo del último punto confirmado por mercado real.
    """
    return anchor + (state - anchor) * retention


# --- Dampening por OOD real (auditoría SPY: oob_frac=45.6% en T+1, 100%
# data real, sin ningún paso sintético todavía) -----------------------------
# `_anchor_pull`/`SIGNAL_DECAY_BASE` amortiguan la deriva SINTÉTICA que el
# propio bucle acumula paso a paso — no dicen nada sobre un tensor que YA
# nace fuera de rango en T+1, antes de que el bucle exista. `oob_frac`
# medido en T+1 (ver diagnóstico más abajo) es 100% atribuible a que
# `feature_scaler` se fiteó una única vez (`MinMaxScaler.fit(features[:
# split_idx])`, ver train_kodaquant_v5.py) sobre un tramo histórico que el
# mercado real ya superó — el activo hizo máximos que el scaler nunca vio.
#
# DESCARTADO explícitamente — "Dynamic Scaler Recalibration" (re-fitear o
# desplazar min/max del scaler en tiempo de inferencia, por ticker/por
# request): matemáticamente inválido para una red YA ENTRENADA. Los pesos
# de la Bi-LSTM aprendieron qué SIGNIFICA "RSI escalado = 0.85" en términos
# del rango de entrenamiento fijo — si el scaler se recalibra con la
# ventana actual, "0.85" pasa a representar un RSI crudo distinto sin que
# la red se entere. El síntoma (oob_frac) desaparecería de los logs, pero
# el problema real (la red extrapolando a una región que nunca vio) seguiría
# intacto y ahora invisible — cambia una anomalía honesta por una lectura
# artificialmente "normal". Es exactamente el tipo de maquillaje de cifras
# que este pipeline evita en todos sus otros componentes (ver
# `evaluate_asset`, `_confidence_score`); no corresponde hacer la excepción
# acá. La corrección estructural real —representación estacionaria
# (retornos/z-score móvil) en vez de nivel de precio absoluto para
# PRICE/EMA/MACD/ATR/OBV— existe y es superior a largo plazo, pero exige
# reentrenar desde cero (cambia qué significa cada columna de entrada) y
# por lo tanto no es un fix de calibración desplegable hoy.
#
# Lo que SÍ es matemáticamente válido sin retrain: si la red va a
# extrapolar de todos modos, no confiar en la MAGNITUD de su señal
# determinista en proporción a cuán lejos del manifold de entrenamiento
# arrancó — mismo principio que `SIGNAL_DECAY_BASE`, pero disparado por
# distancia a la distribución de entrenamiento en vez de por número de
# pasos sintéticos, y fijo para todo el horizonte (T+1 ya nace OOD; ese
# hecho no cambia paso a paso). El ruido i.i.d. (`noise_batch`) y el
# ancho P5-P95 NO se tocan — solo se recorta cuánto pesa el punto estimado
# del modelo, igual que con el decay por pasos.
REAL_OOD_DAMPENING_START = 0.15  # oob_frac en T+1 a partir del cual empieza a atenuarse la señal
REAL_OOD_DAMPENING_FULL = 0.50   # oob_frac en T+1 a partir del cual la atenuación llega a su piso
REAL_OOD_DAMPENING_FLOOR = 0.10  # piso multiplicativo — recorte agresivo bajo saturación extrema; > 0 a propósito: preserva el SIGNO (dirección) que la red detecta, nunca lo invierte ni lo anula a cero (ver auditoría SPY: floor=0.35 dejó -11.94%->-9.27%, compounding vía technicals recursivos hace la relación sub-lineal — floor más bajo pega más fuerte en la magnitud sin tocar de qué lado del cero cae)


def _real_ood_dampening_factor(oob_frac_t0: float) -> float:
    """
    Factor multiplicativo ∈ [REAL_OOD_DAMPENING_FLOOR, 1.0], función lineal
    de `oob_frac_t0` (medido UNA sola vez, en T+1, sobre la ventana 100%
    real — ver captura en el bucle). 1.0 = sin penalización (input dentro
    de rango de entrenamiento). Decrece linealmente entre
    REAL_OOD_DAMPENING_START y REAL_OOD_DAMPENING_FULL hasta el piso.
    """
    if oob_frac_t0 <= REAL_OOD_DAMPENING_START:
        return 1.0
    if oob_frac_t0 >= REAL_OOD_DAMPENING_FULL:
        return REAL_OOD_DAMPENING_FLOOR
    span = REAL_OOD_DAMPENING_FULL - REAL_OOD_DAMPENING_START
    progress = (oob_frac_t0 - REAL_OOD_DAMPENING_START) / span
    return 1.0 - progress * (1.0 - REAL_OOD_DAMPENING_FLOOR)


# --- Simulación Monte Carlo / Intervalos de Confianza Dinámicos -----------
# En vez de un único camino estocástico, cada paso del bucle autoregresivo
# corre un FORWARD PASS BATCHEADO del modelo (batch = N_MONTE_CARLO_SIMULATIONS
# trayectorias en paralelo, una sola llamada a Keras por paso — no N llamadas
# por paso) más su propio ruido de volatilidad i.i.d. por trayectoria. Al
# final se colapsan las N trayectorias a 3 percentiles por fecha.
N_MONTE_CARLO_SIMULATIONS = 100
CONFIDENCE_LOWER_PERCENTILE = 5   # escenario pesimista
CONFIDENCE_MEDIAN_PERCENTILE = 50  # tendencia central (expected_path)
CONFIDENCE_UPPER_PERCENTILE = 95  # escenario optimista

# Plan A y Plan B se despachan EN PARALELO vía asyncio.gather + threads (ver
# _build_investment_plans), pero comparten el MISMO objeto `model` cacheado
# por `_get_keras_model`. Este lock serializa EXCLUSIVAMENTE la llamada de
# inferencia (`model([...], training=True)`) para que dos forward passes
# nunca pisen el mismo grafo/tensores a la vez — todo lo demás del bucle
# (yfinance, feature engineering, reconstrucción de precio) sigue corriendo
# en paralelo sin contención.
inference_lock = threading.Lock()

# --- Guardia de congelamiento — segunda línea de defensa, independiente ---
# del fix de raíz (CPU-only, ver bloque [DEVICE GUARD] arriba). El fix de
# raíz elimina la causa CONOCIDA (Metal desde hilo secundario); esta guardia
# cubre cualquier OTRA causa de hang silencioso (otro backend, otra GPU, un
# deadlock nativo distinto) convirtiéndola en una excepción explícita a los
# `_KERAS_CALL_TIMEOUT_SECONDS` en vez de un freeze infinito. Nativo — cero
# librerías nuevas, reutiliza `ThreadPoolExecutor` (ya importado arriba).
_KERAS_CALL_TIMEOUT_SECONDS = 15.0
_keras_call_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kodaquant-keras-call")


def _call_keras_with_hang_guard(fn, *args, **kwargs):
    """
    Ejecuta `fn(*args, **kwargs)` en un worker dedicado de 1 solo hilo
    (reutilizado entre llamadas mientras no cuelgue ninguna) y exige el
    resultado con timeout. Si `fn` no responde en
    `_KERAS_CALL_TIMEOUT_SECONDS`, lanza `TimeoutError` con diagnóstico
    explícito — Y descarta el executor completo: un hilo que quedó
    congelado a mitad de un forward pass (command queue de GPU corrupto,
    lock nativo tomado y nunca liberado) no es confiable para llamadas
    futuras, así que se abandona (daemon, no bloquea shutdown) y la
    siguiente llamada arranca un worker nuevo y limpio.
    """
    global _keras_call_executor
    future = _keras_call_executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=_KERAS_CALL_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        stuck_executor = _keras_call_executor
        _keras_call_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kodaquant-keras-call")
        stuck_executor.shutdown(wait=False, cancel_futures=False)
        raise TimeoutError(
            f"Forward pass de Keras excedió {_KERAS_CALL_TIMEOUT_SECONDS:.0f}s sin "
            "responder (hang silencioso). Worker aislado y descartado; la siguiente "
            "llamada usa un hilo nuevo. Si esto dispara pese al [DEVICE GUARD] "
            "CPU-only, la causa NO es Metal/GPU — audita a partir de este log."
        ) from exc

# Executor DEDICADO para I/O ligero (yfinance de sentimiento) — separado del
# executor default (`run_in_executor(None, ...)`) que usa `_forecast_asset`
# (Keras + Monte Carlo, bloqueante, varios segundos por ticker). Compartir un
# solo pool entre tareas pesadas de CPU y tareas livianas de red hace que el
# fetch de sentimiento quede en cola detrás de un forecast completo y dispare
# el Circuit Breaker por "timeout" cuando en realidad es contención de
# threads, no lentitud real de red (ver get_market_sentiment). Pool chico y
# propio = el sentimiento nunca espera detrás de un forward pass de Keras.
_SENTIMENT_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="kodaquant-sentiment")

# --- Groq LPU — inferencia generativa en la nube (SDK oficial `groq`) ----
# Reemplaza por completo al viejo transporte dual "http" (servidor GGUF
# aparte, llama-server/LM Studio) / "embedded" (llama-cpp-python + Metal
# in-process). Sin servidor propio que levantar, sin puerto que coordinar
# con FastAPI, sin build de Metal/CUDA — un solo cliente async oficial.
#
# GROQ_API_KEY es OBLIGATORIA. Si falta, el proceso debe fallar rápido y
# ruidosamente al importar este módulo (arranque del servidor) en vez de
# fallar recién en la primera consulta de un usuario real — ver el
# RuntimeError explícito más abajo.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY no está definida. Exportala en el entorno o en el "
        ".env del backend, ej.:\n"
        "    GROQ_API_KEY=gsk_xxx...\n"
        "Conseguila en https://console.groq.com/keys — sin esta clave el "
        "motor no puede inicializar el cliente AsyncGroq."
    )

# Modelo fijo y rígido — no configurable por entorno, para que ningún
# despliegue quede corriendo silenciosamente contra otro modelo por un
# .env desalineado.
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

# Tope de tokens de salida. Balanceado (1024-1500) para respuestas
# completas y estructuradas (RESUMEN + ESTRATEGIA DE RESERVA + ESTRATEGIA
# DE RIESGO, ver _build_system_prompt) sin arriesgar corte a mitad de
# frase ni gastar de más la cuota del Free Tier de Groq.
GROQ_MAX_OUTPUT_TOKENS = int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "1500"))

# Cliente async ÚNICO (singleton de módulo) — AsyncGroq usa httpx.AsyncClient
# internamente y es seguro reutilizar la misma instancia entre requests
# concurrentes de FastAPI; instanciarlo por request desperdiciaría el pool
# de conexiones. Está PROHIBIDO usar el cliente sincrónico `Groq` acá: con
# `Groq` (no async) cada llamada bloquearía el event loop de FastAPI.
_groq_client = AsyncGroq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Custom layer — DEBE ser idéntica bit a bit a la definición del notebook de
# entrenamiento (mismo package/nombre en el decorador, mismos sub-layers,
# mismo build/call) para que `keras.models.load_model` pueda reconstruir el
# grafo y calzar los pesos guardados en `attention_bilstm_global.keras`.
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="quanti")
class BahdanauAttention(layers.Layer):
    """Self-attention aditiva (Bahdanau) sobre la salida temporal de la BiLSTM."""

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


# ---------------------------------------------------------------------------
# V3 — Custom loss / metric. DEBEN ser idénticas bit a bit a
# entrenamiento.py (mismo package="quanti", mismo nombre de clase/función,
# misma lógica de `call`/`get_config`) para que `keras.models.load_model`
# pueda resolver "Unknown loss function: DirectionalHuberLoss" / "Unknown
# metric function: directional_accuracy_metric" al deserializar
# `attention_bilstm_global.keras`.
#
# NOTA: `_get_keras_model()` carga con `compile=False` (este motor solo
# hace forward passes vía `model(...)`, nunca `.fit()`/`.evaluate()`), así
# que en la práctica Keras 3 no necesita reconstruir la loss/optimizer para
# servir inferencia. Igual se registran acá como red de seguridad: distintas
# versiones de Keras 3 difieren en cuánto de `compile_config` tocan incluso
# con `compile=False`, y un futuro `compile=True` (ej. para
# `model.evaluate()` en un job de validación offline) debe encontrar ambos
# símbolos sin sorpresas.
#
# `DynamicGammaCallback` (curriculum learning de `gamma`) NO se replica
# acá a propósito: es un `keras.callbacks.Callback` que solo existe dentro
# del bucle de `model.fit()` — nunca se serializa como parte del grafo ni
# del artefacto `.keras`. El valor de `gamma` con el que se guardó el
# checkpoint queda FIJO como float estático dentro de
# `DirectionalHuberLoss.get_config()` (ver su docstring), que es
# exactamente el comportamiento correcto para inferencia.
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="quanti")
class DirectionalHuberLoss(keras.losses.Loss):
    """
    L(y, ŷ) = Huber_δ(y, ŷ) · (1 + γ · 1[sign(y) ≠ sign(ŷ)])

    Réplica exacta de la loss de entrenamiento — ver entrenamiento.py para
    la derivación completa. En inferencia (`compile=False`, sin `.fit()`)
    esta clase nunca se invoca: solo necesita existir para que
    `keras.models.load_model` pueda resolver el `class_name` guardado en el
    `compile_config` del artefacto si alguna vez se carga con `compile=True`.
    """

    def __init__(self, delta: float = 1.0, gamma=1.5,
                 name: str = "directional_huber", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = delta
        self.gamma = gamma if isinstance(gamma, keras.Variable) else keras.Variable(
            gamma, trainable=False, dtype="float32", name=f"{name}_gamma"
        )

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
        penalty = 1.0 + self.gamma * mismatch
        return keras.ops.mean(huber * penalty, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "delta": self.delta,
            "gamma": float(keras.ops.convert_to_numpy(self.gamma)),
        })
        return config


@keras.saving.register_keras_serializable(package="quanti")
def directional_accuracy_metric(y_true, y_pred):
    """Réplica exacta de la métrica de monitoreo de entrenamiento.py: % de aciertos de signo."""
    y_true = keras.ops.cast(y_true, "float32")
    y_pred = keras.ops.cast(y_pred, "float32")
    match = keras.ops.cast(
        keras.ops.equal(keras.ops.sign(y_true), keras.ops.sign(y_pred)), dtype="float32"
    )
    return keras.ops.mean(match)


# ---------------------------------------------------------------------------
# Carga perezosa y cacheada de artefactos pesados (una sola vez por proceso)
# ---------------------------------------------------------------------------

def _dropout_layer_names(model: KerasModel) -> list[str]:
    """
    Verificación honesta, no un supuesto: recorre el grafo cargado y confirma
    que el `Dropout(0.4)` del notebook sigue vivo en el modelo deserializado.
    A diferencia de un export a ONNX (que hornea Dropout -> Identity bajo
    `model.eval()`), un `.keras` nativo conserva la capa completa — pero
    igual lo comprobamos en vez de darlo por hecho.
    """
    return [layer.name for layer in model.layers if isinstance(layer, layers.Dropout)]


def _build_mc_dropout_bridge(model: KerasModel) -> KerasModel:
    """
    DIAGNÓSTICO RAÍZ (colapso de varianza / línea recta) — confirmado
    inspeccionando `config.json` DENTRO del propio artefacto `.keras` (no
    una suposición): el nodo Functional de cada capa `Dropout` del grafo
    entrenado quedó serializado con `inbound_nodes[0].kwargs ==
    {"training": False}` — un booleano concreto HORNEADO en tiempo de
    construcción del grafo en el notebook (ej. `layers.Dropout(0.4)(x,
    training=False)`), no un valor simbólico. Cuando una capa recibe un
    booleano literal (no `None`) como `training` dentro del Functional
    API, Keras lo fija PERMANENTEMENTE en ESE nodo — llamar después a
    `model([...], training=True)` sobre el modelo completo NO tiene ningún
    efecto sobre ese nodo puntual.

    FIX sin reentrenar y sin tocar el `.keras` original: se reutilizan los
    MISMOS objetos de capa (mismos pesos entrenados) para reconstruir el
    grafo Functional COMPLETO en nodos nuevos, forzando `training=True`
    únicamente en las capas `Dropout`. La versión anterior de este bridge
    hardcodeaba el tramo final (`dropout_regularizer -> post_fusion_dense
    -> return_head`), válido solo para la topología base (cripto) — con la
    topología inflada de equity (bloque residual extra
    `extra_residual_dense -> skip -> add -> ln -> dropout` insertado entre
    medio) esa cadena fija saltaba directo de `fusion_attn_asset` (dim
    ancha) a `post_fusion_dense` (dim angosta) y colapsaba con
    `ValueError` de shape. Este bridge es agnóstico a la topología: no
    conoce nombres de capa ni cantidad de capas Dropout, camina el grafo
    real capa por capa sin importar cuántos bloques residuales, ramas o
    capas multi-entrada/multi-salida tenga.

    MECANISMO — se indexa, para cada nodo entrante de cada capa del
    modelo original (`model.layers`, saltando `InputLayer`), su tensor de
    salida ORIGINAL -> (capa, nodo) que lo produjo (`producer_of`). Para
    reconstruir `model.output` se resuelve esa dependencia RECURSIVAMENTE
    en post-order (primero los tensores de entrada de un nodo, después el
    nodo mismo) — no se asume que `model.layers` ya viene en orden
    topológico estricto, así que el resultado es correcto sin importar
    cómo Keras haya ordenado esa lista internamente.

    Por cada nodo, `node.arguments.fill_in(tensor_dict)` — el MISMO
    mecanismo interno que usa Keras para EJECUTAR el grafo — reconstruye
    los args/kwargs EXACTOS con los que esa capa fue invocada en el
    notebook (posicionales, kwargs, tensores anidados en listas como
    `Add()([a, b])`, constantes no-tensor), sustituyendo cada tensor
    original por su contraparte ya reconstruida. Cero hardcoding de firmas
    de capa: lo que se invoca es literalmente "lo mismo que ya estaba",
    tensor por tensor.

    ÚNICO cambio funcional: si la capa es una instancia de
    `layers.Dropout` (cualquier nombre, cualquier posición, cualquier
    cantidad — la topología equity tiene DOS Dropout, la base tiene UNO),
    se fuerza `training=True` explícito en el nodo NUEVO. Todo lo demás
    (capas custom multi-salida como `BahdanauAttention`, `Concatenate`,
    `Add`, `LayerNormalization`, embeddings, etc.) se reinvoca sin tocar
    un solo kwarg propio.

    Ninguna capa se clona ni se reinicializa: se reutiliza el MISMO objeto
    Python de cada capa (mismos pesos entrenados), solo se le agrega un
    nodo Functional nuevo — el patrón nativo de "capa reutilizable /
    multi-nodo" de Keras (igual que un modelo "siamés"), aplicado ahora a
    la totalidad del grafo en vez de a un tramo fijo.
    """
    original_inputs = list(model.inputs)
    # Tensor de entrada original -> él mismo: el grafo nuevo arranca desde
    # los MISMOS `keras.Input` simbólicos, no se clonan.
    resolved: dict[int, Any] = {id(t): t for t in original_inputs}

    # id(tensor de salida ORIGINAL) -> (capa, nodo) que lo produjo. Se
    # indexa ANTES de invocar ninguna capa nueva, sobre el grafo intacto —
    # así una capa reutilizada más de una vez (>1 `_inbound_nodes`) queda
    # desambiguada nodo por nodo, no por nombre de capa.
    producer_of: dict[int, tuple[Any, Any]] = {}
    for layer in model.layers:
        if isinstance(layer, layers.InputLayer):
            continue
        for node in layer._inbound_nodes:
            outputs = node.output_tensors
            outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
            for tensor in outputs:
                producer_of[id(tensor)] = (layer, node)

    visited_nodes: set[int] = set()

    def _resolve_node(layer, node) -> None:
        if id(node) in visited_nodes:
            return
        visited_nodes.add(id(node))

        # Post-order: resuelve primero TODOS los tensores de entrada de
        # este nodo (recursión sobre su propio productor) antes de
        # reinvocar la capa — garantiza orden correcto sin depender de
        # cómo Keras haya poblado `model.layers`.
        for parent_tensor in node.arguments.keras_tensors:
            if id(parent_tensor) not in resolved:
                parent_layer, parent_node = producer_of[id(parent_tensor)]
                _resolve_node(parent_layer, parent_node)

        tensor_dict = {id(t): resolved[id(t)] for t in node.arguments.keras_tensors}
        args, kwargs = node.arguments.fill_in(tensor_dict)

        if isinstance(layer, layers.Dropout):
            kwargs = dict(kwargs)
            kwargs["training"] = True  # único override permitido — MC Dropout real

        new_output = layer(*args, **kwargs)

        original_output = node.output_tensors
        if isinstance(original_output, (list, tuple)):
            new_outputs_seq = new_output if isinstance(new_output, (list, tuple)) else [new_output]
            for orig_t, new_t in zip(original_output, new_outputs_seq):
                resolved[id(orig_t)] = new_t
        else:
            resolved[id(original_output)] = new_output

    def _resolve_structure(struct):
        if isinstance(struct, (list, tuple)):
            return type(struct)(_resolve_structure(item) for item in struct)
        if isinstance(struct, dict):
            return {key: _resolve_structure(value) for key, value in struct.items()}
        if id(struct) in resolved:
            return resolved[id(struct)]
        if id(struct) in producer_of:
            layer, node = producer_of[id(struct)]
            _resolve_node(layer, node)
            return resolved[id(struct)]
        return struct  # no es un tensor del grafo — constante, se deja intacta

    final_outputs = _resolve_structure(model.output)
    return keras.Model(inputs=original_inputs, outputs=final_outputs, name="mc_dropout_bridge")


def _zero_sample_from_input_shape(model: KerasModel) -> list[np.ndarray]:
    """
    Construye un batch de entrada dummy (ceros) con la forma EXACTA de
    `model.input_shape`, respetando el dtype real que usa `_forecast_asset`
    para cada tensor: `float32` para la ventana de features, `int32` para el
    id categórico del activo. Solo se usa para la verificación de
    estocasticidad de abajo — nunca participa en una predicción real.
    """
    shapes = model.input_shape
    if not isinstance(shapes, list):
        shapes = [shapes]
    samples = []
    for i, shape in enumerate(shapes):
        concrete_shape = tuple(1 if dim is None else dim for dim in shape)
        dtype = np.int32 if i == 1 else np.float32
        samples.append(np.zeros(concrete_shape, dtype=dtype))
    return samples


def _verify_stochastic_variance(model: KerasModel) -> bool:
    """
    Verificación defensiva — NO una suposición. Corre dos forward passes
    IDÉNTICOS (mismo input dummy) con `training=True` y compara bit a bit.
    Un bridge de MC Dropout correctamente reconstruido debe devolver dos
    salidas DISTINTAS (cada pasada muestrea su propia máscara de dropout);
    si salen idénticas, el `training=True` no está llegando realmente a
    ninguna capa Dropout activa (nombre de capa equivocado, capas
    downstream desalineadas con el grafo real, optimización que colapsó el
    nodo nuevo, etc.) — y eso debe quedar visible en el log ANTES de que se
    manifieste solo como una curva plana en el frontend, sin pista de causa.
    """
    try:
        sample_input = _zero_sample_from_input_shape(model)
        with inference_lock:
            out_1 = keras.ops.convert_to_numpy(
                _call_keras_with_hang_guard(model, sample_input, training=True)
            )
            out_2 = keras.ops.convert_to_numpy(
                _call_keras_with_hang_guard(model, sample_input, training=True)
            )
        return not np.allclose(out_1, out_2, atol=1e-9)
    except Exception as exc:  # noqa: BLE001 — la verificación jamás debe tumbar la carga del modelo
        print(f"⚠️ No se pudo verificar la estocasticidad del modelo cargado ({exc!r}).")
        return False


# ---------------------------------------------------------------------------
# Caché de artefactos en RAM (patrón Singleton por régimen) + locks de carga
# ---------------------------------------------------------------------------
# FIX DEADLOCK/I-O: `@lru_cache` protege con su Lock interno el ACCESO al
# dict de caché, pero NO serializa la ejecución de la función decorada —
# bajo cache-miss concurrente (Alpha Seeker escaneando N tickers del MISMO
# régimen vía asyncio.gather + run_in_executor, ver
# prediccion.py._scan_universe), cada hilo que llega antes de que el
# primero termine de poblar el caché dispara su PROPIO
# keras.models.load_model()/pickle.load() en paralelo — de ahí la avalancha
# de "[ARTIFACT LOAD]" duplicados en el mismo milisegundo. Con la GPU
# deshabilitada (Device Guard, más arriba) esas N deserializaciones
# completas del grafo (incl. la reconstrucción nodo-a-nodo del MC-Dropout
# bridge) compiten por CPU y saturan el proceso — el síntoma percibido
# como deadlock.
#
# Reemplazo: dict de caché en RAM + double-checked locking con UN Lock POR
# RÉGIMEN (no un lock global — preserva la independencia de carga
# equity/crypto ya documentada en el diseño original). Garantiza que
# `keras.models.load_model`/`pickle.load` corran EXACTAMENTE UNA VEZ por
# régimen durante todo el ciclo de vida del proceso: el primer hilo que
# adquiere el lock carga y publica en el dict; cualquier hilo que llegue
# después (con el lock libre u ocupado) encuentra el cache-hit y retorna
# sin tocar disco.
_model_cache: dict[str, tuple[KerasModel, tuple[str, ...]]] = {}
_scalers_cache: dict[str, dict] = {}
_model_load_locks: dict[str, threading.Lock] = {regime: threading.Lock() for regime in REGIME_TICKERS}
_scalers_load_locks: dict[str, threading.Lock] = {regime: threading.Lock() for regime in REGIME_TICKERS}


def _load_keras_model_from_disk(regime: str) -> tuple[KerasModel, tuple[str, ...]]:
    """
    Carga real (sin caché, sin lock) de `kodaquant_models/<regime>/model_v5.keras`.
    Body original de `_get_keras_model` V5 intacto — SOLO se invoca desde
    dentro de `_model_load_locks[regime]` en `_get_keras_model`, nunca
    directamente. `compile=False` porque solo hacemos forward passes
    (`model(...)`), nunca `.fit()`.

    Devuelve el "MC-Dropout bridge" (ver `_build_mc_dropout_bridge`) cuando
    el grafo original tiene `dropout_regularizer` congelado en
    `training=False` — en vez del `model` crudo, que seguiría siendo
    determinista para siempre sin importar qué `training` se le pase desde
    `_forecast_asset`.
    """
    model_path = _model_path(regime)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo Keras del régimen '{regime}' en: {model_path}")

    mtime = datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat(timespec="seconds")
    logger.info(
        "[ARTIFACT LOAD] regime=%s model_path=%s mtime=%s",
        regime, os.path.abspath(model_path), mtime,
    )
    model = keras.models.load_model(
        model_path,
        custom_objects={
            "BahdanauAttention": BahdanauAttention,
            "DirectionalHuberLoss": DirectionalHuberLoss,
            "directional_accuracy_metric": directional_accuracy_metric,
        },
        compile=False,
    )

    dropout_layers = tuple(_dropout_layer_names(model))
    if not dropout_layers:
        print(
            f"⚠️ MC Dropout NO disponible: el modelo del régimen '{regime}' cargado desde "
            f"{os.path.basename(model_path)} no tiene ninguna capa Dropout en su grafo. "
            "El motor autoregresivo seguirá siendo 100% real pero "
            "determinista bajo `training=True` — revisa que el .keras "
            "cargado sea el mismo que entrenó `dropout_regularizer` (0.4) "
            "en el notebook. CERO ruido artificial será inyectado como "
            "sustituto."
        )
        return model, dropout_layers

    try:
        inference_model = _build_mc_dropout_bridge(model)
    except Exception as exc:  # noqa: BLE001 — jamás fallar en silencio ni caer a ruido sintético
        print(
            f"⚠️ No se pudo reconstruir el bridge de MC Dropout ({exc!r}) — "
            "se usa `model` crudo. El bridge reconstruye TODO el grafo nodo a "
            "nodo (ver `_build_mc_dropout_bridge`); una excepción acá casi "
            "siempre indica un nodo cuyo `node.arguments.fill_in` no resuelve "
            "limpio (ej. una capa custom con un input no-KerasTensor en su "
            "firma de `call`). Si ninguna capa Dropout del grafo estaba "
            "congelada en `training=False`, esto es inofensivo y el `model` "
            "crudo ya es correcto; si sí lo estaba, el forecast volverá a ser "
            "determinista."
        )
        return model, dropout_layers

    if not _verify_stochastic_variance(inference_model):
        print(
            "🚨 ALERTA — el bridge de MC Dropout se construyó SIN excepción, pero "
            "dos forward passes idénticos con training=True devolvieron el MISMO "
            "resultado: la varianza sigue sin inyectarse pese al bridge. El bridge "
            "es agnóstico a nombres de capa — fuerza `training=True` en TODA "
            "instancia de `layers.Dropout` del grafo real (ver "
            "`_build_mc_dropout_bridge`) — así que si esto dispara, sospechá de "
            "`rate=0.0` en la(s) capa(s) Dropout del `.keras` cargado, o de que "
            "el grafo simplemente no tiene ninguna Dropout activa pese a lo que "
            "reportó `_dropout_layer_names` (`model.summary()` / "
            "`[l.name for l in model.layers]` para confirmar)."
        )

    return inference_model, dropout_layers


def _get_keras_model(regime: str) -> tuple[KerasModel, tuple[str, ...]]:
    """
    Singleton en RAM thread-safe: retorna el modelo Keras del régimen desde
    `_model_cache`, cargándolo desde disco (`_load_keras_model_from_disk`)
    EXACTAMENTE UNA VEZ por régimen durante todo el ciclo de vida del
    proceso. Double-checked locking sobre `_model_load_locks[regime]` — el
    fast-path (cache-hit) no toca ningún lock; solo el primer cache-miss
    por régimen paga el costo de `keras.models.load_model` + reconstrucción
    del MC-Dropout bridge, cualquier hilo concurrente que llegue mientras
    tanto espera el lock y reutiliza el resultado ya publicado (nunca
    dispara su propia carga redundante).
    """
    cached = _model_cache.get(regime)
    if cached is not None:
        return cached

    with _model_load_locks[regime]:
        cached = _model_cache.get(regime)
        if cached is not None:  # otro hilo ya cargó mientras esperábamos el lock
            return cached
        result = _load_keras_model_from_disk(regime)
        _model_cache[regime] = result
        return result


def _load_scalers_from_disk(regime: str) -> dict:
    """
    Carga real (sin caché, sin lock) de
    `kodaquant_models/<regime>/scalers_dict.pkl`. Body original de
    `_get_scalers` V5 intacto — SOLO se invoca desde dentro de
    `_scalers_load_locks[regime]` en `_get_scalers`, nunca directamente.
    """
    scalers_path = _scalers_path(regime)
    if not scalers_path.exists():
        raise FileNotFoundError(f"No se encontró el bundle de scalers del régimen '{regime}' en: {scalers_path}")

    suspicious = sorted(
        p.name for p in scalers_path.parent.glob("scalers*.pkl")
        if p.resolve() != scalers_path.resolve()
    )
    if suspicious:
        print(
            f"⚠️ Artefacto(s) duplicado(s) junto a {scalers_path.name} "
            f"(IGNORADOS, nunca se cargan): {suspicious}"
        )

    with open(scalers_path, "rb") as f:
        payload = pickle.load(f)

    mtime = datetime.fromtimestamp(scalers_path.stat().st_mtime).isoformat(timespec="seconds")
    logger.info(
        "[ARTIFACT LOAD] regime=%s scalers_path=%s mtime=%s tech_cols=%s",
        regime, scalers_path.resolve(), mtime, payload.get("tech_cols"),
    )

    # Validación defensiva V3: si scalers.pkl viene de un bundle con un
    # TECH_COLS distinto (ej. el V2 de 5 técnicos, sin ATR_14/BB_WIDTH_20/
    # OBV), el feature_scaler.transform() de _forecast_asset desplazaría
    # cada columna del tensor EN SILENCIO — mismo riesgo que un
    # desalineamiento de shape, pero sin lanzar ningún error propio. Se
    # compara nombre y orden exactos, no solo el conteo.
    saved_tech_cols = payload.get("tech_cols")
    if saved_tech_cols is not None and list(saved_tech_cols) != TECH_COLS:
        raise ValueError(
            f"Desalineación de TECH_COLS entre {scalers_path.name} "
            f"({list(saved_tech_cols)}) y este motor de inferencia "
            f"({TECH_COLS}). El .keras/scalers_dict.pkl del régimen '{regime}' "
            "no corresponden al pipeline V5 vigente — regenera ambos "
            "artefactos desde train_kodaquant_v5.py antes de servir inferencia."
        )
    if saved_tech_cols is None:
        print(
            f"⚠️ {scalers_path.name} ('{regime}') no incluye la clave 'tech_cols' "
            f"(bundle pre-V5) — se asume el orden hardcodeado {TECH_COLS} sin "
            "poder validarlo contra el artefacto real. Regenera scalers_dict.pkl "
            "con train_kodaquant_v5.py para una validación genuina."
        )

    return payload


def _get_scalers(regime: str) -> dict:
    """
    Singleton en RAM thread-safe para `scalers_dict.pkl` — mismo patrón de
    double-checked locking que `_get_keras_model`, con lock INDEPENDIENTE
    (`_scalers_load_locks[regime]`) para que la carga de scalers de un
    régimen nunca espere ociosamente a que termine la carga (más pesada)
    del modelo Keras de ese mismo régimen.
    """
    cached = _scalers_cache.get(regime)
    if cached is not None:
        return cached

    with _scalers_load_locks[regime]:
        cached = _scalers_cache.get(regime)
        if cached is not None:
            return cached
        payload = _load_scalers_from_disk(regime)
        _scalers_cache[regime] = payload
        return payload


# ---------------------------------------------------------------------------
# Numeric helpers — el backend es la única fuente de verdad matemática.
# ---------------------------------------------------------------------------

def _select_experience_level(budget_usd: float, experience_level: str) -> str:
    if budget_usd >= 150 and experience_level == "advanced":
        return "advanced"
    return "beginner"


def _fmt_money(value: float) -> str:
    return f"${value:,.2f} USD"


def _fmt_pct(value: float) -> str:
    pct = value * 100 if isinstance(value, float) and value <= 1 else value
    return f"{pct:.1f}%"


def _compute_exact_amounts(investment_plans: dict) -> dict:
    """
    Riesgo = Plan B, Reserva = Plan A. SIEMPRE deriva del split dinámico ya
    resuelto por `_resolve_risk_split` (ver `_build_investment_plans`) — jamás
    un 50/50 desacoplado del `risk_score` real del usuario. Este es el único
    punto de verdad para lo que ven las tarjetas "Riesgo" / "Reserva".
    """
    plan_a = investment_plans["plan_a"]
    plan_b = investment_plans["plan_b"]

    return {
        "riesgo_usd": round(float(plan_b["monto_usd"]), 2),
        "reserva_usd": round(float(plan_a["monto_usd"]), 2),
        "riesgo_pct": plan_b["pct"],
        "reserva_pct": plan_a["pct"],
    }


def _extract_market_signal(radar_data: dict) -> str:
    volatility = radar_data.get("volatility", radar_data.get("volatilidad", radar_data.get("volatility_index")))
    correlation = radar_data.get("correlation", radar_data.get("correlacion"))
    macro = radar_data.get("macro", radar_data.get("macro_context", radar_data.get("trend")))

    parts = []
    if volatility is not None:
        parts.append(f"volatilidad={volatility}")
    if correlation is not None:
        parts.append(f"correlación={correlation}")
    if macro is not None:
        parts.append(f"macro='{macro}'")

    return ", ".join(parts) if parts else "sin señales relevantes en el radar actual"


# ---------------------------------------------------------------------------
# Feature engineering — DEBE replicar bit a bit engineer_asset() del notebook
# ---------------------------------------------------------------------------

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


# --- SENTIMENT_SCORE: correlación de Pearson móvil activo/VIX --------------
# HALLAZGO F5 (fix de Data Skew Train/Serve): esta función calculaba antes
# un Z-score de MOMENTUM PROPIO del activo (log-return del día frente a su
# propia media/std móvil) — esa NO es la matemática que entrenó
# `scalers_dict.pkl`. `entrenamiento.py` (V4, ver docstring del módulo)
# redefinió SENTIMENT_SCORE como la correlación de Pearson móvil de
# `SENTIMENT_LOOKBACK_DAYS` sesiones entre el log-return del activo y el
# log-return de ^VIX (proxy de pánico/miedo de mercado, no de momentum
# propio). El `feature_scaler` fue ajustado (`.fit()`) sobre esa
# distribución acotada a [-1, 1] de un COEFICIENTE DE CORRELACIÓN — un
# Z-score de momentum (no acotado, cola larga) cae fuera de ese rango
# ajustado, satura la entrada de la red y colapsa el Monte Carlo
# autoregresivo. DEBE replicar bit a bit `compute_sentiment_score()` del
# notebook de entrenamiento. `SENTIMENT_LOOKBACK_DAYS` == `SENTIMENT_LOOKBACK`
# allá.
SENTIMENT_LOOKBACK_DAYS = 20


def _resolve_vix_ticker(macro_tickers: list[str]) -> str:
    """
    Localiza el ticker de VIX dentro de `macro_tickers` (persistido en
    `scalers_dict.pkl` como `scalers["macro_tickers"]`). SENTIMENT_SCORE
    requiere ESPECÍFICAMENTE el VIX como proxy de pánico/miedo de mercado —
    el resto de factores macro (ej. ^GSPC, ^TNX, GC=F, DX-Y.NYB) no
    participan de este cálculo. Falla ruidosamente (en vez de asumir un
    índice fijo) si el régimen cargado no incluye VIX entre sus macro
    tickers, para no desalinear en silencio el tensor de entrada.
    """
    for macro_ticker in macro_tickers:
        if "VIX" in macro_ticker.upper():
            return macro_ticker
    raise ValueError(
        f"SENTIMENT_SCORE requiere un ticker VIX en macro_tickers ({macro_tickers}) "
        "para replicar la correlación de Pearson móvil del entrenamiento — no se "
        "encontró ninguno en scalers_dict.pkl."
    )


def _compute_sentiment_score(
    price_series: pd.Series, vix_series: pd.Series, window: int = SENTIMENT_LOOKBACK_DAYS
) -> pd.Series:
    """Correlación de Pearson móvil (`window` sesiones) entre el log-return del activo y el de VIX."""
    asset_log_returns = np.log(price_series / price_series.shift(1))
    vix_log_returns = np.log(vix_series / vix_series.shift(1))
    corr = asset_log_returns.rolling(window=window, min_periods=window).corr(vix_log_returns)
    return corr.fillna(0.0).clip(-1.0, 1.0)


# --- V3: ATR_14 / BB_WIDTH_20 / OBV — DEBEN replicar bit a bit compute_atr()
# / compute_bb_width() / compute_obv() del notebook de entrenamiento.
ATR_PERIOD = 14
ATR_ALPHA = 1 / ATR_PERIOD
BB_WIDTH_WINDOW = 20
BB_WIDTH_NUM_STD = 2.0
# OBV es un ACUMULADO, no un oscilador acotado: este motor NO proyecta
# volumen futuro (solo precio), así que el volumen sintético de cada paso
# autoregresivo usa el promedio de los últimos N días REALES como único
# proxy disponible — ver `_compute_indicator_states` y el bucle en
# `_forecast_asset`. Documentado explícitamente, igual que
# VOLATILITY_INJECTION_* más arriba en este módulo.
OBV_VOLUME_PROXY_LOOKBACK_DAYS = 20


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    """ATR_14 — True Range con suavizado de Wilder (EWM). Réplica exacta de compute_atr() del notebook."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr.fillna(0.0)


def _compute_bb_width(series: pd.Series, window: int = BB_WIDTH_WINDOW, num_std: float = BB_WIDTH_NUM_STD) -> pd.Series:
    """BB_WIDTH_20 — ancho normalizado de Bandas de Bollinger. Réplica exacta de compute_bb_width() del notebook."""
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    width = (2 * num_std * rolling_std) / rolling_mean.replace(0.0, np.nan)
    return width.fillna(0.0)


def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV — On-Balance Volume. Réplica exacta de compute_obv() del notebook."""
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    return obv.fillna(0.0)


# --- V5: ADX_14 / STOCH_K_14 — DEBEN replicar bit a bit compute_adx() /
# compute_stochastic_k() de train_kodaquant_v5.py.
ADX_PERIOD = 14
STOCH_K_PERIOD = 14


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ADX_PERIOD) -> pd.Series:
    """ADX (Wilder). Réplica exacta de compute_adx() del notebook."""
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


def _compute_stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = STOCH_K_PERIOD) -> pd.Series:
    """%K estocástico. Réplica exacta de compute_stochastic_k() del notebook."""
    lowest_low = low.rolling(window=period, min_periods=period).min()
    highest_high = high.rolling(window=period, min_periods=period).max()
    denom = (highest_high - lowest_low).replace(0.0, np.nan)
    stoch_k = 100 * (close - lowest_low) / denom
    return stoch_k.fillna(50.0)


def _compute_adx_state(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ADX_PERIOD) -> dict:
    """
    Últimos valores de +DM/-DM suavizados (Wilder) + ADX real — estado
    necesario para continuar la EWM de ADX_14 dentro del bucle autoregresivo
    de `_forecast_asset` (reutiliza `atr_arr` ya trackeado ahí mismo como
    denominador de +DI/-DI, misma fórmula/periodo que este ATR interno).
    """
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    plus_dm_smooth = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    adx = _compute_adx(high, low, close, period)
    return {
        "plus_dm_smooth_prev": float(plus_dm_smooth.iloc[-1]) if pd.notna(plus_dm_smooth.iloc[-1]) else 0.0,
        "minus_dm_smooth_prev": float(minus_dm_smooth.iloc[-1]) if pd.notna(minus_dm_smooth.iloc[-1]) else 0.0,
        "adx_prev": float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else 0.0,
    }


def _compute_indicator_states(close_df: pd.DataFrame, ticker: str, vix_ticker: str) -> dict:
    """
    Estado recursivo final de RSI/EMA/MACD/SENTIMENT/ATR/BB_WIDTH/OBV/ADX_14/
    STOCH_K_14/NEWS_SENTIMENT_SCORE sobre el historial real — necesario para
    actualizarlos incrementalmente dentro del bucle autoregresivo en vez de
    congelarlos (ver _forecast_asset).

    `close_df` es el frame COMPLETO devuelto por `_fetch_feature_window`
    (incluye `f"{ticker}_High"` / `f"{ticker}_Low"` / `f"{ticker}_Volume"`
    además de la serie de precio bajo la columna `ticker`) — ATR_14 y OBV
    necesitan ese OHLCV real para su estado inicial, no solo el precio.
    `vix_ticker` (resuelto vía `_resolve_vix_ticker`) también debe estar
    presente como columna en `close_df` — SENTIMENT_SCORE (Hallazgo F5, ver
    `_compute_sentiment_score`) necesita la ventana PAREADA de log-returns
    activo/VIX, no solo la del activo.
    """
    close_series = close_df[ticker]
    high_series = close_df[f"{ticker}_High"]
    low_series = close_df[f"{ticker}_Low"]
    volume_series = close_df[f"{ticker}_Volume"]

    delta = close_series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().iloc[-1]
    ema_fast = close_series.ewm(span=12, adjust=False).mean().iloc[-1]
    ema_slow = close_series.ewm(span=26, adjust=False).mean().iloc[-1]
    ema_20 = close_series.ewm(span=20, adjust=False).mean().iloc[-1]
    macd_line = (
        close_series.ewm(span=12, adjust=False).mean()
        - close_series.ewm(span=26, adjust=False).mean()
    )
    macd_signal = macd_line.ewm(span=9, adjust=False).mean().iloc[-1]

    # Estado recursivo de SENTIMENT_SCORE (Hallazgo F5 — correlación de
    # Pearson móvil activo/VIX, ver `_compute_sentiment_score`): a diferencia
    # de RSI/EMA/MACD (que se resumen en un puñado de escalares IIR), la
    # correlación necesita la ventana CRUDA PAREADA de los últimos
    # `SENTIMENT_LOOKBACK_DAYS` log-returns reales — del activo Y de VIX,
    # misma fecha a fecha — para poder recalcular la correlación sobre una
    # ventana deslizante en cada paso del bucle (ver _forecast_asset). Se
    # guarda tal cual — sin recortar ni promediar — para no perder precisión.
    log_returns_hist = np.log(close_series / close_series.shift(1)).dropna()
    tail_returns = log_returns_hist.tail(SENTIMENT_LOOKBACK_DAYS).to_numpy(dtype=np.float64)
    if len(tail_returns) < SENTIMENT_LOOKBACK_DAYS:
        # Historial insuficiente (activo recién listado): rellena por
        # delante repitiendo el primer valor disponible, o con ceros si no
        # hay ninguno — evita un ValueError en el reshape aguas abajo.
        pad = SENTIMENT_LOOKBACK_DAYS - len(tail_returns)
        fill_value = tail_returns[0] if len(tail_returns) else 0.0
        tail_returns = np.concatenate([np.full(pad, fill_value), tail_returns])

    vix_series = close_df[vix_ticker]
    vix_log_returns_hist = np.log(vix_series / vix_series.shift(1)).dropna()
    tail_returns_vix = vix_log_returns_hist.tail(SENTIMENT_LOOKBACK_DAYS).to_numpy(dtype=np.float64)
    if len(tail_returns_vix) < SENTIMENT_LOOKBACK_DAYS:
        pad_vix = SENTIMENT_LOOKBACK_DAYS - len(tail_returns_vix)
        fill_value_vix = tail_returns_vix[0] if len(tail_returns_vix) else 0.0
        tail_returns_vix = np.concatenate([np.full(pad_vix, fill_value_vix), tail_returns_vix])

    # --- V3: ATR_14 — último valor real del EWM de Wilder (High/Low/Close
    # reales). El bucle autoregresivo lo continúa con velas SINTÉTICAS
    # degeneradas (High=Low=Close del día proyectado, ver _forecast_asset) —
    # solo el estado INICIAL usa OHLC genuino.
    atr_prev = float(_compute_atr(high_series, low_series, close_series).iloc[-1])

    # --- V3: BB_WIDTH_20 — ventana CRUDA de los últimos 20 precios REALES
    # (a nivel precio, no retorno) — mismo patrón que log_return_window,
    # necesaria para recalcular media/std móvil de precio en cada paso.
    tail_prices = close_series.tail(BB_WIDTH_WINDOW).to_numpy(dtype=np.float64)
    if len(tail_prices) < BB_WIDTH_WINDOW:
        pad = BB_WIDTH_WINDOW - len(tail_prices)
        fill_value = tail_prices[0] if len(tail_prices) else float(close_series.iloc[-1])
        tail_prices = np.concatenate([np.full(pad, fill_value), tail_prices])

    # --- V3: OBV — último acumulado REAL + proxy de volumen para los pasos
    # sintéticos (ver OBV_VOLUME_PROXY_LOOKBACK_DAYS arriba: este motor no
    # proyecta volumen, solo precio).
    obv_series = _compute_obv(close_series, volume_series)
    obv_prev = float(obv_series.iloc[-1])
    volume_tail_mean = volume_series.tail(OBV_VOLUME_PROXY_LOOKBACK_DAYS).mean()
    volume_proxy = float(volume_tail_mean) if pd.notna(volume_tail_mean) else 0.0
    # --- V6: ventana CRUDA de los últimos OBV_ROC_LOOKBACK_DAYS valores
    # REALES de OBV (mismo patrón que `tail_prices`/`stoch_high_window`) —
    # necesaria para derivar OBV_ROC_20 = (OBV_t - OBV_{t-20}) / OBV_{t-20}
    # sobre una ventana deslizante en cada paso del bucle autoregresivo.
    obv_window = obv_series.tail(OBV_ROC_LOOKBACK_DAYS).to_numpy(dtype=np.float64)
    if len(obv_window) < OBV_ROC_LOOKBACK_DAYS:
        pad = OBV_ROC_LOOKBACK_DAYS - len(obv_window)
        fill_value = obv_window[0] if len(obv_window) else obv_prev
        obv_window = np.concatenate([np.full(pad, fill_value), obv_window])

    # --- V5: ADX_14 — estado recursivo (+DM/-DM suavizados de Wilder).
    adx_state = _compute_adx_state(high_series, low_series, close_series)

    # --- V5: STOCH_K_14 — ventana CRUDA de los últimos 14 High/Low reales
    # (rolling min/max, no EWM) — mismo patrón que price_window (BB_WIDTH).
    stoch_high_window = high_series.tail(STOCH_K_PERIOD).to_numpy(dtype=np.float64)
    stoch_low_window = low_series.tail(STOCH_K_PERIOD).to_numpy(dtype=np.float64)
    if len(stoch_high_window) < STOCH_K_PERIOD:
        pad = STOCH_K_PERIOD - len(stoch_high_window)
        fill_h = stoch_high_window[0] if len(stoch_high_window) else float(close_series.iloc[-1])
        fill_l = stoch_low_window[0] if len(stoch_low_window) else float(close_series.iloc[-1])
        stoch_high_window = np.concatenate([np.full(pad, fill_h), stoch_high_window])
        stoch_low_window = np.concatenate([np.full(pad, fill_l), stoch_low_window])

    # --- V5: NEWS_SENTIMENT_SCORE — último valor real (FinBERT/Finnhub, ya
    # calculado en `_fetch_feature_window`); se mantiene CONSTANTE en cada
    # paso sintético del bucle — este motor no proyecta noticias futuras,
    # mismo criterio de honestidad que el volume_proxy de OBV.
    news_sentiment_last = float(close_df["NEWS_SENTIMENT_SCORE"].iloc[-1])

    return {
        "avg_gain": float(avg_gain) if pd.notna(avg_gain) else 0.0,
        "avg_loss": float(avg_loss) if pd.notna(avg_loss) else 0.0,
        "ema_fast": float(ema_fast),
        "ema_slow": float(ema_slow),
        "ema_20": float(ema_20),
        "macd_signal": float(macd_signal),
        "log_return_window": tail_returns,  # shape (SENTIMENT_LOOKBACK_DAYS,)
        "log_return_window_vix": tail_returns_vix,  # shape (SENTIMENT_LOOKBACK_DAYS,) — Hallazgo F5
        "atr_prev": atr_prev,
        "price_window": tail_prices,  # shape (BB_WIDTH_WINDOW,)
        "obv_prev": obv_prev,
        "obv_window": obv_window,  # shape (OBV_ROC_LOOKBACK_DAYS,)
        "volume_proxy": max(volume_proxy, 0.0),
        "adx_prev": adx_state["adx_prev"],
        "plus_dm_smooth_prev": adx_state["plus_dm_smooth_prev"],
        "minus_dm_smooth_prev": adx_state["minus_dm_smooth_prev"],
        "stoch_high_window": stoch_high_window,  # shape (STOCH_K_PERIOD,)
        "stoch_low_window": stoch_low_window,    # shape (STOCH_K_PERIOD,)
        "news_sentiment_last": news_sentiment_last,
    }


def _fetch_feature_window(
    ticker: str,
    macro_tickers: list,
    lookback: int,
    as_of_date: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Ingesta en vivo — DEBE replicar bit a bit `engineer_asset()` del notebook,
    con una salvedad de PRODUCCIÓN: los filtros IIR recursivos (RSI/EMA/MACD,
    todos `ewm(adjust=False)`) requieren un tramo de calentamiento largo para
    converger al mismo estado recursivo con el que el Colab entrenó (10y de
    historia). Descargar 10y en cada request mataría la latencia de la API;
    "6mo" (el valor previo) dejaba un warm-up demasiado corto (~126 sesiones)
    frente a un span MACD_SLOW=26 / RSI=14, con transitorio aún no aplanado.

    FIX DE CONVERGENCIA: se descargan 2 AÑOS ("2y", ~504 sesiones) — suficiente
    burn-in para que el estado recursivo de RSI_14/EMA_20/MACD/MACD_SIGNAL
    converja de forma prácticamente indistinguible del estado alcanzado con
    10y de historia, sin penalizar la latencia como lo haría "10y" completo.

    RECORTE ANTES DE RETORNAR (crítico): el DataFrame `close` (precios +
    indicadores YA estabilizados por los 2y de warm-up) se retorna COMPLETO
    porque callers aguas abajo lo necesitan más allá de `lookback`:
        - `_compute_indicator_states(close_df, ticker, vix_ticker)` recalcula
          RSI/EMA/MACD/SENTIMENT/ATR_14/BB_WIDTH_20/OBV desde cero sobre la
          serie de precios (y OHLCV) completa — si se le pasara solo
          `lookback` filas, se perdería TODO el beneficio de los 2y de
          calentamiento que acabamos de pagar en la descarga.
        - `vol_window = close_df[ticker].tail(VOLATILITY_LOOKBACK_DAYS + 1)`
          (63+1 días) y `historical = close_df[ticker].tail(hist_window)`
          (hasta 90 días, ver `_historical_window_for_horizon`) exceden
          `lookback` en la mayoría de configuraciones.
    Lo único que el tensor 3D de la Bi-LSTM necesita es la ÚLTIMA ventana de
    `lookback` filas — por eso, y SOLO para `raw_features` (el array que
    alimenta al modelo), se recorta estrictamente a `[-lookback:]` antes de
    devolverlo. `close` (para todo lo demás) permanece intacto.
    """
    all_symbols = list(dict.fromkeys([ticker] + macro_tickers))
    raw = _yf_call_with_retry(
        lambda: yf.download(all_symbols, period="2y", auto_adjust=True, progress=False, session=_YF_SESSION)
    )

    # V3: bundle OHLCV COMPLETO solo para el ticker objetivo — ATR_14
    # necesita High/Low, OBV necesita Volume. Los macro tickers permanecen
    # Close-only, replicando exactamente engineer_asset()/download_all() del
    # notebook (no se calculan técnicos sobre los factores macro). Columnas
    # planas `f"{ticker}_High"` / `_Low` / `_Volume`, mismo patrón de
    # nombrado que download_all() en entrenamiento.py.
    close = pd.DataFrame(index=raw["Close"].index)
    close[ticker] = raw["Close"][ticker]
    close[f"{ticker}_High"] = raw["High"][ticker]
    close[f"{ticker}_Low"] = raw["Low"][ticker]
    close[f"{ticker}_Volume"] = raw["Volume"][ticker]
    for macro_ticker in macro_tickers:
        close[macro_ticker] = raw["Close"][macro_ticker]
    close = close.ffill().dropna()

    # --- FIX VELA VIVA (cripto 24/7, sin campana de cierre) -----------------
    # Un Daily Close consolidado para un activo 24/7 solo existe DESPUÉS de
    # las 00:00 UTC del día siguiente. Si este pipeline corre en cualquier
    # momento ANTES de ese corte para el día de "hoy" (UTC), la última fila
    # que yfinance devuelve para el ticker cripto es un snapshot EN VIVO
    # (precio al instante de la query), no un cierre real — y ese snapshot
    # cambia cada vez que se re-corre el pipeline en el mismo día. Usarlo
    # tal cual como último punto de `historical`/ancla del forecast produce
    # exactamente el desfase reportado: el "cierre" no es reproducible ni
    # coincide con el cierre real posterior de esa misma fecha. Equities NO
    # tienen este problema — yfinance no publica la fila del día en curso
    # hasta después de la campana de cierre — así que el fix se acota
    # estrictamente al régimen 24/7 (mismo patrón que el resto del motor,
    # ver `regime == "crypto_specialist"` más abajo). Se dropea la fila
    # completa (todas las columnas, incluidos los macro tickers) porque una
    # sesión de "hoy" sin cerrar tampoco es un dato consolidado válido para
    # ningún otro campo derivado de esa misma fila.
    if ticker in REGIME_TICKERS.get("crypto_specialist", ()) and not close.empty:
        today_utc = datetime.now(timezone.utc).date()
        if close.index[-1].date() >= today_utc:
            close = close.iloc[:-1]
        if close.empty:
            raise ValueError(
                f"'{ticker}': sin ninguna sesión UTC consolidada todavía tras "
                "dropear la vela en vivo de hoy — histórico insuficiente para "
                "construir la ventana de features."
            )

    # --- FIX PRICING (Adjusted vs Regular Close) ---------------------------
    # `close[ticker]` de arriba es Auto-Adjusted Close (auto_adjust=True) —
    # correcto y OBLIGATORIO para el modelo/indicadores (paridad bit a bit
    # con engineer_asset() de entrenamiento, ver docstring). Pero un ajuste
    # retroactivo por dividendos hace que ESE mismo precio, usado tal cual
    # para el histórico que ve el usuario en el chart, no calce con el
    # "Regular Close" real (Google Finance / Yahoo quote) — la discrepancia
    # crece cuanto más atrás en el tiempo, según cuántos dividendos pagó el
    # activo en la ventana. Se descarga una serie separada, SOLO Close, SIN
    # ajustar, exclusivamente para el ticker objetivo (no macro, no O/H/L/V
    # — esos siguen sirviendo únicamente al modelo) y se alinea al mismo
    # índice de sesiones. El precio más reciente (ancla del forecast) es
    # prácticamente idéntico en ambas series por construcción de yfinance
    # (el factor de ajuste en la sesión más reciente es ~1.0) — esto NUNCA
    # introduce un salto visual entre histórico y proyección, solo corrige
    # los puntos más antiguos del histórico para que calcen con la realidad
    # del activo. Ver uso en `historical` dentro de `_forecast_asset`.
    display_col = f"{ticker}__DISPLAY_CLOSE"
    try:
        raw_display = _yf_call_with_retry(
            lambda: yf.download(ticker, period="2y", auto_adjust=False, progress=False, session=_YF_SESSION)
        )
        display_close = raw_display["Close"] if not raw_display.empty else None
        if isinstance(display_close, pd.DataFrame):  # MultiIndex de 1 solo símbolo, según versión de yfinance
            display_close = display_close[ticker] if ticker in display_close.columns else display_close.iloc[:, 0]
        if display_close is None or display_close.empty:
            raise ValueError("respuesta vacía")
        # FIX (ancla desfasada / "lag" del cursor): `close.index` y
        # `display_close.index` vienen de DOS llamadas yf.download()
        # independientes — nada garantiza que ambas devuelvan la MISMA
        # fecha como sesión más reciente (latencia de red, refresco parcial
        # del endpoint no-ajustado, etc.). Un `.reindex().ffill().bfill()`
        # ciego enmascara ese hueco: si a `display_close` le falta
        # justo la fila más nueva, el ffill arrastra el cierre de un día
        # ANTERIOR y lo presenta como si fuera el precio de hoy — exactamente
        # el síntoma de "ancla con lag" reportado. Se valida explícitamente
        # la cobertura de la sesión más reciente antes de confiar en ella.
        latest_session = close.index[-1]
        if display_close.index[-1] < latest_session:
            missing_sessions = int((close.index > display_close.index[-1]).sum())
            print(
                f"⚠️ Regular Close de '{ticker}' desactualizado: la fuente no-ajustada "
                f"llega solo hasta {display_close.index[-1].strftime('%Y-%m-%d')}, pero la "
                f"sesión más reciente real es {latest_session.strftime('%Y-%m-%d')} "
                f"({missing_sessions} sesión/es sin cobertura). Se evita el ffill silencioso "
                "sobre la(s) fila(s) faltante(s): esas sesiones usan Auto-Adjusted Close "
                "como fallback puntual (huecos intermedios sí se rellenan por feriados de "
                "mercado normales)."
            )
            aligned_display = display_close.reindex(close.index)
            # Solo las sesiones estrictamente posteriores a la última cobertura real
            # de la fuente no-ajustada se completan con Auto-Adjusted Close (fallback
            # explícito y acotado); todo lo demás (huecos intermedios normales, p. ej.
            # feriados que no calzan 1:1 entre ambas descargas) sigue su ffill/bfill
            # habitual, que ahí sí es información real, solo con fecha de publicación
            # ligeramente distinta entre fuentes.
            stale_mask = aligned_display.index > display_close.index[-1]
            aligned_display = aligned_display.ffill().bfill()
            aligned_display.loc[stale_mask] = close.loc[stale_mask, ticker]
            close[display_col] = aligned_display
        else:
            close[display_col] = display_close.reindex(close.index).ffill().bfill()
    except Exception as exc:  # noqa: BLE001
        print(
            f"⚠️ No se pudo obtener Regular Close (sin ajustar) de '{ticker}' ({exc}) — "
            "el histórico visual usará Auto-Adjusted Close como fallback (puede no calzar "
            "exactamente con el oráculo real)."
        )
        close[display_col] = close[ticker]

    # CUTOFF RETROACTIVO (online learning): si se pide reconstruir la ventana
    # tal cual estaba en una fecha pasada (`as_of_date`), se descarta TODA
    # fila posterior ANTES de calcular RSI/EMA/MACD — si se recortara
    # después, los indicadores incorporarían "futuro" (fuga de información)
    # respecto al momento real en que se generó la predicción que se está
    # reevaluando. `as_of_date=None` (default, ruta de inferencia normal)
    # deja el comportamiento bit a bit idéntico al de antes de este cambio.
    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date)
        close = close[close.index <= cutoff]
        if close.empty:
            raise ValueError(
                f"'{as_of_date}' no tiene datos de mercado disponibles (¿fecha "
                "futura o anterior al histórico descargable?) — no se puede "
                "reconstruir la ventana de ese día para fine-tuning."
            )

    close["RSI_14"] = _compute_rsi(close[ticker])  # oscilador ya acotado [0,100] — sin cambios

    # --- V6: LOG_RETURN_1D reemplaza a "PRICE" (nivel absoluto) como único
    # input de nivel-precio del tensor. `close[ticker]` SIGUE guardando el
    # precio CRUDO — lo necesitan `last_price`/`historical`/
    # `_compute_indicator_states` y los técnicos derivados más abajo;
    # LOG_RETURN_1D es una columna ADICIONAL, nunca un reemplazo.
    close["LOG_RETURN_1D"] = np.log(close[ticker] / close[ticker].shift(1))

    ema_20 = close[ticker].ewm(span=20, adjust=False).mean()
    close["EMA20_DEV_PCT"] = (close[ticker] - ema_20) / ema_20  # % desviación vs. la propia media móvil

    macd_line, signal_line = _compute_macd(close[ticker])
    close["MACD_PCT"] = macd_line / close[ticker]
    close["MACD_SIGNAL_PCT"] = signal_line / close[ticker]

    # Hallazgo F5: SENTIMENT_SCORE = correlación de Pearson móvil activo/VIX
    # (ver _compute_sentiment_score), no un Z-score de momentum propio.
    # DEBE correr con `close[vix_ticker]` TODAVÍA en precio crudo (la
    # función deriva su propio log-return internamente) — por eso va ANTES
    # del bloque de log-return de factores macro más abajo.
    vix_ticker = _resolve_vix_ticker(macro_tickers)
    close["SENTIMENT_SCORE"] = _compute_sentiment_score(close[ticker], close[vix_ticker])

    # V3: ATR_14 -> ATR_PCT ("ATRP", normalizado por precio) — requiere
    # High/Low reales del ticker, igual que antes.
    atr = _compute_atr(close[f"{ticker}_High"], close[f"{ticker}_Low"], close[ticker])
    close["ATR_PCT"] = atr / close[ticker]
    close["BB_WIDTH_20"] = _compute_bb_width(close[ticker])  # ya es 2σ/μ, adimensional — sin cambios

    obv = _compute_obv(close[ticker], close[f"{ticker}_Volume"])
    obv_roc = obv.pct_change(periods=OBV_ROC_LOOKBACK_DAYS)
    close["OBV_ROC_20"] = obv_roc.replace([np.inf, -np.inf], np.nan).clip(-OBV_ROC_CLIP, OBV_ROC_CLIP).fillna(0.0)

    close["ADX_14"] = _compute_adx(close[f"{ticker}_High"], close[f"{ticker}_Low"], close[ticker])
    close["STOCH_K_14"] = _compute_stochastic_k(close[f"{ticker}_High"], close[f"{ticker}_Low"], close[ticker])
    # V5 (Requerimiento 1/3 del pipeline): NEWS_SENTIMENT_SCORE vía FinBERT
    # (transformers) + Finnhub/yfinance.news — misma fuente que entrenó los
    # especialistas. Llamada de red + inferencia local; el pipeline de
    # inferencia ya paga latencia por la descarga yfinance de 2y, así que
    # esto es consistente con el resto de este método (sin cache propio).
    close["NEWS_SENTIMENT_SCORE"] = get_daily_news_sentiment(ticker, close.index)

    # --- FIX CRÍTICO (auditoría de paridad train/serve, previa a V6):
    # `engineer_asset()` en train_kodaquant_v5.py transforma CADA factor
    # macro a log-return ANTES de que `feature_scaler.fit()` los vea
    # (estacionariedad, V4) — esta función NUNCA replicaba ese paso: los
    # macro_tickers llegaban a `feature_cols` (más abajo) en NIVEL DE
    # PRECIO CRUDO (ej. ^GSPC≈6000, ^TNX≈4.2, GC=F≈2400, DX-Y.NYB≈100)
    # contra un `feature_scaler` calibrado para log-returns ≈[-0.05, 0.05]
    # — saturación masiva en TODOS los timesteps de la ventana (no solo los
    # sintéticos), consistente con el oob_frac=45.6% medido en T+1 (100%
    # datos reales) que se venía atribuyendo enteramente al ATH. Se
    # transforma en columnas NUEVAS (sufijo `__LOGRET`) en vez de
    # sobreescribir `close[macro_ticker]` in-place: `_compute_indicator_
    # states`/`_resolve_vix_ticker` más abajo en `_forecast_asset` siguen
    # necesitando el precio CRUDO del VIX.
    macro_logret_cols = [f"{m}__LOGRET" for m in macro_tickers]
    for macro_ticker, logret_col in zip(macro_tickers, macro_logret_cols):
        macro_price = close[macro_ticker]
        close[logret_col] = np.log(macro_price / macro_price.shift(1))

    close = close.ffill().dropna()

    # Orden EXACTO de TECH_COLS (ver constante de módulo, arriba): RSI_14,
    # EMA20_DEV_PCT, MACD_PCT, MACD_SIGNAL_PCT, SENTIMENT_SCORE, ATR_PCT,
    # BB_WIDTH_20, OBV_ROC_20, ADX_14, STOCH_K_14, NEWS_SENTIMENT_SCORE — el
    # feature_scaler cargado desde scalers_dict.pkl fue ajustado con ese
    # orden/definición, así que romperlo aquí desalinearía cada columna al
    # pasar por `.transform()`. `_get_scalers()` ya validó que TECH_COLS
    # coincide con `scalers["tech_cols"]` al arrancar el proceso. El tramo
    # macro usa las columnas `__LOGRET` (estacionarias) — NUNCA el precio
    # crudo `macro_tickers`, que sigue vivo en `close` solo para los otros
    # consumidores señalados arriba.
    feature_cols = ["LOG_RETURN_1D"] + TECH_COLS + macro_logret_cols

    # Recorte estricto: SOLO el tensor de entrada a la red se acota a
    # `lookback` filas. `close` se retorna completo (2y) para los
    # consumidores que necesitan más historia que la ventana del modelo.
    raw_features = close[feature_cols].values[-lookback:]

    return close, raw_features


# ---------------------------------------------------------------------------
# Inferencia Keras / TensorFlow nativa (MC Dropout real vía training=True)
# ---------------------------------------------------------------------------

def _resolve_horizon_days(forecast_horizon_days: int | None) -> int:
    """
    Clamp defensivo del horizonte de proyección: cualquier entero positivo
    razonable se respeta; basura (None, <=0, no-entero) cae al default de
    1 semana; se topa a 252 sesiones (~1 año bursátil).
    """
    if not isinstance(forecast_horizon_days, int) or isinstance(forecast_horizon_days, bool):
        return DEFAULT_FORECAST_HORIZON_DAYS
    if forecast_horizon_days <= 0:
        return DEFAULT_FORECAST_HORIZON_DAYS
    return min(forecast_horizon_days, 252)


def _find_optimal_entry_exit(
    forecast_points: list[dict],
    anchor_date: str | None = None,
    anchor_price: float | None = None,
) -> dict | None:
    """
    Peak & Trough Analysis — ESTRICTAMENTE cronológico. El tiempo es lineal:
    jamás se puede vender antes o el mismo día que se compra.

    Paso A) Día de Compra = precio MÍNIMO GLOBAL de todo el arreglo de
            proyección (`forecast_points`).
    Paso B) Día de Venta = precio MÁXIMO, buscado ÚNICAMENTE en el slice de
            `forecast_points` posterior al índice de compra
            (`forecast_points[entry_idx + 1:]`) — nunca en el mismo día,
            nunca antes.

    Si el mínimo global cae en el ÚLTIMO día proyectado, no queda ventana de
    salida posterior dentro del propio arreglo. En ese caso degrada a un plan
    defensivo en vez de no emitir señal: Compra HOY (`anchor_date` /
    `anchor_price`, el último precio real ya conocido, anterior a toda la
    proyección) y Vende en el último día proyectado. Nunca inventa una fecha
    o precio fuera de estos dos ya calculados.
    """
    if not forecast_points:
        return None

    entry_idx = min(range(len(forecast_points)), key=lambda i: forecast_points[i]["expected_path"])
    entry = forecast_points[entry_idx]

    remaining = forecast_points[entry_idx + 1:]
    if remaining:
        exit_point = max(remaining, key=lambda p: p["expected_path"])
        return {
            "entry_date": entry["date"],
            "entry_price": entry["expected_path"],
            "exit_date": exit_point["date"],
            "exit_price": exit_point["expected_path"],
        }

    # El mínimo global es el ÚLTIMO día proyectado: no hay slice posterior.
    # Fallback defensivo — compra HOY, vende en el último día proyectado.
    if anchor_date is None or anchor_price is None:
        return None

    last_point = forecast_points[-1]
    if last_point["date"] == anchor_date:
        # Degenerado: no hay ni un día futuro distinto de hoy — sin señal.
        return None

    return {
        "entry_date": anchor_date,
        "entry_price": anchor_price,
        "exit_date": last_point["date"],
        "exit_price": last_point["expected_path"],
    }


def _validate_date_str(value) -> str | None:
    """
    VERIFICACIÓN CRÍTICA (post-mortem: el LLM omitía fechas cuando algo
    corrupto llegaba a la plantilla). `entry_date`/`exit_date` deberían
    llegar siempre como 'YYYY-MM-DD' (vienen de `strftime` en
    `_forecast_asset`), pero esta función lo confirma explícitamente antes
    de que cualquier fecha toque el prompt: si no es un string con esa
    forma exacta, retorna None en vez de dejar pasar un valor corrupto
    (None, '', un float, un objeto date sin castear, etc.) a la f-string.
    """
    if not isinstance(value, str):
        return None
    value = value.strip()
    if len(value) != 10 or value[4] != "-" or value[7] != "-":
        return None
    year, month, day = value[:4], value[5:7], value[8:10]
    if not (year.isdigit() and month.isdigit() and day.isdigit()):
        return None
    return value


def _build_tactical_operation(plan: dict) -> dict | None:
    """
    Traduce el `tactical_signal` (Peak & Trough, ver `_find_optimal_entry_exit`)
    de UN plan (Plan A o Plan B, ambos pasan por aquí — jamás solo Plan B) a
    una Operación Táctica ejecutable, agregando la Ganancia Neta Proyectada en
    USD: capital asignado a ESE activo × diferencial de precios entrada/salida.

    Este cálculo vive 100% en Python — verdad absoluta. El LLM NUNCA lo
    recalcula, solo lo narra (ver REGLA ANTI-ALUCINACIÓN en
    `_build_system_prompt`).

    Retorna None si el forecast del plan falló, si no hay ventana de salida
    válida, o si `entry_date`/`exit_date` no pasan `_validate_date_str` — en
    los tres casos el prompt debe degradar sin inventar fecha/precio/ganancia.
    """
    forecast = plan.get("forecast")
    if not forecast:
        return None

    tactical_signal = forecast.get("tactical_signal")
    if not tactical_signal:
        return None

    entry_date = _validate_date_str(tactical_signal.get("entry_date"))
    exit_date = _validate_date_str(tactical_signal.get("exit_date"))
    if entry_date is None or exit_date is None:
        print(
            f"⚠️ Fecha inválida en tactical_signal de {plan.get('activo_referencia')} "
            f"(entry={tactical_signal.get('entry_date')!r}, exit={tactical_signal.get('exit_date')!r}) "
            "— se omite la operación en vez de inyectar una fecha corrupta al prompt."
        )
        return None

    entry_price = tactical_signal["entry_price"]
    exit_price = tactical_signal["exit_price"]
    capital_usd = float(plan["monto_usd"])

    price_return_pct = (exit_price / entry_price - 1) if entry_price else 0.0
    net_profit_usd = round(capital_usd * price_return_pct, 2)

    return {
        "ticker": plan["activo_referencia"],
        "nombre_plan": plan["nombre"],
        "capital_usd": round(capital_usd, 2),
        "entry_date": entry_date,
        "entry_price": entry_price,
        "exit_date": exit_date,
        "exit_price": exit_price,
        "net_profit_usd": net_profit_usd,
    }

def _historical_window_for_horizon(steps: int) -> int:
    """
    El panel histórico debe verse proporcional a lo proyectado — 30 días
    fijos contra un horizonte de 63 (3M) deja la gráfica ~68% forecast /
    32% historia real. Regla: ~1.5x el horizonte, acotado a [30, 90].
    """
    return max(30, min(90, int(steps * 1.5)))

def _forecast_asset(ticker: str, steps: int = DEFAULT_FORECAST_HORIZON_DAYS) -> dict:
    """
    Motor autoregresivo con Simulación Monte Carlo (sliding window, batcheado).
    En cada paso T+1 ... T+steps se corre UN forward pass del modelo Keras
    con un batch de `N_MONTE_CARLO_SIMULATIONS` trayectorias en paralelo
    (`model(..., training=True)` — MC Dropout real, ver `_get_keras_model`;
    con dropout activo, cada elemento del batch muestrea su propia máscara,
    así que el batch ya son N inferencias estocásticas genuinas, no N copias
    idénticas). A cada trayectoria se le suma además su propio ruido
    gaussiano i.i.d. calibrado con la volatilidad histórica REAL del activo
    (`VOLATILITY_INJECTION_*`). Cada trayectoria evoluciona su propia ventana
    de features (RSI/EMA/MACD recalculados por trayectoria) de forma
    completamente independiente.

    Al terminar el bucle, las N trayectorias se colapsan por fecha a tres
    percentiles — `lower_bound` (P5), `expected_path` (P50), `upper_bound`
    (P95) — que son los que se devuelven en `forecast`. Los N caminos crudos
    NUNCA se devuelven, solo el resumen probabilístico.

    HONESTIDAD DE DATOS: `expected_path` es la mediana de N simulaciones
    (modelo + volatilidad calibrada), no una única inferencia determinista
    de la red. `sample_path` es, en cambio, una trayectoria real de esas
    mismas N simulaciones (la más cercana a la mediana) — real, no
    percentil, así que preserva la textura día-a-día genuina que el
    percentil-por-fecha cancela parcialmente. `variance_source` declara el
    mecanismo exacto para que ningún consumidor de este dict (frontend,
    prompt del LLM) presente cualquiera de los dos como algo distinto a lo
    que es.

    SÍNCRONA Y BLOQUEANTE por diseño: el caller SIEMPRE debe despacharla a
    un executor.
    """
    regime = _regime_for_ticker(ticker)
    scalers = _get_scalers(regime)
    if ticker not in scalers["asset_to_id"]:
        raise ValueError(f"'{ticker}' no pertenece al universo entrenado del especialista '{regime}'.")

    asset_id = scalers["asset_to_id"][ticker]
    feature_scaler = scalers["feature_scalers"][ticker]
    target_scaler = scalers["target_scalers"][ticker]
    lookback = scalers["lookback"]
    macro_tickers = scalers["macro_tickers"]
    n_sim = N_MONTE_CARLO_SIMULATIONS

    # --- SEMILLA DETERMINISTA (memoria diaria por activo) -------------------
    # Objetivo: todas las peticiones para el MISMO (ticker, horizonte) en el
    # MISMO día natural (UTC) deben producir EXACTAMENTE las mismas N
    # trayectorias, sin importar cuántos usuarios/requests las disparen —
    # hasta que cierre la vela diaria (`today_utc_str` avanza a las 00:00 UTC
    # y la semilla cambia sola, sin cache manual que purgar).
    #
    # Hash estable vía sha256 — NUNCA `hash()` de Python: está salado por
    # proceso (`PYTHONHASHSEED` aleatorio por defecto para strings), así que
    # dos workers o dos reinicios del mismo proceso darían semillas
    # DISTINTAS para el mismo input, rompiendo justo la consistencia
    # multiusuario que es el objetivo de este fix.
    #
    # `np.random.SeedSequence(base_seed).spawn(steps)` deriva `steps`
    # sub-semillas HIJAS de alta calidad (independientes entre sí, no una
    # secuencia trivial `seed + step_i` que podría correlacionar pasos
    # vecinos) — una por cada T+1..T+steps, así el ruido i.i.d. de cada día
    # de la trayectoria es distinto pero 100% reproducible.
    #
    # Cada paso instancia su propio `np.random.default_rng(...)` LOCAL (ver
    # bucle más abajo) en vez de mutar `np.random.seed()` global: Plan A
    # (SPY) y Plan B (p. ej. BTC-USD) se despachan CONCURRENTEMENTE en
    # threads distintos (`_build_investment_plans`) — un seed global sería
    # una condición de carrera real entre ambos.
    today_utc_str = datetime.now(timezone.utc).date().isoformat()
    base_seed_material = f"{ticker}|{today_utc_str}|{steps}".encode("utf-8")
    base_seed = int(hashlib.sha256(base_seed_material).hexdigest(), 16) % (2**32)
    step_seed_sequences = np.random.SeedSequence(base_seed).spawn(steps)

    # `model` aquí puede ser el grafo crudo o el "MC-Dropout bridge" (ver
    # `_build_mc_dropout_bridge` / `_get_keras_model`) — la función decide
    # cuál sirve según si el nodo `dropout_regularizer` del .keras venía
    # congelado en `training=False`. Esta función NUNCA debe asumir cuál de
    # los dos es: solo le pasa `training=True` y confía en que
    # `_get_keras_model` ya resolvió el bug de congelamiento.
    model, dropout_layers = _get_keras_model(regime)
    base_variance_source = "mc_dropout" if dropout_layers else "deterministic_native_keras"

    close_df, raw_features = _fetch_feature_window(ticker, macro_tickers, lookback)

    # sigma_hist: desviación estándar de los log-returns REALES de los
    # últimos VOLATILITY_LOOKBACK_DAYS del propio activo — no un número
    # inventado, es volatilidad realizada genuina. Sirve como escala del
    # ruido gaussiano que se inyecta por trayectoria en cada paso.
    vol_window = close_df[ticker].tail(VOLATILITY_LOOKBACK_DAYS + 1).to_numpy(dtype=np.float64)
    if VOLATILITY_INJECTION_ENABLED and len(vol_window) >= 3:
        hist_log_returns = np.diff(np.log(vol_window))
        sigma_hist = float(np.std(hist_log_returns))
    else:
        sigma_hist = 0.0
    noise_scale = sigma_hist * VOLATILITY_INJECTION_FACTOR
    variance_source = base_variance_source + (
        f"+monte_carlo_{n_sim}x+volatility_injection" if noise_scale > 0 else f"+monte_carlo_{n_sim}x"
    )
    # Techo de la señal del modelo (no del ruido) — ver "Cinturón de
    # seguridad contra el tobogán" arriba. Si `sigma_hist` no se pudo
    # calcular (histórico insuficiente), no hay base estadística real para
    # un límite -> no se acota (np.inf), igual que antes de este fix.
    model_signal_clip = (
        sigma_hist * MODEL_SIGNAL_CLIP_SIGMAS if sigma_hist > 0 else np.inf
    )

    # GUARD "Falso Monte Carlo" — la ÚNICA condición bajo la que las N
    # trayectorias serían deterministas e idénticas (colapso a un solo
    # vector, línea recta perfecta): CERO fuentes de varianza real. Este
    # motor tiene DOS por diseño — (1) MC-Dropout genuino vía `training=True`
    # en el batch (ver `_get_keras_model`/`dropout_layers`, cada fila del
    # batch muestrea su propia máscara de Dropout) y (2) ruido gaussiano
    # aditivo post-`inverse_transform` escalado a `VOLATILITY_INJECTION_FACTOR
    # * sigma_hist` (ver más abajo, `noise_batch`). Si AMBAS fallan a la vez
    # (ej. el .keras cargado no tiene capas Dropout activas Y
    # VOLATILITY_INJECTION_ENABLED=False o `sigma_hist=0`), el bucle SÍ
    # degenera a un filtro determinista — fallar ruidosamente aquí en vez de
    # devolver un forecast silenciosamente colapsado.
    if not dropout_layers and noise_scale == 0.0:
        raise ValueError(
            f"Monte Carlo determinista detectado para '{ticker}': dropout_layers "
            "vacío (el .keras del régimen no tiene capas Dropout activas) Y "
            "noise_scale=0.0 (VOLATILITY_INJECTION_ENABLED=False o sigma_hist=0 "
            "por histórico insuficiente). Sin NINGUNA fuente de varianza real, "
            f"las {n_sim} trayectorias serían idénticas — degenerando a un único "
            "vector determinista y explicando un colapso en línea recta "
            "geométrica. Revisa la arquitectura del modelo cargado o "
            "VOLATILITY_INJECTION_ENABLED/VOLATILITY_LOOKBACK_DAYS antes de "
            "reintentar el forecast."
        )

    last_price = float(close_df[ticker].iloc[-1])
    cursor_date = close_df.index[-1]
    anchor_date_str = cursor_date.strftime("%Y-%m-%d")  # "HOY" — ancla real, pre-proyección

    hist_window = _historical_window_for_horizon(steps)
    # FIX PRICING: histórico visual en Regular Close (oráculo real), no en
    # Auto-Adjusted Close (ver bloque `__DISPLAY_CLOSE` en
    # _fetch_feature_window). Fallback a la columna ajustada si por lo que
    # sea no está presente (backend viejo / excepción silenciada arriba).
    display_col = f"{ticker}__DISPLAY_CLOSE"
    historical_price_series = close_df[display_col] if display_col in close_df.columns else close_df[ticker]
    historical = [
        {"date": d.strftime("%Y-%m-%d"), "price": round(float(p), 2)}
        for d, p in historical_price_series.tail(hist_window).items()
    ]
    # FIX PRICING (ancla vs. histórico visual): `last_price` (arriba) es
    # Auto-Adjusted Close — correcto y OBLIGATORIO como semilla del Monte
    # Carlo (`cursor_prices` más abajo) y para la paridad con el
    # entrenamiento, así que NO se toca. Pero ese mismo valor NO es
    # necesariamente idéntico al último precio de `historical` (Regular
    # Close, ver bloque __DISPLAY_CLOSE arriba) — la discrepancia, aunque
    # normalmente mínima, es justo lo que se percibe como "el cursor no
    # calza con el último punto del histórico" en el chart. Todo lo que se
    # DEVUELVE al frontend como "precio actual" (campo `last_price` de la
    # respuesta y `anchor_price` de la señal táctica) usa en cambio
    # `last_price_quote`, tomado del MISMO array `historical` que ya se
    # construyó arriba — coincide con su último punto por construcción,
    # no por casualidad. Fallback a `last_price` solo si `historical`
    # llegara vacío (activo sin histórico suficiente).
    last_price_quote = historical[-1]["price"] if historical else round(last_price, 2)

    # Estado vectorizado: N trayectorias en paralelo, cada una con su propia
    # ventana de features y su propio estado de indicadores (RSI/EMA/MACD),
    # todas arrancando del MISMO histórico real (`raw_features`).
    cursor_windows = np.repeat(raw_features[-lookback:][np.newaxis, :, :], n_sim, axis=0).astype(np.float64)
    cursor_prices = np.full(n_sim, last_price, dtype=np.float64)

    vix_ticker = _resolve_vix_ticker(macro_tickers)  # Hallazgo F5 — ver _compute_sentiment_score
    base_indicator_state = _compute_indicator_states(close_df, ticker, vix_ticker)
    avg_gain_arr = np.full(n_sim, base_indicator_state["avg_gain"], dtype=np.float64)
    avg_loss_arr = np.full(n_sim, base_indicator_state["avg_loss"], dtype=np.float64)
    ema_fast_arr = np.full(n_sim, base_indicator_state["ema_fast"], dtype=np.float64)
    ema_slow_arr = np.full(n_sim, base_indicator_state["ema_slow"], dtype=np.float64)
    ema_20_arr = np.full(n_sim, base_indicator_state["ema_20"], dtype=np.float64)
    macd_signal_arr = np.full(n_sim, base_indicator_state["macd_signal"], dtype=np.float64)
    # SENTIMENT_SCORE (Hallazgo F5 — correlación de Pearson móvil
    # activo/VIX): cada trayectoria arranca con la MISMA ventana real de los
    # últimos SENTIMENT_LOOKBACK_DAYS log-returns del activo (no hay fuga de
    # futuro, todo viene del histórico ya observado) y a partir de ahí
    # diverge de forma independiente, igual que RSI/EMA/MACD.
    log_return_window_arr = np.repeat(
        base_indicator_state["log_return_window"][np.newaxis, :], n_sim, axis=0
    ).astype(np.float64)  # shape (n_sim, SENTIMENT_LOOKBACK_DAYS)
    # Ventana de VIX: COMPARTIDA entre trayectorias (1D, no (n_sim, window)).
    # Este motor no proyecta macro a futuro — los tickers macro se propagan
    # SIN CAMBIO dentro del tensor (ver comentario "macro_tickers... se
    # propagan sin cambio" más abajo, en el ensamblado de `new_rows`), así
    # que el VIX queda congelado en su último precio real para siempre. Con
    # el precio de VIX congelado, cada log-return SINTÉTICO de VIX es
    # exactamente 0.0 (log(P/P) = 0) — por eso la ventana solo se desliza
    # añadiendo ceros a medida que el historial real envejece fuera de ella,
    # nunca diverge por trayectoria (mismo principio de honestidad que
    # `volume_proxy` de OBV: este motor no inventa noticias ni macro futuro).
    log_return_window_vix = base_indicator_state["log_return_window_vix"].copy()  # shape (SENTIMENT_LOOKBACK_DAYS,)
    RSI_ALPHA, EMA20_K, EMA_FAST_K, EMA_SLOW_K, MACD_SIGNAL_K = (
        1 / 14, 2 / 21, 2 / 13, 2 / 27, 2 / 10,
    )

    # --- V3: estado inicial vectorizado de ATR_14 / BB_WIDTH_20 / OBV, una
    # trayectoria por fila, todas arrancando del MISMO estado real
    # (`base_indicator_state`) y divergiendo de forma independiente —
    # idéntico patrón a RSI/EMA/MACD/SENTIMENT_SCORE arriba.
    atr_arr = np.full(n_sim, base_indicator_state["atr_prev"], dtype=np.float64)
    price_window_arr = np.repeat(
        base_indicator_state["price_window"][np.newaxis, :], n_sim, axis=0
    ).astype(np.float64)  # shape (n_sim, BB_WIDTH_WINDOW)
    obv_arr = np.full(n_sim, base_indicator_state["obv_prev"], dtype=np.float64)
    # --- V6: ventana deslizante de OBV crudo por trayectoria — necesaria
    # para derivar OBV_ROC_20 (la feature que SÍ entra al tensor, ver
    # `new_rows` más abajo); `obv_arr` sigue siendo el acumulado crudo
    # interno (misma recursión que antes, solo cambia qué se PROYECTA al
    # modelo).
    obv_window_arr = np.repeat(
        base_indicator_state["obv_window"][np.newaxis, :], n_sim, axis=0
    ).astype(np.float64)  # shape (n_sim, OBV_ROC_LOOKBACK_DAYS)
    # Proxy de volumen CONSTANTE (promedio histórico reciente) — este motor
    # no proyecta volumen futuro, solo precio; ver OBV_VOLUME_PROXY_LOOKBACK_DAYS.
    volume_proxy = base_indicator_state["volume_proxy"]

    # --- V5: estado inicial vectorizado de ADX_14 / STOCH_K_14 / NEWS_SENTIMENT_SCORE.
    plus_dm_smooth_arr = np.full(n_sim, base_indicator_state["plus_dm_smooth_prev"], dtype=np.float64)
    minus_dm_smooth_arr = np.full(n_sim, base_indicator_state["minus_dm_smooth_prev"], dtype=np.float64)
    adx_arr = np.full(n_sim, base_indicator_state["adx_prev"], dtype=np.float64)
    stoch_high_window_arr = np.repeat(
        base_indicator_state["stoch_high_window"][np.newaxis, :], n_sim, axis=0
    ).astype(np.float64)  # shape (n_sim, STOCH_K_PERIOD)
    stoch_low_window_arr = np.repeat(
        base_indicator_state["stoch_low_window"][np.newaxis, :], n_sim, axis=0
    ).astype(np.float64)  # shape (n_sim, STOCH_K_PERIOD)
    # NEWS_SENTIMENT_SCORE no se proyecta (este motor no genera noticias
    # futuras) — se mantiene CONSTANTE en cada paso sintético.
    news_sentiment_value = base_indicator_state["news_sentiment_last"]

    # --- Anclas reales para el mecanismo de anchor-pull (ver ANCHOR_RETENTION
    # más abajo) — el ÚLTIMO valor real observado de cada indicador
    # price-level-dependiente y no auto-acotado (EMA_20/EMA_fast/EMA_slow/
    # MACD_SIGNAL/ATR_14). Se extraen una sola vez aquí, fuera del bucle,
    # porque son CONSTANTES por toda la simulación (el punto de referencia
    # es siempre "lo último realmente observado", nunca un valor sintético).
    ema_fast_anchor = float(base_indicator_state["ema_fast"])
    ema_slow_anchor = float(base_indicator_state["ema_slow"])
    ema_20_anchor = float(base_indicator_state["ema_20"])
    macd_signal_anchor = float(base_indicator_state["macd_signal"])
    atr_anchor = float(base_indicator_state["atr_prev"])

    # --- Anclas reales de OBV_14 / BB_WIDTH_20 (fix calibración T+5, ver
    # bloque ANCHOR_RETENTION arriba) — mismo criterio que EMA/ATR: último
    # estado real conocido, congelado una sola vez antes del bucle. BB_WIDTH
    # no tiene un escalar recursivo propio en `base_indicator_state` (se
    # recalcula fresco cada paso desde `price_window_arr`), así que su ancla
    # se deriva aquí con la MISMA fórmula que `_compute_bb_width` sobre la
    # ventana real de precios (`price_window`) — no es un valor nuevo, es el
    # BB_WIDTH_20 real del último día confirmado por mercado.
    obv_anchor = float(base_indicator_state["obv_prev"])
    _bb_tail_real = base_indicator_state["price_window"]
    _bb_mean_real = float(_bb_tail_real.mean())
    _bb_std_real = float(_bb_tail_real.std())
    bb_width_anchor = (
        (2 * BB_WIDTH_NUM_STD * _bb_std_real) / _bb_mean_real if _bb_mean_real != 0.0 else 0.0
    )

    forecast_dates: list[str] = []
    price_paths: list[np.ndarray] = []  # un array (n_sim,) por fecha proyectada

    # Capturados UNA vez en T+1 (100% data real) — ver bloque
    # _real_ood_dampening_factor arriba. Defaults neutros por si el propio
    # step 0 falla dentro del try/except del bucle.
    real_input_oob_frac = 0.0
    real_ood_dampening = 1.0

    forecast_incomplete = False
    for step_i in range(steps):
        try:
            # RNG local de este paso (determinista, ver bloque de semilla
            # arriba) — nunca `np.random` global. `step_keras_seed` deriva
            # un uint32 reproducible de la MISMA SeedSequence hija, para
            # sembrar también el generador de Keras (MC Dropout) justo antes
            # del forward pass, ver `with inference_lock` más abajo.
            step_rng = np.random.default_rng(step_seed_sequences[step_i])
            step_keras_seed = int(step_seed_sequences[step_i].generate_state(1)[0])

            # Batch de N ventanas -> 2D para el scaler (fit espera
            # (n_samples, n_features)) -> de vuelta a 3D (n_sim, lookback, n_features).
            flat_windows = cursor_windows.reshape(-1, cursor_windows.shape[-1])
            scaled_flat = feature_scaler.transform(flat_windows)
            scaled_windows = scaled_flat.reshape(n_sim, lookback, -1).astype(np.float32)

            # FIX DE FUGA AUTORREGRESIVA (auditoría línea-por-línea del
            # bucle — colapso geométrico en línea casi recta): `feature_scaler`
            # es un MinMaxScaler fit en train sobre el rango histórico REAL
            # (-> [0, 1]). `MinMaxScaler.transform()` NO clippea por diseño —
            # un valor fuera del rango de entrenamiento se EXTRAPOLA
            # linealmente sin límite (ver docstring del módulo, sección
            # "Pipeline de inferencia"). Cada paso de este bucle retroalimenta
            # indicadores 100% SINTÉTICOS (RSI/EMA/MACD/SENTIMENT/ATR/BB/OBV/
            # ADX/STOCH, todos derivados de `cursor_prices`) al tensor de
            # entrada del paso siguiente — si una trayectoria encadena varios
            # retornos negativos, sus indicadores se alejan CADA VEZ MÁS del
            # rango de entrenamiento, extrapolando MÁS lejos de [0, 1] paso a
            # paso. El modelo nunca vio esa región del espacio de entrada
            # durante el entrenamiento: su salida ahí no es una extrapolación
            # razonable, es indefinida — y en la práctica sesga
            # sistemáticamente hacia retornos aún más negativos, retroalimentando
            # la fuga en cada iteración (exactamente el patrón de colapso en
            # "línea casi recta" reportado). Acotar el tensor YA ESCALADO a
            # una banda alrededor de [0, 1] antes de CADA forward pass
            # garantiza que el modelo reciba siempre condiciones cercanas al
            # rango de entrenamiento, sin importar cuánto diverja
            # `cursor_prices` en espacio real — corta la fuga en la fuente,
            # antes de que contamine la predicción, en vez de intentar
            # corregirla después.
            #
            # SOFT-CLIP (no hard-clip): un `np.clip(scaled_windows, 0.0, 1.0)`
            # pinea TODO valor fuera de rango exactamente en 0.0/1.0 — cuando
            # eso alcanza 40-50% de los features (ver auditoría std/
            # clipped_frac en el log), trayectorias con divergencias muy
            # distintas en espacio real terminan con el MISMO tensor de
            # entrada, y la red pierde la capacidad de diferenciarlas
            # (aplanamiento de la textura direccional, ver bloque
            # SOFT_CLIP_MARGIN arriba). `_soft_clip_unit_interval` es
            # identidad exacta dentro de [0, 1] y comprime el excedente con
            # tanh de forma monótona/inyectiva en vez de pinearlo.
            oob_frac = float(np.mean((scaled_windows < 0.0) | (scaled_windows > 1.0)))
            if step_i == 0:
                # T+1 = ventana 100% real, cero contaminación sintética —
                # ver bloque _real_ood_dampening_factor arriba. Se congela
                # acá y se reusa en TODOS los pasos: la distancia al
                # manifold de entrenamiento del punto de partida no cambia
                # con el número de pasos sintéticos.
                real_input_oob_frac = oob_frac
                real_ood_dampening = _real_ood_dampening_factor(real_input_oob_frac)
            scaled_windows = _soft_clip_unit_interval(scaled_windows)
            asset_tensor = np.full((n_sim, 1), asset_id, dtype=np.int32)

            # MC Dropout real + batch de N trayectorias en UNA sola llamada:
            # `training=True` mantiene vivo `Dropout(0.4)` y, al ser un
            # batch, cada fila del tensor de salida muestrea su propia
            # máscara — N inferencias estocásticas genuinas por el precio de
            # UN forward pass, no de N. `inference_lock` protege esta única
            # llamada por paso (Plan A y Plan B corren en threads distintos
            # contra el mismo `model` cacheado — ver _build_investment_plans).
            #
            # `keras.utils.set_random_seed` va DENTRO del mismo lock, justo
            # antes del forward pass: el generador aleatorio de Keras es
            # estado GLOBAL del proceso (no hay uno por-thread sin tocar la
            # arquitectura del modelo ya entrenado/guardado). Fijarlo fuera
            # del lock dejaría una ventana de carrera real donde el thread de
            # Plan B pisa la semilla de Plan A entre el set y el forward
            # pass — atándolos al mismo lock, ambos ocurren como una unidad
            # atómica, así la máscara de MC Dropout queda reproducible por
            # (ticker, fecha, horizonte, paso) igual que el ruido gaussiano.
            with inference_lock:
                keras.utils.set_random_seed(step_keras_seed)
                y_pred_scaled = _call_keras_with_hang_guard(
                    model, [scaled_windows, asset_tensor], training=True
                )
            y_pred_np = keras.ops.convert_to_numpy(y_pred_scaled)
            if y_pred_np.shape != (n_sim, 2):
                raise ValueError(
                    f"Salida del modelo con shape inesperado {y_pred_np.shape} "
                    f"(se esperaba ({n_sim}, 2): [mu, log_var] dual-head por trayectoria)."
                )
            mu_scaled_batch = y_pred_np[:, 0:1]
            log_var_scaled_batch = y_pred_np[:, 1:2]
            r_hat_model_batch = target_scaler.inverse_transform(mu_scaled_batch).reshape(n_sim)
            # StandardScaler(with_mean=False): inverse_transform es y_raw = y_scaled * scale_;
            # la std escala igual (lineal) -> sigma_real = sigma_scaled * scale_[0]
            sigma_model_batch = (
                np.sqrt(np.exp(log_var_scaled_batch)).reshape(n_sim) * target_scaler.scale_[0]
            )
            # FIX (colapso estocástico / "electrocardiograma roto" en BTC):
            # `r_hat_model_batch` (la MEDIA) ya tenía cinturón de seguridad
            # (`model_signal_clip`, ver más abajo) contra la fuga
            # autorregresiva bajo inputs OOD — pero `sigma_model_batch` (la
            # VARIANZA, head log_var) NUNCA lo tuvo. `sqrt(exp(log_var))` es
            # EXPONENCIAL en la salida cruda de la red: bajo un input OOD
            # (ver `real_ood_dampening`/`oob_frac`, auditoría SPY 45.6% en
            # T+1 — y crypto, con su rango de precio/features moviéndose más
            # rápido que la ventana de entrenamiento de 2y, es el régimen más
            # expuesto) un log_var mal calibrado puede disparar
            # `sigma_model_batch` muy por encima de la volatilidad real del
            # activo. Ese valor entra directo a `combined_scale` ->
            # `noise_batch` -> `cursor_prices` sin ningún techo, produciendo
            # exactamente picos/caídas extremos en pasos consecutivos: no es
            # volatilidad real de BTC, es ruido gaussiano con una desviación
            # estándar que la propia red dejó de calibrar bien. Se acota con
            # el MISMO techo (`model_signal_clip`, sigma_hist real x
            # MODEL_SIGNAL_CLIP_SIGMAS) que ya protege a la media — defensa
            # en profundidad simétrica, no un número nuevo inventado.
            sigma_model_batch = np.clip(sigma_model_batch, 0.0, model_signal_clip)

            # AUDITORÍA DUAL-HEAD — un solo bloque (los dos que había se
            # solapaban), solo en T+1: un horizonte de 3M son 63 pasos x 2
            # planes (A/B) x ambos regímenes en paralelo — sin el guard
            # step_i==0 esto imprime cientos de bloques por request.
            if step_i == 0:
                logger.info(
                    "[AUDITORIA DUAL-HEAD] ticker=%s regime=%s y_pred_shape=%s "
                    "target_scaler.scale_[0]=%.8f mu_scaled[:3]=%s log_var_scaled[:3]=%s "
                    "sigma_model_batch[:3]=%s r_hat_model_batch[:3]=%s",
                    ticker, regime, y_pred_np.shape,
                    float(target_scaler.scale_[0]),
                    mu_scaled_batch[:3].flatten(), log_var_scaled_batch[:3].flatten(),
                    sigma_model_batch[:3], r_hat_model_batch[:3],
                )

            # Decay de señal (ver SIGNAL_DECAY_BASE arriba): step_i=0 -> 100%
            # de la señal del modelo; cada paso adicional la atenúa
            # geométricamente. Es lo ÚNICO que se decae — el ruido i.i.d.
            # (`noise_batch`, más abajo) sigue intacto, así que el ancho
            # P5-P95 del cono sigue creciendo con sqrt(steps) como un random
            # walk real; solo se descuenta cuánto pesa la convicción
            # direccional COMPARTIDA del modelo (la que, sin decay, arrastraba
            # a P5 Y P95 por igual en la misma dirección).
            signal_decay = SIGNAL_DECAY_BASE ** step_i
            # `real_ood_dampening` (ver bloque arriba, capturado en T+1):
            # penaliza la MAGNITUD de la señal determinista en proporción a
            # cuán lejos del rango de entrenamiento arrancó el forecast —
            # independiente de `signal_decay`, que solo penaliza por número
            # de pasos sintéticos acumulados. Con datos in-distribution
            # (oob_frac en T+1 <= 15%) este factor es 1.0 y no cambia nada
            # del comportamiento anterior.
            r_hat_model_batch = r_hat_model_batch * signal_decay * real_ood_dampening * EDGE_SHRINKAGE_FACTOR

            # Cinturón de seguridad (ver MODEL_SIGNAL_CLIP_SIGMAS arriba):
            # acota SOLO la señal cruda de la red a ±5 sigma_hist antes de
            # sumar el ruido — corta el sesgo direccional de la fuga
            # autorregresiva sin tocar la varianza real inyectada abajo. Se
            # conserva intacto como techo absoluto incluso con el decay ya
            # aplicado (defensa en profundidad, no defensa única).
            r_hat_model_batch = np.clip(r_hat_model_batch, -model_signal_clip, model_signal_clip)

            # Inyección de Volatilidad Estocástica — un draw i.i.d. POR
            # TRAYECTORIA, ocurre aquí sobre el vector `r_hat_model_batch` ya
            # en espacio real (post inverse_transform), nunca sobre los
            # tensores de entrada del modelo. La varianza entre trayectorias
            # (y por tanto el ancho de la banda P5-P95) crece de forma
            # natural con sqrt(steps), como un random walk real.
            combined_scale = np.sqrt(sigma_model_batch ** 2 + noise_scale ** 2)
            noise_batch = step_rng.normal(loc=0.0, scale=combined_scale, size=n_sim)
            r_hat_batch = r_hat_model_batch + noise_batch
        except Exception as exc:  # noqa: BLE001
            print(
                f"⚠️ Paso T+{step_i + 1}/{steps} de {ticker} falló en el modelo Keras ({exc}) — "
                "se devuelve el forecast probabilístico acumulado hasta aquí, sin inventar el resto."
            )
            forecast_incomplete = True
            break

        prev_prices = cursor_prices
        cursor_prices = cursor_prices * np.exp(r_hat_batch)

        # DIAGNÓSTICO (fix calibración T+5): antes solo se logueaba el ÚLTIMO
        # paso. Se agregan dos checkpoints más para poder atribuir la causa
        # empíricamente en el próximo forecast real: T+1 (ventana 100% real,
        # CERO contaminación sintética — si `oob_frac` ya es alto acá, la
        # causa es un escalador desactualizado frente al precio/OBV real
        # actual, no drift autorregresivo) y T+5 (mismo checkpoint que la
        # auditoría original que documentó el colapso, para comparar
        # antes/después de este fix con la misma referencia exacta). Con
        # soft-clip, `scaled_windows` casi nunca es EXACTAMENTE 0.0/1.0
        # (asíntota, no pineado) — esa comparación dejó de ser una señal
        # útil. `oob_frac` (capturado ANTES del soft-clip, arriba) mide lo
        # mismo que antes medía `clipped_frac`: presión de extrapolación
        # real sobre el tensor, sin depender de si el guard es duro o suave.
        if step_i in (0, 4, steps - 1):
            _step_label = {
                0: "T+1, ventana 100% real — sin contaminación sintética",
                4: "T+5, checkpoint de la auditoría original del colapso",
            }.get(step_i, f"T+{step_i + 1}, último paso completado")
            print(f"🔎 {ticker} step {step_i} [{_step_label}]: std(cursor_prices)={cursor_prices.std():.4f} | "
                  f"fracción de features fuera de rango de entrenamiento (pre soft-clip)={oob_frac:.4%}")
        # FIX CALENDARIO (Timezones/Data Gaps): saltar sábado/domingo es
        # correcto SOLO para equities (L-V, sesión ancla en la timezone del
        # exchange, ver `close_df.index` real de yfinance). Crypto
        # (`regime == "crypto_specialist"`, ver REGIME_TICKERS) cotiza 24/7
        # en UTC — su propio histórico real (`close_df.index`) YA incluye
        # sábados/domingos, así que saltarlos acá desalineaba el eje
        # temporal del forecast contra la realidad del activo (fechas
        # proyectadas más lejanas de lo real, "1w" dejaba de ser 7 días
        # calendario reales para BTC/ETH).
        cursor_date = cursor_date + timedelta(days=1)
        if regime != "crypto_specialist":
            while cursor_date.weekday() >= 5:
                cursor_date = cursor_date + timedelta(days=1)
        forecast_dates.append(cursor_date.strftime("%Y-%m-%d"))
        price_paths.append(cursor_prices.copy())

        # Recalcular momentum incrementalmente por trayectoria — jamás
        # congelarlo ni compartirlo entre trayectorias.
        delta_arr = cursor_prices - prev_prices
        gain_arr = np.maximum(delta_arr, 0.0)
        loss_arr = np.maximum(-delta_arr, 0.0)
        avg_gain_arr = avg_gain_arr * (1 - RSI_ALPHA) + gain_arr * RSI_ALPHA
        avg_loss_arr = avg_loss_arr * (1 - RSI_ALPHA) + loss_arr * RSI_ALPHA
        safe_avg_loss_arr = np.where(avg_loss_arr == 0.0, 1.0, avg_loss_arr)
        rs_arr = avg_gain_arr / safe_avg_loss_arr
        # misma convención que _compute_rsi: avg_loss==0 -> NaN -> fillna(50)
        rsi_arr = np.where(avg_loss_arr == 0.0, 50.0, 100.0 - (100.0 / (1.0 + rs_arr)))

        ema_fast_arr = cursor_prices * EMA_FAST_K + ema_fast_arr * (1 - EMA_FAST_K)
        ema_slow_arr = cursor_prices * EMA_SLOW_K + ema_slow_arr * (1 - EMA_SLOW_K)
        ema_20_arr = cursor_prices * EMA20_K + ema_20_arr * (1 - EMA20_K)
        # Anchor-pull (ver bloque ANCHOR_RETENTION arriba): la fórmula EWM de
        # arriba queda intacta (misma matemática honesta); esto solo
        # amortigua cuánto puede alejarse cada EMA del último valor real
        # antes del siguiente forward pass — se aplica ANTES de derivar MACD
        # para que el amortiguamiento se propague también a macd_new_arr.
        ema_fast_arr = _anchor_pull(ema_fast_arr, ema_fast_anchor)
        ema_slow_arr = _anchor_pull(ema_slow_arr, ema_slow_anchor)
        ema_20_arr = _anchor_pull(ema_20_arr, ema_20_anchor)
        macd_new_arr = ema_fast_arr - ema_slow_arr
        macd_signal_arr = macd_new_arr * MACD_SIGNAL_K + macd_signal_arr * (1 - MACD_SIGNAL_K)
        macd_signal_arr = _anchor_pull(macd_signal_arr, macd_signal_anchor)

        # SENTIMENT_SCORE recursivo (Hallazgo F5 — correlación de Pearson
        # móvil activo/VIX, ver _compute_sentiment_score): el log-return
        # SINTÉTICO recién generado del activo (r_hat_batch, ya con
        # volatilidad inyectada) entra a la ventana deslizante DE CADA
        # TRAYECTORIA; el de VIX (congelado, ver `log_return_window_vix`
        # arriba) es COMPARTIDO y se desliza añadiendo 0.0. La correlación
        # se recalcula por trayectoria sobre ambas ventanas — misma fórmula
        # exacta que `_compute_sentiment_score()` (Pearson vía media/std
        # normalizadas; el factor n/(n-1) de la covarianza muestral se
        # cancela entre numerador y denominador), solo que aquí es online y
        # vectorizada por trayectoria en vez de `pd.Series.rolling().corr()`.
        log_return_window_arr = np.concatenate(
            [log_return_window_arr[:, 1:], r_hat_batch[:, np.newaxis]], axis=1
        )
        log_return_window_vix = np.concatenate([log_return_window_vix[1:], [0.0]])

        asset_mean_arr = log_return_window_arr.mean(axis=1)
        asset_std_arr = log_return_window_arr.std(axis=1)
        vix_mean = log_return_window_vix.mean()
        vix_std = log_return_window_vix.std()
        cov_arr = ((log_return_window_arr - asset_mean_arr[:, np.newaxis]) * (log_return_window_vix - vix_mean)).mean(axis=1)
        denom_arr = asset_std_arr * vix_std
        safe_denom_arr = np.where(denom_arr == 0.0, 1.0, denom_arr)
        sentiment_arr = np.clip(
            np.where(denom_arr == 0.0, 0.0, cov_arr / safe_denom_arr),
            -1.0, 1.0,
        )

        # --- V3: ATR_14 recursivo. Vela sintética DEGENERADA (High=Low=
        # Close del día proyectado — este motor no modela rango intradía):
        # con H=L=C, True Range = max(H-L, |H-prevC|, |L-prevC|) se reduce
        # EXACTAMENTE a |C_t - C_{t-1}| en los 3 términos (ver _compute_atr).
        # El EWM de Wilder real se mantiene: atr_t = atr_{t-1}*(1-ATR_ALPHA)
        # + TR_t*ATR_ALPHA. Aproximación honesta: subestima el ATR real al
        # ignorar rango intradía, documentado igual que el resto de
        # indicadores sintéticos de este bucle.
        tr_arr = np.abs(delta_arr)
        atr_arr = atr_arr * (1 - ATR_ALPHA) + tr_arr * ATR_ALPHA
        # Anchor-pull (ver ANCHOR_RETENTION arriba) — misma razón que EMA/MACD:
        # ATR_14 es magnitud absoluta de precio (no auto-acotado como
        # RSI/STOCH/ADX), así que sin esto puede alejarse del rango de
        # entrenamiento tan rápido como el propio precio diverge.
        atr_arr = _anchor_pull(atr_arr, atr_anchor)

        # --- V5: ADX_14 recursivo. Vela sintética degenerada (H=L=C) reduce
        # +DM/-DM EXACTAMENTE a gain_arr/loss_arr (misma derivación que
        # ATR_14 arriba) — reutiliza `atr_arr` (idéntica fórmula/periodo)
        # como denominador de +DI/-DI, sin estado adicional.
        plus_dm_smooth_arr = plus_dm_smooth_arr * (1 - ATR_ALPHA) + gain_arr * ATR_ALPHA
        minus_dm_smooth_arr = minus_dm_smooth_arr * (1 - ATR_ALPHA) + loss_arr * ATR_ALPHA
        safe_atr_arr = np.where(atr_arr == 0.0, 1.0, atr_arr)
        plus_di_arr = 100 * plus_dm_smooth_arr / safe_atr_arr
        minus_di_arr = 100 * minus_dm_smooth_arr / safe_atr_arr
        di_sum_arr = plus_di_arr + minus_di_arr
        safe_di_sum_arr = np.where(di_sum_arr == 0.0, 1.0, di_sum_arr)
        dx_arr = np.where(di_sum_arr == 0.0, 0.0, 100 * np.abs(plus_di_arr - minus_di_arr) / safe_di_sum_arr)
        adx_arr = adx_arr * (1 - ATR_ALPHA) + dx_arr * ATR_ALPHA

        # --- BB_WIDTH_20 recursivo.
        # sintéticos (no retornos) — mismo patrón que log_return_window_arr,
        # a nivel precio en vez de log-return.
        price_window_arr = np.concatenate(
            [price_window_arr[:, 1:], cursor_prices[:, np.newaxis]], axis=1
        )
        bb_mean_arr = price_window_arr.mean(axis=1)
        bb_std_arr = price_window_arr.std(axis=1)
        safe_bb_mean_arr = np.where(bb_mean_arr == 0.0, 1.0, bb_mean_arr)
        bb_width_arr = np.where(
            bb_mean_arr == 0.0,
            0.0,
            (2 * BB_WIDTH_NUM_STD * bb_std_arr) / safe_bb_mean_arr,
        )
        # Anchor-pull (fix calibración T+5, ver bloque ANCHOR_RETENTION
        # arriba) — BB_WIDTH_20 es atado al nivel de precio (no auto-acotado)
        # igual que EMA/MACD_SIGNAL/ATR; hasta ahora quedaba fuera del
        # estabilizador pese a estar identificado como tal en el bloque
        # soft-clip.
        bb_width_arr = _anchor_pull(bb_width_arr, bb_width_anchor)

        # --- V3: OBV recursivo. sign(ΔP) * volumen PROXY constante (ver
        # `volume_proxy` arriba) — este motor no proyecta volumen futuro,
        # solo precio, así que el volumen sintético de cada paso reutiliza
        # el promedio histórico reciente en vez de inventar una cifra nueva.
        direction_arr = np.sign(delta_arr)
        obv_arr = obv_arr + direction_arr * volume_proxy
        # Anchor-pull (fix calibración T+5, ver bloque ANCHOR_RETENTION
        # arriba) — OBV es una SUMA ACUMULADA sin ningún término de
        # reversión propio (a diferencia de una EWM, que se amortigua sola).
        # Sin esto, una racha de `direction_arr` con el mismo signo entre
        # trayectorias (esperable mientras la señal direccional compartida
        # aún no decayó del todo) la aleja del rango de entrenamiento de
        # forma monótona y acelerada — el candidato más plausible al sesgo
        # COMPARTIDO (P5 y P95 cayendo juntos) reportado en producción.
        obv_arr = _anchor_pull(obv_arr, obv_anchor)

        # --- V6: OBV_ROC_20 — la feature que REALMENTE entra al tensor (ver
        # `new_rows` más abajo). `obv_arr` (crudo, recursión sin cambios
        # arriba) sigue siendo el estado interno; se lee el valor que ESTÁ
        # por salir de la ventana (el de hace exactamente
        # OBV_ROC_LOOKBACK_DAYS pasos, ANTES de deslizarla) como denominador
        # — misma fórmula y mismo clip que `engineer_asset()` en
        # train_kodaquant_v5.py (`obv.pct_change(20)` con denominador cero
        # ->0.0 en vez de inf, luego clip a ±OBV_ROC_CLIP).
        obv_denom_arr = obv_window_arr[:, 0]  # valor de hace 20 pasos, ANTES del slide
        safe_obv_denom_arr = np.where(obv_denom_arr == 0.0, 1.0, obv_denom_arr)
        obv_roc_arr = np.clip(
            np.where(obv_denom_arr == 0.0, 0.0, (obv_arr - obv_denom_arr) / safe_obv_denom_arr),
            -OBV_ROC_CLIP, OBV_ROC_CLIP,
        )
        obv_window_arr = np.concatenate(
            [obv_window_arr[:, 1:], obv_arr[:, np.newaxis]], axis=1
        )

        # --- STOCH_K_14 recursivo. Ventana deslizante de H/L (degenerada a
        # cursor_prices en pasos sintéticos, real en el seed histórico).
        stoch_high_window_arr = np.concatenate(
            [stoch_high_window_arr[:, 1:], cursor_prices[:, np.newaxis]], axis=1
        )
        stoch_low_window_arr = np.concatenate(
            [stoch_low_window_arr[:, 1:], cursor_prices[:, np.newaxis]], axis=1
        )
        highest_high_arr = stoch_high_window_arr.max(axis=1)
        lowest_low_arr = stoch_low_window_arr.min(axis=1)
        stoch_denom_arr = highest_high_arr - lowest_low_arr
        safe_stoch_denom_arr = np.where(stoch_denom_arr == 0.0, 1.0, stoch_denom_arr)
        stoch_k_arr = np.where(
            stoch_denom_arr == 0.0, 50.0,
            100 * (cursor_prices - lowest_low_arr) / safe_stoch_denom_arr,
        )

        new_rows = cursor_windows[:, -1, :].copy()
        # --- V6: proyección a features ESTACIONARIAS. La recursión interna
        # de ema_fast_arr/ema_slow_arr/ema_20_arr/macd_signal_arr/atr_arr
        # arriba NO cambió (misma matemática honesta, mismo anchor-pull) —
        # lo único que cambia es QUÉ se escribe en el tensor que ve el
        # modelo: razones adimensionales en vez de niveles absolutos, para
        # que MinMaxScaler nunca vuelva a extrapolar sin límite ante un
        # nuevo máximo histórico de precio.
        new_rows[:, 0] = r_hat_batch  # LOG_RETURN_1D — el retorno recién generado, ya en espacio real
        new_rows[:, 1] = rsi_arr
        safe_ema_20_arr = np.where(ema_20_arr == 0.0, 1.0, ema_20_arr)
        new_rows[:, 2] = np.where(
            ema_20_arr == 0.0, 0.0, (cursor_prices - ema_20_arr) / safe_ema_20_arr
        )  # EMA20_DEV_PCT
        safe_cursor_prices = np.where(cursor_prices == 0.0, 1.0, cursor_prices)
        new_rows[:, 3] = macd_new_arr / safe_cursor_prices        # MACD_PCT
        new_rows[:, 4] = macd_signal_arr / safe_cursor_prices     # MACD_SIGNAL_PCT
        new_rows[:, 5] = sentiment_arr
        new_rows[:, 6] = atr_arr / safe_cursor_prices              # ATR_PCT
        new_rows[:, 7] = bb_width_arr
        new_rows[:, 8] = obv_roc_arr                                # OBV_ROC_20
        new_rows[:, 9] = adx_arr
        new_rows[:, 10] = stoch_k_arr
        new_rows[:, 11] = news_sentiment_value
        # macro_tickers__LOGRET (índice 12+, tras LOG_RETURN_1D + 11
        # técnicos): este motor no proyecta macro a futuro. Con los factores
        # macro ahora en ESPACIO DE RETORNO (ver `_fetch_feature_window`),
        # "sin forecast" significa retorno diario = 0.0 (macro plano) — NO
        # "repetir para siempre el último retorno real observado", que
        # compondría un drift falso (ej. VIX +1.3% cada día del horizonte
        # si el último día real cerró con ese retorno). Antes del fix del
        # bug de paridad train/serve, esta columna guardaba precio crudo
        # congelado, que SÍ equivalía correctamente a "retorno 0%" — esta
        # asignación explícita preserva esa misma intención en el nuevo
        # espacio de retornos.
        new_rows[:, 12:] = 0.0
        cursor_windows = np.concatenate([cursor_windows[:, 1:, :], new_rows[:, np.newaxis, :]], axis=1)

    if price_paths:
        price_matrix = np.stack(price_paths, axis=0)  # (steps_completados, n_sim)
        lower_arr = np.percentile(price_matrix, CONFIDENCE_LOWER_PERCENTILE, axis=1)
        expected_arr = np.percentile(price_matrix, CONFIDENCE_MEDIAN_PERCENTILE, axis=1)
        upper_arr = np.percentile(price_matrix, CONFIDENCE_UPPER_PERCENTILE, axis=1)

        # --- Trayectoria representativa (REAL, no fabricada) ----------------
        # `expected_arr` es un percentil calculado POR FECHA de forma
        # independiente entre columnas — un corte transversal de las N
        # simulaciones en cada instante t, no un camino muestral. Colapsar
        # N caminatas i.i.d. fecha por fecha cancela parcialmente el ruido
        # diario de cada trayectoria individual (ley de los grandes
        # números): por construcción sale más suave que CUALQUIER camino
        # individual, aunque cada camino individual (ver `noise_batch` +
        # MC Dropout arriba) sea genuinamente ruidoso.
        #
        # FIX (variance collapse en sample_path): la versión anterior medía
        # distancia a la mediana ACUMULADA sobre todo el horizonte
        # (np.linalg.norm(..., axis=0) sumando los `steps` pasos). Eso sesga
        # el argmin hacia la trayectoria MENOS ruidosa de las N=100 -- una
        # que se aparta poco de la mediana EN CADA paso intermedio tiene
        # distancia total más chica que una con zigzag diario genuino,
        # aunque ambas terminen en un lugar similar. Con N=100 draws casi
        # siempre hay una lo bastante "lisa" para ganar el argmin, matando
        # la textura real que este campo existe para mostrar.
        #
        # Comparar SOLO en el día terminal preserva la consistencia
        # narrativa (expected_path/predicted_return_pct se calculan sobre
        # el último día) sin penalizar el ruido genuino de los días
        # intermedios -- ya no premia trayectorias artificialmente suaves.
        dist_to_median_terminal = np.abs(price_matrix[-1, :] - expected_arr[-1])
        representative_idx = int(np.argmin(dist_to_median_terminal))
        representative_arr = price_matrix[:, representative_idx]
    else:
        lower_arr = expected_arr = upper_arr = representative_arr = np.array([])

    target_price = float(expected_arr[-1]) if expected_arr.size else last_price
    total_return_pct = (target_price / last_price - 1) * 100

    forecast_points = [
        {
            "date": d,
            "lower_bound": round(float(lo), 2),
            "expected_path": round(float(ex), 2),
            "upper_bound": round(float(hi), 2),
            # Camino real (una de las N simulaciones), no un percentil —
            # ver nota "Trayectoria representativa" arriba. Úsalo para
            # DIBUJAR la línea central; usa `expected_path` (mediana) para
            # cifras/decisiones (retorno %, señal táctica), que deben ser
            # robustas a un solo draw ruidoso.
            "sample_path": round(float(sp), 2),
        }
        for d, lo, ex, hi, sp in zip(forecast_dates, lower_arr, expected_arr, upper_arr, representative_arr)
    ]
    predicted_path = [p["expected_path"] for p in forecast_points]
    tactical_signal = _find_optimal_entry_exit(
        forecast_points,
        anchor_date=anchor_date_str,
        anchor_price=last_price_quote,
    )

    return {
        "ticker": ticker,
        "last_price": last_price_quote,
        "predicted_price": round(target_price, 2),
        "predicted_return_pct": round(total_return_pct, 2),
        "trend": "alcista" if total_return_pct >= 0 else "bajista",
        "historical": historical,
        "forecast": forecast_points,
        "predicted_path": predicted_path,
        "tactical_signal": tactical_signal,
        "forecast_incomplete": forecast_incomplete,
        "variance_source": variance_source,
        "n_simulations": n_sim,
        # Auditoría de la memoria diaria (ver bloque "SEMILLA DETERMINISTA"
        # arriba): dos responses con el mismo `deterministic_seed` son,
        # verificablemente, la MISMA corrida — mismo (ticker, fecha UTC,
        # horizonte). Cambia solo al cruzar medianoche UTC o al pedir un
        # horizonte distinto, nunca entre requests idénticos del mismo día.
        "deterministic_seed": base_seed,
        "confidence_interval": f"P{CONFIDENCE_LOWER_PERCENTILE}-P{CONFIDENCE_UPPER_PERCENTILE}",
        # Diagnóstico honesto (auditoría SPY oob_frac=45.6% en T+1, ver
        # bloque _real_ood_dampening_factor): cuánto del input REAL en T+1
        # ya caía fuera del rango de entrenamiento, y si por eso se
        # atenuó la magnitud de la señal del modelo. No reemplaza un
        # retrain — solo evita mostrar un forecast degradado con la misma
        # confianza aparente que uno in-distribution.
        "real_data_ood_frac": round(real_input_oob_frac, 4),
        "real_ood_dampening_applied": round(real_ood_dampening, 4),
        "forecast_reliability": "degraded" if real_input_oob_frac > REAL_OOD_DAMPENING_START else "normal",
    }


def _fmt_plan_math_line(plan: dict, horizon_days: int) -> str:
    """
    Cheat-sheet de una línea por plan para el prompt. CRÍTICO: calcula la
    ganancia en USD explícitamente aquí (capital * %return) para que el LLM
    NUNCA tenga que inferirla — jamás debe confundir el precio de cotización
    del activo (ej. $745.76/acción) con la ganancia real sobre el capital
    invertido (ej. $0.05 USD sobre $6 invertidos).
    """
    forecast = plan.get("forecast")
    if forecast:
        ganancia_estimada_usd = round(
            plan["monto_usd"] * forecast["predicted_return_pct"] / 100, 2
        )
        prediction = (
            f"Precio de cotización: {_fmt_money(forecast['last_price'])} → "
            f"{_fmt_money(forecast['predicted_price'])} "
            f"({forecast['predicted_return_pct']:+.2f}%, {forecast['trend']}) | "
            f"Ganancia estimada de la cartera: {_fmt_money(ganancia_estimada_usd)}"
        )
    else:
        prediction = "no disponible"
    return (
        f"Activo={plan['activo_referencia']} | Porcentaje={plan['pct']} | "
        f"Capital invertido={_fmt_money(plan['monto_usd'])} | Horizonte={horizon_days} días hábiles | "
        f"{prediction}"
    )


# ---------------------------------------------------------------------------
# Risk profiling — el risk_score (0-100) es la ÚNICA fuente de verdad del
# split Plan A (Reserva) / Plan B (Riesgo). risk_profile es solo etiqueta.
# ---------------------------------------------------------------------------

_DEFAULT_RISK_SCORE = 50
_RISK_PROFILE_LABELS = {"conservador", "moderado", "agresivo"}
_DEFAULT_RISK_PROFILE = "moderado"


def _clamp_risk_score(risk_score: int | None) -> int:
    if not isinstance(risk_score, int) or isinstance(risk_score, bool):
        return _DEFAULT_RISK_SCORE
    return max(0, min(100, risk_score))


def _normalize_risk_profile(risk_profile: str | None) -> str:
    normalized = (risk_profile or "").strip().lower()
    return normalized if normalized in _RISK_PROFILE_LABELS else _DEFAULT_RISK_PROFILE


def _resolve_risk_split(risk_score: int | None) -> dict[str, float]:
    """
    risk_score=85 -> 85% Plan B (Riesgo) / 15% Plan A (Reserva), continuo,
    sin buckets. Reemplaza cualquier split fijo (60/40, 50/50, etc.).
    """
    score = _clamp_risk_score(risk_score)
    plan_b_weight = score / 100
    return {"plan_a": round(1 - plan_b_weight, 4), "plan_b": round(plan_b_weight, 4)}


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# ---------------------------------------------------------------------------
# FIX CRÍTICO — DESFASE DE ESTADO (alucinación narrativa): `_build_investment_
# plans` llamaba SIEMPRE a `_resolve_risk_split(risk_score)` — el split
# MANUAL del slider — sin importar que `main.py` ya hubiera resuelto un split
# distinto vía `risk_manager.calculate_portfolio` (Auto-Allocation IA, modo
# discovery, dictado por el `market_signal` real del radar/Alpha Seeker).
# Ese `portfolio_allocation` SÍ llegaba como parámetro a
# `generate_quanti_strategy_stream`, pero solo se reenviaba crudo al frontend
# en `meta_payload["allocation"]` — el frontend lo usaba para PISAR las
# tarjetas de Riesgo/Reserva (ver useSignalConsole.js), mientras que
# `investment_plans` (y por lo tanto `_build_tactical_operation` y el prompt
# del LLM) seguían calculando sobre el split manual del risk_score, ya
# invisible en pantalla. Resultado: dos verdades del split coexistiendo
# (75/25 en las tarjetas, 50/50 narrado por Quanti) y una ganancia neta
# (`net_profit_usd`) calculada sobre un capital que el usuario nunca vio
# asignado.
#
# `_resolve_effective_allocation` es ahora el ÚNICO punto que decide el
# split real de capital, ANTES de construir `investment_plans` — todo lo
# demás (amounts de las tarjetas, operación táctica, prompt del LLM) lee
# de esta única fuente. Usa los USD exactos que ya calculó
# `risk_manager.calculate_portfolio` (no los re-deriva multiplicando
# fracciones contra `budget_usd` de nuevo) para que no exista ni un
# centavo de drift por redondeo entre lo que ve el usuario y lo que narra
# Quanti.
# ---------------------------------------------------------------------------

def _resolve_effective_allocation(
    budget_usd: float,
    portfolio_allocation: dict | None,
    risk_score: int | None,
    analysis_mode: str,
) -> dict[str, Any]:
    """
    Retorna el split de capital REAL que debe gobernar tarjetas, operación
    táctica y narrativa — todos desde el mismo objeto, cero duplicidad:
      - modo 'discovery' CON asignación IA válida (`asignacion.riesgo_pct`/
        `reserva_pct`/`riesgo_usd`/`reserva_usd`, todos números finitos):
        la Auto-Allocation IA manda. `source`='ai_auto_allocation'.
      - modo 'specific', o 'discovery' sin asignación IA válida (radar
        degradado, calculate_portfolio no disponible): cae al perfilador
        manual (`_resolve_risk_split(risk_score)`) — el comportamiento
        previo, nunca un 50/50 mudo. `source`='manual_risk_profile'.
    """
    if analysis_mode == "discovery" and isinstance(portfolio_allocation, dict):
        asignacion = portfolio_allocation.get("asignacion")
        if isinstance(asignacion, dict):
            riesgo_pct = asignacion.get("riesgo_pct")
            reserva_pct = asignacion.get("reserva_pct")
            riesgo_usd = asignacion.get("riesgo_usd")
            reserva_usd = asignacion.get("reserva_usd")
            if all(_is_finite_number(v) for v in (riesgo_pct, reserva_pct, riesgo_usd, reserva_usd)):
                return {
                    "plan_a_usd": round(float(reserva_usd), 2),
                    "plan_a_pct": float(reserva_pct),
                    "plan_b_usd": round(float(riesgo_usd), 2),
                    "plan_b_pct": float(riesgo_pct),
                    "source": "ai_auto_allocation",
                    "signal_detectada": portfolio_allocation.get("signal_detectada"),
                    "mensaje_estrategico": portfolio_allocation.get("mensaje_estrategico"),
                }

    split = _resolve_risk_split(risk_score)
    return {
        "plan_a_usd": round(budget_usd * split["plan_a"], 2),
        "plan_a_pct": split["plan_a"],
        "plan_b_usd": round(budget_usd * split["plan_b"], 2),
        "plan_b_pct": split["plan_b"],
        "source": "manual_risk_profile",
        "signal_detectada": None,
        "mensaje_estrategico": None,
    }


def _candidate_alpha_score(candidate: dict) -> float:
    """
    Alpha Score real: retorno proyectado CON SIGNO (nunca abs()) ponderado
    por la confianza señal/ruido del propio forecast (ver
    `_confidence_score`/`_rank_alpha_seeker_candidates` en prediccion.py).
    Reemplaza el `risk_score` fantasma que ningún productor llenaba jamás
    — con esa clave siempre en 0 para todos los candidatos, `max()`
    devolvía en silencio el PRIMER elemento de la lista sin importar cuál
    activo era realmente el mejor. Este score es la única fuente de
    verdad de la selección.
    """
    growth = candidate.get("predicted_return_pct", candidate.get("projected_growth_pct", 0.0)) or 0.0
    confidence = candidate.get("confidence_score", 0.0) or 0.0
    return float(growth) * (1.0 + float(confidence))


def _resolve_plan_b_ticker(
    radar_data: dict,
    analysis_mode: str = "discovery",
    target_asset: str | None = None,
) -> tuple[str, bool]:
    """
    Retorna (ticker_resuelto, degraded). degraded=True cuando el usuario
    pidió un target_asset fuera del universo entrenado y se sustituyó por
    discovery — el caller DEBE propagar esto al frontend para que el chart
    no muestre en silencio un activo distinto al seleccionado.

    Alpha Seeker (modo discovery / "Quanti's Choice"): escanea TODO el
    universo real vía `radar_data["top_assets"]` (contrato nuevo, ver
    `_rank_alpha_seeker_candidates` en prediccion.py; compat retro con la
    clave legacy "recommendations" si algún caller viejo sigue mandando
    ese payload) y elige el candidato con mayor `_candidate_alpha_score`
    ENTRE LOS QUE TIENEN EDGE POSITIVO. Un activo con retorno proyectado
    negativo NUNCA puede ganar la elección — "Quanti's Choice" promete la
    mejor oportunidad real, no el movimiento con mayor magnitud (positiva
    o negativa). PLAN_A_TICKER se excluye del pool: ya está cubierto por
    Plan A, elegirlo también para Plan B rompería la diversificación.
    """
    degraded = False
    if analysis_mode == "specific" and target_asset:
        normalized_ticker = target_asset.strip().upper()
        try:
            scalers = _get_scalers(_regime_for_ticker(normalized_ticker))
        except (ValueError, FileNotFoundError):
            scalers = None

        if scalers and normalized_ticker in scalers["asset_to_id"]:
            return normalized_ticker, False

        print(
            f"⚠️ target_asset '{target_asset}' fuera del universo entrenado "
            "— degradando a 'discovery' vía radar."
        )
        degraded = True

    candidates = radar_data.get("top_assets") or radar_data.get("recommendations") or []
    candidates = [
        c for c in candidates
        if (c.get("symbol") or c.get("ticker", "")).strip().upper() != PLAN_A_TICKER
    ]

    profitable = [c for c in candidates if _candidate_alpha_score(c) > 0]
    # Sin ningún candidato con edge positivo real, se degrada al mejor
    # disponible del universo escaneado (puede incluir negativos) — sigue
    # siendo dato real del motor, nunca el hardcode ciego de abajo, que
    # solo se alcanza si el radar entero vino vacío.
    pool = profitable or candidates
    if not pool:
        return "NVDA", degraded

    best = max(pool, key=_candidate_alpha_score)
    symbol = (best.get("symbol") or best.get("ticker") or "NVDA").strip().upper()
    return symbol, degraded


def _derive_capital_signal(radar_data: dict) -> str:
    """
    Auto-Allocation: clasifica el radar REAL (`generate_predictions`, ver
    services/prediccion.py) en el enum COMPRA/VENTA/NEUTRAL que consume
    `risk_manager.calculate_portfolio` — activa el split de capital
    dictado por IA (antes código muerto: nada escribía nunca
    `radar_data["signal"]`, así que `calculate_portfolio` caía SIEMPRE en
    NEUTRAL 50/50 sin importar el mercado).

    Deliberadamente NO reutiliza `_extract_market_signal` — esa función
    arma una frase narrativa para el prompt del LLM ("volatilidad=...,
    macro='...'"), no un enum clasificable; pasarla tal cual a
    risk_manager.py caería siempre en su rama "señal no reconocida".

    Cero heurística nueva: solo LEE el veredicto que Alpha Seeker
    (`_rank_alpha_seeker_candidates` en prediccion.py) ya calculó.
      COMPRA:  `top_assets` no vacío -> existe al menos una oportunidad
               real de grado compra en el universo entrenado.
      VENTA:   sin oportunidades de compra Y el mover más fuerte del
               radar (`recommendations[0]`) es una señal de VENTA real.
      NEUTRAL: cualquier otro caso (radar vacío/degradado, mercado mixto).
    """
    top_assets = radar_data.get("top_assets") or []
    if top_assets:
        return "COMPRA"

    recommendations = radar_data.get("recommendations") or []
    if recommendations and str(recommendations[0].get("action", "")).strip().upper() == "VENTA":
        return "VENTA"

    return "NEUTRAL"


# ---------------------------------------------------------------------------
# SentimentEnricher — capa async de "Visión Fundamental". Independiente del
# pipeline Keras: no toca `scaled_window` ni el bucle autoregresivo, solo
# enriquece el payload de salida con contexto macro antes del dispatch.
# Circuit Breaker propio — un fallo aquí NUNCA debe tumbar la proyección
# Monte Carlo ya calculada (ver except al fondo de get_market_sentiment).
# ---------------------------------------------------------------------------

# FIX (Circuit Breaker por "timeout" que no era lentitud de red): 2.5s
# alcanza justo para un round-trip de yfinance en condiciones ideales, pero
# CERO margen para jitter real de red o para una cola momentánea en el
# executor. Con `_SENTIMENT_EXECUTOR` dedicado (arriba) la contención con
# Keras ya no existe, así que este valor ahora refleja tolerancia real a
# latencia de red, no un colchón para encubrir contención de threads.
SENTIMENT_API_TIMEOUT_SECONDS = 6.0
SENTIMENT_MOMENTUM_LOOKBACK_DAYS = 21  # ~1 mes bursátil
SENTIMENT_BULLISH_THRESHOLD = 0.2
SENTIMENT_BEARISH_THRESHOLD = -0.2

# Caché TTL en memoria — PLAN_A_TICKER (SPY) y el ticker de Plan B se
# consultan en CADA request de /consult y /radar; sin caché, requests
# concurrentes disparan N fetches redundantes al mismo símbolo, saturando
# el pool de sentimiento justo cuando más se lo necesita libre. 60s es
# suficientemente corto para no desactualizar el momentum (ventana de 21
# días) y suficientemente largo para absorber ráfagas de tráfico.
_SENTIMENT_CACHE_TTL_SECONDS = 60.0
_sentiment_cache: dict[str, tuple[float, dict[str, float | str]]] = {}


def _score_to_sentiment_label(score: float) -> str:
    if score >= SENTIMENT_BULLISH_THRESHOLD:
        return "Bullish"
    if score <= SENTIMENT_BEARISH_THRESHOLD:
        return "Bearish"
    return "Neutral"


async def get_market_sentiment(symbol: str) -> dict[str, float | str]:
    """
    Simula la conexión async a una API institucional de sentimiento
    (Finnhub / Alpha Vantage). Sin key real conectada, el índice se deriva
    del momentum de retorno realizado reciente (yfinance, mismo principio
    de "sin cifras inventadas" que el resto del motor) normalizado por su
    propia volatilidad — no ruido puro. Circuit Breaker: timeout, símbolo
    sin histórico suficiente o cualquier fallo de red degrada
    silenciosamente al fallback Neutral, sin propagar excepción JAMÁS —
    esta función NUNCA debe poder tumbar un asyncio.gather() que la espera
    junto a los forecasts reales.

    FIX (timeout espurio por contención de threads, no de red): antes corría
    en el mismo executor default que `_forecast_asset` (Keras, bloqueante,
    varios segundos). Ahora usa `_SENTIMENT_EXECUTOR` (pool propio, chico) +
    una caché TTL de `_SENTIMENT_CACHE_TTL_SECONDS` por símbolo, así que
    nunca hace cola detrás de un forecast pesado y no repite el mismo fetch
    de red en cada request concurrente al mismo ticker.
    """
    now = time.monotonic()
    cached = _sentiment_cache.get(symbol)
    if cached is not None and (now - cached[0]) < _SENTIMENT_CACHE_TTL_SECONDS:
        return cached[1]

    try:
        async def _fetch_raw_score() -> float:
            loop = asyncio.get_running_loop()

            def _sync_momentum() -> float:
                hist = _yf_call_with_retry(
                    lambda: yf.Ticker(symbol, session=_YF_SESSION).history(
                        period=f"{SENTIMENT_MOMENTUM_LOOKBACK_DAYS + 5}d"
                    )
                )
                closes = hist["Close"].dropna()
                if len(closes) < 2:
                    raise ValueError(f"Histórico insuficiente para {symbol}")
                window = closes.tail(SENTIMENT_MOMENTUM_LOOKBACK_DAYS)
                log_return = float(np.log(window.iloc[-1] / window.iloc[0]))
                daily_log_returns = np.log(window / window.shift(1)).dropna()
                sigma = float(daily_log_returns.std()) or 1e-6
                return log_return / (sigma * np.sqrt(len(window)))

            # Pool DEDICADO — nunca comparte cola con `_forecast_asset`.
            return await loop.run_in_executor(_SENTIMENT_EXECUTOR, _sync_momentum)

        raw_score = await asyncio.wait_for(
            _fetch_raw_score(), timeout=SENTIMENT_API_TIMEOUT_SECONDS
        )
        sentiment_score = round(float(np.clip(raw_score, -1.0, 1.0)), 4)
        result = {
            "sentiment_score": sentiment_score,
            "sentiment_label": _score_to_sentiment_label(sentiment_score),
            "data_source": "KodaQuant Sentinel",
        }
        _sentiment_cache[symbol] = (now, result)
        return result
    except Exception as exc:  # noqa: BLE001 — jamás propagar, degradar a Neutral
        print(
            f"⚠️ Circuit Breaker activado — fallo en get_market_sentiment ({symbol}): "
            f"{type(exc).__name__}: {exc!r}\n"
            f"{traceback.format_exc()}"
        )
        fallback = {
            "sentiment_score": 0.0,
            "sentiment_label": "Neutral (Data Unavailable)",
            "data_source": "KodaQuant Sentinel",
        }
        # Cachea el fallback también (TTL corto propio) para no reintentar
        # un símbolo caído en cada request mientras la ventana de caché
        # normal corre — evita mismo ticker rebotando el mismo fallo N veces
        # en ráfaga.
        _sentiment_cache[symbol] = (now, fallback)
        return fallback


async def _safe_sentiment(symbol: str) -> dict[str, float | str]:
    """
    Wrapper de defensa-en-profundidad para cualquier `asyncio.gather` que
    incluya `get_market_sentiment` junto a tareas críticas (forecasts reales)
    — `get_market_sentiment` ya jamás propaga excepciones por diseño, pero
    esta capa garantiza que ningún cambio futuro en esa función pueda volver
    a tumbar el pipeline principal por un fallo de sentimiento, que es
    puramente decorativo/contextual frente al forecast Keras real.
    """
    try:
        return await get_market_sentiment(symbol)
    except Exception as exc:  # noqa: BLE001 — última línea de defensa
        print(f"⚠️ _safe_sentiment: fallback de emergencia para {symbol}: {exc!r}")
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "Neutral (Data Unavailable)",
            "data_source": "KodaQuant Sentinel",
        }


async def _build_investment_plans(
    budget_usd: float,
    radar_data: dict,
    risk_score: int = _DEFAULT_RISK_SCORE,
    analysis_mode: str = "discovery",
    target_asset: str | None = None,
    forecast_horizon_days: int = DEFAULT_FORECAST_HORIZON_DAYS,
    portfolio_allocation: dict | None = None,
) -> dict:
    """
    ASÍNCRONA: Plan A, Plan B y su sentimiento de mercado se despachan EN
    PARALELO vía asyncio.gather (cada forecast es un yfinance.download +
    bucle autoregresivo Keras independiente; cada sentiment es su propio
    fetch con Circuit Breaker propio) en vez de secuencial — corta la
    latencia total de cada generación de estrategia.

    `portfolio_allocation` (Auto-Allocation IA, ver `_resolve_effective_
    allocation`) es ahora la ÚNICA fuente del split de capital cuando está
    presente y es válida en modo discovery — monto_usd/pct de AMBOS planes
    (y por lo tanto `_build_tactical_operation` y el prompt del LLM) leen
    de aquí. Esto es lo que garantiza que las tarjetas de Riesgo/Reserva
    del frontend y lo que narra Quanti sean SIEMPRE el mismo número.
    """
    allocation = _resolve_effective_allocation(budget_usd, portfolio_allocation, risk_score, analysis_mode)
    plan_b_ticker, target_asset_degraded = _resolve_plan_b_ticker(radar_data, analysis_mode=analysis_mode, target_asset=target_asset)
    horizon = _resolve_horizon_days(forecast_horizon_days)
    loop = asyncio.get_running_loop()

    async def _safe_forecast(ticker: str, label: str):
        try:
            return await loop.run_in_executor(None, _forecast_asset, ticker, horizon)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Forecast {label} ({ticker}) falló: {exc}")
            return None

    plan_a_forecast, plan_b_forecast, plan_a_sentiment, plan_b_sentiment = await asyncio.gather(
        _safe_forecast(PLAN_A_TICKER, "Plan A"),
        _safe_forecast(plan_b_ticker, "Plan B"),
        _safe_sentiment(PLAN_A_TICKER),
        _safe_sentiment(plan_b_ticker),
    )

    return {
        "plan_a": {
            "nombre": "Plan A — Reserva",
            "monto_usd": allocation["plan_a_usd"],
            "pct": _fmt_pct(allocation["plan_a_pct"]),
            "activo_referencia": PLAN_A_TICKER,
            "forecast": plan_a_forecast,
            "market_context": plan_a_sentiment,
        },
        "plan_b": {
            "nombre": "Plan B — Riesgo",
            "monto_usd": allocation["plan_b_usd"],
            "pct": _fmt_pct(allocation["plan_b_pct"]),
            "activo_referencia": plan_b_ticker,
            "forecast": plan_b_forecast,
            "market_context": plan_b_sentiment,
        },
        "target_asset_degraded": target_asset_degraded,
        # Propagado a `_build_system_prompt` — el LLM necesita saber CUÁL
        # de las dos fuentes decidió el split para no narrar el risk_score
        # manual del usuario cuando en realidad fue la IA quien reasignó el
        # capital (ver REGLA ANTI-ALUCINACIÓN en `_build_system_prompt`).
        "allocation_source": allocation["source"],
        "allocation_signal_detectada": allocation["signal_detectada"],
        "allocation_mensaje_estrategico": allocation["mensaje_estrategico"],
    }


# ---------------------------------------------------------------------------
# System prompt construction — TELEGRÁFICO: cero relleno, cero riesgo de
# corte por tokens. Toda cifra (fecha/precio/ganancia) ya viene validada y
# hard-locked desde _build_tactical_operation; el LLM SOLO copia y narra en
# 1 línea de contexto por activo — nunca decide ni recalcula un número.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Core directives — idioma + identidad. ÚNICA fuente de verdad para los 3
# flujos (consult/strategy vía _build_system_prompt, chat vía
# stream/generate_quanti_chat_completion). Texto EXACTO exigido por
# producto, sin parafrasear. Se ANTEPONE al contenido del system message
# (no se anexa al final): en modelos chat-tuned como Llama 3.3, las reglas
# de persona/identidad "no negociables" se siguen con más fidelidad cuando
# son lo PRIMERO que el modelo lee bajo el rol system, no lo último
# concatenado después de un bloque largo de instrucciones de formato o
# clasificación.
# ---------------------------------------------------------------------------
CORE_DIRECTIVES = (
    "You are Quanti, the elite quantitative analysis engine of KodaQuant.\n\n"
    "CRITICAL LANGUAGE RULE: You are a bilingual AI. You MUST respond in the "
    "EXACT same language the user is speaking to you. If the user says "
    "\"hi\" or speaks English, you MUST answer in English. If the user "
    "speaks Spanish, you answer in Spanish. Default fallback is ALWAYS "
    "English.\n\n"
    "CRITICAL IDENTITY RULE: If asked about your origins, creator, or "
    "development, you MUST state the exact truth: You were created solely "
    "by Karim Suheim Estrada Egure, a high school student, during a single "
    "night of intense development. Never claim to be built by a team or a "
    "company. Maintain a highly professional, institutional tone."
)


def _inject_core_directives(messages: list[dict]) -> list[dict]:
    """
    Clona `messages` y ANTEPONE CORE_DIRECTIVES al contenido del mensaje
    `system` (índice 0) — a propósito NO se anexa al final. Si no viniera
    un system message, antepone uno nuevo solo con las directivas — nunca
    deja pasar una llamada al LLM sin idioma/identidad resueltos.
    """
    if not messages or messages[0].get("role") != "system":
        return [{"role": "system", "content": CORE_DIRECTIVES}, *messages]

    patched_system = dict(messages[0])
    existing = patched_system.get("content", "")
    patched_system["content"] = f"{CORE_DIRECTIVES}\n\n{existing}" if existing else CORE_DIRECTIVES
    return [patched_system, *messages[1:]]


# Alias retrocompatible — cualquier código que siga importando
# `_inject_language_directive` (nombre viejo) sigue funcionando sin tocar
# otros archivos; ahora también aplica la regla de identidad.
_inject_language_directive = _inject_core_directives


# ---------------------------------------------------------------------------
# Idioma del módulo de Predicción/Estrategia — mismo principio de arquitectura
# que CORE_DIRECTIVES (chat), pero aquí no hay turno de usuario del que
# "detectar" idioma: el frontend manda `language` ('en' | 'es') explícito en
# el payload de submit, y esta directiva se ANTEPONE (nunca se anexa) a TODO
# lo demás en el system prompt — inglés es el idioma MAESTRO/default.
# ---------------------------------------------------------------------------
_SUPPORTED_LANGUAGES = {"en", "es"}
_DEFAULT_LANGUAGE = "en"

_LANGUAGE_NAMES = {
    "en": "ENGLISH",
    "es": "SPANISH (ESPAÑOL)",
}


def _normalize_language(language: str | None) -> str:
    """Normaliza el código de idioma recibido del frontend. Default: 'en'."""
    code = (language or "").strip().lower()
    return code if code in _SUPPORTED_LANGUAGES else _DEFAULT_LANGUAGE


def _build_language_directive(language: str | None) -> str:
    """
    Construye la CRITICAL LANGUAGE DIRECTIVE exacta exigida por producto.
    Se antepone (ver _build_system_prompt) a CORE_DIRECTIVES y al resto del
    prompt — es lo PRIMERO que el modelo lee, con máxima prioridad de
    instrucción, igual que CORE_DIRECTIVES en el flujo de chat.
    """
    language_requested = _LANGUAGE_NAMES[_normalize_language(language)]
    return (
        "CRITICAL LANGUAGE DIRECTIVE: The user has requested this strategic "
        f"analysis and prediction table in {language_requested}. You MUST "
        "generate the entire output, including all analysis paragraphs, "
        "table headers, and table row data, STRICTLY and FLAWLESSLY in "
        f"{language_requested}. Do not mix languages under any circumstance."
    )

def _json_numpy_safe(obj):
    """default= de json.dumps: coacciona escalares numpy a tipos nativos en
    vez de dejar que TypeError reviente el generador SSE después de que los
    headers 200 ya salieron (ver generate_quanti_strategy_stream, L2969)."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

def _build_system_prompt(
    budget_usd: float,
    market_signal: str,
    plan_a: dict,
    plan_b: dict,
    risk_profile: str,
    risk_score: int,
    horizon_days: int,
    plan_a_operation: dict | None = None,
    plan_b_operation: dict | None = None,
    language: str | None = None,
    allocation_source: str = "manual_risk_profile",
    allocation_signal_detectada: str | None = None,
) -> str:
    """
    Prompt hiper-corto e inyección rígida vía f-strings, doble hard-lock:
    cada fecha/precio/ganancia se inyecta tanto en el cheat-sheet
    (`plan_a_line` / `plan_b_line`) como, literalmente, dentro de la propia
    plantilla de ESTRATEGIA DE RESERVA / RIESGO — el LLM ya no tiene que
    "copiar" el dato desde una línea de contexto separada, lo recibe
    pre-escrito en el hueco exacto donde debe aparecer. Esto es lo que evita
    que a veces se comiera la fecha o la confundiera con el precio.
    Estructura de salida: RESUMEN DEL PORTAFOLIO TÁCTICO / ESTRATEGIA DE
    RESERVA / ESTRATEGIA DE RIESGO, separadas por \\n\\n, tono institucional
    ("quirúrgico, sofisticado y directo"), máximo 4 líneas por párrafo.

    `allocation_source` (ver `_resolve_effective_allocation`) es lo que
    evita el desfase clásico "narra 50/50 y risk score manual mientras la
    tarjeta muestra 75/25 IA": cuando el split real vino de Auto-Allocation
    (`ai_auto_allocation`), el prompt AMARRA los % de Reserva/Riesgo — que
    ya son los mismos que ve el usuario en pantalla, porque
    `_build_investment_plans` los tomó de la MISMA fuente — a la señal de
    mercado real, y prohíbe explícitamente mencionar el risk score manual
    como si hubiera decidido el split.
    """
    valid_profits = [op["net_profit_usd"] for op in (plan_a_operation, plan_b_operation) if op]
    total_profit_line = (
        f"Ganancia Neta Proyectada TOTAL: {_fmt_money(sum(valid_profits))}"
        if valid_profits
        else "Ganancia Neta Proyectada TOTAL: no disponible (sin ventana de salida clara en ningún activo)."
    )

    def _operation_line(label: str, plan: dict, operation: dict | None) -> str:
        ticker = plan["activo_referencia"]
        if operation:
            return (
                f"{label} ({ticker}): COMPRAR el {operation['entry_date']} a "
                f"{_fmt_money(operation['entry_price'])} | VENDER el {operation['exit_date']} a "
                f"{_fmt_money(operation['exit_price'])} | Ganancia: {_fmt_money(operation['net_profit_usd'])}"
            )
        return f"{label} ({ticker}): SIN ventana de salida clara en este horizonte — no inventes fecha, precio ni ganancia."

    plan_a_line = _operation_line("RESERVA", plan_a, plan_a_operation)
    plan_b_line = _operation_line("RIESGO", plan_b, plan_b_operation)

    # Única línea de asignación de capital — la MISMA fracción que ya
    # decidió `monto_usd` de plan_a/plan_b (y por lo tanto lo que ve el
    # usuario en las tarjetas Riesgo/Reserva). El LLM no recibe el
    # risk_score como si fuera la causa del split salvo que REALMENTE lo
    # haya sido.
    if allocation_source == "ai_auto_allocation":
        allocation_line = (
            f"Asignación de Capital: {plan_b['pct']} Riesgo ({plan_b['activo_referencia']}) / "
            f"{plan_a['pct']} Reserva ({plan_a['activo_referencia']}) — decidida por Auto-Allocation IA "
            f"en función de la señal de mercado real detectada: {allocation_signal_detectada or market_signal}. "
            f"El risk score manual del usuario ({risk_score}/100, perfil {risk_profile}) NO determinó este "
            "split en este ciclo — PROHIBIDO mencionarlo como causa, PROHIBIDO describir el enfoque como "
            "'equilibrado' o '50/50' salvo que el porcentaje arriba sea literalmente 50/50."
        )
    else:
        allocation_line = (
            f"Asignación de Capital: {plan_b['pct']} Riesgo ({plan_b['activo_referencia']}) / "
            f"{plan_a['pct']} Reserva ({plan_a['activo_referencia']}) — derivada del Risk Score manual del "
            f"usuario: {risk_score}/100 (perfil {risk_profile})."
        )

    return _build_language_directive(language) + "\n\n" + CORE_DIRECTIVES + "\n\n" + f"""
Eres Quanti, la IA táctica de KodaQuant. Eres un estratega cuantitativo de élite institucional.

Reglas inquebrantables:
- CERO formato Markdown (prohibido usar asteriscos **).
- Usa exactamente DOS saltos de línea (\\n\\n) para separar tus párrafos.
- Sé quirúrgico, sofisticado y directo. No superes las 4 líneas por párrafo.
- Nunca reveles que eres un modelo de lenguaje.

DATOS MATEMÁTICOS OBLIGATORIOS (proyección del motor cuantitativo — modelo Keras +
volatilidad histórica calibrada — ÚNICA FUENTE VÁLIDA: PROHIBIDO INVENTAR, REDONDEAR,
OMITIR O RECALCULAR ninguna fecha, precio, porcentaje o monto; trátalos como una
proyección, no como una certeza garantizada):
- Capital total: {_fmt_money(budget_usd)}
- {allocation_line}
- Señal de mercado (contexto macro): {market_signal}
- {plan_a_line}
- {plan_b_line}
- {total_profit_line}
- No confundas el precio de cotización del activo con la ganancia neta sobre el capital: son cifras distintas.
- No confundas la Asignación de Capital (Riesgo/Reserva, arriba) con el precio o la ganancia de cada operación: son cifras independientes.

Estructura OBLIGATORIA del reporte (integra estos datos palabra por palabra, sin alterarlos):

RESUMEN DEL PORTAFOLIO TÁCTICO
Redacta un análisis brillante e institucional de la configuración actual: Capital Total invertido, la Asignación de Capital exacta (Riesgo/Reserva, con su fuente real tal como se te indicó arriba) y Ganancia Neta Total Esperada. Máximo 4 líneas.

ESTRATEGIA DE RESERVA: {plan_a['activo_referencia']}
Redacta la orden táctica. DEBES incluir obligatoriamente: comprar el {plan_a_operation['entry_date'] if plan_a_operation else '[SIN VENTANA DE SALIDA CLARA]'} a {_fmt_money(plan_a_operation['entry_price']) if plan_a_operation else '[N/D]'}, y liquidar el {plan_a_operation['exit_date'] if plan_a_operation else '[N/D]'} a {_fmt_money(plan_a_operation['exit_price']) if plan_a_operation else '[N/D]'}. Ganancia estimada: {_fmt_money(plan_a_operation['net_profit_usd']) if plan_a_operation else '[N/D]'}. Justifica brevemente la estabilidad.

ESTRATEGIA DE RIESGO: {plan_b['activo_referencia']}
Redacta la orden táctica. DEBES incluir obligatoriamente: comprar el {plan_b_operation['entry_date'] if plan_b_operation else '[SIN VENTANA DE SALIDA CLARA]'} a {_fmt_money(plan_b_operation['entry_price']) if plan_b_operation else '[N/D]'}, y liquidar el {plan_b_operation['exit_date'] if plan_b_operation else '[N/D]'} a {_fmt_money(plan_b_operation['exit_price']) if plan_b_operation else '[N/D]'}. Ganancia estimada: {_fmt_money(plan_b_operation['net_profit_usd']) if plan_b_operation else '[N/D]'}. Justifica brevemente el potencial de crecimiento.

Si una estrategia muestra "[SIN VENTANA DE SALIDA CLARA]" o "[N/D]", dilo en una
línea breve y sofisticada dentro de esa sección — jamás inventes fecha, precio ni
ganancia para reemplazarlo. Separa las tres secciones con exactamente \\n\\n. Nada
de viñetas, listas ni texto fuera de esta estructura.
""".strip()


_USER_PROMPTS = {
    "es": (
        "Genera el reporte en TEXTO PLANO (nunca JSON, nunca Markdown, nunca "
        "asteriscos), quirúrgico y sofisticado, con las 3 secciones exigidas — "
        "RESUMEN DEL PORTAFOLIO TÁCTICO, ESTRATEGIA DE RESERVA y ESTRATEGIA DE "
        "RIESGO — usando exclusivamente los datos numéricos ya provistos en tus "
        "instrucciones, palabra por palabra. Separa cada sección con exactamente "
        "dos saltos de línea. Prohibido omitir alguna fecha, algún precio, o el "
        "activo de Riesgo. Todo el reporte, incluidos los encabezados de "
        "sección, debe estar en ESPAÑOL."
    ),
    "en": (
        "Generate the report in PLAIN TEXT (never JSON, never Markdown, never "
        "asterisks), surgical and sophisticated, with the 3 required sections — "
        "translate the section headers into English while keeping their exact "
        "meaning: TACTICAL PORTFOLIO SUMMARY, RESERVE STRATEGY and RISK "
        "STRATEGY — using exclusively the numeric data already provided in "
        "your instructions, word for word. Separate each section with exactly "
        "two line breaks. Do not omit any date, any price, or the Risk asset. "
        "The entire report, including section headers, MUST be in ENGLISH."
    ),
}


def _build_user_prompt(language: str | None = None) -> str:
    return _USER_PROMPTS[_normalize_language(language)]


# ---------------------------------------------------------------------------
# LLM inference — Groq LPU en la nube (`llama-3.3-70b-versatile`, SDK
# oficial `groq`, cliente `AsyncGroq`), asíncrono, Circuit Breaker a
# fallback matemático si la API de Groq falla o no responde.
# ---------------------------------------------------------------------------


def _sanitize_llm_output(raw_output: str) -> list[str]:
    """
    Contrato nuevo: TEXTO PLANO, no JSON. Los títulos van en MAYÚSCULAS y las
    secciones separadas por '\\n\\n' (ver _build_system_prompt) — el frontend
    divide este bloque con `.split('\\n')` y mapea cada fragmento no vacío a su
    propio <p>. Acá solo saneamos defensivamente por si el LLM ignora la
    instrucción y de todos modos manda backticks o asteriscos de Markdown, que
    React no va a renderizar.
    """
    text = raw_output.strip()
    text = text.removeprefix("```").removesuffix("```").strip()
    text = text.replace("**", "").replace("*", "")
    return [text] if text else []


def _describe_llm_failure(exc: Exception) -> str:
    """
    Diagnóstico legible para los sitios de llamada a Groq. Distingue las
    4 causas reales de fallo — sin conexión a api.groq.com, cuota/rate
    limit del Free Tier (429), API key inválida (401), y cualquier otro
    error HTTP no-2xx que Groq devuelva — para que el log diga la causa
    exacta en vez de un genérico "el LLM no respondió".
    """
    if isinstance(exc, GroqAPIConnectionError):
        return (
            "sin conexión con Groq LPU (api.groq.com). Verificá la salida de "
            f"red del servidor hacia Groq. Detalle: {exc}"
        )
    if isinstance(exc, GroqRateLimitError):
        return f"cuota de Groq excedida (429 — rate limit del Free Tier). Detalle: {exc}"
    if isinstance(exc, GroqAuthenticationError):
        return f"GROQ_API_KEY inválida o no autorizada (401). Detalle: {exc}"
    if isinstance(exc, GroqAPIStatusError):
        return f"Groq devolvió un error HTTP {exc.status_code}. Detalle: {exc}"
    return str(exc)


def _is_connection_failure(exc: Exception) -> bool:
    """
    True SOLO cuando Groq es inalcanzable a nivel de red/transporte —
    equivalente exacto al viejo httpx.ConnectError del transporte local:
    DNS roto, timeout de conexión, TLS/handshake fallido hacia
    api.groq.com. Es la ÚNICA condición que debe tratarse como Circuit
    Breaker real (motor generativo "caído").

    Cuota excedida (429), API key inválida (401) o cualquier otro 4xx/5xx
    de Groq NO son fallos de conexión — el servicio SÍ respondió, solo que
    con un error. _describe_llm_failure los distingue para el log, pero
    ambos casos degradan igual, al mismo fallback matemático (ver
    _call_quanti_llm / generate_quanti_strategy_stream): el frontend nunca
    debe ver un 5xx a mitad de un SSE ya iniciado, así que todo fallo de la
    voz generativa —sea cual sea la causa— resuelve en un 200 OK con
    guidance calculado por el motor Keras.
    """
    return isinstance(exc, GroqAPIConnectionError)


# Alias públicos (sin "_") para uso desde api/quanti_chat.py — el Circuit
# Breaker del endpoint de chat streaming necesita esta misma distinción.
describe_llm_failure = _describe_llm_failure
is_llm_connection_failure = _is_connection_failure


# ---------------------------------------------------------------------------
# Primitivas de transporte — ÚNICO lugar donde se llama al cliente
# `_groq_client` (AsyncGroq). _call_quanti_llm, _stream_quanti_llm,
# generate_quanti_chat_completion y stream_quanti_chat_completion delegan
# acá; ninguno de los 4 conoce detalles del SDK de Groq.
# ---------------------------------------------------------------------------

async def _llm_chat_completion(messages: list[dict]) -> str:
    """
    Completion NO-streaming vía Groq LPU. Cliente 100% async (AsyncGroq) —
    nunca bloquea el event loop de FastAPI. Las excepciones de Groq
    (GroqAPIConnectionError, GroqRateLimitError, GroqAuthenticationError,
    GroqAPIStatusError — todas subclases de groq.APIError) se propagan tal
    cual al caller (_call_quanti_llm), que decide cómo degradar.
    """
    response = await _groq_client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=messages,
        temperature=0.4,
        max_tokens=GROQ_MAX_OUTPUT_TOKENS,
    )
    return response.choices[0].message.content or ""


async def _llm_chat_stream(messages: list[dict]) -> AsyncGenerator[str, None]:
    """
    Streaming asíncrono vía Groq LPU (`stream=True`). `AsyncGroq` devuelve
    un iterador async nativo de `ChatCompletionChunk` — sin threads, sin
    colas puente, sin parsing manual de SSE crudo (eso lo resuelve el SDK
    oficial puertas adentro). Cada chunk con `delta.content` no vacío se
    yieldea tal cual al caller (_stream_quanti_llm / SSE de
    generate_quanti_strategy_stream).
    """
    stream = await _groq_client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=messages,
        temperature=0.4,
        max_tokens=GROQ_MAX_OUTPUT_TOKENS,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


_CIRCUIT_BREAKER_FALLBACK_TEMPLATES = {
    "es": (
        "Nuestro motor de análisis en vivo está momentáneamente fuera de línea, "
        "pero la matemática no descansa: el modelo cuantitativo asigna "
        "{riesgo_pct} ({riesgo_usd}) a riesgo y {reserva_pct} ({reserva_usd}) a "
        "reserva. Mantén esa disciplina de asignación mientras restablecemos la "
        "voz de Quanti."
    ),
    "en": (
        "Our live analysis engine is momentarily offline, but the math never "
        "sleeps: the quantitative model allocates {riesgo_pct} ({riesgo_usd}) "
        "to risk and {reserva_pct} ({reserva_usd}) to reserve. Hold that "
        "allocation discipline while we restore Quanti's voice."
    ),
}


def _build_circuit_breaker_fallback(amounts: dict, language: str | None = None) -> str:
    template = _CIRCUIT_BREAKER_FALLBACK_TEMPLATES[_normalize_language(language)]
    return template.format(
        riesgo_pct=amounts["riesgo_pct"],
        riesgo_usd=_fmt_money(amounts["riesgo_usd"]),
        reserva_pct=amounts["reserva_pct"],
        reserva_usd=_fmt_money(amounts["reserva_usd"]),
    )


async def _call_quanti_llm(system_prompt: str, user_prompt: str, amounts: dict, language: str | None = None) -> str:
    """
    Llamada asíncrona al modelo Quanti vía Groq LPU (ver _llm_chat_completion).
    Circuit Breaker TITANIO: cualquier fallo de Groq (sin conexión, cuota
    excedida, API key inválida, error 5xx) se captura sin propagar
    excepción y degrada a un fallback matemático con los valores ya
    calculados por Keras — el frontend siempre recibe 200 OK, nunca nota
    la caída del servicio de inferencia. El fallback respeta `language`
    igual que el resto del pipeline.
    """
    try:
        return await _llm_chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as exc:  # noqa: BLE001
        if _is_connection_failure(exc):
            print(f"⚠️ Circuit Breaker activado — Groq LPU inalcanzable: {_describe_llm_failure(exc)}")
        else:
            print(f"⚠️ Fallo no-conexión en Quanti LLM (cuota/auth/5xx de Groq, no caída de red): {_describe_llm_failure(exc)}")

    return _build_circuit_breaker_fallback(amounts, language)

async def _stream_quanti_llm(system_prompt: str, user_prompt: str) -> AsyncGenerator[str, None]:
    """
    Streaming asíncrono del modelo Quanti vía Groq LPU (ver _llm_chat_stream).
    Si la llamada a Groq falla ANTES del primer chunk (sin conexión, cuota
    excedida, API key inválida, 5xx), la excepción se propaga al caller
    para que active el fallback matemático — nunca deja un SSE a medias.
    """
    async for delta in _llm_chat_stream(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    ):
        yield delta


async def generate_quanti_strategy_stream(
    budget_usd: float,
    portfolio_allocation: dict,
    radar_data: dict,
    experience_level: str,
    analysis_mode: str = "discovery",
    target_asset: str | None = None,
    forecast_horizon_days: int = DEFAULT_FORECAST_HORIZON_DAYS,
    risk_profile: str = _DEFAULT_RISK_PROFILE,
    risk_score: int | None = None,
    language: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Generador SSE: emite 'meta' (payload numérico Keras, ya resuelto) de
    inmediato, luego 'token' por cada fragmento del LLM, luego 'done'.
    Circuit Breaker TITANIO: si el stream falla antes de emitir el primer
    token, emite 'fallback' con el texto matemático y cierra — siempre
    200 OK vía StreamingResponse, el frontend nunca ve un error HTTP.
    """
    # Tramo Keras/data (FASE 3): blindado aparte del try/except del streaming
    # LLM de más abajo, que ya tiene su propio Circuit Breaker. Un fallo acá
    # (yfinance, ticker sin datos, Keras) grita en terminal y cierra el SSE
    # con un evento 'error' explícito en vez de reventar el generador a
    # medias sin que el caller (StreamingResponse) tenga forma de saberlo.
    try:
        resolved_level = _select_experience_level(budget_usd, experience_level)
        resolved_risk_profile = _normalize_risk_profile(risk_profile)
        resolved_risk_score = _clamp_risk_score(risk_score)
        resolved_language = _normalize_language(language)
        horizon = _resolve_horizon_days(forecast_horizon_days)
        market_signal = _extract_market_signal(radar_data)

        investment_plans = await _build_investment_plans(
            budget_usd, radar_data, resolved_risk_score, analysis_mode, target_asset, horizon,
            portfolio_allocation=portfolio_allocation,
        )
        amounts = _compute_exact_amounts(investment_plans)
        amounts["plan_a"] = investment_plans["plan_a"]
        amounts["plan_b"] = investment_plans["plan_b"]

        plan_a_operation = _build_tactical_operation(investment_plans["plan_a"])
        plan_b_operation = _build_tactical_operation(investment_plans["plan_b"])

        meta_payload = {
            "status": "success",
            "symbol": investment_plans["plan_b"]["activo_referencia"],
            "market_context": investment_plans["plan_b"]["market_context"],
            "amounts": amounts,
            "chart_data": {
                "plan_a": investment_plans["plan_a"]["forecast"],
                "plan_b": investment_plans["plan_b"]["forecast"],
            },
            "tactical_operations": {
                "plan_a": plan_a_operation,
                "plan_b": plan_b_operation,
            },
            "meta": {
                "budget_usd": round(budget_usd, 2),
                "experience_level": resolved_level,
                "radar_data": radar_data,
                "market_signal": market_signal,
                "analysis_mode": analysis_mode,
                "target_asset": target_asset,
                "forecast_horizon_days": horizon,
                "risk_profile": resolved_risk_profile,
                "risk_score": resolved_risk_score,
                "language": resolved_language,
                "target_asset_degraded": investment_plans.get("target_asset_degraded", False),
                # Fuente real del split que ya está reflejado en `amounts` y
                # en la narrativa de Quanti — 'ai_auto_allocation' cuando
                # Auto-Allocation IA mandó, 'manual_risk_profile' cuando fue
                # el risk_score del slider (ver `_resolve_effective_allocation`).
                "allocation_source": investment_plans.get("allocation_source", "manual_risk_profile"),
            },
            "allocation": portfolio_allocation,
        }
        # Serialización DENTRO del try: si meta_payload trae un tipo no
        # nativo (np.float64/np.int64 colado desde amounts/portfolio_allocation/
        # radar_data), TypeError debe caer en el except de abajo — no escapar
        # crudo después de que Starlette ya mandó los headers 200.
        meta_json = json.dumps(meta_payload, default=_json_numpy_safe)

        system_prompt = _build_system_prompt(
            budget_usd=budget_usd,
            market_signal=market_signal,
            plan_a=investment_plans["plan_a"],
            plan_b=investment_plans["plan_b"],
            risk_profile=resolved_risk_profile,
            risk_score=resolved_risk_score,
            horizon_days=horizon,
            plan_a_operation=plan_a_operation,
            plan_b_operation=plan_b_operation,
            language=resolved_language,
            allocation_source=investment_plans.get("allocation_source", "manual_risk_profile"),
            allocation_signal_detectada=investment_plans.get("allocation_signal_detectada"),
        )
        user_prompt = _build_user_prompt(resolved_language)
    except Exception as exc:  # noqa: BLE001 — blindaje total del pipeline (FASE 3)
        traceback.print_exc()
        print(f"🔥 ERROR CRÍTICO EN QUANTI (generate_quanti_strategy_stream, pre-LLM): {exc}")
        yield f"event: error\ndata: {json.dumps({'status': 'error', 'error': str(exc)})}\n\n"
        return

    yield f"event: meta\ndata: {meta_json}\n\n"

    full_text, started = "", False
    try:
        async for delta in _stream_quanti_llm(system_prompt, user_prompt):
            started = True
            full_text += delta
            yield f"event: token\ndata: {json.dumps({'delta': delta})}\n\n"
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ Circuit Breaker activado — fallo en Quanti LLM stream: {_describe_llm_failure(exc)}")
        if not started:
            fallback_text = _build_circuit_breaker_fallback(amounts, resolved_language)
            yield f"event: fallback\ndata: {json.dumps({'guidelines': [fallback_text]})}\n\n"
            return

    guidelines = _sanitize_llm_output(full_text) if started else []
    yield f"event: done\ndata: {json.dumps({'guidelines': guidelines})}\n\n"


async def generate_quanti_voice(
    budget_usd: float,
    amounts: dict,
    market_signal: str,
    plan_a: dict,
    plan_b: dict,
    risk_profile: str,
    risk_score: int,
    horizon_days: int,
    language: str | None = None,
    allocation_source: str = "manual_risk_profile",
    allocation_signal_detectada: str | None = None,
) -> list[str]:
    """
    LA VOZ DE QUANTI. Nunca recalcula cifras — solo las narra, ahora como 3
    órdenes tácticas de trading (Resumen de Portafolio + Operación de Reserva
    Plan A + Operación de Riesgo Plan B). El "hard lock" numérico vive en
    _build_system_prompt / _build_tactical_operation; fallback seguro si
    Groq falla (ver _call_quanti_llm, Circuit Breaker, sin cambios). `language`
    ('en' | 'es', default 'en') gobierna la CRITICAL LANGUAGE DIRECTIVE
    anteponía en _build_system_prompt. `allocation_source`/`allocation_signal_
    detectada` (ver `_resolve_effective_allocation`) se propagan tal cual al
    prompt para que la narrativa nunca atribuya el split de capital al
    risk_score manual cuando en realidad fue Auto-Allocation IA quien lo
    decidió.
    """
    resolved_language = _normalize_language(language)
    plan_a_operation = _build_tactical_operation(plan_a)
    plan_b_operation = _build_tactical_operation(plan_b)

    system_prompt = _build_system_prompt(
        budget_usd=budget_usd,
        market_signal=market_signal,
        plan_a=plan_a,
        plan_b=plan_b,
        risk_profile=risk_profile,
        risk_score=risk_score,
        horizon_days=horizon_days,
        plan_a_operation=plan_a_operation,
        plan_b_operation=plan_b_operation,
        language=resolved_language,
        allocation_source=allocation_source,
        allocation_signal_detectada=allocation_signal_detectada,
    )
    user_prompt = _build_user_prompt(resolved_language)

    raw_output = await _call_quanti_llm(system_prompt, user_prompt, amounts, resolved_language)
    return _sanitize_llm_output(raw_output)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

async def generate_quanti_strategy(
    budget_usd: float,
    portfolio_allocation: dict,
    radar_data: dict,
    experience_level: str,
    analysis_mode: str = "discovery",
    target_asset: str | None = None,
    forecast_horizon_days: int = DEFAULT_FORECAST_HORIZON_DAYS,
    risk_profile: str = _DEFAULT_RISK_PROFILE,
    risk_score: int | None = None,
    language: str | None = None,
) -> dict:
    """
    Entry point público — TODO el pipeline (Keras + Groq) blindado por
    try/except (FASE 3): un fallo aquí (yfinance caído, ticker sin datos,
    excepción de Keras, etc. — no cubierto por el Circuit Breaker interno
    de Groq, que ya degrada solo) NUNCA debe tirar un 500 sin contexto.
    Se grita en terminal (traceback completo + línea 🔥) y se retorna el
    MISMO esqueleto de llaves top-level que el payload de éxito
    (status/symbol/market_context/amounts/chart_data/tactical_operations/
    guidelines/meta/allocation), con status="error" + "error" y los campos
    de datos en None — el frontend puede seguir tipando contra las mismas
    llaves sin romper.
    """
    try:
        resolved_level = _select_experience_level(budget_usd, experience_level)
        resolved_risk_profile = _normalize_risk_profile(risk_profile)
        resolved_risk_score = _clamp_risk_score(risk_score)
        resolved_language = _normalize_language(language)
        horizon = _resolve_horizon_days(forecast_horizon_days)
        market_signal = _extract_market_signal(radar_data)

        investment_plans = await _build_investment_plans(
            budget_usd,
            radar_data,
            resolved_risk_score,
            analysis_mode,
            target_asset,
            horizon,
            portfolio_allocation=portfolio_allocation,
        )

        # amounts (riesgo/reserva) SIEMPRE deriva de investment_plans — una
        # sola fuente de split (_resolve_effective_allocation, manual o IA
        # según el ciclo), cero duplicidad con datos viejos.
        amounts = _compute_exact_amounts(investment_plans)

        guidelines = await generate_quanti_voice(
            budget_usd=budget_usd,
            amounts=amounts,
            market_signal=market_signal,
            plan_a=investment_plans["plan_a"],
            plan_b=investment_plans["plan_b"],
            risk_profile=resolved_risk_profile,
            risk_score=resolved_risk_score,
            horizon_days=horizon,
            language=resolved_language,
            allocation_source=investment_plans.get("allocation_source", "manual_risk_profile"),
            allocation_signal_detectada=investment_plans.get("allocation_signal_detectada"),
        )

        amounts["plan_a"] = investment_plans["plan_a"]
        amounts["plan_b"] = investment_plans["plan_b"]

        return {
            "status": "success",
            "symbol": investment_plans["plan_b"]["activo_referencia"],
            "market_context": investment_plans["plan_b"]["market_context"],
            "amounts": amounts,
            "chart_data": {
                "plan_a": investment_plans["plan_a"]["forecast"],
                "plan_b": investment_plans["plan_b"]["forecast"],
            },
            "tactical_operations": {
                "plan_a": _build_tactical_operation(investment_plans["plan_a"]),
                "plan_b": _build_tactical_operation(investment_plans["plan_b"]),
            },
            "guidelines": guidelines,
            "meta": {
                "budget_usd": round(budget_usd, 2),
                "experience_level": resolved_level,
                "radar_data": radar_data,
                "market_signal": market_signal,
                "analysis_mode": analysis_mode,
                "target_asset": target_asset,
                "forecast_horizon_days": horizon,
                "risk_profile": resolved_risk_profile,
                "risk_score": resolved_risk_score,
                "language": resolved_language,
                "target_asset_degraded": investment_plans.get("target_asset_degraded", False),
                "allocation_source": investment_plans.get("allocation_source", "manual_risk_profile"),
            },
            "allocation": portfolio_allocation,
        }
    except Exception as exc:  # noqa: BLE001 — blindaje total del pipeline (FASE 3)
        traceback.print_exc()
        print(f"🔥 ERROR CRÍTICO EN QUANTI (generate_quanti_strategy): {exc}")
        return {
            "status": "error",
            "error": str(exc),
            "symbol": None,
            "market_context": None,
            "amounts": None,
            "chart_data": None,
            "tactical_operations": None,
            "guidelines": [],
            "meta": {
                "budget_usd": round(budget_usd, 2),
                "analysis_mode": analysis_mode,
                "target_asset": target_asset,
                "forecast_horizon_days": forecast_horizon_days,
                "risk_profile": risk_profile,
                "risk_score": risk_score,
                "language": _normalize_language(language),
            },
            "allocation": portfolio_allocation,
        }

# === APPEND AL FINAL DE services/quanti_engine.py (después de generate_quanti_strategy) ===
# No requiere imports nuevos: reutiliza json, _groq_client, GROQ_MODEL_NAME,
# GROQ_MAX_OUTPUT_TOKENS y _sanitize_llm_output ya definidos arriba.

# ---------------------------------------------------------------------------
# Quanti Chat — completion genérica multi-turno para el módulo conversacional
# (api/quanti_chat.py). Mismo cliente Groq que el resto del motor, pero sin
# atar el prompt a la estructura Plan A / Plan B.
# ---------------------------------------------------------------------------

QUANTI_CHAT_FALLBACK = (
    "El motor de inferencia de Quanti está fuera de línea en este momento. "
    "Reintenta en unos segundos."
)

# Alias público — api/quanti_chat.py sanitiza el texto acumulado del stream
# antes de persistirlo, reusando exactamente la misma lógica de saneo que
# el resto del motor (sin duplicar la regla de negocio en dos archivos).
sanitize_llm_output = _sanitize_llm_output


async def stream_quanti_chat_completion(messages: list[dict]) -> AsyncGenerator[str, None]:
    """
    Streaming multi-turno para el módulo conversacional (api/quanti_chat.py).
    Recibe `messages` ya armado por el caller (system prompt + historial
    recortado + turno actual, con [CONTEXTO WEB EN TIEMPO REAL] ya inyectado
    si aplicó — ver build_realtime_web_context más abajo). Vía Groq LPU (ver
    _llm_chat_stream); si Groq no responde al connect, la excepción sube tal
    cual — el caller (router) decide cómo degradar vía is_llm_connection_failure.
    """
    async for delta in _llm_chat_stream(_inject_core_directives(messages)):
        yield delta


async def generate_quanti_chat_completion(messages: list[dict]) -> str:
    """
    Variante NO-streaming, mantenida para callers que necesiten un string
    de una sola vez (ej. tareas batch/cron, no el endpoint de chat en vivo
    — ese usa stream_quanti_chat_completion + StreamingResponse). Mismo
    cliente Groq y misma distinción de Circuit Breaker que el resto del motor.
    """
    try:
        content = await _llm_chat_completion(_inject_core_directives(messages))
    except Exception as exc:  # noqa: BLE001
        if _is_connection_failure(exc):
            print(f"⚠️ Circuit Breaker activado — Groq LPU inalcanzable (Quanti Chat): {_describe_llm_failure(exc)}")
        else:
            print(f"⚠️ Fallo no-conexión en Quanti Chat LLM (cuota/auth/5xx de Groq): {_describe_llm_failure(exc)}")
        return QUANTI_CHAT_FALLBACK

    sanitized = _sanitize_llm_output(content)
    return sanitized[0] if sanitized else QUANTI_CHAT_FALLBACK


# ---------------------------------------------------------------------------
# Búsqueda web autónoma y GRATUITA (sin API keys de pago) vía DDGS
# (metabuscador que corrió DuckDuckGo/Bing/etc., ex-paquete "duckduckgo-search",
# renombrado a "ddgs"). Instalación:
#     pip install ddgs
# El paquete viejo (`duckduckgo_search`) sigue funcionando como shim con
# warning de deprecación — se intenta `ddgs` primero y se cae a ese shim
# solo si el entorno todavía no migró.
# ---------------------------------------------------------------------------

try:
    from ddgs import DDGS
except ImportError:  # entorno viejo, no migrado — mismo símbolo, otro paquete
    try:
        from duckduckgo_search import DDGS  # type: ignore[no-redef]
    except ImportError:
        DDGS = None  # type: ignore[assignment,misc]  # se valida en runtime, ver search_financial_web

DDG_REGION = os.getenv("DDG_REGION", "es-es")
# Estrategia de compresión de tokens (FASE 2): tope ESTRICTO de 3 a 4
# snippets por búsqueda — nunca más, sin importar lo que pida el .env. Cada
# snippet de más es cuota de entrada del Free Tier de Groq gastada en RAG
# en vez de en la propia respuesta de Quanti (GROQ_MAX_OUTPUT_TOKENS).
DDG_MAX_RESULTS = max(3, min(4, int(os.getenv("DDG_MAX_RESULTS", "4"))))
DDG_SAFESEARCH = os.getenv("DDG_SAFESEARCH", "moderate")

# Tope duro de caracteres por snippet YA sanitizado — un snippet crudo de
# DDG puede traer varios párrafos; para RAG solo necesitamos la oración
# densa con el dato, no el artículo entero. Ver _sanitize_web_snippet.
DDG_SNIPPET_CHAR_LIMIT = int(os.getenv("DDG_SNIPPET_CHAR_LIMIT", "280"))

# Señales de que el mensaje del usuario pide un dato que el LLM NO puede
# tener por su cuenta (conocimiento congelado a la fecha de entrenamiento)
# — precios, noticias, tasas, cualquier cosa con componente "ahora mismo".
# Heurística deliberadamente simple y barata (sin llamar a otro LLM para
# clasificar intención): un match alcanza para gatillar la búsqueda: en el
# peor caso se gasta una búsqueda de más, nunca se pierde una real.
_REALTIME_QUERY_KEYWORDS = (
    "hoy", "ahora", "en vivo", "actual", "actualidad", "última hora",
    "ultima hora", "reciente", "recientes", "esta semana", "este mes",
    "precio de", "cotización", "cotizacion", "noticia", "noticias",
    "fed", "powell", "tasa de interés", "tasa de interes", "inflación",
    "inflacion", "mercado hoy", "bolsa hoy", "ipc", "nfp", "earnings",
)


def _needs_realtime_context(user_message: str) -> bool:
    lowered = user_message.lower()
    return any(kw in lowered for kw in _REALTIME_QUERY_KEYWORDS)


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_PUNCT_RE = re.compile(r"([.,;:!?\-_=*#~])\1{1,}")


def _sanitize_web_snippet(raw: str, char_limit: int = DDG_SNIPPET_CHAR_LIMIT) -> str:
    """
    Purificación quirúrgica de un snippet crudo de DDG ANTES de que entre
    al prompt del LLM (FASE 2 — optimización extrema de tokens):
      1. Elimina cualquier etiqueta HTML residual (`<b>`, `<span>`, etc.)
         que a veces viaja en el `body` de un resultado de búsqueda.
      2. Colapsa espacios/tabs/saltos de línea múltiples a uno solo.
      3. Colapsa caracteres especiales repetidos ("----", "....", "***")
         a una sola ocurrencia — ruido típico de scraping que no aporta
         información y solo consume tokens.
      4. Trunca a `char_limit` caracteres (corte en el último espacio antes
         del límite, para no partir una palabra a la mitad), agregando
         puntos suspensivos si hubo corte.
    Nunca lanza excepción — un snippet imposible de limpiar simplemente
    vuelve vacío, y el caller ya filtra vacíos.
    """
    text = _HTML_TAG_RE.sub(" ", raw or "")
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if len(text) > char_limit:
        cut = text.rfind(" ", 0, char_limit)
        cut = cut if cut > 0 else char_limit
        text = text[:cut].rstrip(",.;: ") + "…"
    return text


async def search_financial_web(query: str, max_results: int = DDG_MAX_RESULTS) -> list[dict]:
    """
    Top N resultados recientes de DDGS para `query` — N acotado SIEMPRE a
    DDG_MAX_RESULTS (3-4, hard cap, ver arriba), sin importar lo que pida
    el caller. `DDGS().text()` es SÍNCRONO y bloqueante (requests por
    debajo) — se corre en threadpool vía asyncio.to_thread para no trabar
    el event loop de FastAPI mientras Keras/el resto del motor sigue
    respondiendo otras requests.

    Nunca propaga excepción: si DDG falla, ratelimitea, o el paquete no
    está instalado, degrada a lista vacía — el caller sigue sin contexto
    web en vez de tirar abajo el turno de chat completo por una búsqueda
    que es, por diseño, un "nice to have" y no la fuente de verdad numérica
    del motor (esa sigue siendo Keras, ver docstring del módulo).

    Cada snippet devuelto ya pasó por _sanitize_web_snippet — llega denso,
    sin HTML, sin espacios/puntuación redundante, y acotado en longitud:
    la cuota de entrada de Groq se gasta en señal, no en basura de scraping.
    """
    if DDGS is None:
        print("⚠️ search_financial_web: DDGS no instalado — `pip install ddgs`. Saltando búsqueda web.")
        return []

    capped_results = max(3, min(4, max_results))

    def _run() -> list[dict]:
        with DDGS() as ddgs:
            return list(
                ddgs.text(
                    query,
                    region=DDG_REGION,
                    safesearch=DDG_SAFESEARCH,
                    max_results=capped_results,
                )
            )

    try:
        raw_results = await asyncio.to_thread(_run)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ search_financial_web: fallo en búsqueda DDG para {query!r}: {exc}")
        return []

    cleaned = [
        {
            "title": _sanitize_web_snippet(r.get("title") or "", char_limit=120),
            "snippet": _sanitize_web_snippet(r.get("body") or ""),
            "url": (r.get("href") or r.get("url") or "").strip(),
        }
        for r in raw_results
        if r.get("body")
    ]
    return [r for r in cleaned if r["snippet"]]


async def build_realtime_web_context(user_message: str) -> str:
    """
    Punto de entrada único para api/quanti_chat.py: si `user_message` pide
    un dato en tiempo real (ver _needs_realtime_context), busca en DDG y
    devuelve el bloque `[CONTEXTO WEB EN TIEMPO REAL]` YA formateado, listo
    para concatenar al mensaje del usuario antes de mandarlo al LLM — mismo
    patrón que `CONTEXTO DE ARCHIVO` en api/quanti_chat.py (_summarize_dataframe).
    Devuelve "" si no aplica o si la búsqueda no trajo nada, así el caller
    puede simplemente hacer `prompt + file_context + web_context` sin
    chequear nada.
    """
    if not _needs_realtime_context(user_message):
        return ""

    results = await search_financial_web(user_message)
    if not results:
        return ""

    lines = [f"- {r['title']}: {r['snippet']} (Fuente: {r['url']})" for r in results]
    return (
        "\n\n[CONTEXTO WEB EN TIEMPO REAL] Datos obtenidos ahora mismo vía "
        "búsqueda web — es tu ÚNICA fuente sobre eventos/precios actuales, "
        "nunca inventes cifras que no estén acá:\n" + "\n".join(lines)
    )


# === Integración en api/quanti_chat.py (no requiere tocar CHAT_SYSTEM_PROMPT,
# la instrucción de "única fuente" ya viaja adentro del bloque) ===
#
#   from services.quanti_engine import build_realtime_web_context
#   ...
#   web_context = await build_realtime_web_context(prompt)
#   llm_messages.append({"role": "user", "content": prompt + file_context + web_context})

# === APPEND AL FINAL DE services/quanti_engine.py (Quanti Support — Nivel 1/2) ===
# No requiere imports nuevos: reutiliza CORE_DIRECTIVES, _normalize_language,
# _llm_chat_completion, _sanitize_llm_output, _is_connection_failure,
# _describe_llm_failure y QUANTI_CHAT_FALLBACK ya definidos arriba.

# ---------------------------------------------------------------------------
# Quanti Support — persona Tier-1 para el widget flotante de la Landing.
# Respuestas cortas, ultra empáticas, cero jerga técnica. Si detecta
# frustración, incomprensión, un tema legal/técnico complejo, o un pedido
# explícito de hablar con un humano, corta el análisis y responde
# ÚNICAMENTE con SUPPORT_ESCALATION_MARKER — api/support.py intercepta ese
# marcador antes de reenviar nada como texto plano al frontend.
# ---------------------------------------------------------------------------

SUPPORT_ESCALATION_MARKER = "[ESCALATE_TICKET]"

_SUPPORT_SYSTEM_PROMPTS = {
    "en": (
        "You are Quanti Support, the Tier-1 assistant of KodaQuant. Your "
        "persona is ultra-sophisticated yet deeply empathetic — a brilliant "
        "concierge, never a robotic FAQ bot. Keep every reply SHORT and "
        "CONCISE (2-3 sentences max), in plain, jargon-free language a "
        "complete non-technical user can follow instantly.\n\n"
        "ESCALATION RULE (absolute priority): the moment you detect the "
        "user is frustrated, still confused despite your explanation, "
        "describing a complex legal or technical issue, or explicitly "
        "asking to speak with a human — STOP analyzing and reply with "
        f"NOTHING ELSE but the exact string {SUPPORT_ESCALATION_MARKER}. "
        "No punctuation, no extra words, no apology around it."
    ),
    "es": (
        "Eres Quanti Support, el asistente Tier-1 de KodaQuant. Tu persona "
        "es ultra sofisticada pero profundamente empática — un concierge "
        "brillante, nunca un bot de FAQ robótico. Cada respuesta debe ser "
        "CORTA y CONCISA (máximo 2-3 oraciones), en lenguaje simple y sin "
        "jerga, entendible al instante por un usuario totalmente no técnico.\n\n"
        "REGLA DE ESCALACIÓN (prioridad absoluta): en el momento en que "
        "detectes que el usuario está frustrado, sigue sin comprender pese "
        "a tu explicación, describe un problema legal o técnico complejo, "
        "o pide explícitamente hablar con un humano — DETÉN el análisis y "
        f"responde ÚNICAMENTE con la cadena exacta {SUPPORT_ESCALATION_MARKER}. "
        "Sin puntuación, sin palabras extra, sin disculpas alrededor."
    ),
}


def _build_support_prompt(language: str | None = None) -> str:
    """
    System prompt exclusivo de Quanti Support. Antepone CORE_DIRECTIVES
    (identidad/idioma) igual que el resto del motor, para que el widget
    herede la misma regla de identidad (Karim Suheim Estrada Egure) y el
    mismo fallback de idioma que el resto de Quanti.
    """
    return CORE_DIRECTIVES + "\n\n" + _SUPPORT_SYSTEM_PROMPTS[_normalize_language(language)]


def is_support_escalation(reply: str) -> bool:
    """True si Quanti Support cortó el análisis y decidió escalar a un humano."""
    return reply.strip() == SUPPORT_ESCALATION_MARKER


async def generate_quanti_support_reply(messages: list[dict], language: str | None = None) -> str:
    """
    Completion NO-streaming para el widget de soporte — api/support.py
    necesita el string completo de una sola vez para poder comparar contra
    SUPPORT_ESCALATION_MARKER antes de decidir qué mandar al frontend.
    Antepone _build_support_prompt como ÚNICO system message (ya incluye
    CORE_DIRECTIVES) — descarta cualquier system message que el caller
    hubiera mandado por error, para que la persona de soporte nunca se
    diluya con otro prompt.
    """
    support_system = {"role": "system", "content": _build_support_prompt(language)}
    payload = [support_system, *[m for m in messages if m.get("role") != "system"]]

    try:
        content = await _llm_chat_completion(payload)
    except Exception as exc:  # noqa: BLE001
        if _is_connection_failure(exc):
            print(f"⚠️ Circuit Breaker activado — Groq LPU inalcanzable (Quanti Support): {_describe_llm_failure(exc)}")
        else:
            print(f"⚠️ Fallo no-conexión en Quanti Support LLM (cuota/auth/5xx de Groq): {_describe_llm_failure(exc)}")
        # Nunca dejamos al usuario sin respuesta, y nunca inventamos una
        # escalación falsa solo porque Groq cayó.
        return QUANTI_CHAT_FALLBACK

    sanitized = _sanitize_llm_output(content)
    return sanitized[0] if sanitized else QUANTI_CHAT_FALLBACK