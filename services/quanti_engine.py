# services/quanti_engine.py
"""
Quanti AI Engine — KodaQuant Terminal
======================================
Inferencia matemática 100% Keras 3 / TensorFlow NATIVO — carga directa de
`attention_bilstm_global.keras` in-process (CERO ONNX Runtime en este
proceso) + voz generativa vía Groq LPU en la nube (`llama-3.3-70b-versatile`,
SDK oficial `groq`, cliente `AsyncGroq`), con Circuit Breaker a fallback
matemático si la API de Groq no responde. Soporte de inferencia local
(llama.cpp/Metal) retirado permanentemente — ver GROQ_API_KEY más abajo.

Requisitos locales (Mac): `pip install keras tensorflow scikit-learn pandas
yfinance httpx` — sin re-entrenar nada, sin Colab. El modelo y los scalers
ya entrenados (`attention_bilstm_global.keras`, `scalers.pkl`) viven en
ML_ENGINE_DIR tal cual salieron del notebook.

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
import os
import pickle
import re
import threading
import traceback
from pathlib import Path
from datetime import timedelta
from functools import lru_cache
from typing import TYPE_CHECKING, Any, AsyncGenerator

import numpy as np
import pandas as pd
import yfinance as yf
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
# Configuración — rutas y endpoints
# ---------------------------------------------------------------------------

# Resolución de rutas ANCLADA al propio archivo (no al cwd del proceso que
# arranca el servidor) — `attention_bilstm_global.keras` y `scalers.pkl`
# viven en la MISMA carpeta que este módulo (ej. backend/services/).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ML_MODEL_PATH = os.path.join(_THIS_DIR, "attention_bilstm_global.keras")
ML_SCALERS_PATH = Path(_THIS_DIR) / "scalers.pkl"

# Fallback defensivo únicamente. El valor real de cada request viaja en
# `forecastHorizonDays` desde el Command Center y pasa por
# `_resolve_horizon_days()` antes de tocar el motor Keras.
DEFAULT_FORECAST_HORIZON_DAYS = 5
PLAN_A_TICKER = "SPY"  # Reserva — activo ancla de baja volatilidad

# --- V3: TECH_COLS — fuente de verdad ÚNICA del orden de columnas técnicas
# del tensor de 14 features (PRICE + 8 técnicos + 5 macro). DEBE coincidir
# EXACTO (mismo orden) con TECH_COLS en entrenamiento.py — el feature_scaler
# cargado desde scalers.pkl fue ajustado con ESTE orden, así que desalinearlo
# acá desplazaría cada columna en silencio dentro de feature_scaler.transform().
# Se valida contra scalers["tech_cols"] en `_get_scalers()` (ver más abajo),
# para fallar ruidosamente antes de la primera inferencia en vez de producir
# un forecast matemáticamente incorrecto sin ninguna excepción visible.
TECH_COLS = ["RSI_14", "EMA_20", "MACD", "MACD_SIGNAL", "SENTIMENT_SCORE",
             "ATR_14", "BB_WIDTH_20", "OBV"]

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


_MC_DROPOUT_LAYER_NAME = "dropout_regularizer"
_MC_DROPOUT_DOWNSTREAM_LAYERS = ("post_fusion_dense", "return_head")


def _build_mc_dropout_bridge(model: KerasModel) -> KerasModel:
    """
    DIAGNÓSTICO RAÍZ (colapso de varianza / línea recta) — confirmado
    inspeccionando `config.json` DENTRO del propio
    `attention_bilstm_global.keras` (no una suposición): el nodo Functional
    del layer `dropout_regularizer` quedó serializado con
    `inbound_nodes[0].kwargs == {"training": False}` — un booleano concreto
    HORNEADO en tiempo de construcción del grafo en el notebook (ej.
    `layers.Dropout(0.4)(fusion, training=False)`), no un valor simbólico.

    Cuando una capa recibe un booleano literal (no `None`) como `training`
    dentro del Functional API, Keras lo fija PERMANENTEMENTE en ESE nodo.
    Llamar después a `model([...], training=True)` sobre el modelo completo
    (como ya hacía `_forecast_asset`) NO tiene ningún efecto sobre ese nodo
    puntual — el bug nunca estuvo en esta capa de inferencia (el
    `training=True` explícito siempre fue correcto), sino horneado en el
    artefacto `.keras` mismo. Resultado: Dropout(0.4) nunca se activaba,
    cada paso autoregresivo era 100% determinista, y el ruido de MC Dropout
    que debía generar picos/caídas realistas jamás existió — de ahí la
    línea recta/curva suave pese al bucle autoregresivo estar bien escrito.

    FIX sin reentrenar y sin tocar el .keras original: se reutilizan los
    MISMOS objetos de capa (mismos pesos entrenados) para reconstruir, en
    un grafo Functional nuevo, únicamente el tramo final —
    `dropout_regularizer -> post_fusion_dense -> return_head` — volviendo a
    invocar `dropout_regularizer` con `training=True` explícito en un NODO
    NUEVO. Esto es un patrón nativo y soportado de Keras (capas
    reutilizables / multi-nodo, igual que un modelo "siamés"): no duplica
    pesos, no reentrenar nada, solo corrige la invocación congelada.
    """
    dropout_layer = model.get_layer(_MC_DROPOUT_LAYER_NAME)

    # CRÍTICO: leer `.input` ANTES de volver a invocar la capa. `Layer.input`
    # solo es válido mientras la capa tenga exactamente UN nodo entrante —
    # el nodo original congelado (`training=False`). Si se leyera después de
    # añadir el nodo nuevo, Keras lanzaría ambigüedad (dos nodos posibles).
    fusion_tensor = dropout_layer.input

    x_mc = dropout_layer(fusion_tensor, training=True)  # nodo NUEVO, no congelado
    for layer_name in _MC_DROPOUT_DOWNSTREAM_LAYERS:
        x_mc = model.get_layer(layer_name)(x_mc)

    return keras.Model(inputs=model.input, outputs=x_mc, name="mc_dropout_bridge")


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
            out_1 = keras.ops.convert_to_numpy(model(sample_input, training=True))
            out_2 = keras.ops.convert_to_numpy(model(sample_input, training=True))
        return not np.allclose(out_1, out_2, atol=1e-9)
    except Exception as exc:  # noqa: BLE001 — la verificación jamás debe tumbar la carga del modelo
        print(f"⚠️ No se pudo verificar la estocasticidad del modelo cargado ({exc!r}).")
        return False


@lru_cache(maxsize=1)
def _get_keras_model() -> tuple[KerasModel, tuple[str, ...]]:
    """
    Carga `attention_bilstm_global.keras` UNA sola vez por proceso, en
    memoria, sin ONNX Runtime de por medio. `compile=False` porque solo
    hacemos forward passes (`model(...)`), nunca `.fit()` — evita reconstruir
    optimizer/loss innecesariamente al cargar.

    Devuelve el "MC-Dropout bridge" (ver `_build_mc_dropout_bridge`) cuando
    el grafo original tiene `dropout_regularizer` congelado en
    `training=False` — que es el caso real de este artefacto — en vez del
    `model` crudo, que seguiría siendo determinista para siempre sin
    importar qué `training` se le pase desde `_forecast_asset`.
    """
    if not os.path.exists(ML_MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo Keras en: {ML_MODEL_PATH}")

    model = keras.models.load_model(
        ML_MODEL_PATH,
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
            "⚠️ MC Dropout NO disponible: el modelo cargado desde "
            f"{os.path.basename(ML_MODEL_PATH)} no tiene ninguna capa Dropout en su grafo. "
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
            "se usa `model` crudo. Si el nodo de "
            f"'{_MC_DROPOUT_LAYER_NAME}' NO estaba congelado en `training=False` "
            "en tu grafo, esto es esperado y el `model` crudo ya es correcto; "
            "si SÍ lo estaba, el forecast volverá a ser determinista."
        )
        return model, dropout_layers

    if not _verify_stochastic_variance(inference_model):
        print(
            "🚨 ALERTA — el bridge de MC Dropout se construyó SIN excepción, pero "
            "dos forward passes idénticos con training=True devolvieron el MISMO "
            "resultado: la varianza sigue sin inyectarse pese al bridge. Revisa que "
            f"'{_MC_DROPOUT_LAYER_NAME}' y {_MC_DROPOUT_DOWNSTREAM_LAYERS} coincidan "
            "EXACTAMENTE con los nombres reales del grafo (`model.summary()` / "
            "`[l.name for l in model.layers]`) — un nombre desalineado es la causa "
            "más probable de un forecast que sigue siendo una línea recta."
        )

    return inference_model, dropout_layers


@lru_cache(maxsize=1)
def _get_scalers() -> dict:
    scalers_path = Path(ML_SCALERS_PATH)
    if not scalers_path.exists():
        raise FileNotFoundError(f"No se encontró el bundle de scalers en: {scalers_path}")

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
            f"({TECH_COLS}). El .keras/scalers.pkl cargados no corresponden "
            "al pipeline V3 vigente — regenera ambos artefactos desde "
            "entrenamiento.py antes de servir inferencia."
        )
    if saved_tech_cols is None:
        print(
            f"⚠️ {scalers_path.name} no incluye la clave 'tech_cols' (bundle "
            f"pre-V3) — se asume el orden hardcodeado {TECH_COLS} sin poder "
            "validarlo contra el artefacto real. Regenera scalers.pkl con "
            "el notebook V3 (entrenamiento.py) para una validación genuina."
        )

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


# --- SENTIMENT_SCORE: proxy de noticias vía Z-score de momentum ------------
# DEBE replicar bit a bit `compute_sentiment_score()` del notebook de
# entrenamiento. `SENTIMENT_LOOKBACK_DAYS` == `SENTIMENT_LOOKBACK` allá.
SENTIMENT_LOOKBACK_DAYS = 20


def _compute_sentiment_score(price_series: pd.Series, window: int = SENTIMENT_LOOKBACK_DAYS) -> pd.Series:
    """Z-score del log-return del día frente a su media/std móvil de `window` sesiones."""
    log_returns = np.log(price_series / price_series.shift(1))
    rolling_mean = log_returns.rolling(window=window, min_periods=window).mean()
    rolling_std = log_returns.rolling(window=window, min_periods=window).std()
    z_score = (log_returns - rolling_mean) / rolling_std.replace(0.0, np.nan)
    return z_score.fillna(0.0)


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


def _compute_indicator_states(close_df: pd.DataFrame, ticker: str) -> dict:
    """
    Estado recursivo final de RSI/EMA/MACD/SENTIMENT/ATR/BB_WIDTH/OBV sobre
    el historial real — necesario para actualizarlos incrementalmente dentro
    del bucle autoregresivo en vez de congelarlos (ver _forecast_asset).

    `close_df` es el frame COMPLETO devuelto por `_fetch_feature_window`
    (incluye `f"{ticker}_High"` / `f"{ticker}_Low"` / `f"{ticker}_Volume"`
    además de la serie de precio bajo la columna `ticker`) — ATR_14 y OBV
    necesitan ese OHLCV real para su estado inicial, no solo el precio.
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

    # Estado recursivo de SENTIMENT_SCORE: a diferencia de RSI/EMA/MACD (que
    # se resumen en un puñado de escalares IIR), el Z-score de momentum
    # necesita la ventana CRUDA de los últimos `SENTIMENT_LOOKBACK_DAYS`
    # log-returns reales para poder recalcular media/std sobre una ventana
    # deslizante en cada paso del bucle (ver _forecast_asset). Se guarda tal
    # cual — sin recortar ni promediar — para no perder precisión.
    log_returns_hist = np.log(close_series / close_series.shift(1)).dropna()
    tail_returns = log_returns_hist.tail(SENTIMENT_LOOKBACK_DAYS).to_numpy(dtype=np.float64)
    if len(tail_returns) < SENTIMENT_LOOKBACK_DAYS:
        # Historial insuficiente (activo recién listado): rellena por
        # delante repitiendo el primer valor disponible, o con ceros si no
        # hay ninguno — evita un ValueError en el reshape aguas abajo.
        pad = SENTIMENT_LOOKBACK_DAYS - len(tail_returns)
        fill_value = tail_returns[0] if len(tail_returns) else 0.0
        tail_returns = np.concatenate([np.full(pad, fill_value), tail_returns])

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
    obv_prev = float(_compute_obv(close_series, volume_series).iloc[-1])
    volume_tail_mean = volume_series.tail(OBV_VOLUME_PROXY_LOOKBACK_DAYS).mean()
    volume_proxy = float(volume_tail_mean) if pd.notna(volume_tail_mean) else 0.0

    return {
        "avg_gain": float(avg_gain) if pd.notna(avg_gain) else 0.0,
        "avg_loss": float(avg_loss) if pd.notna(avg_loss) else 0.0,
        "ema_fast": float(ema_fast),
        "ema_slow": float(ema_slow),
        "ema_20": float(ema_20),
        "macd_signal": float(macd_signal),
        "log_return_window": tail_returns,  # shape (SENTIMENT_LOOKBACK_DAYS,)
        "atr_prev": atr_prev,
        "price_window": tail_prices,  # shape (BB_WIDTH_WINDOW,)
        "obv_prev": obv_prev,
        "volume_proxy": max(volume_proxy, 0.0),
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
        - `_compute_indicator_states(close_df, ticker)` recalcula RSI/EMA/
          MACD/SENTIMENT/ATR_14/BB_WIDTH_20/OBV desde cero sobre la serie de
          precios (y OHLCV) completa — si se le pasara solo `lookback`
          filas, se perdería TODO el beneficio de los 2y de calentamiento
          que acabamos de pagar en la descarga.
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
    raw = yf.download(all_symbols, period="2y", auto_adjust=True, progress=False)

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

    close["RSI_14"] = _compute_rsi(close[ticker])
    close["EMA_20"] = close[ticker].ewm(span=20, adjust=False).mean()
    macd_line, signal_line = _compute_macd(close[ticker])
    close["MACD"] = macd_line
    close["MACD_SIGNAL"] = signal_line
    close["SENTIMENT_SCORE"] = _compute_sentiment_score(close[ticker])
    # V3: ATR_14 (volatilidad absoluta, requiere High/Low reales del
    # ticker), BB_WIDTH_20 (volatilidad relativa vía Bollinger) y OBV
    # (volumen firmado acumulado, requiere Volume real del ticker).
    close["ATR_14"] = _compute_atr(close[f"{ticker}_High"], close[f"{ticker}_Low"], close[ticker])
    close["BB_WIDTH_20"] = _compute_bb_width(close[ticker])
    close["OBV"] = _compute_obv(close[ticker], close[f"{ticker}_Volume"])
    close = close.ffill().dropna()

    # Orden EXACTO de TECH_COLS (ver constante de módulo, arriba): RSI_14,
    # EMA_20, MACD, MACD_SIGNAL, SENTIMENT_SCORE, ATR_14, BB_WIDTH_20, OBV —
    # el feature_scaler cargado desde scalers.pkl fue ajustado con ese
    # orden, así que romperlo aquí desalinearía cada columna al pasar por
    # `.transform()`. `_get_scalers()` ya validó que TECH_COLS coincide con
    # `scalers["tech_cols"]` al arrancar el proceso.
    feature_cols = [ticker] + TECH_COLS + macro_tickers

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
    scalers = _get_scalers()
    if ticker not in scalers["asset_to_id"]:
        raise ValueError(f"'{ticker}' no pertenece al universo entrenado del modelo global.")

    asset_id = scalers["asset_to_id"][ticker]
    feature_scaler = scalers["feature_scalers"][ticker]
    target_scaler = scalers["target_scalers"][ticker]
    lookback = scalers["lookback"]
    macro_tickers = scalers["macro_tickers"]
    n_sim = N_MONTE_CARLO_SIMULATIONS

    # `model` aquí puede ser el grafo crudo o el "MC-Dropout bridge" (ver
    # `_build_mc_dropout_bridge` / `_get_keras_model`) — la función decide
    # cuál sirve según si el nodo `dropout_regularizer` del .keras venía
    # congelado en `training=False`. Esta función NUNCA debe asumir cuál de
    # los dos es: solo le pasa `training=True` y confía en que
    # `_get_keras_model` ya resolvió el bug de congelamiento.
    model, dropout_layers = _get_keras_model()
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

    last_price = float(close_df[ticker].iloc[-1])
    cursor_date = close_df.index[-1]
    anchor_date_str = cursor_date.strftime("%Y-%m-%d")  # "HOY" — ancla real, pre-proyección

    hist_window = _historical_window_for_horizon(steps)
    historical = [
        {"date": d.strftime("%Y-%m-%d"), "price": round(float(p), 2)}
        for d, p in close_df[ticker].tail(hist_window).items()
    ]

    # Estado vectorizado: N trayectorias en paralelo, cada una con su propia
    # ventana de features y su propio estado de indicadores (RSI/EMA/MACD),
    # todas arrancando del MISMO histórico real (`raw_features`).
    cursor_windows = np.repeat(raw_features[-lookback:][np.newaxis, :, :], n_sim, axis=0).astype(np.float64)
    cursor_prices = np.full(n_sim, last_price, dtype=np.float64)

    base_indicator_state = _compute_indicator_states(close_df, ticker)
    avg_gain_arr = np.full(n_sim, base_indicator_state["avg_gain"], dtype=np.float64)
    avg_loss_arr = np.full(n_sim, base_indicator_state["avg_loss"], dtype=np.float64)
    ema_fast_arr = np.full(n_sim, base_indicator_state["ema_fast"], dtype=np.float64)
    ema_slow_arr = np.full(n_sim, base_indicator_state["ema_slow"], dtype=np.float64)
    ema_20_arr = np.full(n_sim, base_indicator_state["ema_20"], dtype=np.float64)
    macd_signal_arr = np.full(n_sim, base_indicator_state["macd_signal"], dtype=np.float64)
    # SENTIMENT_SCORE: cada trayectoria arranca con la MISMA ventana real de
    # los últimos SENTIMENT_LOOKBACK_DAYS log-returns (no hay fuga de futuro,
    # todo viene del histórico ya observado) y a partir de ahí diverge de
    # forma independiente, igual que RSI/EMA/MACD.
    log_return_window_arr = np.repeat(
        base_indicator_state["log_return_window"][np.newaxis, :], n_sim, axis=0
    ).astype(np.float64)  # shape (n_sim, SENTIMENT_LOOKBACK_DAYS)
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
    # Proxy de volumen CONSTANTE (promedio histórico reciente) — este motor
    # no proyecta volumen futuro, solo precio; ver OBV_VOLUME_PROXY_LOOKBACK_DAYS.
    volume_proxy = base_indicator_state["volume_proxy"]

    forecast_dates: list[str] = []
    price_paths: list[np.ndarray] = []  # un array (n_sim,) por fecha proyectada

    forecast_incomplete = False
    for step_i in range(steps):
        try:
            # Batch de N ventanas -> 2D para el scaler (fit espera
            # (n_samples, n_features)) -> de vuelta a 3D (n_sim, lookback, n_features).
            flat_windows = cursor_windows.reshape(-1, cursor_windows.shape[-1])
            scaled_flat = feature_scaler.transform(flat_windows)
            scaled_windows = scaled_flat.reshape(n_sim, lookback, -1).astype(np.float32)
            asset_tensor = np.full((n_sim, 1), asset_id, dtype=np.int32)

            # MC Dropout real + batch de N trayectorias en UNA sola llamada:
            # `training=True` mantiene vivo `Dropout(0.4)` y, al ser un
            # batch, cada fila del tensor de salida muestrea su propia
            # máscara — N inferencias estocásticas genuinas por el precio de
            # UN forward pass, no de N. `inference_lock` protege esta única
            # llamada por paso (Plan A y Plan B corren en threads distintos
            # contra el mismo `model` cacheado — ver _build_investment_plans).
            with inference_lock:
                y_pred_scaled = model([scaled_windows, asset_tensor], training=True)
            y_pred_np = keras.ops.convert_to_numpy(y_pred_scaled)
            if y_pred_np.size != n_sim:
                # Verificación defensiva: un batch de N debe devolver N
                # log-returns escalares. Cualquier otra cardinalidad es una
                # señal de que la capa de salida ('return_head') no coincide
                # con lo esperado — mejor fallar ruidosamente aquí que
                # colapsar/broadcastear datos entre trayectorias en silencio.
                raise ValueError(
                    f"Salida del modelo con tamaño inesperado {y_pred_np.shape} "
                    f"(se esperaban {n_sim} log-returns, uno por trayectoria)."
                )
            r_scaled_batch = y_pred_np.reshape(n_sim, 1)
            r_hat_model_batch = target_scaler.inverse_transform(r_scaled_batch).reshape(n_sim)

            # Inyección de Volatilidad Estocástica — un draw i.i.d. POR
            # TRAYECTORIA, ocurre aquí sobre el vector `r_hat_model_batch` ya
            # en espacio real (post inverse_transform), nunca sobre los
            # tensores de entrada del modelo. La varianza entre trayectorias
            # (y por tanto el ancho de la banda P5-P95) crece de forma
            # natural con sqrt(steps), como un random walk real.
            if noise_scale > 0:
                noise_batch = np.random.normal(loc=0.0, scale=noise_scale, size=n_sim)
            else:
                noise_batch = np.zeros(n_sim)
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
        cursor_date = cursor_date + timedelta(days=1)
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
        macd_new_arr = ema_fast_arr - ema_slow_arr
        macd_signal_arr = macd_new_arr * MACD_SIGNAL_K + macd_signal_arr * (1 - MACD_SIGNAL_K)

        # SENTIMENT_SCORE recursivo: el log-return SINTÉTICO recién generado
        # (r_hat_batch, ya con volatilidad inyectada) entra a la ventana
        # deslizante de cada trayectoria; el Z-score se recalcula sobre esa
        # ventana propia — misma fórmula exacta que compute_sentiment_score()
        # del notebook, solo que aquí es online y vectorizada por trayectoria.
        log_return_window_arr = np.concatenate(
            [log_return_window_arr[:, 1:], r_hat_batch[:, np.newaxis]], axis=1
        )
        sentiment_mean_arr = log_return_window_arr.mean(axis=1)
        sentiment_std_arr = log_return_window_arr.std(axis=1)
        safe_std_arr = np.where(sentiment_std_arr == 0.0, 1.0, sentiment_std_arr)
        sentiment_arr = np.where(
            sentiment_std_arr == 0.0,
            0.0,
            (r_hat_batch - sentiment_mean_arr) / safe_std_arr,
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

        # --- V3: BB_WIDTH_20 recursivo. Ventana deslizante de PRECIOS
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

        # --- V3: OBV recursivo. sign(ΔP) * volumen PROXY constante (ver
        # `volume_proxy` arriba) — este motor no proyecta volumen futuro,
        # solo precio, así que el volumen sintético de cada paso reutiliza
        # el promedio histórico reciente en vez de inventar una cifra nueva.
        direction_arr = np.sign(delta_arr)
        obv_arr = obv_arr + direction_arr * volume_proxy

        new_rows = cursor_windows[:, -1, :].copy()
        new_rows[:, 0] = cursor_prices
        new_rows[:, 1] = rsi_arr
        new_rows[:, 2] = ema_20_arr
        new_rows[:, 3] = macd_new_arr
        new_rows[:, 4] = macd_signal_arr
        new_rows[:, 5] = sentiment_arr
        new_rows[:, 6] = atr_arr
        new_rows[:, 7] = bb_width_arr
        new_rows[:, 8] = obv_arr
        # macro_tickers (índice 9+, tras PRICE + 8 técnicos) se propagan sin
        # cambio por diseño — no hay forecast de esos activos en este motor.
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
        # Para exponer esa textura real día a día SIN inventar un solo
        # dato, se selecciona la trayectoria ya simulada (una de las N=100
        # reales) más cercana al camino mediano en distancia euclídea sobre
        # todo el horizonte. Sigue siendo salida 100% genuina del modelo +
        # ruido calibrado con volatilidad histórica real — solo que sin el
        # efecto de cancelación del percentil-por-fecha. `variance_source`
        # sigue describiendo el mecanismo real; esto no lo cambia.
        dist_to_median = np.linalg.norm(price_matrix - expected_arr[:, np.newaxis], axis=0)
        representative_idx = int(np.argmin(dist_to_median))
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
        anchor_price=round(last_price, 2),
    )

    return {
        "ticker": ticker,
        "last_price": round(last_price, 2),
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
        "confidence_interval": f"P{CONFIDENCE_LOWER_PERCENTILE}-P{CONFIDENCE_UPPER_PERCENTILE}",
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
    """
    degraded = False
    if analysis_mode == "specific" and target_asset:
        normalized_ticker = target_asset.strip().upper()
        try:
            scalers = _get_scalers()
        except FileNotFoundError:
            scalers = None

        if scalers and normalized_ticker in scalers["asset_to_id"]:
            return normalized_ticker, False

        print(
            f"⚠️ target_asset '{target_asset}' fuera del universo entrenado "
            "— degradando a 'discovery' vía radar."
        )
        degraded = True

    top_assets = radar_data.get("top_assets", [])
    if not top_assets:
        return "NVDA", degraded
    best = max(top_assets, key=lambda a: a.get("risk_score", 0))
    return best.get("symbol", "NVDA"), degraded


# ---------------------------------------------------------------------------
# SentimentEnricher — capa async de "Visión Fundamental". Independiente del
# pipeline Keras: no toca `scaled_window` ni el bucle autoregresivo, solo
# enriquece el payload de salida con contexto macro antes del dispatch.
# Circuit Breaker propio — un fallo aquí NUNCA debe tumbar la proyección
# Monte Carlo ya calculada (ver except al fondo de get_market_sentiment).
# ---------------------------------------------------------------------------
SENTIMENT_API_TIMEOUT_SECONDS = 2.5
SENTIMENT_MOMENTUM_LOOKBACK_DAYS = 21  # ~1 mes bursátil
SENTIMENT_BULLISH_THRESHOLD = 0.2
SENTIMENT_BEARISH_THRESHOLD = -0.2


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
    silenciosamente al fallback Neutral, sin propagar excepción.
    """
    try:
        async def _fetch_raw_score() -> float:
            loop = asyncio.get_running_loop()

            def _sync_momentum() -> float:
                hist = yf.Ticker(symbol).history(
                    period=f"{SENTIMENT_MOMENTUM_LOOKBACK_DAYS + 5}d"
                )
                closes = hist["Close"].dropna()
                if len(closes) < 2:
                    raise ValueError(f"Histórico insuficiente para {symbol}")
                window = closes.tail(SENTIMENT_MOMENTUM_LOOKBACK_DAYS)
                log_return = float(np.log(window.iloc[-1] / window.iloc[0]))
                daily_log_returns = np.log(window / window.shift(1)).dropna()
                sigma = float(daily_log_returns.std()) or 1e-6
                return log_return / (sigma * np.sqrt(len(window)))

            return await loop.run_in_executor(None, _sync_momentum)

        raw_score = await asyncio.wait_for(
            _fetch_raw_score(), timeout=SENTIMENT_API_TIMEOUT_SECONDS
        )
        sentiment_score = round(float(np.clip(raw_score, -1.0, 1.0)), 4)
        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": _score_to_sentiment_label(sentiment_score),
            "data_source": "KodaQuant Sentinel",
        }
    except Exception as exc:  # noqa: BLE001 — jamás propagar, degradar a Neutral
        print(
            f"⚠️ Circuit Breaker activado — fallo en get_market_sentiment ({symbol}): "
            f"{type(exc).__name__}: {exc!r}\n"
            f"{traceback.format_exc()}"
        )
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
) -> dict:
    """
    ASÍNCRONA: Plan A, Plan B y su sentimiento de mercado se despachan EN
    PARALELO vía asyncio.gather (cada forecast es un yfinance.download +
    bucle autoregresivo Keras independiente; cada sentiment es su propio
    fetch con Circuit Breaker propio) en vez de secuencial — corta la
    latencia total de cada generación de estrategia.
    """
    split = _resolve_risk_split(risk_score)
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
        get_market_sentiment(PLAN_A_TICKER),
        get_market_sentiment(plan_b_ticker),
    )

    return {
        "plan_a": {
            "nombre": "Plan A — Reserva",
            "monto_usd": round(budget_usd * split["plan_a"], 2),
            "pct": _fmt_pct(split["plan_a"]),
            "activo_referencia": PLAN_A_TICKER,
            "forecast": plan_a_forecast,
            "market_context": plan_a_sentiment,
        },
        "plan_b": {
            "nombre": "Plan B — Riesgo",
            "monto_usd": round(budget_usd * split["plan_b"], 2),
            "pct": _fmt_pct(split["plan_b"]),
            "activo_referencia": plan_b_ticker,
            "forecast": plan_b_forecast,
            "market_context": plan_b_sentiment,
        },
        "target_asset_degraded": target_asset_degraded,
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

    return _build_language_directive(language) + "\n\n" + CORE_DIRECTIVES + "\n\n" + f"""
Eres Quanti, la IA táctica de KodaQuant. Eres un estratega cuantitativo de élite institucional.

Reglas inquebrantables:
- CERO formato Markdown (prohibido usar asteriscos **).
- Usa exactamente DOS saltos de línea (\\n\\n) para separar tus párrafos.
- Sé quirúrgico, sofisticado y directo. No superes las 4 líneas por párrafo.
- Nunca reveles que eres un modelo de lenguaje.

DATOS MATEMÁTICOS OBLIGATORIOS (proyección del motor cuantitativo — modelo Keras +
volatilidad histórica calibrada — ÚNICA FUENTE VÁLIDA: PROHIBIDO INVENTAR, REDONDEAR,
OMITIR O RECALCULAR ninguna fecha, precio o monto; trátalos como una proyección, no
como una certeza garantizada):
- Capital total: {_fmt_money(budget_usd)} | Risk Score: {risk_score}/100 ({risk_profile})
- Señal de mercado: {market_signal}
- {plan_a_line}
- {plan_b_line}
- {total_profit_line}
- No confundas el precio de cotización del activo con la ganancia neta sobre el capital: son cifras distintas.

Estructura OBLIGATORIA del reporte (integra estos datos palabra por palabra, sin alterarlos):

RESUMEN DEL PORTAFOLIO TÁCTICO
Redacta un análisis brillante e institucional de la configuración actual: Capital Total invertido, Perfil de Riesgo y Ganancia Neta Total Esperada. Máximo 4 líneas.

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
            },
            "allocation": portfolio_allocation,
        }

        system_prompt = _build_system_prompt(
            budget_usd=budget_usd, market_signal=market_signal,
            plan_a=investment_plans["plan_a"], plan_b=investment_plans["plan_b"],
            risk_profile=resolved_risk_profile, risk_score=resolved_risk_score,
            horizon_days=horizon,
            plan_a_operation=plan_a_operation, plan_b_operation=plan_b_operation,
            language=resolved_language,
        )
        user_prompt = _build_user_prompt(resolved_language)
    except Exception as exc:  # noqa: BLE001 — blindaje total del pipeline (FASE 3)
        traceback.print_exc()
        print(f"🔥 ERROR CRÍTICO EN QUANTI (generate_quanti_strategy_stream, pre-LLM): {exc}")
        yield f"event: error\ndata: {json.dumps({'status': 'error', 'error': str(exc)})}\n\n"
        return

    yield f"event: meta\ndata: {json.dumps(meta_payload)}\n\n"

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
) -> list[str]:
    """
    LA VOZ DE QUANTI. Nunca recalcula cifras — solo las narra, ahora como 3
    órdenes tácticas de trading (Resumen de Portafolio + Operación de Reserva
    Plan A + Operación de Riesgo Plan B). El "hard lock" numérico vive en
    _build_system_prompt / _build_tactical_operation; fallback seguro si
    Groq falla (ver _call_quanti_llm, Circuit Breaker, sin cambios). `language`
    ('en' | 'es', default 'en') gobierna la CRITICAL LANGUAGE DIRECTIVE
    anteponía en _build_system_prompt.
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
        )

        # amounts (riesgo/reserva) SIEMPRE deriva de investment_plans — un solo
        # cálculo de split (_resolve_risk_split), cero duplicidad con datos viejos.
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