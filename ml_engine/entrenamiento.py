# %%
#!pip install yfinance -q

"""
Pipeline Predictivo GLOBAL Multi-Activo: Attention-BiLSTM (Log-Returns) — V3
=============================================================================
Arquitectura (heredada de la V2, SIN cambios estructurales — ver sección 9
más abajo, los cambios de la V3 son de entrenamiento/loss/regularización,
no de topología): BiLSTM(256) -> LayerNorm -> [MultiHeadAttention + Residual]
                 -> LayerNorm -> [BahdanauAttention || BiLSTM(128)+LayerNorm]
                 -> Concat + Embedding(asset_id) -> Dropout(0.45)
                 -> [Dense(64)+L2 + Residual(proyección+L2)] -> LayerNorm -> Dense(1)+L2
Entorno objetivo: local (VSCode / macOS), TensorFlow/Keras 3.x

CAMBIOS RESPECTO A LA V1 (single-asset, precio absoluto):

1) GLOBAL MULTI-ACTIVO: se entrena UN SOLO modelo sobre 10 activos distintos,
   combinados en un dataset masivo. Un input categórico `asset_id` (Embedding)
   le da al modelo la capacidad de diferenciar el "régimen" de cada activo
   dentro del mismo espacio de pesos compartido — sin esto, el modelo no
   podría distinguir un día de BTC-USD de un día de SPY.

2) TARGET = LOG-RETURN, no precio absoluto. Un LSTM entrenado a predecir el
   precio absoluto de un activo casi-random-walk aprende trivialmente a
   "copiar" el último precio conocido (efecto shadow/lag: la predicción
   parece la serie real desplazada un día). Al predecir
   log(P_t / P_{t-1}) forzamos al modelo a aprender señal real, no inercia.
   El precio se RECONSTRUYE matemáticamente en evaluación:
       P_hat_t = P_{t-1} * exp(r_hat_t)

3) NORMALIZACIÓN POR ACTIVO: cada activo tiene su propio feature_scaler y su
   propio target_scaler, ajustados SOLO con el tramo de train de ESE activo.
   Esto evita que activos de alta volatilidad (BTC-USD) dominen la pérdida
   frente a activos de baja volatilidad (SPY), y evita fuga de información
   train/test.

4) MÉTRICA NUEVA: directional accuracy (% de aciertos en el SIGNO del
   retorno), la métrica que realmente importa en un modelo de retornos —
   MAE/RMSE de precio reconstruido pueden verse "bien" incluso si el modelo
   no acierta la dirección.

CAMBIOS V2 (este archivo):

5) FEATURES DE VOLATILIDAD/VOLUMEN: ATR_14 (volatilidad absoluta), BB_WIDTH_20
   (volatilidad relativa vía Bollinger) y OBV (volumen firmado acumulado).
   Requieren descargar High/Low/Volume además de Close -> `download_all`
   ahora trae el bundle OHLCV completo, no solo Close.

6) ARQUITECTURA: LayerNormalization tras cada bloque recurrente/atencional
   (estabiliza la escala de los hidden states entre capas), un bloque de
   MultiHeadAttention auto-atencional CON conexión residual antes del
   Bahdanau (que sigue actuando como pooling final de contexto), y una
   conexión residual (con proyección lineal para igualar dimensiones) en el
   bloque denso profundo.

7) PÉRDIDA ASIMÉTRICA DIRECCIONAL: `DirectionalHuberLoss` escala Huber por
   (1 + gamma) cuando sign(y_true) != sign(y_pred). CORRECCIÓN NECESARIA:
   para que "signo en espacio escalado" == "signo del retorno real", el
   target_scaler pasa de StandardScaler() a StandardScaler(with_mean=False)
   (solo divide por sigma, no resta la media) — de lo contrario, en activos
   con drift fuerte (BTC, NVDA) el "cruce por cero" quedaría desplazado a
   y_raw=mu≠0 y la pérdida direccional penalizaría el signo EQUIVOCADO en un
   sub-conjunto de muestras cercanas a cero. Ver docstring de la clase.

8) SCHEDULING: CosineDecayRestarts reemplaza a EarlyStopping/ReduceLROnPlateau
   como mecanismo de escape de mínimos locales, inyectado directamente en el
   optimizador (así es como Keras 3 espera un LearningRateSchedule — NO como
   callback). ModelCheckpoint(save_best_only=True) + recarga post-fit se
   mantienen para no terminar el entrenamiento con los pesos de la ÚLTIMA
   época (que tras un reinicio de LR puede ser peor que un punto intermedio).

CAMBIOS V3 (este archivo) — FIX DEL COLAPSO "COBARDE":

9) DIAGNÓSTICO V2: con gamma=1.5 FIJO desde el Epoch 0, el gradiente de la
   penalización direccional domina antes de que el modelo aprenda siquiera
   la ESCALA de los retornos. La salida más "segura" para minimizar
   Huber*(1+gamma) bajo incertidumbre de signo es predecir ŷ≈0 (error de
   magnitud pequeño) -> loss y mae siguen bajando mientras
   directional_accuracy se estanca en ~38% (peor que el 50% de una moneda
   justa, señal de que el sesgo hacia 0 rompe incluso el empate esperable
   de un ŷ aleatorio). Los 4 cambios de abajo atacan esto sin tocar la
   arquitectura (Fases A/B se mantienen idénticas a la V2).

9.1) CURRICULUM LEARNING (GAMMA DINÁMICO). `gamma` deja de ser un float
     Python y pasa a ser un `keras.Variable` no-entrenable, mutado por
     `DynamicGammaCallback` en `on_epoch_begin` según un warmup de
     GAMMA_WARMUP_EPOCHS épocas:

         progress(e) = clip(e / GAMMA_WARMUP_EPOCHS, 0, 1)

         # Lineal:
         gamma(e) = GAMMA_MAX * progress(e)

         # Sigmoide (GAMMA_SCHEDULE="sigmoid", por defecto) — la mayor
         # parte del cambio se concentra a mitad de camino en vez de
         # repartirse uniforme; k=GAMMA_SIGMOID_STEEPNESS controla qué tan
         # abrupta es esa zona central. Se renormaliza para que
         # gamma(0)=0 exacto y gamma(GAMMA_WARMUP_EPOCHS)=GAMMA_MAX exacto
         # (un sigmoide crudo nunca toca 0 ni 1 en sus extremos):
         sig(x)   = 1 / (1 + e^(-k*(x - 0.5)))
         gamma(e) = GAMMA_MAX * (sig(progress(e)) - sig(0)) / (sig(1) - sig(0))

     Con gamma≈0 en las primeras épocas la pérdida es Huber prácticamente
     puro -> el modelo explora magnitudes de retorno sin miedo al castigo
     direccional. Solo cuando gamma empieza a crecer el gradiente también
     empuja a acertar el SIGNO, ya sobre una base de magnitud razonable,
     evitando el óptimo local "predecir 0" que domina cuando ambos
     objetivos compiten desde el Epoch 0.

9.2) GRADIENT CLIPPING. `AdamW(clipnorm=GRAD_CLIPNORM=1.0)`: recorta la
     norma global del gradiente antes de cada update. Con 14 features
     (ATR/OBV con escalas y ruido propios) y los picos de LR que introduce
     cada reinicio de CosineDecayRestarts, un batch con outliers puede
     generar un gradiente desproporcionado que explota los pesos justo
     tras un reinicio -> clipnorm pone un techo duro a esa norma,
     complementando (no reemplazando) a TerminateOnNaN().

9.3) REINICIOS DE LR MÁS FRECUENTES. LR_RESTART_PERIOD_EPOCHS baja de 10 a
     8: reinicios más seguidos agitan la superficie de pérdida con más
     regularidad, ayudando a escapar de mínimos locales -> no se lleva a 5
     para no superponer dos fuentes de inestabilidad (reinicio de LR +
     rampa de gamma) dentro de la misma ventana de épocas.

9.4) REGULARIZACIÓN. Dropout del bloque de fusión sube de 0.4 a 0.45
     (DROPOUT_RATE) + L2 leve (DENSE_L2_REG=1e-5) en las 3 capas Dense de
     la cabeza (post_fusion_dense, fused_projection_skip, return_head):
     fuerza a esas capas a apoyarse en pesos más pequeños y distribuidos,
     en vez de memorizar combinaciones específicas de features ruidosas
     del set de entrenamiento.
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# CONFIG — 100% LOCAL (VSCode / macOS Intel). CERO Colab, CERO Drive.
# ============================================================
BASE_DIR = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
DATA_DIR = BASE_DIR / "data_cache"       # cache local del OHLCV descargado (parquet)
MODEL_DIR = BASE_DIR / "artifacts"       # copia de trabajo/dev (opcional, se mantiene por compatibilidad)
PROD_DIR = BASE_DIR.parent / "services"  # ENRUTAMIENTO DE PRODUCCIÓN: sube un nivel desde BASE_DIR y entra a /services
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROD_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "BTC-USD", "ETH-USD", "SPY"]
MACRO_TICKERS = ["^GSPC", "^TNX", "^VIX", "GC=F", "DX-Y.NYB"]  # SPX, UST10Y, VIX, Oro, Dollar Index
PERIOD = "10y"
LOOKBACK = 60
TRAIN_RATIO = 0.8
SEED = 42
ASSET_EMBED_DIM = 8

SENTIMENT_LOOKBACK = 20  # ventana del Z-score de momentum (proxy de "noticias")

# TECH_COLS es la ÚNICA fuente de verdad de qué técnicos entran al tensor —
# agregar/quitar un nombre aquí basta para que N_FEATURES y feature_cols
# (en engineer_asset / build_asset_dataset) se re-dimensionen solos, SIN
# tocar la arquitectura del modelo a mano.
TECH_COLS = ["RSI_14", "EMA_20", "MACD", "MACD_SIGNAL", "SENTIMENT_SCORE",
             "ATR_14", "BB_WIDTH_20", "OBV"]
N_FEATURES = 1 + len(TECH_COLS) + len(MACRO_TICKERS)  # precio propio + técnicos + macro = 14

ASSET_TO_ID = {t: i for i, t in enumerate(TICKERS)}

# --- Hiperparámetros de entrenamiento V2 (antes hardcodeados dentro de las
#     funciones; centralizados acá para que build_model/train_model queden
#     como funciones puras parametrizadas) ---
BATCH_SIZE = 64
EPOCHS = 100
VALIDATION_SPLIT = 0.1
MHA_HEADS = 4
MHA_KEY_DIM = 32
HUBER_DELTA = 1.0

# --- V3: CURRICULUM LEARNING (GAMMA DINÁMICO) --------------------------
# GAMMA_MAX reemplaza al antiguo DIRECTIONAL_GAMMA=1.5 ESTÁTICO: ahora es
# el TECHO al que converge el schedule, no el valor desde la Epoch 0 (ver
# DynamicGammaCallback y la sección 9.1 del docstring del módulo).
GAMMA_INITIAL = 0.0             # arranque: Huber prácticamente puro (sin penalización de signo)
GAMMA_MAX = 1.5                 # techo de penalización (antiguo DIRECTIONAL_GAMMA)
GAMMA_WARMUP_EPOCHS = 35        # rampa completa entre época 30-40 (elegido: 35)
GAMMA_SCHEDULE = "sigmoid"      # "sigmoid" (rampa suave en S) o "linear"
GAMMA_SIGMOID_STEEPNESS = 10.0  # mayor = transición más abrupta en el punto medio del warmup

# --- V3: REGULARIZACIÓN --------------------------------------------------
DROPOUT_RATE = 0.45          # antes 0.4 -> refuerzo anti-overfitting del bloque de fusión
DENSE_L2_REG = 1e-5          # L2 leve en las cabezas densas (post-fusión + salida)

LR_INITIAL = 1e-3
LR_ALPHA = 1e-2             # piso del LR en cada valle, como fracción del inicial
LR_T_MUL = 2.0              # cada ciclo de reinicio dura 2x el anterior
LR_M_MUL = 0.85             # cada reinicio arranca en 0.85x el LR pico anterior
# V3: de 10 -> 8. Reinicios más frecuentes agitan la superficie de pérdida con
# más regularidad (útil contra el colapso "cobarde"); no se baja a 5 para no
# superponer dos fuentes de inestabilidad (reinicio de LR + rampa de gamma)
# dentro de la misma ventana de épocas -> ver sección 9.3 del docstring.
LR_RESTART_PERIOD_EPOCHS = 8
GRAD_CLIPNORM = 1.0          # V3: clipping explícito -> evita explosión de gradiente en los reinicios de LR

keras.utils.set_random_seed(SEED)  # fija numpy/python/backend de forma consistente (Keras 3)


# ============================================================
# FASE A: ETL — DESCARGA UNIFICADA (OHLCV) + FEATURE ENGINEERING POR ACTIVO
# ============================================================
MAX_DOWNLOAD_RETRIES = 5
BACKOFF_BASE_SECONDS = 2.0


def _download_with_backoff(symbols: list, period: str,
                            max_retries: int = MAX_DOWNLOAD_RETRIES,
                            backoff_base: float = BACKOFF_BASE_SECONDS) -> pd.DataFrame:
    """
    Reintenta la descarga con backoff exponencial (2s, 4s, 8s, 16s, ...) ante
    errores TRANSITORIOS de red o de rate-limit (HTTP 429 / timeouts) — algo
    común en runners de CI/CD que comparten IP de salida con muchos otros
    jobs. Esto es simplemente "esperar y reintentar", el manejo estándar y
    sostenible de un 429 (ver `Retry-After` / RFC 6585); NO altera headers
    para disfrazar el origen del tráfico ni intenta sortear CAPTCHAs — ver
    la nota al final del script sobre por qué esa parte se dejó fuera.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(symbols, period=period, auto_adjust=True, progress=False)
            if data is None or data.empty:
                raise ValueError("yfinance devolvió un DataFrame vacío")
            return data
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            wait_s = backoff_base * (2 ** (attempt - 1))
            print(f"      [reintento {attempt}/{max_retries}] descarga falló ({exc!r}); "
                  f"backoff exponencial -> esperando {wait_s:.0f}s...")
            time.sleep(wait_s)
    raise RuntimeError(
        f"download_all: no se pudo descargar tras {max_retries} intentos"
    ) from last_exc


def download_all(tickers: list, macro_tickers: list, period: str) -> pd.DataFrame:
    """
    Descarga en UNA sola llamada el bundle OHLCV completo (Open/High/Low/
    Close/Volume) de los activos principales + factores macro, con cache
    local en DATA_DIR (parquet). A diferencia de la V1 (que solo traía
    Close), acá se necesita el bundle completo porque ATR requiere
    High/Low y OBV requiere Volume. Para los macro tickers solo se
    conserva Close (son inputs de contexto, no se calculan técnicos sobre
    ellos). Columnas planas "TICKER_FIELD" para que el cache parquet sea
    trivial (evita el manejo de MultiIndex en disco).
    """
    all_symbols = list(dict.fromkeys(tickers + macro_tickers))  # dedup preservando orden
    fields = ["Open", "High", "Low", "Close", "Volume"]
    cache_path = DATA_DIR / f"all_ohlcv_{period}.parquet"

    required_cols = set()
    for t in tickers:
        required_cols.update(f"{t}_{f}" for f in fields)
    for m in macro_tickers:
        required_cols.add(f"{m}_Close")

    if cache_path.exists():
        print(f"      [cache local] leyendo {cache_path.relative_to(BASE_DIR)}")
        cached = pd.read_parquet(cache_path)
        if required_cols.issubset(set(cached.columns)):
            return cached

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
    print(f"      [cache local] guardado en {cache_path.relative_to(BASE_DIR)}")
    return flat


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI nativo (suavizado de Wilder vía EWM)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD nativo: línea MACD y línea de señal."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_sentiment_score(price_series: pd.Series, window: int = SENTIMENT_LOOKBACK) -> pd.Series:
    """
    SENTIMENT_SCORE — proxy de noticias vía Z-score de MOMENTUM histórico.
    No hay feed de noticias real: se aproxima el "shock" de sentimiento del
    día como cuán extremo fue el log-return de ESE día frente a su propia
    distribución reciente (media/std móviles de `window` sesiones). Un
    Z-score fuertemente positivo/negativo se interpreta como un movimiento
    "noticioso" (sorpresa), mientras que valores cerca de 0 son ruido normal
    del régimen de volatilidad vigente del activo.

        z_t = (r_t - mean(r_{t-window:t})) / std(r_{t-window:t})

    donde r_t = log(P_t / P_{t-1}). `std==0` (activo sin movimiento en la
    ventana) se guarda contra división por cero -> NaN -> fillna(0.0).
    """
    log_returns = np.log(price_series / price_series.shift(1))
    rolling_mean = log_returns.rolling(window=window, min_periods=window).mean()
    rolling_std = log_returns.rolling(window=window, min_periods=window).std()
    z_score = (log_returns - rolling_mean) / rolling_std.replace(0.0, np.nan)
    return z_score.fillna(0.0)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    ATR_14 — Average True Range (suavizado de Wilder vía EWM): proxy de
    volatilidad ABSOLUTA (en unidades de precio) que RSI/MACD no capturan.
        TR_t  = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
        ATR_t = EWM_Wilder(TR)_t
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr.fillna(0.0)


def compute_bb_width(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    BB_WIDTH_20 — ancho normalizado de Bandas de Bollinger:
        width_t = (banda_sup - banda_inf) / SMA_t = (2*num_std*std_t) / SMA_t
    Proxy de volatilidad RELATIVA (independiente de la escala de precio),
    complementa a ATR_14 (volatilidad en términos absolutos).
    """
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    width = (2 * num_std * rolling_std) / rolling_mean.replace(0.0, np.nan)
    return width.fillna(0.0)


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    OBV — On-Balance Volume: acumulado de volumen firmado por la dirección
    del precio. Único feature del tensor que incorpora PARTICIPACIÓN
    (volumen); ayuda a detectar divergencias precio/volumen.
        OBV_t = OBV_{t-1} + sign(C_t - C_{t-1}) * V_t
    """
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    return obv.fillna(0.0)


def engineer_asset(all_data: pd.DataFrame, ticker: str, macro_tickers: list) -> pd.DataFrame:
    """Construye el frame de features de UN activo: OHLCV propio + técnicos + macro (solo Close)."""
    close_col = f"{ticker}_Close"
    high_col = f"{ticker}_High"
    low_col = f"{ticker}_Low"
    vol_col = f"{ticker}_Volume"
    macro_close_cols = [f"{m}_Close" for m in macro_tickers]

    df = all_data[[close_col, high_col, low_col, vol_col] + macro_close_cols].copy()
    df = df.ffill().dropna()

    close, high, low, volume = df[close_col], df[high_col], df[low_col], df[vol_col]

    df["RSI_14"] = compute_rsi(close)
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    macd_line, signal_line = compute_macd(close)
    df["MACD"] = macd_line
    df["MACD_SIGNAL"] = signal_line
    df["SENTIMENT_SCORE"] = compute_sentiment_score(close)
    df["ATR_14"] = compute_atr(high, low, close)
    df["BB_WIDTH_20"] = compute_bb_width(close)
    df["OBV"] = compute_obv(close, volume)
    df = df.ffill().dropna()

    feature_cols = [close_col] + TECH_COLS + macro_close_cols
    df = df[feature_cols]
    df.columns = ["PRICE"] + TECH_COLS + macro_tickers  # nombre genérico -> permite apilar activos distintos
    return df


def build_asset_dataset(df: pd.DataFrame, ticker: str, lookback: int, train_ratio: float):
    """
    Ventanas (X) + targets de LOG-RETURN (y) para UN activo, con normalización
    propia (feature_scaler + target_scaler) ajustada SOLO en el tramo de train
    de ese activo. Split cronológico: sin fuga de información test -> train.

    target_scaler = StandardScaler(with_mean=False): SOLO divide por sigma,
    NO resta la media. Es una corrección deliberada respecto a la V1: con
    centrado de media, sign(y_scaled) puede diferir de sign(y_raw) para
    activos con drift fuerte (el "cero" quedaría desplazado a y_raw=mu), lo
    que rompería silenciosamente la semántica de DirectionalHuberLoss. Dividir
    solo por una constante positiva (sigma) preserva el signo exactamente.
    """
    feature_cols = ["PRICE"] + TECH_COLS + MACRO_TICKERS
    prices = df["PRICE"].values
    features = df[feature_cols].values
    log_returns = np.diff(np.log(prices))  # log_returns[k] = log(P[k+1] / P[k])

    n = len(df)
    split_idx = int(n * train_ratio)

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(features[:split_idx])
    features_scaled = feature_scaler.transform(features)

    X, y_raw, last_price, dates = [], [], [], []
    # t recorre el índice del DÍA OBJETIVO. Predecimos el retorno que ocurre
    # entre t-1 y t usando SOLO features hasta t-1 inclusive (sin fuga).
    for t in range(lookback, n):
        X.append(features_scaled[t - lookback: t])
        y_raw.append(log_returns[t - 1])
        last_price.append(prices[t - 1])
        dates.append(df.index[t])

    X = np.array(X, dtype=np.float32)
    y_raw = np.array(y_raw, dtype=np.float32).reshape(-1, 1)
    last_price = np.array(last_price, dtype=np.float32)
    dates = pd.DatetimeIndex(dates)

    window_split = split_idx - lookback  # offset por la ventana de lookback

    target_scaler = StandardScaler(with_mean=False)  # ver docstring: preserva el signo del retorno
    target_scaler.fit(y_raw[:window_split])
    y_scaled = target_scaler.transform(y_raw).astype(np.float32)

    asset_id = np.full((len(X),), ASSET_TO_ID[ticker], dtype=np.int32)

    train = dict(
        X=X[:window_split], y=y_scaled[:window_split], asset_id=asset_id[:window_split],
        last_price=last_price[:window_split], dates=dates[:window_split],
    )
    test = dict(
        X=X[window_split:], y=y_scaled[window_split:], asset_id=asset_id[window_split:],
        last_price=last_price[window_split:], dates=dates[window_split:],
    )
    return train, test, feature_scaler, target_scaler


# ============================================================
# FASE B: ARQUITECTURA GLOBAL — MHA+Residual + Bahdanau + BiLSTM + Asset Embedding
# ============================================================
@keras.saving.register_keras_serializable(package="quanti")
class BahdanauAttention(layers.Layer):
    """Self-attention aditiva (Bahdanau) — pooling final de contexto sobre la secuencia enriquecida por MHA."""

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


# ============================================================
# FASE B.1: FUNCIÓN DE PÉRDIDA — ASYMMETRIC DIRECTIONAL HUBER
# ============================================================
@keras.saving.register_keras_serializable(package="quanti")
class DirectionalHuberLoss(keras.losses.Loss):
    """
    L(y, ŷ) = Huber_δ(y, ŷ) · (1 + γ · 1[sign(y) ≠ sign(ŷ)])

    Extiende Huber (robusta a outliers de retornos extremos) con un
    multiplicador γ que SOLO se activa cuando el modelo falla el SIGNO del
    retorno — el error que realmente destruye una estrategia direccional
    (ir LONG cuando debía ir SHORT, o viceversa), a diferencia de un simple
    error de magnitud con el signo correcto.

    Requiere que y_true/y_pred lleguen en un espacio donde el signo del
    valor escalado coincide EXACTAMENTE con el signo del retorno real; por
    eso `build_asset_dataset` usa StandardScaler(with_mean=False) para el
    target en vez de un StandardScaler centrado en la media (ver su
    docstring). Con centrado de media este término penalizaría el signo
    equivocado para activos con drift fuerte.

    V3 — CURRICULUM LEARNING: `gamma` ahora acepta un `keras.Variable` NO
    entrenable (además del float estático de la V2, por compatibilidad).
    `DynamicGammaCallback` reescribe su VALOR en cada `on_epoch_begin` (de
    0.0 a GAMMA_MAX en rampa sigmoide/lineal a lo largo de
    GAMMA_WARMUP_EPOCHS) -> como `call()` lee `self.gamma` en cada forward
    pass, el nuevo valor se refleja automáticamente sin recompilar el
    modelo. Ver sección 9.1 del docstring del módulo para el fundamento
    matemático completo.
    """

    def __init__(self, delta: float = 1.0, gamma=1.5,
                 name: str = "directional_huber", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = delta
        # Compatibilidad dual: si `gamma` ya es una Variable (curriculum
        # dinámico -> caso normal en V3) se usa TAL CUAL, de modo que el
        # callback y la loss comparten la misma celda de memoria. Si es un
        # float (ej. al recargar un .keras guardado) se envuelve en una
        # Variable no-entrenable nueva, así `call()` nunca tiene que
        # ramificar entre "float" y "Variable".
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
        # self.gamma es una Variable: se lee su valor CORRIENTE en cada
        # forward pass. DynamicGammaCallback la actualiza al inicio de cada
        # época, así que este término crece época a época sin recompilar.
        penalty = 1.0 + self.gamma * mismatch
        return keras.ops.mean(huber * penalty, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "delta": self.delta,
            # Se serializa el VALOR ESCALAR actual (no el objeto Variable,
            # que no es JSON-serializable). Al recargar con
            # keras.models.load_model() este número queda FIJO -> correcto
            # para inferencia; para REANUDAR el curriculum tras una carga
            # habría que re-envolverlo a mano en una Variable nueva.
            "gamma": float(keras.ops.convert_to_numpy(self.gamma)),
        })
        return config


@keras.saving.register_keras_serializable(package="quanti")
def directional_accuracy_metric(y_true, y_pred):
    """Métrica de monitoreo en vivo durante fit(): % de aciertos de signo (espacio escalado, signo-preservado)."""
    y_true = keras.ops.cast(y_true, "float32")
    y_pred = keras.ops.cast(y_pred, "float32")
    match = keras.ops.cast(
        keras.ops.equal(keras.ops.sign(y_true), keras.ops.sign(y_pred)), dtype="float32"
    )
    return keras.ops.mean(match)


# ============================================================
# FASE B.2: CURRICULUM LEARNING — CALLBACK DE GAMMA DINÁMICO (V3)
# ============================================================
class DynamicGammaCallback(keras.callbacks.Callback):
    """
    Reescribe, al inicio de cada época, el `keras.Variable` de gamma que
    consume `DirectionalHuberLoss`, siguiendo un warmup de
    `warmup_epochs` épocas desde `0.0` hasta `gamma_max`. Ver sección 9.1
    del docstring del módulo para la derivación matemática completa del
    schedule sigmoide/lineal.

    Se implementa como Callback (no como parte de la loss) porque el
    schedule depende del ÍNDICE DE ÉPOCA, una noción que la loss —evaluada
    por batch, sin conocimiento del bucle de entrenamiento— no tiene; el
    Callback sí recibe `epoch` en `on_epoch_begin` desde `model.fit()`.
    """

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
        # Normalización del sigmoide para que _progress_to_fraction(0)=0.0 y
        # _progress_to_fraction(1)=1.0 EXACTOS -> un sigmoide "crudo" nunca
        # toca 0 ni 1 en sus extremos (asíntotas), lo que dejaría gamma(0)
        # ligeramente por encima de 0 (penalización direccional desde el
        # primer batch, justo lo que V3 busca evitar).
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
            print(f"      [DynamicGamma] epoch={epoch:3d}  gamma={new_gamma:.4f}{tag}")


def build_model(n_timesteps: int, n_features: int, n_assets: int,
                 embed_dim: int = ASSET_EMBED_DIM, attention_units: int = 64,
                 mha_heads: int = MHA_HEADS, mha_key_dim: int = MHA_KEY_DIM,
                 steps_per_epoch: int = 100,
                 huber_delta: float = HUBER_DELTA,
                 gamma_initial: float = GAMMA_INITIAL,
                 dropout_rate: float = DROPOUT_RATE,
                 dense_l2_reg: float = DENSE_L2_REG) -> tuple:
    # `n_features` NUNCA se hardcodea acá — llega calculado desde
    # N_FEATURES = 1 + len(TECH_COLS) + len(MACRO_TICKERS). Con ATR/BB/OBV
    # agregados a TECH_COLS, `n_features` pasa de 11 a 14 y `input_sequence`
    # se redimensiona solo, sin tocar la arquitectura.
    seq_input = keras.Input(shape=(n_timesteps, n_features), name="input_sequence")
    asset_input = keras.Input(shape=(1,), dtype="int32", name="input_asset_id")

    bilstm_1 = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True), name="bilstm_256"
    )(seq_input)
    bilstm_1 = layers.LayerNormalization(name="ln_post_bilstm_256")(bilstm_1)

    # Bloque auto-atencional multi-cabeza (estilo encoder de Transformer) con
    # conexión residual: cada posición temporal puede atender a TODA la
    # secuencia (no solo al promedio, como hace Bahdanau) antes de que
    # Bahdanau haga el pooling final de contexto. La suma residual +
    # LayerNormalization evita que el bloque MHA degrade el gradiente que
    # fluye desde la BiLSTM (identidad como piso, nunca peor que sin MHA).
    mha_out = layers.MultiHeadAttention(
        num_heads=mha_heads, key_dim=mha_key_dim, name="multi_head_self_attention"
    )(query=bilstm_1, value=bilstm_1, key=bilstm_1)
    attn_block = layers.Add(name="mha_residual_add")([bilstm_1, mha_out])
    attn_block = layers.LayerNormalization(name="ln_post_mha_residual")(attn_block)

    context_vector, attention_weights = BahdanauAttention(
        attention_units, name="bahdanau_attention"
    )(attn_block)

    bilstm_2 = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False), name="bilstm_128"
    )(attn_block)
    bilstm_2 = layers.LayerNormalization(name="ln_post_bilstm_128")(bilstm_2)

    asset_embed = layers.Embedding(n_assets, embed_dim, name="asset_embedding")(asset_input)
    asset_embed = layers.Flatten(name="asset_embedding_flat")(asset_embed)

    fused = layers.Concatenate(name="fusion_attn_bilstm_asset")([context_vector, bilstm_2, asset_embed])
    # V3: dropout_rate sube de 0.4 a 0.45 por defecto (ver sección 9.4 del
    # docstring del módulo) -> parametrizado, ya no hardcodeado.
    fused = layers.Dropout(dropout_rate, name="dropout_regularizer")(fused)

    # Bloque denso con conexión residual: `fused_projection_skip` proyecta el
    # vector de fusión (776-d) a 64-d para poder sumarlo con `post_fusion_dense`
    # (necesario porque las dimensiones no coinciden — proyección lineal, no
    # una identidad pura, siguiendo el patrón de "projection shortcut" de ResNet).
    # V3: las 3 capas Dense de la cabeza llevan L2 leve (dense_l2_reg) ->
    # fuerza pesos más pequeños/distribuidos en vez de memorizar combinaciones
    # específicas de features ruidosas (ver sección 9.4 del docstring).
    l2 = keras.regularizers.l2(dense_l2_reg)
    dense_hidden = layers.Dense(64, activation="relu", name="post_fusion_dense",
                                 kernel_regularizer=l2)(fused)
    fused_projection = layers.Dense(64, name="fused_projection_skip",
                                     kernel_regularizer=l2)(fused)
    dense_block = layers.Add(name="dense_residual_add")([dense_hidden, fused_projection])
    dense_block = layers.LayerNormalization(name="ln_post_dense_residual")(dense_block)

    output = layers.Dense(1, name="return_head", kernel_regularizer=l2)(dense_block)

    model = keras.Model(inputs=[seq_input, asset_input], outputs=output,
                         name="Global_Attention_BiLSTM_Returns_V3")

    # CosineDecayRestarts: LearningRateSchedule inyectado DIRECTO en el
    # optimizador (no es un callback) — cada `first_decay_steps` el LR baja
    # de LR_INITIAL a LR_ALPHA*LR_INITIAL en coseno y luego "reinicia" a un
    # pico más alto, forzando al optimizador a re-explorar la vecindad de un
    # mínimo en vez de estancarse en el primero que encuentra. Los ciclos se
    # alargan (LR_T_MUL=2.0) y el pico decae levemente en cada reinicio
    # (LR_M_MUL=0.85) para converger hacia el final del entrenamiento.
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LR_INITIAL,
        first_decay_steps=max(1, steps_per_epoch * LR_RESTART_PERIOD_EPOCHS),
        t_mul=LR_T_MUL,
        m_mul=LR_M_MUL,
        alpha=LR_ALPHA,
    )
    # V3: clipnorm=GRAD_CLIPNORM recorta la norma global del gradiente antes
    # de cada update -> red de seguridad contra explosiones de peso justo
    # tras un reinicio de LR (ver sección 9.2 del docstring del módulo).
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=GRAD_CLIPNORM
    )

    # V3 — CURRICULUM LEARNING: gamma deja de ser el float estático de la V2
    # y pasa a ser un keras.Variable NO entrenable (no es un peso del
    # modelo: es un hiperparámetro mutable). DynamicGammaCallback la
    # reescribe en `on_epoch_begin`; como DirectionalHuberLoss.call() lee
    # `self.gamma` en cada forward pass, el nuevo valor se refleja solo,
    # sin recompilar el modelo. Ver sección 9.1 del docstring del módulo.
    gamma_variable = keras.Variable(
        gamma_initial, trainable=False, dtype="float32", name="directional_gamma"
    )
    model.compile(
        optimizer=optimizer,
        loss=DirectionalHuberLoss(delta=huber_delta, gamma=gamma_variable),
        metrics=["mae", directional_accuracy_metric],
    )
    # NO se guarda `gamma_variable` como atributo de `model` (ej.
    # `model.gamma_variable = ...`): keras.Model rastrea automáticamente
    # cualquier keras.Variable asignada como atributo directo de un Layer/
    # Model (igual que trackea pesos), lo que la suma al conteo de
    # variables del Functional top-level. Al guardar el checkpoint esa
    # variable extra queda serializada, pero al reconstruir el modelo desde
    # su config en `keras.models.load_model()` el grafo Functional
    # reconstruido NO la vuelve a crear (no es parte de la arquitectura) ->
    # desalineación "expected 0 variables, but received 1" y el reload
    # revienta. Se devuelve por separado para que la comparta explícitamente
    # con `train_model()` sin pasar por la maquinaria de tracking de Keras.
    return model, gamma_variable


# ============================================================
# FASE C: ENTRENAMIENTO GLOBAL
# ============================================================
def train_model(model: keras.Model, X_train, asset_id_train, y_train,
                 epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
                 validation_split: float = VALIDATION_SPLIT,
                 checkpoint_path: str = "attention_bilstm_global_best.keras",
                 gamma_variable=None,
                 gamma_max: float = GAMMA_MAX,
                 gamma_warmup_epochs: int = GAMMA_WARMUP_EPOCHS,
                 gamma_schedule: str = GAMMA_SCHEDULE,
                 gamma_sigmoid_steepness: float = GAMMA_SIGMOID_STEEPNESS):
    """
    Sin EarlyStopping: CosineDecayRestarts ya fuerza reinicios periódicos del
    LR (el mecanismo que normalmente cubriría un EarlyStopping por plateau),
    así que se deja correr el ciclo completo de `epochs`. ModelCheckpoint
    sigue siendo indispensable — sin él, el modelo resultante sería el de la
    ÚLTIMA época, que justo después de un reinicio de LR puede tener peor
    val_loss que un punto intermedio. Tras fit(), se recarga el checkpoint de
    mejor val_loss, replicando el efecto de `restore_best_weights=True`.
    TerminateOnNaN() es una red de seguridad barata: un schedule más agresivo
    (picos de LR más altos en cada reinicio) tiene más riesgo de un salto que
    haga explotar el loss en algún batch.

    V3 — nota sobre `gamma_variable`: si `save_best_only=True` guarda el
    checkpoint en una época donde el curriculum aún no llegó a gamma_max (un
    val_loss más bajo es "más fácil" con gamma pequeño), el `best_model`
    recargado por `keras.models.load_model()` queda con un gamma ESTÁTICO
    congelado en ese valor (ver `DirectionalHuberLoss.get_config`) — esto es
    correcto para inferencia (la loss no se usa en `predict()`) pero si se
    quisiera REANUDAR entrenamiento habría que reconstruir la Variable a mano.
    """
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
        keras.callbacks.TerminateOnNaN(),
    ]
    if gamma_variable is not None:
        # V3: registra el callback de curriculum learning SOLO si se pasó una
        # gamma_variable (permite seguir usando train_model en modo V2 puro,
        # con una loss de gamma estático, sin romper la firma de la función).
        callbacks.append(DynamicGammaCallback(
            gamma_variable=gamma_variable,
            gamma_max=gamma_max,
            warmup_epochs=gamma_warmup_epochs,
            schedule=gamma_schedule,
            sigmoid_steepness=gamma_sigmoid_steepness,
        ))
    history = model.fit(
        [X_train, asset_id_train], y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )
    best_model = keras.models.load_model(checkpoint_path)
    return history, best_model


# ============================================================
# FASE D: EVALUACIÓN — RECONSTRUCCIÓN DE PRECIOS + DIRECTIONAL ACCURACY
# ============================================================
def evaluate_asset(model: keras.Model, test: dict, ticker: str, target_scaler: StandardScaler) -> dict:
    asset_id_col = test["asset_id"].reshape(-1, 1)
    y_pred_scaled = model.predict([test["X"], asset_id_col], verbose=0)

    r_hat = target_scaler.inverse_transform(y_pred_scaled).flatten()
    r_true = target_scaler.inverse_transform(test["y"]).flatten()

    price_pred = test["last_price"] * np.exp(r_hat)
    price_true = test["last_price"] * np.exp(r_true)

    mae_price = mean_absolute_error(price_true, price_pred)
    rmse_price = np.sqrt(mean_squared_error(price_true, price_pred))
    directional_acc = float(np.mean(np.sign(r_hat) == np.sign(r_true)))

    return {
        "ticker": ticker,
        "mae_price": mae_price,
        "rmse_price": rmse_price,
        "directional_accuracy": directional_acc,
        "price_true": price_true,
        "price_pred": price_pred,
        "dates": test["dates"],
    }


def plot_asset(result: dict, ticker: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(result["dates"], result["price_true"], label="Real", color="#1f77b4", linewidth=1.8)
    ax.plot(result["dates"], result["price_pred"], label="Predicho (reconstruido)", color="#d62728",
            linewidth=1.5, linestyle="--")
    ax.set_title(f"{ticker} — Global Attention-BiLSTM V3 (Log-Return): Predicho vs. Real (Test)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio de Cierre (USD)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"forecast_{ticker.replace('=', '_').replace('^', '')}.png", dpi=150)
    plt.show()


# ============================================================
# ORQUESTACIÓN (nivel de módulo -> variables quedan en scope global)
# ============================================================
print(f"[1/5] Descargando {len(TICKERS)} activos (OHLCV) + {len(MACRO_TICKERS)} factores macro (Close) ({PERIOD})...")
all_market_data = download_all(TICKERS, MACRO_TICKERS, PERIOD)

print("[2/5] Ingeniería de features y construcción de dataset GLOBAL por activo...")
feature_scalers, target_scalers = {}, {}
X_train_parts, y_train_parts, asset_id_train_parts = [], [], []
test_sets = {}

for ticker in TICKERS:
    df_asset = engineer_asset(all_market_data, ticker, MACRO_TICKERS)
    train, test, f_scaler, t_scaler = build_asset_dataset(df_asset, ticker, LOOKBACK, TRAIN_RATIO)

    feature_scalers[ticker] = f_scaler
    target_scalers[ticker] = t_scaler
    test_sets[ticker] = test

    X_train_parts.append(train["X"])
    y_train_parts.append(train["y"])
    asset_id_train_parts.append(train["asset_id"])
    print(f"      {ticker:10s} train={len(train['X']):5d}  test={len(test['X']):5d}")

X_train = np.concatenate(X_train_parts, axis=0)
y_train = np.concatenate(y_train_parts, axis=0)
asset_id_train = np.concatenate(asset_id_train_parts, axis=0).reshape(-1, 1)
print(f"      Dataset global combinado -> X_train {X_train.shape}")

print("[3/5] Construyendo y entrenando el modelo GLOBAL...")
# Verificación de dimensionalidad ANTES de instanciar el modelo: si
# TECH_COLS y N_FEATURES se desalinearan del tensor real, mejor un
# AssertionError explícito acá que un colapso de shape dentro de la BiLSTM.
assert X_train.shape[-1] == N_FEATURES, (
    f"Desalineación de features: X_train tiene {X_train.shape[-1]} columnas "
    f"pero N_FEATURES={N_FEATURES} (TECH_COLS={TECH_COLS}). Revisa "
    "engineer_asset()/build_asset_dataset()."
)
print(f"      n_features dinámico = {N_FEATURES} (PRICE + {len(TECH_COLS)} técnicos incl. ATR/BB/OBV + {len(MACRO_TICKERS)} macro)")

# steps_per_epoch estimado sobre el tramo EFECTIVO de entrenamiento (90% de
# X_train, ya que model.fit reserva VALIDATION_SPLIT=10% para validación) —
# necesario para calibrar `first_decay_steps` de CosineDecayRestarts en
# términos de ÉPOCAS y no de steps crudos.
n_train_effective = int(X_train.shape[0] * (1 - VALIDATION_SPLIT))
steps_per_epoch = max(1, n_train_effective // BATCH_SIZE)

model, gamma_variable = build_model(
    n_timesteps=LOOKBACK, n_features=N_FEATURES, n_assets=len(TICKERS),
    mha_heads=MHA_HEADS, mha_key_dim=MHA_KEY_DIM, steps_per_epoch=steps_per_epoch,
    huber_delta=HUBER_DELTA, gamma_initial=GAMMA_INITIAL,
    dropout_rate=DROPOUT_RATE, dense_l2_reg=DENSE_L2_REG,
)
model.summary()
print(f"      [V3] Curriculum gamma: {GAMMA_INITIAL} -> {GAMMA_MAX} en {GAMMA_WARMUP_EPOCHS} "
      f"épocas (schedule={GAMMA_SCHEDULE})  |  clipnorm={GRAD_CLIPNORM}  |  "
      f"LR_RESTART_PERIOD_EPOCHS={LR_RESTART_PERIOD_EPOCHS}")
history, model = train_model(
    model, X_train, asset_id_train, y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
    gamma_variable=gamma_variable, gamma_max=GAMMA_MAX,
    gamma_warmup_epochs=GAMMA_WARMUP_EPOCHS, gamma_schedule=GAMMA_SCHEDULE,
    gamma_sigmoid_steepness=GAMMA_SIGMOID_STEEPNESS,
)

print("[4/5] Evaluación por activo (reconstrucción de precio + directional accuracy)...")
results = {}
for ticker in TICKERS:
    results[ticker] = evaluate_asset(model, test_sets[ticker], ticker, target_scalers[ticker])
    r = results[ticker]
    print(f"      {ticker:10s} MAE=${r['mae_price']:.2f}  RMSE=${r['rmse_price']:.2f}  "
          f"DirAcc={r['directional_accuracy'] * 100:.1f}%")

plot_asset(results["AAPL"], "AAPL")  # muestra representativa; repetir con cualquier ticker de TICKERS

print("[5/5] Exportando modelo global + scalers -> ENRUTAMIENTO DE PRODUCCIÓN (/services)...")
# Rutas FIJAS (nombre de archivo estable, sin timestamp/versión) para que cada
# corrida del pipeline SOBREESCRIBA el artefacto anterior en vez de generar
# duplicados tipo "scalers 1.pkl".
model_path = PROD_DIR / "attention_bilstm_global.keras"
scalers_path = PROD_DIR / "scalers.pkl"

model.save(model_path, overwrite=True)  # overwrite explícito: en un runner de CI no hay prompt interactivo que confirmar

scalers_payload = {
    "feature_scalers": feature_scalers,
    "target_scalers": target_scalers,
    "asset_to_id": ASSET_TO_ID,
    "lookback": LOOKBACK,
    "macro_tickers": MACRO_TICKERS,
    "tickers": TICKERS,
    "tech_cols": TECH_COLS,  # incluye ATR_14/BB_WIDTH_20/OBV -> permite validar orden/dimensión en producción
}
with open(scalers_path, "wb") as f:  # modo "wb" trunca y reescribe el archivo existente, no lo duplica
    pickle.dump(scalers_payload, f)

print(f"      Modelo  -> {model_path.relative_to(BASE_DIR.parent)}")
print(f"      Scalers -> {scalers_path.relative_to(BASE_DIR.parent)}")
print("      Archivos sobreescritos exitosamente en /services")

# Variables disponibles en el scope global tras la ejecución:
# model, history, feature_scalers, target_scalers, test_sets, results,
# X_train, y_train, asset_id_train
# %%