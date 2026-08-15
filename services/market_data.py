# services/market_data.py
"""
Proveedor de datos de mercado — KodaQuant Terminal V5
=======================================================
Reemplaza por completo `yfinance` + el pool de proxies gratuitos
(bloqueos recurrentes de Yahoo Finance / errores "Proxy CONNECT aborted"
en producción) por:

  1. **Twelve Data** (https://twelvedata.com) — API REST oficial, HTTPS
     puro, sin scraping ni impersonation TLS. Fuente PRIMARIA para el
     universo cerrado y verificado de tickers que el motor realmente
     opera (`REGIME_TICKERS` ∪ `PLAN_A_TICKER` en quanti_engine.py:
     AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META, SPY, BTC-USD, ETH-USD).
     Plan Basic (gratuito, sin tarjeta): 800 créditos/día, 8 requests/min,
     hasta 5.000 puntos de datos por request — verificado contra la
     documentación pública de Twelve Data (agosto 2026). Registro:
     https://twelvedata.com/pricing → plan "Basic" → Dashboard → API Keys.

  2. **Stooq** (https://stooq.com, CSV público, sin key) como fallback de
     ÚLTIMO RECURSO si Twelve Data no responde (rate limit agotado, caída
     puntual, símbolo momentáneamente indisponible). Mismo rol que ya
     cumplía en quanti_engine.py antes de esta migración — solo entrega
     Close diario, así que bajo este fallback ATR_14/OBV_ROC_20/ADX_14/
     STOCH_K_14 quedan temporalmente degradados (Open/High/Low se
     replican desde Close, Volume queda en NaN→0) hasta que Twelve Data
     vuelva a responder. Nunca se usa como fuente silenciosa: cada vez
     que se activa, queda logueado con `logger.warning`.

CACHÉ EN DISCO (JSON, TTL configurable vía `KODAQUANT_DATA_CACHE_TTL_SECONDS`,
default 6h) + RATE LIMITER propio (ventana deslizante, 7 req/min — margen
de seguridad bajo el límite real de 8/min del plan Basic). El universo
completo escaneado por `services/prediccion.py` (10 tickers operables +
los macro tickers de `scalers["macro_tickers"]`, ~14-15 símbolos en
total) jamás se acerca al tope diario de 800 créditos incluso bajo
tráfico real, porque las barras diarias no cambian más de una vez por
sesión de mercado — no hay motivo para re-descargar 2 años de historia
en cada request de un usuario.

--- AUDITORÍA DE PARIDAD TRAIN/SERVE (contra train_kodaquant_v5.py) ---
Reconciliado línea a línea contra el script de entrenamiento real. Hallazgos:

  • UNIVERSO DE TICKERS: `REGIME_TICKERS` en train_kodaquant_v5.py
    (equity_specialist: AAPL/MSFT/NVDA/TSLA/GOOGL/AMZN/META/SPY;
    crypto_specialist: BTC-USD/ETH-USD) coincide EXACTO con el universo
    que ya cubren `_TICKER_MAP_TD`/`_STOOQ_SYMBOL_MAP` — sin cambios.

  • MACRO_TICKERS: `MACRO_TICKERS = ["^GSPC", "^TNX", "^VIX", "GC=F",
    "DX-Y.NYB"]` (train_kodaquant_v5.py, línea 149) — exactamente los 5
    tickers que ya cubría `MACRO_SYMBOL_MAP` (abajo). Sin faltantes, sin
    sobrantes.

  • TODOS los macro tickers entran al tensor ÚNICAMENTE como LOG-RETURN
    diario (`engineer_asset()`: `df[macro_close_col] = np.log(macro_price
    / macro_price.shift(1))`, aplicado ANTES del feature_scaler, idéntico
    en `_fetch_feature_window` del lado de inferencia). Consecuencia
    matemática directa: `log(k·x_t / k·x_{t-1}) = log(x_t / x_{t-1})` —
    un factor de escala CONSTANTE y multiplicativo en cualquier macro
    ticker (ej. un proveedor que reporte el mismo activo ×10 o ×100
    frente a otro) es matemáticamente INVISIBLE para el modelo. Esto no
    exime de verificar la escala (una serie con escala INCONSISTENTE en
    el tiempo, o con timestamps desalineados, sí correspería fuga/ruido
    real) — pero sí acota el riesgo real de un desajuste de escala
    puramente constante a cero impacto en el tensor de entrada.

  • ^TNX — HALLAZGO QUE CORRIGE UNA ADVERTENCIA PREVIA DE ESTE MÓDULO:
    una versión anterior de este archivo asumía que yfinance publicaba
    ^TNX bajo la vieja convención CBOE "rendimiento ×10" (ej. 42.0 =
    4.20%). Verificado en vivo contra la cotización real de Yahoo Finance
    para ^TNX (agosto 2026): el valor publicado es 4.5580 — es decir,
    yfinance/Yahoo publican ^TNX en PORCENTAJE DIRECTO (4.558 = 4.558%),
    NO en la vieja convención ×10. Stooq (`10yusy.b`, "U.S. 10-Year
    Government Bond Yield") publica la misma convención de porcentaje
    directo. CONCLUSIÓN: no existe el desajuste de escala ×10 que se
    sospechaba — la advertencia anterior era incorrecta y fue retirada.
    No se aplica ninguna transformación compensatoria (no hay nada que
    compensar), y por el punto anterior (macro = solo log-return) aunque
    lo hubiera habido, habría sido matemáticamente inerte para el tensor.

  • auto_adjust=True — confirmado en train_kodaquant_v5.py línea 359
    (`yf.download(symbols, period=period, auto_adjust=True, ...)`),
    aplicado de forma UNIFORME a tickers Y macro. Para los 5 macro
    tickers (todos índices/futuros, sin dividendos ni splits) esto es un
    no-op exacto — auto_adjust=True/False no cambia un solo valor. Para
    los tickers de equity (AAPL, MSFT, ...) SÍ es relevante: ver el
    caveat ya documentado en quanti_engine.py (`_fetch_feature_window`,
    bloque "FIX PRICING") sobre la diferencia de metodología de ajuste
    por dividendos entre Yahoo y Twelve Data — no es un bug de este
    módulo, es una diferencia real entre proveedores que ninguna
    transformación en `market_data.py` puede eliminar sin una fuente de
    dividendos propia.

  • GC=F vs XAU/USD — riesgo real, NO corregible con un factor fijo.
    GC=F es el futuro de oro COMEX del contrato más próximo; Twelve
    Data's XAU/USD es el spot. Verificado en vivo (agosto 2026): ambos
    cotizan en el mismo orden de magnitud (spot ~$4.320-4.410 vs futuro
    ~$4.408-4.430) pero con una base (contango/backwardation) variable
    en el tiempo — no una proporción constante, así que NO se puede
    "corregir" con un multiplicador fijo sin inventar un número. Se deja
    como proxy de mejor esfuerzo, documentado explícitamente en
    `MACRO_SYMBOL_MAP` — igual que arriba, al entrar como log-return el
    impacto se limita a la diferencia de VOLATILIDAD diaria entre spot y
    futuro (correlacionadas >0.98 en la práctica), no a un sesgo de nivel.

  • ^GSPC↔SPX, ^VIX↔VIX, DX-Y.NYB↔DXY — mismo índice subyacente en
    ambos proveedores (S&P 500 cash, CBOE VIX, ICE US Dollar Index
    respectivamente), sin dualidad de convención conocida. `DX-Y.NYB`
    verificado en vivo (~99.94) contra el nivel públicamente reportado
    de DXY (~99-101) — mismo orden de magnitud, consistente.

--- ADVERTENCIA VIGENTE — LEER ANTES DE DESPLEGAR A PRODUCCIÓN ---
Esta auditoría se hizo contra el CÓDIGO de entrenamiento (train_kodaquant_v5.py)
y contra cotizaciones en vivo de terceros (Yahoo, Investing.com, Barchart,
TradingView) — no contra tu `scalers_dict.pkl` real ni contra una llamada
en vivo a la API de Twelve Data (este entorno no tiene salida de red hacia
api.twelvedata.com/stooq.com). Si `MACRO_TICKERS` en tu copia real de
train_kodaquant_v5.py difiere de la verificada acá, o si tu plan de Twelve
Data no cubre alguno de estos símbolos, corregí `MACRO_SYMBOL_MAP` antes
de servir tráfico real — `resolve_macro_symbol()` sigue fallando ruidoso
ante cualquier ticker fuera del diccionario, nunca adivina un símbolo.

--- ROOT CAUSE REAL DE "degradación total del radar" (FIX 2026-08-15) ---
El fix del 2026-08-15 de arriba (^GSPC/^VIX/DX-Y.NYB/^TNX directo a Stooq)
resolvía el 404 de Twelve Data pero destapó el bug real, más grave:
`services/prediccion._scan_universe` despacha los 10 tickers de
REGIME_TICKERS en PARALELO real (`asyncio.gather` sobre 10
`run_in_executor`, ver prediccion.py) y CADA uno de esos 10 hilos llama a
`fetch_feature_ohlcv(ticker, macro_tickers, ...)`, que a su vez pide los
MISMOS 5 macro tickers compartidos por todo el universo. Sin coordinación
entre hilos, eso dispara hasta 10 requests SIMULTÁNEAS por cada uno de los
4 macro tickers que van directo a Stooq (~40 requests en una ráfaga de
milisegundos) contra un endpoint CSV público, anónimo, sin key y sin rate
limit documentado del lado servidor — el patrón clásico que Stooq
responde bloqueando/sirviendo una página de error en vez de CSV real
(`_stooq_daily_close` ya detectaba ese caso devolviendo `None`, pero
`_fetch_symbol_ohlcv` no tenía ningún colchón: en cuanto Stooq fallaba
para el símbolo, lanzaba `RuntimeError` de inmediato). Como ^GSPC (y el
resto de los macro tickers) es un input COMPARTIDO por los 10 modelos, un
solo bloqueo de Stooq tumba el forecast de LOS 10 tickers a la vez —
exactamente "degradación total del radar", no un fallo aislado.

Tres capas de fix, todas en este módulo (`_fetch_symbol_ohlcv` +
`_stooq_daily_close`), cero cambios de contrato hacia quanti_engine.py/
prediccion.py:
  1. SINGLE-FLIGHT por símbolo (`_lock_for` + `threading.Lock`): con 10
     hilos pidiendo '^GSPC' a la vez, solo el PRIMERO golpea la red: los
     otros 9 esperan el lock y reusan lo que ese hilo dejó en caché.
     Colapsa la ráfaga de ~10 requests por macro ticker a 1 sola.
  2. THROTTLE + REINTENTOS propios para Stooq (`_STOOQ_CONCURRENCY` +
     `_stooq_throttle` + backoff en `_stooq_daily_close`) — antes Stooq
     no tenía NINGÚN control del lado cliente (a diferencia de Twelve
     Data, que sí tenía semáforo + rate limiter); un solo intento fallido
     (glitch transitorio, no necesariamente bloqueo) mataba el símbolo
     entero sin reintentar.
  3. CACHÉ STALE COMO ÚLTIMO RECURSO (`_cache_get(..., ignore_ttl=True)`
     en `_fetch_symbol_ohlcv`): si Twelve Data Y Stooq fallan los DOS en
     vivo, antes de lanzar `RuntimeError` (y tumbar el universo completo)
     se busca la última descarga exitosa en disco sin importar el TTL de
     6h — para un macro factor que entra al tensor como log-return diario,
     servir el dato de ayer en vez de "nada" es estrictamente mejor y
     evita la cascada total. Solo se lanza `RuntimeError` si JAMÁS hubo
     una descarga exitosa para ese símbolo (arranque en frío + ambos
     proveedores caídos a la vez, caso genuinamente irrecuperable).
"""

import io
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("kodaquant.market_data")

# ---------------------------------------------------------------------------
# Config / credenciales
# ---------------------------------------------------------------------------
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "").strip()
TWELVE_DATA_BASE_URL = "https://api.twelvedata.com"
TWELVE_DATA_TIMEOUT_S = 20.0

# ~2 años de sesiones diarias (mismo horizonte que el "period=2y" que usaba
# yfinance) — suficiente burn-in para que RSI_14/EMA_20/MACD/MACD_SIGNAL
# (todos IIR recursivos, `ewm(adjust=False)`) converjan al mismo estado que
# vio el notebook de entrenamiento. +16 sesiones de margen sobre las ~504
# de 2 años calendario, para no quedar corto en años con más feriados.
DEFAULT_HISTORY_SESSIONS = 520

if not TWELVE_DATA_API_KEY:
    logger.warning(
        "TWELVE_DATA_API_KEY no configurada -- TODA descarga de mercado va "
        "a caer directo al fallback Stooq (cobertura reducida: solo Close "
        "diario real; ATR_14/OBV_ROC_20/ADX_14/STOCH_K_14 quedarán "
        "degradados hasta que se configure la key). Registrate gratis en "
        "https://twelvedata.com/pricing (plan Basic) y seteá la variable "
        "de entorno / Secret 'TWELVE_DATA_API_KEY'."
    )

# ---------------------------------------------------------------------------
# Rate limiter -- ventana deslizante de 60s, 7 requests (margen de
# seguridad bajo el límite real de 8/min del plan Basic gratuito).
# ---------------------------------------------------------------------------
_RATE_LOCK = threading.Lock()
_RATE_WINDOW_S = 60.0
_RATE_MAX_CALLS = 7
_rate_call_log: list[float] = []

# Concurrencia real hacia Twelve Data -- evita disparar 10-15 requests
# simultáneas cuando el radar escanea todo REGIME_TICKERS en paralelo
# (mismo rol que _YF_CONCURRENCY cumplía bajo yfinance).
_TD_CONCURRENCY = threading.BoundedSemaphore(4)

# FIX 2026-08-15 -- Stooq (fallback CSV público, sin key) NO tenía ningún
# control de concurrencia propio, a diferencia de Twelve Data arriba. Bajo
# un scan paralelo de las 10 tickers de REGIME_TICKERS, eso permitía
# ráfagas de hasta ~40 requests simultáneas contra un endpoint anónimo sin
# rate limit documentado -- el patrón real que dispara los bloqueos/CSV
# vacíos observados en producción (ver ROOT CAUSE en el docstring del
# módulo). El single-flight lock (`_lock_for`, más abajo) ya colapsa la
# mayor parte de esa ráfaga, pero este semáforo + throttle es la segunda
# capa de defensa para cualquier request que sí llegue a golpear la red.
_STOOQ_CONCURRENCY = threading.BoundedSemaphore(2)
_STOOQ_RATE_LOCK = threading.Lock()
_STOOQ_RATE_WINDOW_S = 10.0
_STOOQ_RATE_MAX_CALLS = 3
_stooq_call_log: list[float] = []


def _stooq_throttle() -> None:
    with _STOOQ_RATE_LOCK:
        now = time.monotonic()
        while _stooq_call_log and now - _stooq_call_log[0] > _STOOQ_RATE_WINDOW_S:
            _stooq_call_log.pop(0)
        if len(_stooq_call_log) >= _STOOQ_RATE_MAX_CALLS:
            wait_s = _STOOQ_RATE_WINDOW_S - (now - _stooq_call_log[0]) + 0.05
            if wait_s > 0:
                logger.debug("Rate limiter Stooq: esperando %.2fs.", wait_s)
                time.sleep(wait_s)
        _stooq_call_log.append(time.monotonic())


# ---------------------------------------------------------------------------
# Single-flight por símbolo -- ver ROOT CAUSE en el docstring del módulo.
# Un `threading.Lock` dedicado por `internal_key` (creado on-demand, nunca
# liberado -- son ~15 símbolos como mucho, memoria irrelevante) asegura que
# de los N hilos que puedan pedir el MISMO ticker en paralelo (ej. '^GSPC'
# pedido por los 10 forecasts del universo a la vez), solo UNO llega a
# tocar la red; el resto espera el lock y reusa lo que ese hilo dejó en
# caché -- sin este mecanismo, cada hilo repetía la descarga completa.
# ---------------------------------------------------------------------------
_SYMBOL_LOCKS: dict[str, threading.Lock] = {}
_SYMBOL_LOCKS_META = threading.Lock()


def _lock_for(internal_key: str) -> threading.Lock:
    with _SYMBOL_LOCKS_META:
        lock = _SYMBOL_LOCKS.get(internal_key)
        if lock is None:
            lock = threading.Lock()
            _SYMBOL_LOCKS[internal_key] = lock
        return lock


def _throttle() -> None:
    with _RATE_LOCK:
        now = time.monotonic()
        while _rate_call_log and now - _rate_call_log[0] > _RATE_WINDOW_S:
            _rate_call_log.pop(0)
        if len(_rate_call_log) >= _RATE_MAX_CALLS:
            wait_s = _RATE_WINDOW_S - (now - _rate_call_log[0]) + 0.05
            if wait_s > 0:
                logger.debug("Rate limiter Twelve Data: esperando %.2fs.", wait_s)
                time.sleep(wait_s)
        _rate_call_log.append(time.monotonic())


# ---------------------------------------------------------------------------
# Caché en disco (JSON, TTL) -- clave = ticker INTERNO (AAPL, BTC-USD,
# ^VIX, ...), nunca el símbolo externo, así todo el motor comparte una
# única entrada de caché por activo sin importar qué proveedor lo sirvió.
# ---------------------------------------------------------------------------
_CACHE_DIR = Path(os.environ.get("KODAQUANT_DATA_CACHE_DIR", "/tmp/kodaquant_market_cache"))
_CACHE_TTL_S = int(os.environ.get("KODAQUANT_DATA_CACHE_TTL_SECONDS", str(6 * 3600)))
_CACHE_LOCK = threading.Lock()


def _cache_path(cache_key: str) -> Path:
    safe = cache_key.replace("/", "_").replace(":", "_").replace("^", "caret_")
    return _CACHE_DIR / f"{safe}.json"


def _cache_get(cache_key: str, ignore_ttl: bool = False) -> Optional[pd.DataFrame]:
    """
    `ignore_ttl=True` -- FIX 2026-08-15, ver ROOT CAUSE en el docstring del
    módulo: devuelve lo último cacheado en disco SIN importar si el TTL de
    6h ya venció. Uso exclusivo del fallback de última instancia en
    `_fetch_symbol_ohlcv` cuando Twelve Data Y Stooq fallan los dos en
    vivo -- para un macro factor que entra al tensor como log-return
    diario, un dato de ayer sigue siendo estrictamente mejor que tumbar el
    universo entero. El camino normal (`ignore_ttl=False`, default) NO
    cambia de comportamiento.
    """
    path = _cache_path(cache_key)
    try:
        with _CACHE_LOCK:
            if not path.exists():
                return None
            payload = json.loads(path.read_text())
        age_s = time.time() - payload.get("_cached_at", 0)
        if not ignore_ttl and age_s > _CACHE_TTL_S:
            return None
        df = pd.read_json(io.StringIO(payload["data"]), orient="split")
        df.index = pd.to_datetime(df.index)
        if ignore_ttl and age_s > _CACHE_TTL_S:
            df.attrs["_stale_age_hours"] = round(age_s / 3600, 1)
        return df
    except Exception as exc:  # noqa: BLE001 -- caché corrupta/ilegible jamás tumba el fetch real
        logger.debug("Caché ilegible para %s (%r) -- se ignora y se re-descarga.", cache_key, exc)
        return None


def _cache_set(cache_key: str, df: pd.DataFrame) -> None:
    path = _cache_path(cache_key)
    payload = {"data": df.to_json(orient="split", date_format="iso"), "_cached_at": time.time()}
    try:
        with _CACHE_LOCK:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
    except Exception as exc:  # noqa: BLE001 -- falla de disco (fs read-only, etc.) jamás tumba el boot
        logger.debug("No se pudo escribir caché para %s (%r).", cache_key, exc)


# ---------------------------------------------------------------------------
# Mapeo de símbolos
# ---------------------------------------------------------------------------

# Universo cerrado y VERIFICADO (equities + crypto) -- idéntico a
# REGIME_TICKERS/PLAN_A_TICKER en quanti_engine.py. Formato Twelve Data
# para cripto es "BASE/QUOTE" (ver docs.twelvedata.com); equities usan el
# mismo ticker que Yahoo/NYSE/NASDAQ 1:1, sin traducción.
_TICKER_MAP_TD: dict[str, str] = {
    "BTC-USD": "BTC/USD",
    "ETH-USD": "ETH/USD",
}

# Fallback Stooq para el mismo universo -- reutiliza el mapeo que ya vivía
# en quanti_engine.py (_STOOQ_SYMBOL_MAP) antes de esta migración.
_STOOQ_SYMBOL_MAP: dict[str, str] = {
    "AAPL": "aapl.us", "MSFT": "msft.us", "NVDA": "nvda.us", "TSLA": "tsla.us",
    "GOOGL": "googl.us", "AMZN": "amzn.us", "META": "meta.us", "SPY": "spy.us",
    "BTC-USD": "btc.v", "ETH-USD": "eth.v",
}

# Macro tickers -- auditado contra MACRO_TICKERS de train_kodaquant_v5.py
# (línea 149: ^GSPC, ^TNX, ^VIX, GC=F, DX-Y.NYB) -- ver la sección
# "AUDITORÍA DE PARIDAD TRAIN/SERVE" en el docstring del módulo para el
# detalle matemático completo de cada fila. `twelvedata: None` significa
# "este proveedor no lo cubre de forma confiable en el plan free -- se
# sirve directo desde Stooq".
MACRO_SYMBOL_MAP: dict[str, dict[str, Optional[str]]] = {
    "^GSPC":    {"twelvedata": None,      "stooq": "^spx"},     # FIX 2026-08-15: Twelve Data devuelve
                                                                  # 404 real en producción para "SPX"
                                                                  # (time_series) -- índices cash NO
                                                                  # están cubiertos de forma confiable
                                                                  # en el plan Basic/free (mismo patrón
                                                                  # públicamente documentado para VIX,
                                                                  # ver soporte oficial TD). Se sirve
                                                                  # SIEMPRE desde Stooq (mismo patrón
                                                                  # que ya usa ^TNX abajo) -- no un
                                                                  # 404 silencioso seguido de fallback
                                                                  # lento, sino la ruta directa.
    "^VIX":     {"twelvedata": None,      "stooq": "^vix"},     # FIX 2026-08-15: mismo caso -- VIX
                                                                  # confirmado sin cobertura en TD free
                                                                  # (documentado públicamente por TD).
                                                                  # Directo a Stooq.
    "GC=F":     {"twelvedata": "XAU/USD", "stooq": "xauusd"},   # Oro SPOT -- GC=F en entrenamiento
                                                                  # es el FUTURO COMEX del contrato
                                                                  # próximo. Verificado en vivo
                                                                  # (ago-2026): mismo orden de
                                                                  # magnitud, base variable en el
                                                                  # tiempo (no una proporción fija) --
                                                                  # proxy de mejor esfuerzo, NO
                                                                  # corregible con un multiplicador
                                                                  # constante. Ver docstring del módulo.
    "DX-Y.NYB": {"twelvedata": None,      "stooq": "usdx.f"},   # FIX 2026-08-15: mismo caso -- índice
                                                                  # ICE crudo, mismo patrón de
                                                                  # no-cobertura en TD free que ^GSPC/
                                                                  # ^VIX. Directo a Stooq.
    "^TNX":     {"twelvedata": None,      "stooq": "10yusy.b"}, # Rendimiento UST 10Y, PORCENTAJE
                                                                  # DIRECTO (4.558 = 4.558%) en AMBOS
                                                                  # yfinance/Yahoo y Stooq -- verificado
                                                                  # en vivo, ago-2026. CORRIGE una
                                                                  # advertencia previa de este módulo
                                                                  # que asumía, incorrectamente, la
                                                                  # vieja convención CBOE "rendimiento
                                                                  # ×10" -- esa convención NO aplica acá,
                                                                  # sin transformación de escala. No
                                                                  # cubierto por Twelve Data free -- se
                                                                  # sirve siempre desde Stooq.
}


def _resolve_td_symbol(internal_ticker: str) -> Optional[str]:
    return _TICKER_MAP_TD.get(internal_ticker, internal_ticker)


def _resolve_stooq_symbol(internal_ticker: str) -> Optional[str]:
    return _STOOQ_SYMBOL_MAP.get(internal_ticker.strip().upper())


def resolve_macro_symbol(macro_ticker: str) -> dict[str, Optional[str]]:
    """
    Fail-loud por diseño: un macro ticker sin entrada acá NUNCA se sirve
    con un símbolo adivinado. Agregalo a MACRO_SYMBOL_MAP (arriba) con el
    símbolo Twelve Data y/o Stooq correcto antes de desplegar.
    """
    entry = MACRO_SYMBOL_MAP.get(macro_ticker)
    if entry is None:
        raise ValueError(
            f"'{macro_ticker}' no tiene mapeo registrado en MACRO_SYMBOL_MAP "
            "(services/market_data.py). Completá esa entrada (símbolo "
            "Twelve Data y/o Stooq real) antes de servir tráfico -- nunca "
            "se adivina un símbolo macro (misma regla 'cero cifras "
            "inventadas' del resto del motor)."
        )
    return entry


# ---------------------------------------------------------------------------
# Twelve Data -- cliente HTTP crudo
# ---------------------------------------------------------------------------

def _td_time_series(td_symbol: str, outputsize: int, attempts: int = 3) -> Optional[pd.DataFrame]:
    """
    Una llamada a `/time_series` (1 crédito) para UN símbolo, `interval=1day`.
    Reintenta ante 429/5xx/timeout con backoff simple; nunca reintenta ante
    un símbolo inválido (400 "symbol not found") -- eso es un error de
    mapeo, no algo que un backoff pueda arreglar.
    """
    if not TWELVE_DATA_API_KEY:
        return None

    params = {
        "symbol": td_symbol,
        "interval": "1day",
        "outputsize": str(outputsize),
        "order": "ASC",
        "timezone": "UTC",
        "apikey": TWELVE_DATA_API_KEY,
    }

    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        with _TD_CONCURRENCY:
            _throttle()
            try:
                resp = requests.get(
                    f"{TWELVE_DATA_BASE_URL}/time_series", params=params, timeout=TWELVE_DATA_TIMEOUT_S
                )
                if resp.status_code == 429:
                    raise RuntimeError(f"Twelve Data rate limit (429): {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and data.get("status") == "error":
                    code = data.get("code")
                    message = data.get("message", "")
                    if code == 400:  # símbolo inválido / no encontrado -- no reintentar
                        logger.warning("Twelve Data: símbolo inválido '%s' (%s).", td_symbol, message)
                        return None
                    raise RuntimeError(f"Twelve Data error {code}: {message}")
                return _td_payload_to_df(data)
            except Exception as exc:  # noqa: BLE001 -- reintentamos cualquier fallo transitorio
                last_exc = exc
                if attempt < attempts - 1:
                    time.sleep(1.5 * (attempt + 1))

    logger.warning("Twelve Data agotó reintentos para '%s': %r", td_symbol, last_exc)
    return None


def _td_payload_to_df(payload: dict) -> Optional[pd.DataFrame]:
    values = payload.get("values") if isinstance(payload, dict) else None
    if not values:
        return None
    df = pd.DataFrame(values)
    if "datetime" not in df.columns:
        return None
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    return df if not df.empty else None


# ---------------------------------------------------------------------------
# Stooq -- fallback CSV sin key (mismo endpoint que ya usaba quanti_engine.py)
# ---------------------------------------------------------------------------

def _stooq_daily_close(stooq_symbol: str, tail_days: int = 400, attempts: int = 3) -> Optional[pd.Series]:
    """
    FIX 2026-08-15 (ver ROOT CAUSE en el docstring del módulo): antes esto
    era un único intento sin throttle ni semáforo -- bajo ráfaga paralela
    (10 tickers x hasta 4 macro tickers directo a Stooq) un solo glitch
    transitorio o un bloqueo momentáneo del endpoint anónimo mataba el
    símbolo entero de inmediato. Ahora reintenta con backoff (glitches
    transitorios: timeout, 5xx, CSV vacío que Stooq a veces sirve bajo
    carga) Y respeta `_STOOQ_CONCURRENCY`/`_stooq_throttle` para no ser,
    en sí mismo, la causa de la próxima ráfaga.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        with _STOOQ_CONCURRENCY:
            _stooq_throttle()
            try:
                resp = requests.get(
                    "https://stooq.com/q/d/l/",
                    # FIX 2026-08-15 (previo): antes se interpolaba
                    # stooq_symbol crudo en el f-string de la URL -- el '^'
                    # de los símbolos de índice (^spx, ^vix) viaja sin
                    # url-encodear (%5E). `params=` deja que `requests` lo
                    # encodee correctamente; evita un CSV vacío/malformado
                    # que silenciaba el fallback justo para índices.
                    params={"s": stooq_symbol, "i": "d"},
                    timeout=10,
                    headers={"User-Agent": "Mozilla/5.0 (KodaQuant market_data fallback)"},
                )
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text))
                if df.empty or "Close" not in df.columns or "Date" not in df.columns:
                    # Stooq devuelve 200 con un cuerpo que NO es el CSV
                    # esperado cuando bloquea/limita un cliente anónimo
                    # (página de error, CSV vacío) -- se trata como fallo
                    # transitorio, reintentable, no como "símbolo inválido".
                    raise ValueError(
                        f"Stooq devolvió una respuesta sin columnas Close/Date "
                        f"para '{stooq_symbol}' (posible bloqueo/rate-limit "
                        f"anónimo bajo carga)."
                    )
                df["Date"] = pd.to_datetime(df["Date"])
                series = df.set_index("Date").sort_index()["Close"].dropna()
                if series.empty:
                    raise ValueError(f"Stooq devolvió CSV sin filas válidas para '{stooq_symbol}'.")
                return series.tail(tail_days)
            except Exception as exc:  # noqa: BLE001 -- reintentamos cualquier fallo transitorio
                last_exc = exc
                if attempt < attempts - 1:
                    time.sleep(1.5 * (attempt + 1))

    logger.warning("Stooq falló para '%s' tras %d intento(s): %r", stooq_symbol, attempts, last_exc)
    return None


# ---------------------------------------------------------------------------
# Fetch unificado por símbolo -- Twelve Data primero (con caché), Stooq
# como último recurso. Usado tanto para el ticker operable como para cada
# macro ticker.
# ---------------------------------------------------------------------------

def _fetch_symbol_ohlcv(
    internal_key: str,
    td_symbol: Optional[str],
    stooq_symbol: Optional[str],
    min_sessions: int,
) -> pd.DataFrame:
    cached = _cache_get(f"ohlcv:{internal_key}")
    if cached is not None and len(cached) >= min(min_sessions, 30):
        return cached

    # FIX 2026-08-15 -- SINGLE-FLIGHT (ver ROOT CAUSE en el docstring del
    # módulo). El universo completo (10 tickers, escaneado en paralelo por
    # `prediccion._scan_universe`) comparte los MISMOS 5 macro tickers: sin
    # este lock, hasta 10 hilos pedían '^GSPC' (u otro macro ticker) a la
    # red AL MISMO TIEMPO, disparando la ráfaga que Stooq interpreta como
    # abuso. Con el lock, solo el primer hilo en tomarlo golpea la red; el
    # resto espera y reusa lo que ese hilo deja en caché -- de ahí el
    # segundo `_cache_get` adentro (double-checked): si otro hilo ya
    # resolvió el símbolo mientras esperábamos el lock, no repetimos nada.
    with _lock_for(internal_key):
        cached = _cache_get(f"ohlcv:{internal_key}")
        if cached is not None and len(cached) >= min(min_sessions, 30):
            return cached

        df = _td_time_series(td_symbol, outputsize=DEFAULT_HISTORY_SESSIONS) if td_symbol else None

        if df is None or df.empty:
            if stooq_symbol is not None:
                series = _stooq_daily_close(stooq_symbol, tail_days=DEFAULT_HISTORY_SESSIONS + 30)
                if series is not None and not series.empty:
                    df = pd.DataFrame({"close": series})
                    df["open"] = df["close"]
                    df["high"] = df["close"]
                    df["low"] = df["close"]
                    df["volume"] = np.nan
                    logger.warning(
                        "'%s': sirviendo desde fallback Stooq -- solo Close real; "
                        "Open/High/Low replicados desde Close y Volume=NaN "
                        "(ATR_14/OBV_ROC_20/ADX_14/STOCH_K_14 quedan degradados para "
                        "este ticker hasta que Twelve Data vuelva a responder).",
                        internal_key,
                    )

            if df is None or df.empty:
                # FIX 2026-08-15 -- CACHÉ STALE COMO ÚLTIMO RECURSO, antes de
                # rendirse. Twelve Data Y Stooq fallaron los dos en vivo; en
                # vez de lanzar de inmediato (y con eso tumbar los 10
                # forecasts que dependen de este mismo macro ticker), se
                # busca la última descarga exitosa en disco sin importar si
                # el TTL de 6h venció. Solo si NUNCA hubo una descarga
                # exitosa (arranque en frío + ambos proveedores caídos a la
                # vez) se lanza el RuntimeError real.
                stale = _cache_get(f"ohlcv:{internal_key}", ignore_ttl=True)
                if stale is not None and not stale.empty:
                    stale_age_h = stale.attrs.get("_stale_age_hours")
                    logger.warning(
                        "'%s': Twelve Data y Stooq fallaron los dos en vivo -- "
                        "sirviendo caché STALE (%s hs de antigüedad) en vez de "
                        "tumbar el forecast completo. Reintentará una descarga "
                        "en vivo en la próxima llamada.",
                        internal_key, stale_age_h if stale_age_h is not None else "?",
                    )
                    return stale

                if stooq_symbol is None:
                    raise RuntimeError(
                        f"Sin datos para '{internal_key}': Twelve Data no respondió, "
                        "no hay símbolo Stooq de fallback registrado para este "
                        "ticker, y no hay caché previa (ni siquiera stale) que servir."
                    )
                raise RuntimeError(
                    f"Sin datos para '{internal_key}': Twelve Data y Stooq "
                    "fallaron ambos y no hay caché previa (ni siquiera stale) "
                    "que servir -- radar degradado para este símbolo."
                )

        _cache_set(f"ohlcv:{internal_key}", df)
        return df


# ---------------------------------------------------------------------------
# API pública -- consumida por services/quanti_engine.py
# ---------------------------------------------------------------------------

def fetch_feature_ohlcv(ticker: str, macro_tickers: list[str], min_sessions: int = DEFAULT_HISTORY_SESSIONS) -> pd.DataFrame:
    """
    Reemplazo directo del antiguo `yf.download(all_symbols, period="2y",
    auto_adjust=True, ...)` en `_fetch_feature_window`. Devuelve un
    DataFrame indexado por fecha con EXACTAMENTE el mismo layout de
    columnas que consumía ese bloque:
        out[ticker]              -- Close diario del ticker objetivo
        out[f"{ticker}_High"]    -- High diario (requerido por ATR_14)
        out[f"{ticker}_Low"]     -- Low diario (requerido por ATR_14)
        out[f"{ticker}_Volume"]  -- Volume diario (requerido por OBV_ROC_20)
        out[macro_ticker]        -- Close diario de cada factor macro
    """
    td_symbol = _resolve_td_symbol(ticker)
    stooq_symbol = _resolve_stooq_symbol(ticker)
    target_df = _fetch_symbol_ohlcv(ticker, td_symbol, stooq_symbol, min_sessions)

    out = pd.DataFrame(index=target_df.index)
    out[ticker] = target_df["close"]
    out[f"{ticker}_High"] = target_df["high"]
    out[f"{ticker}_Low"] = target_df["low"]
    out[f"{ticker}_Volume"] = target_df["volume"]

    for macro_ticker in macro_tickers:
        mapping = resolve_macro_symbol(macro_ticker)
        macro_df = _fetch_symbol_ohlcv(
            macro_ticker, mapping.get("twelvedata"), mapping.get("stooq"), min_sessions
        )
        out[macro_ticker] = macro_df["close"].reindex(out.index)

    out = out.ffill().dropna()
    if out.empty:
        raise ValueError(
            f"'{ticker}': ventana de features vacía tras alinear con los "
            "macro tickers -- histórico insuficiente o símbolos macro "
            "desalineados (fechas sin superposición real)."
        )
    return out


def fetch_close_history(internal_key: str, min_days: int) -> pd.Series:
    """
    Reemplazo directo del antiguo `yf.Ticker(symbol).history(period=...)`
    en `get_market_sentiment` -- serie de Close diario para `internal_key`
    (ticker operable O macro ticker, ambos soportados). Reutiliza la MISMA
    caché de `fetch_feature_ohlcv` cuando el símbolo ya fue descargado
    para un forecast en la misma ventana de TTL -- cero llamadas de red
    adicionales en ese caso.
    """
    if internal_key in MACRO_SYMBOL_MAP:
        mapping = resolve_macro_symbol(internal_key)
        td_symbol, stooq_symbol = mapping.get("twelvedata"), mapping.get("stooq")
    else:
        td_symbol = _resolve_td_symbol(internal_key)
        stooq_symbol = _resolve_stooq_symbol(internal_key)

    df = _fetch_symbol_ohlcv(internal_key, td_symbol, stooq_symbol, min_days)
    return df["close"].dropna()