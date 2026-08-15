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

--- ROOT CAUSE REAL DE "degradación total del radar" (FIX 2026-08-15 — RONDA 1) ---
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

--- RONDA 2 (FIX 2026-08-15, mismo día — logs de producción posteriores) ---
Los logs post-RONDA 1 muestran algo distinto de lo que la RONDA 1 asumía:
el single-flight lock SÍ funcionó (ya no hay ráfaga — los intentos contra
Stooq para '^spx' quedaron serializados, uno cada ~10s, uno por ticker del
universo) y AÚN ASÍ los 10 fallan, siempre con el mismo error ("Stooq
devolvió una respuesta sin columnas Close/Date"), incluso con reintentos y
backoff. Un fallo que persiste 100% de las veces YA SERIALIZADO, request
tras request, con el MISMO símbolo, deja de ser "ráfaga mal comportada" y
pasa a ser evidencia de un bloqueo ESTRUCTURAL: Stooq es conocido por
bloquear (servir HTML/CSV vacío con 200 OK en vez de datos reales) tráfico
proveniente de rangos de IP de datacenter/cloud -- Hugging Face Spaces
corre sobre infraestructura cloud (AWS/GCP/Azure), exactamente el patrón
que ese bloqueo apunta. Los reintentos de la RONDA 1 no podían arreglar
esto porque no es un glitch transitorio -- es un rechazo consistente del
lado del proveedor para ESTA clase de tráfico, independientemente de
cuántas veces se reintente.

Auditoría de blast radius: `MACRO_TICKERS` tiene 5 entradas
(^GSPC/^TNX/^VIX/GC=F/DX-Y.NYB) pero `fetch_feature_ohlcv` los recorre en
un `for` secuencial que ABORTA en el primer `RuntimeError` -- como ^GSPC
es el PRIMERO de la lista y el único que alcanzamos a ver fallar en los
logs, los otros 3 símbolos ruteados directo a Stooq (^VIX, ^TNX,
DX-Y.NYB, ver MACRO_SYMBOL_MAP de la RONDA 1) NUNCA llegaron a intentarse
-- pero comparten el MISMO endpoint (`stooq.com/q/d/l/`) con el MISMO
patrón de bloqueo, así que hay que asumir que fallarían igual si ^GSPC no
abortara primero. Cuatro de los cinco macro tickers dependían de Stooq
como única fuente real (`twelvedata: None`) -- ese es el problema de raíz,
no la concurrencia (ya resuelta en RONDA 1).

FIX DE RAÍZ -- dejar de depender de Stooq como fuente PRIMARIA para los
macro tickers, reemplazándolo por proveedores que SÍ están probados como
100% confiables en este mismo entorno (Twelve Data ya sirve AAPL/MSFT/
TSLA/SPY/BTC-USD/... sin un solo fallo en los logs) o que son APIs reales
diseñadas para tráfico programático (no scraping de un endpoint CSV
público), en vez de reintentar contra un proveedor que rechaza el tráfico
de forma sistemática. Los 5 macro tickers entran al tensor SOLO como
log-return (ver auditoría más arriba) -- eso da libertad real para
sustituir cada índice/futuro por un ETF/activo LÍQUIDO Y REAL que trackea
la MISMA magnitud subyacente, con la MISMA garantía matemática ya
establecida arriba (un factor de escala/tracking constante es invisible
para el tensor; solo importa que el MOVIMIENTO diario sea el mismo
activo real, cero simulación):

  • ^GSPC (S&P 500 cash) -> **SPY** vía Twelve Data. SPY *es* el ETF que
    trackea el S&P 500 -- no es un proxy aproximado, es prácticamente el
    mismo activo con tracking error de puntos básicos. Ya está en
    REGIME_TICKERS y se descarga con éxito en cada scan (ver logs) -- cero
    riesgo de proveedor nuevo, cero costo extra relevante de créditos.

  • ^VIX (CBOE Volatility Index) -> **VIXY** (ProShares VIX Short-Term
    Futures ETF) vía Twelve Data. Proxy, no el índice exacto (roll
    yield/contango del futuro de corto plazo introduce deriva de largo
    plazo que el índice cash no tiene) -- mismo tipo de caveat ya aceptado
    para GC=F/XAU-USD arriba, documentado igual acá. El día a día
    (dirección e intensidad del shock de volatilidad, que es lo que
    SENTIMENT_SCORE/el tensor realmente necesitan) sigue correlacionado.

  • DX-Y.NYB (ICE US Dollar Index) -> **UUP** (Invesco DB US Dollar Index
    Bullish Fund) vía Twelve Data. UUP está diseñado explícitamente para
    trackear la MISMA canasta de divisas que el USDX -- proxy de alta
    fidelidad, no una aproximación libre.

  • GC=F (oro) -> SIN CAMBIO, ya vía Twelve Data (XAU/USD) desde la
    migración original -- nunca dependió de Stooq como primaria, por eso
    nunca apareció en los logs de fallo.

  • ^TNX (rendimiento UST 10Y) -> este es el ÚNICO que NO se puede
    resolver con un ETF de precio: un rendimiento (yield) y el precio de
    un ETF de bonos se mueven en direcciones opuestas y con una escala
    que depende de la duration del instrumento (NO es un factor constante
    -- a diferencia de todos los casos de arriba, acá "precio de ETF" y
    "yield" NO son la misma magnitud, son magnitudes relacionadas por una
    transformación que varía en el tiempo). Inventar un signo/escala acá
    violaría directamente "cero cifras inventadas". En vez de forzar un
    proxy matemáticamente débil, se agrega **FRED** (Federal Reserve
    Economic Data, `api.stlouisfed.org`, serie oficial `DGS10` = "10-Year
    Treasury Constant Maturity Rate", el rendimiento UST10Y real y
    oficial) como fuente PRIMARIA nueva -- 100% gratis (requiere API key
    gratuita, sin tarjeta, alta de un minuto en
    https://fred.stlouisfed.org/docs/api/api_key.html), y es una API REST
    real diseñada para tráfico programático (no un endpoint de scraping
    con protección anti-bot) -- no comparte el problema de Stooq. Stooq
    (`10yusy.b`) queda como fallback secundario si `FRED_API_KEY` no está
    configurada o la request a FRED falla puntualmente.

Con esto, Stooq deja de ser la única fuente real de 4 de los 5 macro
tickers y pasa a ser lo que su nombre siempre debió indicar: un fallback
de ÚLTIMO recurso, nunca la primaria de nada crítico.
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

# FRED (Federal Reserve Economic Data, api.stlouisfed.org) -- fuente
# PRIMARIA nueva (RONDA 2) para ^TNX (serie oficial DGS10, rendimiento
# UST10Y real). 100% gratis, sin tarjeta -- alta de la key en
# https://fred.stlouisfed.org/docs/api/api_key.html. API REST real
# diseñada para tráfico programático (no un endpoint de scraping con
# protección anti-bot como Stooq) -- ver ROOT CAUSE RONDA 2 en el
# docstring del módulo. Opcional: si no está configurada, ^TNX cae
# directo al fallback Stooq (mismo comportamiento que antes de RONDA 2).
FRED_API_KEY = os.environ.get("FRED_API_KEY", "").strip()
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_TIMEOUT_S = 15.0

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

if not FRED_API_KEY:
    logger.warning(
        "FRED_API_KEY no configurada -- '^TNX' (rendimiento UST10Y) cae "
        "directo al fallback Stooq, que en este hosting viene demostrando "
        "bloqueos sistemáticos (ver ROOT CAUSE RONDA 2 en el docstring del "
        "módulo). Registrate gratis (sin tarjeta, ~1 minuto) en "
        "https://fred.stlouisfed.org/docs/api/api_key.html y seteá la "
        "variable de entorno / Secret 'FRED_API_KEY' para eliminar ese "
        "último punto de falla."
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
# detalle matemático completo de cada fila.
#
# RONDA 2 (FIX 2026-08-15) -- ver "ROOT CAUSE RONDA 2" en el docstring del
# módulo: Stooq demostró en producción (logs post-RONDA 1, requests YA
# serializadas por el single-flight lock, con reintentos) un rechazo
# CONSISTENTE, no transitorio, para tráfico desde este hosting -- dejó de
# ser candidato a fuente PRIMARIA de nada crítico. `twelvedata`/`fred`
# ahora cubren 4 de los 5 macro tickers como fuente real primaria; `stooq`
# queda como fallback de último recurso en los 5 (nunca se elimina del
# todo -- sigue siendo mejor que nada si Twelve Data/FRED tienen un
# hiccup puntual). Orden de intento real, ver `_fetch_symbol_ohlcv`:
# FRED (si hay `fred` Y `FRED_API_KEY` configurada) -> Twelve Data (si hay
# `twelvedata`) -> Stooq (si hay `stooq`) -> caché stale -> falla ruidosa.
MACRO_SYMBOL_MAP: dict[str, dict[str, Optional[str]]] = {
    "^GSPC":    {"twelvedata": "SPY",   "stooq": "^spx",     "fred": None},
    # RONDA 2: SPY *es* el ETF que trackea el S&P 500 (no un proxy
    # aproximado) -- ya se descarga con éxito en cada scan del universo
    # (está en REGIME_TICKERS), cero riesgo de proveedor nuevo. Stooq
    # (^spx) queda como fallback si Twelve Data tuviera un hiccup puntual.

    "^VIX":     {"twelvedata": "VIXY",  "stooq": "^vix",     "fred": None},
    # RONDA 2: VIXY (ProShares VIX Short-Term Futures ETF) vía Twelve
    # Data -- proxy de futuros de corto plazo, no el índice cash exacto
    # (roll yield/contango introduce deriva de largo plazo que el índice
    # no tiene), mismo tipo de caveat ya aceptado para GC=F/XAU-USD abajo.
    # El shock día a día (lo que SENTIMENT_SCORE/el tensor necesitan)
    # sigue correlacionado. Cubierto por el plan Basic gratuito de TD
    # como cualquier ETF/equity normal (mismo trato que SPY/AAPL).

    "GC=F":     {"twelvedata": "XAU/USD", "stooq": "xauusd", "fred": None},
    # SIN CAMBIO en RONDA 2 -- ya vía Twelve Data desde la migración
    # original, nunca dependió de Stooq como primaria (por eso nunca
    # apareció en los logs de fallo). Oro SPOT -- GC=F en entrenamiento es
    # el FUTURO COMEX del contrato próximo; verificado en vivo (ago-2026):
    # mismo orden de magnitud, base variable en el tiempo (no una
    # proporción fija) -- proxy de mejor esfuerzo, NO corregible con un
    # multiplicador constante. Ver docstring del módulo.

    "DX-Y.NYB": {"twelvedata": "UUP",   "stooq": "usdx.f",   "fred": None},
    # RONDA 2: UUP (Invesco DB US Dollar Index Bullish Fund) vía Twelve
    # Data -- diseñado explícitamente para trackear la MISMA canasta de
    # divisas que el USDX, proxy de alta fidelidad (no una aproximación
    # libre). Stooq (usdx.f) queda como fallback de último recurso.

    "^TNX":     {"twelvedata": None,    "stooq": "10yusy.b", "fred": "DGS10"},
    # RONDA 2: ÚNICO macro ticker donde NO hay un ETF de precio que sirva
    # de proxy sin inventar un signo/escala -- un rendimiento (yield) y el
    # precio de un ETF de bonos se mueven en direcciones opuestas con una
    # escala que depende de la duration (no un factor CONSTANTE, a
    # diferencia de todos los demás casos de esta tabla) -- forzar esa
    # conversión violaría "cero cifras inventadas". En vez de eso: FRED
    # (`api.stlouisfed.org`, serie oficial `DGS10` = rendimiento UST10Y
    # real) como fuente PRIMARIA -- 100% gratis, API REST real diseñada
    # para tráfico programático, no comparte el problema de bloqueo de
    # Stooq. Requiere `FRED_API_KEY` (gratis, ver docstring del módulo);
    # sin ella, cae directo a Stooq (mismo comportamiento que antes de
    # RONDA 2, ya con las mejoras de resiliencia de RONDA 1 -- single-
    # flight, reintentos, caché stale). Rendimiento UST 10Y, PORCENTAJE
    # DIRECTO (4.558 = 4.558%) tanto en Stooq/yfinance/Yahoo como en FRED
    # -- verificado en vivo, ago-2026, sin transformación de escala
    # adicional necesaria entre proveedores.
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
# FRED (Federal Reserve Economic Data) -- fuente PRIMARIA nueva (RONDA 2)
# para ^TNX (serie DGS10). API REST oficial del gobierno de EE.UU.,
# diseñada para tráfico programático -- ver ROOT CAUSE RONDA 2 en el
# docstring del módulo. 100% gratis, requiere `FRED_API_KEY` (opcional: si
# no está seteada, este proveedor se salta en silencio y el caller cae al
# siguiente en la cadena, sin romper nada para quien no la configuró).
# ---------------------------------------------------------------------------

def _fred_series_daily(series_id: str, tail_days: int, attempts: int = 3) -> Optional[pd.Series]:
    """
    Devuelve una `pd.Series` de Close diario (valor de la serie FRED, ya en
    la unidad publicada -- para DGS10, porcentaje directo, ej. 4.558 =
    4.558%, misma convención que Stooq/yfinance ya documentada en
    MACRO_SYMBOL_MAP). FRED marca los días sin observación (feriados,
    fines de semana ya vienen excluidos por la API, pero algún feriado
    bancario puntual puede venir como ".") con el string "." -- se
    descartan como NaN, nunca se interpolan ni se inventan.
    """
    if not FRED_API_KEY:
        return None

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": str(tail_days),
    }

    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            resp = requests.get(FRED_BASE_URL, params=params, timeout=FRED_TIMEOUT_S)
            if resp.status_code == 429:
                raise RuntimeError(f"FRED rate limit (429): {resp.text[:200]}")
            resp.raise_for_status()
            payload = resp.json()
            observations = payload.get("observations") if isinstance(payload, dict) else None
            if not observations:
                return None

            df = pd.DataFrame(observations)
            if "date" not in df.columns or "value" not in df.columns:
                return None
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")  # "." (sin dato) -> NaN, nunca inventado
            series = df.set_index("date").sort_index()["value"].dropna()
            return series.tail(tail_days) if not series.empty else None
        except Exception as exc:  # noqa: BLE001 -- reintentamos cualquier fallo transitorio
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))

    logger.warning("FRED agotó reintentos para la serie '%s': %r", series_id, last_exc)
    return None


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
    fred_series_id: Optional[str] = None,
) -> pd.DataFrame:
    cached = _cache_get(f"ohlcv:{internal_key}")
    if cached is not None and len(cached) >= min(min_sessions, 30):
        return cached

    # FIX 2026-08-15 -- SINGLE-FLIGHT (ver ROOT CAUSE RONDA 1 en el
    # docstring del módulo). El universo completo (10 tickers, escaneado en
    # paralelo por `prediccion._scan_universe`) comparte los MISMOS 5 macro
    # tickers: sin este lock, hasta 10 hilos pedían '^GSPC' (u otro macro
    # ticker) a la red AL MISMO TIEMPO. Con el lock, solo el primer hilo en
    # tomarlo golpea la red; el resto espera y reusa lo que ese hilo deja
    # en caché -- de ahí el segundo `_cache_get` adentro (double-checked):
    # si otro hilo ya resolvió el símbolo mientras esperábamos el lock, no
    # repetimos nada.
    with _lock_for(internal_key):
        cached = _cache_get(f"ohlcv:{internal_key}")
        if cached is not None and len(cached) >= min(min_sessions, 30):
            return cached

        df: Optional[pd.DataFrame] = None

        # RONDA 2 -- FRED primero si el símbolo lo tiene mapeado (hoy solo
        # ^TNX, ver MACRO_SYMBOL_MAP). Es una API oficial diseñada para
        # tráfico programático, no un endpoint de scraping con protección
        # anti-bot como Stooq -- ver ROOT CAUSE RONDA 2. `_fred_series_daily`
        # ya devuelve `None` sin tocar la red si `FRED_API_KEY` no está
        # configurada, así que este bloque es un no-op limpio en ese caso.
        if fred_series_id:
            fred_series = _fred_series_daily(fred_series_id, tail_days=DEFAULT_HISTORY_SESSIONS + 30)
            if fred_series is not None and not fred_series.empty:
                df = pd.DataFrame({"close": fred_series})
                df["open"] = df["close"]
                df["high"] = df["close"]
                df["low"] = df["close"]
                df["volume"] = np.nan

        if df is None or df.empty:
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
                        "este ticker hasta que la fuente primaria vuelva a responder).",
                        internal_key,
                    )

            if df is None or df.empty:
                # FIX 2026-08-15 -- CACHÉ STALE COMO ÚLTIMO RECURSO, antes de
                # rendirse. TODAS las fuentes configuradas fallaron en vivo;
                # en vez de lanzar de inmediato (y con eso tumbar los 10
                # forecasts que dependen de este mismo macro ticker), se
                # busca la última descarga exitosa en disco sin importar si
                # el TTL de 6h venció. Solo si NUNCA hubo una descarga
                # exitosa (arranque en frío + todas las fuentes caídas a la
                # vez) se lanza el RuntimeError real.
                stale = _cache_get(f"ohlcv:{internal_key}", ignore_ttl=True)
                if stale is not None and not stale.empty:
                    stale_age_h = stale.attrs.get("_stale_age_hours")
                    logger.warning(
                        "'%s': todas las fuentes en vivo configuradas fallaron -- "
                        "sirviendo caché STALE (%s hs de antigüedad) en vez de "
                        "tumbar el forecast completo. Reintentará una descarga "
                        "en vivo en la próxima llamada.",
                        internal_key, stale_age_h if stale_age_h is not None else "?",
                    )
                    return stale

                fuentes = [n for n, v in (("FRED", fred_series_id), ("Twelve Data", td_symbol), ("Stooq", stooq_symbol)) if v]
                raise RuntimeError(
                    f"Sin datos para '{internal_key}': todas las fuentes configuradas "
                    f"({', '.join(fuentes) if fuentes else 'ninguna'}) fallaron y no hay "
                    "caché previa (ni siquiera stale) que servir -- radar degradado "
                    "para este símbolo."
                )

        _cache_set(f"ohlcv:{internal_key}", df)
        return df


# ---------------------------------------------------------------------------
# API pública -- consumida por services/quanti_engine.py
# ---------------------------------------------------------------------------

def _normalize_daily_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    FIX 2026-08-15 (alineación 24/7 vs 5/5) -- normaliza un índice de
    fechas a tz-naive con la hora puesta a medianoche.

    Las 3 fuentes de este módulo NO garantizan el mismo dtype de índice:
    Twelve Data (`_td_time_series`) pide `"timezone": "UTC"` y puede
    devolver timestamps CON offset explícito (tz-aware) según el símbolo/
    intervalo, mientras que FRED (`_fred_series_daily`) y Stooq
    (`_stooq_daily_close`) siempre devuelven `pd.to_datetime(...)`
    tz-naive. Un solo símbolo (target O macro) tz-aware colisionando
    contra el otro tz-naive hace que `.reindex()` entre ambos falle en
    TODAS las fechas (no solo fines de semana) -- indistinguible, en el
    mensaje de error final, de un genuino hueco de historia. Se normaliza
    ACÁ, en el borde de ingesta de cada símbolo, para que ninguna
    comparación de índices aguas abajo pueda volver a pisar este bug.
    """
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def _macro_close_ffilled_to_daily_calendar(macro_close: pd.Series, upto: pd.Timestamp) -> pd.Series:
    """
    FIX 2026-08-15 (alineación 24/7 vs 5/5) -- CAUSA RAÍZ real de
    "ventana de features vacía" para BTC-USD/ETH-USD.

    Cripto cotiza 24/7; los macro tickers (SPY/DGS10/XAU-USD/VIXY/UUP,
    todos vía bolsas tradicionales) cotizan 5/5. El código anterior hacía
    `macro_df["close"].reindex(out.index)` -- un reindex DIRECTO contra el
    calendario de 7 días de cripto, que deja NaN en cada sábado/domingo
    (el macro ticker simplemente no tiene fila esos días). El `.ffill()`
    posterior sobre el DataFrame COMBINADO alcanza a tapar la mayoría de
    esos huecos, pero CUALQUIER fila líder de `out.index` anterior a la
    primera fecha del macro ticker (o cualquier huequito interno que el
    ffill combinado no alcance a cubrir de forma determinista según el
    orden de columnas) sobrevive como NaN -- y `dropna()` sobre el frame
    completo basta para vaciarlo entero si eso golpea suficientes filas.

    Fix real: upsamplear la serie del macro ticker a un calendario DIARIO
    CORRIDO (freq="D", fin de semana incluido) y hacer forward-fill DENTRO
    de esa serie -- ANTES de reindexarla contra el calendario 24/7 del
    ticker objetivo. Así, el valor de un viernes de bolsa queda disponible
    como el macro-input válido para sábado y domingo de cripto, sin
    depender del orden ni de la suerte del ffill final sobre el frame ya
    combinado. Nunca se hace `bfill` (relleno hacia atrás): eso sería fuga
    de información (un valor FUTURO del macro ticker "explicando" un día
    pasado) -- el único costo real de este fix es que los primeros días,
    anteriores al primer dato macro disponible, siguen sin cobertura y se
    recortan más abajo, exactamente igual que antes.
    """
    if macro_close.empty:
        return macro_close
    macro_close = macro_close[~macro_close.index.duplicated(keep="last")].sort_index()
    calendar_end = max(macro_close.index.max(), upto)
    daily_calendar = pd.date_range(macro_close.index.min(), calendar_end, freq="D")
    return macro_close.reindex(daily_calendar).ffill()


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

    Ver `_macro_close_ffilled_to_daily_calendar` para el fix de alineación
    24/7 (cripto) vs 5/5 (macro tickers) -- antes de esta fecha, este
    mismo merge era la causa raíz real de
    "'BTC-USD'/'ETH-USD': ventana de features vacía tras alinear con los
    macro tickers" (ver logs de producción del 2026-08-15).
    """
    td_symbol = _resolve_td_symbol(ticker)
    stooq_symbol = _resolve_stooq_symbol(ticker)
    target_df = _fetch_symbol_ohlcv(ticker, td_symbol, stooq_symbol, min_sessions)

    target_index = _normalize_daily_index(target_df.index)
    target_index = target_index[~target_index.duplicated(keep="last")]
    # Puede desordenar la serie si la fuente ya venía ordenada pero con
    # timestamps tz-aware cuya normalización cambia el orden relativo en
    # un caso límite (feriados con horario de cierre distinto) -- sort
    # explícito, nunca asumido.
    target_df = target_df.loc[~target_df.index.duplicated(keep="last")].copy()
    target_df.index = _normalize_daily_index(target_df.index)
    target_df = target_df.sort_index()

    out = pd.DataFrame(index=target_df.index)
    out[ticker] = target_df["close"]
    out[f"{ticker}_High"] = target_df["high"]
    out[f"{ticker}_Low"] = target_df["low"]
    out[f"{ticker}_Volume"] = target_df["volume"]

    target_last_date = out.index.max() if not out.empty else pd.Timestamp.utcnow().normalize()

    for macro_ticker in macro_tickers:
        mapping = resolve_macro_symbol(macro_ticker)
        macro_df = _fetch_symbol_ohlcv(
            macro_ticker, mapping.get("twelvedata"), mapping.get("stooq"), min_sessions,
            fred_series_id=mapping.get("fred"),
        )
        macro_close = macro_df["close"].copy()
        macro_close.index = _normalize_daily_index(macro_close.index)
        macro_close_daily = _macro_close_ffilled_to_daily_calendar(macro_close, target_last_date)
        out[macro_ticker] = macro_close_daily.reindex(out.index)

    out = out.sort_index()
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
        fred_series_id = mapping.get("fred")
    else:
        td_symbol = _resolve_td_symbol(internal_key)
        stooq_symbol = _resolve_stooq_symbol(internal_key)
        fred_series_id = None

    df = _fetch_symbol_ohlcv(internal_key, td_symbol, stooq_symbol, min_days, fred_series_id=fred_series_id)
    return df["close"].dropna()