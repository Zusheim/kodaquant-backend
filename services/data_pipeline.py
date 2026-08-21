"""
services/data_pipeline.py

Fragmento de Ingeniería de Features — Pipeline NLP de Sentimiento de Noticias
==============================================================================
KodaQuant V5 — Requerimiento 1 (Cero Fricción / Anti-CAPTCHAs).

Añade la variable continua `NEWS_SENTIMENT_SCORE` (rango [-1, 1]) al tensor
de entrada del modelo, calculada EXCLUSIVAMENTE con APIs nativas:

    1) DDGS (DuckDuckGo News Search) -> fuente primaria, sin API key.
       Reemplaza a `yfinance.Ticker(ticker).news` (yfinance retirado por
       completo del proyecto, ver services/market_data.py) manteniendo el
       mismo rol estructural: gratuita, sin key, no depende de que el
       operador configure nada para tener AL MENOS cobertura básica.
    2) Finnhub REST API                -> fallback, requiere FINNHUB_API_KEY.
       `/company-news` para equities, `/news?category=crypto` para cripto
       (ver `_is_crypto_ticker`/`_fetch_headlines_finnhub` — V6, fix de
       "señal muerta" en crypto_specialist).

PROHIBIDO en este módulo: Selenium, BeautifulSoup, requests directos a
portales de noticias, o cualquier scraping HTML iterativo. Cualquiera de
esos vectores colapsa el pipeline productivo ante CAPTCHAs, Cloudflare o
baneos de IP del proveedor de hosting — exactamente lo que este pipeline
está diseñado para evitar. DDGS consume el endpoint JSON de noticias de
DuckDuckGo (misma librería `ddgs` ya usada por
`quanti_engine.search_financial_web`, ya pineada en requirements.txt) —
no es scraping de HTML de portales de noticias individuales.

V7 (auditoría "señal muerta" universal, post-fix-tz V6): dos fallos
independientes que colapsaban NEWS_SENTIMENT_SCORE a 0.0 EXACTO pese al
fix de timezone previo — (a) `_score_text` descartaba en silencio
cualquier etiqueta de FinBERT no reconocida (ej. LABEL_0/1/2 en vez de
positive/negative/neutral) sin loguear nada, para TODOS los tickers por
igual; (b) `get_daily_news_sentiment` exigía calce EXACTO de
fecha-calendario (`Series.reindex`), que nunca aterriza para titulares
publicados fin de semana/feriado en el universo de equities. Ver
docstrings de ambas funciones para el detalle completo.

V8 (auditoría de atribución por gradiente, post-fix-señal-muerta V7): con
la señal ya viva (V7), NEWS_SENTIMENT_SCORE seguía rank 17/17 en
equity_specialist mientras SENTIMENT_SCORE (correlación móvil
activo<->VIX, ver train_kodaquant_v5.py/quanti_engine.py — variable de
régimen/beta derivada de precio, NO redundante con esta) dominaba el
Top 5. Diagnóstico: no es colisión de features, es cobertura de prensa
esporádica en equity (relleno neutro + saltos discretos = ruido de alta
frecuencia que el modelo aprende a ignorar). Fix: EMA regime-aware al
final de `get_daily_news_sentiment` — amortigua equity, no-op exacto en
cripto (flujo de noticias 24/7, la misma densidad que sostiene el
51.2%/52.3% de crypto_specialist). Ver constantes
`NEWS_SENTIMENT_EMA_SPAN_*` más abajo. Cero cambios en
train_kodaquant_v5.py/quanti_engine.py: ambos consumen esta función tal
cual, consistencia train/inferencia automática.

V9 (consolidación del core, auditoría CTO 2026-08-16): `fetch_recent_headlines`
ahora cachea en disco vía `kodaquant_core.CacheManager` (TTL corto,
`KODAQUANT_NEWS_CACHE_TTL_SECONDS`, purga automática local en cada
llamada) -- reduce presión sobre DDGS/Finnhub sin tocar la lógica de
scoring/merge de abajo, que queda intacta. Ver `kodaquant_core.py` para el
resto de la consolidación (caché de OHLCV, cliente FRED, alineación de
calendario macro), compartida ahora con `train_kodaquant_v5.py`.

Analizador de sentimiento: FinBERT (`ProsusAI/finbert` vía `transformers`),
cargado perezosamente (singleton) y con selección automática de device
(GPU si disponible, CPU en runners de CI). `get_daily_news_sentiment` sigue
aceptando un `scorer_fn` intercambiable para no acoplar el resto del módulo
a este modelo en particular.

Integración con `engineer_asset` (ver train_kodaquant_v5.py):
    df["NEWS_SENTIMENT_SCORE"] = get_daily_news_sentiment(ticker, df.index)
Debe ejecutarse ANTES de que `TECH_COLS`/`feature_cols` arme el tensor de
entrada, junto al resto de los indicadores técnicos — nunca después de la
transformación a retornos logarítmicos del target.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

# --- RESOLUCIÓN ABSOLUTA DE IMPORTS (Directiva 2 — fin al Import Hell) ----
# Este archivo vive ÚNICAMENTE en `ml_engine/` (Directiva 1: SINGLE SOURCE
# OF TRUTH). Sube desde este archivo hasta encontrar la raíz del proyecto
# -- el primer ancestro que contiene TANTO `ml_engine/` como `services/`
# como subdirectorios -- y la inserta en `sys.path[0]`. A partir de ahí,
# `ml_engine` y `services` son paquetes válidos sin importar desde qué
# directorio de trabajo se invoque `python .../data_pipeline.py` (directo,
# importado por `train_kodaquant_v5.py`, o vía `services.quanti_engine`).
# Reemplaza por completo el patrón try/except dual + copias duplicadas de
# `kodaquant_core.py` que causaba `ModuleNotFoundError`/`ImportError`.
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

# Núcleo compartido train/inferencia -- SINGLE SOURCE OF TRUTH en
# `ml_engine/kodaquant_core.py` (Directiva 1). Import directo y absoluto,
# sin fallback dual: con la raíz del proyecto ya en sys.path, esta ruta
# resuelve siempre, sin importar el cwd de invocación.
from ml_engine.kodaquant_core import CacheManager  # noqa: E402

# --- PARCHE macOS Intel: runtimes OpenMP duplicados TensorFlow/PyTorch ----
# `quanti_engine.py` carga TensorFlow/Keras (bundla Intel MKL -> libiomp5.dylib)
# EAGER al arrancar Uvicorn. PyTorch (bundla LLVM -> libomp.dylib) se importa
# LAZY aquí, recién en el primer request real. Ambos runtimes OpenMP en el
# mismo proceso es un conflicto conocido en macOS que puede impedir que
# torch cargue su extensión nativa. Debe fijarse ANTES del primer `import
# torch` — no tiene efecto si se fija después.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import requests  # Únicamente para el endpoint JSON de Finnhub — API nativa, NO scraping.

# --- Búsqueda de noticias vía DDGS (DuckDuckGo) — reemplaza a yfinance.news
# como fuente primaria, gratuita y sin API key (yfinance retirado por
# completo del proyecto, ver services/market_data.py). Mismo import
# defensivo (paquete renombrado `duckduckgo_search` -> `ddgs`) que ya usa
# `quanti_engine.search_financial_web`; `ddgs` ya está pineado en
# requirements.txt, cero dependencias nuevas.
try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover
    try:
        from duckduckgo_search import DDGS  # type: ignore[no-redef]
    except ImportError:
        DDGS = None  # type: ignore[assignment,misc]  # se valida en runtime, ver _fetch_headlines_ddg

DDG_REGION = os.getenv("DDG_REGION", "es-es")
DDG_SAFESEARCH = os.getenv("DDG_SAFESEARCH", "moderate")
MAX_DDG_HEADLINES = 10  # techo por-ticker -- Finnhub complementa si esto no alcanza

logger = logging.getLogger("kodaquant.data_pipeline")

# Caché de titulares (DDGS/Finnhub) — vida corta: la cobertura de prensa
# cambia varias veces al día, pero no vale la pena re-golpear DDGS/Finnhub
# en cada llamada de `get_daily_news_sentiment` dentro de la MISMA corrida
# (el radar escanea el universo completo en paralelo, ver
# `data_pipeline.get_daily_news_sentiment` invocado por cada régimen).
# Purga automática local (Requerimiento 1): `_headline_cache.purge_stale()`
# se invoca de forma perezosa en cada `fetch_recent_headlines` -- sin cron
# externo, sin intervención manual.
_HEADLINE_CACHE_DIR = Path(os.getenv("KODAQUANT_NEWS_CACHE_DIR", "cache/news"))
_HEADLINE_CACHE_TTL_SECONDS = int(os.getenv("KODAQUANT_NEWS_CACHE_TTL_SECONDS", str(2 * 3600)))
_headline_cache = CacheManager(_HEADLINE_CACHE_DIR, default_ttl_seconds=_HEADLINE_CACHE_TTL_SECONDS)

# ---------------------------------------------------------------------------
# FinBERT — analizador de sentimiento transformer, carga perezosa (singleton)
# ---------------------------------------------------------------------------
_FINBERT_MODEL_NAME = "ProsusAI/finbert"
_FINBERT_LABEL_SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
# V7 (auditoría "señal muerta" UNIVERSAL — TODOS los activos, no solo cripto):
# ciertas combinaciones de `transformers`/config cacheado devuelven las
# etiquetas GENÉRICAS del clasificador (LABEL_0/LABEL_1/LABEL_2) en vez de
# las humanas positive/negative/neutral, cuando el `id2label` del modelo no
# se resuelve en tiempo de carga. Orden oficial de ProsusAI/finbert
# (config.json del modelo): 0=positive, 1=negative, 2=neutral.
_FINBERT_LABEL_ALIASES = {"label_0": "positive", "label_1": "negative", "label_2": "neutral"}
_finbert_pipeline = None
# Protege la carga perezosa contra llamadas CONCURRENTES: bajo Uvicorn,
# get_daily_news_sentiment se dispara desde varios hilos a la vez (Plan A/Plan B
# en paralelo, radar escaneando el universo completo vía ThreadPoolExecutor).
# Sin lock, el primer request real puede disparar 2+ hilos importando
# torch/transformers simultáneamente -> ImportError transitorio de import
# parcial/circular, indistinguible de "no instalado" sin este fix.
_finbert_init_lock = threading.Lock()


def _get_finbert_pipeline():
    """Carga perezosa (singleton, thread-safe) del pipeline FinBERT — CPU-safe para runners sin GPU."""
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline

    with _finbert_init_lock:
        if _finbert_pipeline is not None:  # doble check: otro hilo ya lo cargó mientras esperábamos el lock
            return _finbert_pipeline

        try:
            import torch
            from transformers import pipeline as _hf_pipeline
        except ModuleNotFoundError as exc:
            # Caso real de "no instalado": el paquete no existe en este intérprete.
            raise ImportError(
                f"torch/transformers no están instalados en {sys.executable}: {exc}. "
                "Ejecuta `pip install torch transformers` en el MISMO venv que corre Uvicorn."
            ) from exc
        except ImportError as exc:
            # CUALQUIER OTRO ImportError (símbolo nativo, dlopen roto, import
            # circular parcial por carrera entre hilos, choque de runtimes
            # OpenMP TF/Torch, etc.) — antes se enmascaraba con el mismo
            # mensaje genérico de "no instalado". Se preserva el tipo y
            # mensaje reales para que el log deje de mentir.
            raise ImportError(
                f"torch/transformers están instalados pero fallaron al importar "
                f"({type(exc).__name__}: {exc}). NO es un paquete faltante — revisa "
                "conflictos nativos (runtimes OpenMP duplicados TF/PyTorch en macOS, "
                "ver KMP_DUPLICATE_LIB_OK) o reinstala ambos paquetes emparejados."
            ) from exc

        device = 0 if torch.cuda.is_available() else -1
        _finbert_pipeline = _hf_pipeline(
            "sentiment-analysis", model=_FINBERT_MODEL_NAME,
            tokenizer=_FINBERT_MODEL_NAME, truncation=True, device=device,
        )
    return _finbert_pipeline

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
# V6 (auditoría "señal muerta" crypto_specialist) — `/company-news` es
# EXCLUSIVO para equities listadas en EE. UU. (doc oficial Finnhub); jamás
# devuelve resultados para tickers cripto. `/news?category=crypto` es el
# endpoint correcto para ese universo — ver `_fetch_headlines_finnhub`.
FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"
FINNHUB_CRYPTO_NEWS_URL = "https://finnhub.io/api/v1/news"

NEWS_LOOKBACK_DAYS = 7           # ventana de titulares recientes a recolectar (fallback Finnhub)
MAX_HEADLINES_PER_TICKER = 60    # techo defensivo (rate-limit / ruido)
NEUTRAL_SCORE = 0.0              # relleno para días/activos sin cobertura de prensa
# V7 — tolerancia del merge_asof temporal (ver get_daily_news_sentiment):
# NEWS_LOOKBACK_DAYS + colchón de fin de semana/feriado, para que un titular
# del viernes siga cubriendo el lunes bursátil sin exigir calce EXACTO de
# fecha (el bug que _NEWS_MERGE_TOLERANCE_DAYS/pd.merge_asof reemplaza) y,
# a la vez, sin arrastrar sentimiento añejo indefinidamente como hacía el
# `.ffill()` sin cota que tenía este merge antes.
_NEWS_MERGE_TOLERANCE_DAYS = NEWS_LOOKBACK_DAYS + 3

_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 1.5

# --- V8 (Filtrado Regime-Aware — aislamiento de ruido equity sin tocar cripto) ---
# Auditoría de atribución por gradiente: NEWS_SENTIMENT_SCORE cae a
# rank 17/17 en equity_specialist. Causa raíz NO es colisión con
# SENTIMENT_SCORE (esa es la correlación móvil activo<->VIX de
# compute_sentiment_score en train_kodaquant_v5.py/quanti_engine.py — un
# proxy de régimen/beta derivado 100% de precio, variable estructuralmente
# distinta de esta, cero solapamiento de información): es que la cobertura
# de prensa por-ticker vía DDGS/Finnhub es ESPORÁDICA en equity (pocos
# titulares/semana), así que la señal día a día alterna tramos planos
# (relleno neutro) con saltos discretos cada vez que aterriza un titular —
# exactamente el patrón de alta frecuencia que un BiLSTM aprende a
# descontar como ruido. crypto_specialist NO sufre esto: `/news?category=
# crypto` entrega flujo 24/7 sin huecos, la misma densidad/volatilidad
# cruda que llevó a crypto_specialist a 51.2%/52.3% de asertividad — jamás
# se suaviza.
# Fix: EMA regime-aware aplicado DENTRO de get_daily_news_sentiment (única
# fuente de esta columna para train_kodaquant_v5.py Y quanti_engine.py) ->
# consistencia train/inferencia automática, cero cambios en esos dos
# archivos, cero impacto en TECH_COLS/N_FEATURES/(batch,60,17).
NEWS_SENTIMENT_EMA_SPAN_EQUITY = 5   # ~1 semana bursátil: amortigua saltos discretos sin borrar la tendencia real
NEWS_SENTIMENT_EMA_SPAN_CRYPTO = 1   # span=1 -> alpha=2/(span+1)=1.0 -> no-op matemático exacto, señal cruda intacta


def _score_text(text: str) -> float:
    """Escalar direccional (∈[-1, 1]) de un titular vía FinBERT (ProsusAI/finbert)."""
    if not text:
        return NEUTRAL_SCORE
    try:
        result = _get_finbert_pipeline()(text[:512])[0]
        raw_label = str(result["label"]).lower()
        label = _FINBERT_LABEL_ALIASES.get(raw_label, raw_label)
        if label not in _FINBERT_LABEL_SIGN:
            # FIX RAÍZ V7 (auditoría "señal muerta" universal — 10 titulares
            # obtenidos, 100% neutro, TODOS los activos, sin ningún otro
            # warning en el log): antes, una etiqueta no reconocida caía en
            # `_FINBERT_LABEL_SIGN.get(label, 0.0)` -> 0.0 SILENCIOSO. No es
            # una excepción (no dispara el except de abajo), así que NUNCA
            # quedaba rastro en el log de por qué el score era siempre 0 —
            # exactamente el síntoma reportado, y explica por qué persistía
            # pese al fix de timezone (esa parte del pipeline nunca fue el
            # problema: los titulares SÍ llegaban, pero cada uno puntuaba
            # 0.0 antes de tocar cualquier lógica de fechas). Ahora es ruidoso.
            logger.warning(
                "FinBERT devolvió una etiqueta no reconocida %r (fuera de "
                "_FINBERT_LABEL_SIGN/_FINBERT_LABEL_ALIASES) -> score forzado "
                "a NEUTRAL_SCORE para este titular. Verifica la versión de "
                "`transformers` / el config.json de %s (id2label esperado: "
                "0=positive, 1=negative, 2=neutral).", result["label"], _FINBERT_MODEL_NAME,
            )
            return NEUTRAL_SCORE
        sign = _FINBERT_LABEL_SIGN[label]
        return float(sign * result["score"])
    except Exception as exc:  # noqa: BLE001 — fallo de inferencia del modelo local
        # exc_info=True: antes se logueaba solo %r del error (a veces ya
        # reescrito con un mensaje genérico), sin traceback — la causa real
        # (incl. __cause__ encadenado) quedaba invisible en los logs de
        # Uvicorn. Ahora se propaga completa.
        logger.warning(
            "FinBERT falló al puntuar titular (%s: %s)", type(exc).__name__, exc,
            exc_info=True,
        )
        return NEUTRAL_SCORE


def _normalize_timestamp(ts_raw) -> Optional[datetime]:
    """Normaliza epoch (int/float) o string de fecha a `datetime` UTC."""
    if ts_raw is None:
        return None
    try:
        if isinstance(ts_raw, (int, float)):
            return datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
        return pd.Timestamp(ts_raw, tz="UTC").to_pydatetime()
    except (ValueError, TypeError, OSError):
        return None


def _fetch_headlines_ddg(ticker: str, max_results: int = MAX_DDG_HEADLINES) -> list[dict]:
    """
    Fuente primaria — búsqueda de noticias vía DDGS (`ddgs.news`, JSON nativo
    de DuckDuckGo, sin scraping de portales individuales, sin API key).
    Reemplaza 1:1 el rol de `yfinance.Ticker(ticker).news` (retirado, ver
    services/market_data.py): misma firma de retorno (`title`/`published_at`),
    mismo lugar en `fetch_recent_headlines` (primaria, Finnhub complementa).
    """
    if DDGS is None:
        logger.warning("DDGS no instalado (`pip install ddgs`) -- fuente primaria de noticias deshabilitada; solo Finnhub (si hay API key).")
        return []

    # Cripto: "-USD" es la convención de símbolo de todo el proyecto
    # (REGIME_TICKERS/MACRO_SYMBOL_MAP), no específica de ningún proveedor
    # -- se usa solo el símbolo base (ej. "BTC") para una query de noticias
    # más natural que "BTC-USD".
    query = f"{ticker.split('-')[0]} crypto" if _is_crypto_ticker(ticker) else f"{ticker} stock"

    try:
        with DDGS() as ddgs:
            raw = list(
                ddgs.news(query, region=DDG_REGION, safesearch=DDG_SAFESEARCH, max_results=max_results)
            )
    except Exception as exc:  # noqa: BLE001 -- red/HTTP transitorio de DDG
        logger.warning("DDGS.news falló para %s: %r", ticker, exc)
        return []

    headlines = []
    for item in raw:
        title = (item.get("title") or "").strip()
        published_at = _normalize_timestamp(item.get("date"))
        if title and published_at is not None:
            headlines.append({"title": title, "published_at": published_at})
    return headlines


def _is_crypto_ticker(ticker: str) -> bool:
    """
    Heurística deliberadamente auto-contenida (sin importar `REGIME_TICKERS`
    de quanti_engine.py/train_kodaquant_v5.py: SON quienes importan
    data_pipeline.py, importar en sentido inverso crearía un ciclo). La
    convención de todo el proyecto para cripto es SIEMPRE '<SÍMBOLO>-USD'
    (BTC-USD, ETH-USD); ningún ticker de equity/ETF del universo entrenado
    usa ese sufijo -> heurística exacta y suficiente para este propósito.
    """
    return ticker.upper().endswith("-USD")


def _fetch_headlines_finnhub(ticker: str, lookback_days: int = NEWS_LOOKBACK_DAYS) -> list[dict]:
    """
    Fallback — API REST gratuita de Finnhub, JSON puro, sin scraping.
    Requiere `FINNHUB_API_KEY` en el entorno; si falta, se omite
    silenciosamente (DDGS sigue siendo la fuente primaria y suficiente
    en la mayoría de los casos).

    V6 (auditoría "señal muerta" `crypto_specialist`, NEWS_SENTIMENT_SCORE
    rank 17/17, atribución 0.000000) — FIX de enrutamiento: `/company-news`
    es EXCLUSIVO para equities listadas en EE. UU. (documentación oficial
    de Finnhub) y nunca devuelve resultados para tickers cripto; antes se
    llamaba ese mismo endpoint también para BTC-USD/ETH-USD, vaciando
    silenciosamente el fallback justo para el régimen que más lo necesita.
    Se enruta ahora vía `_is_crypto_ticker` al endpoint general
    `/news?category=crypto` para cripto (sin filtro por símbolo: el tier
    gratuito de Finnhub no ofrece news cripto por-moneda -> se filtra la
    ventana `lookback_days` del lado del cliente, ya que ese endpoint no
    acepta `from`/`to`) y mantiene `/company-news` intacto para equities.
    """
    if not FINNHUB_API_KEY:
        logger.debug("FINNHUB_API_KEY no configurada; se omite el fallback de Finnhub.")
        return []

    is_crypto = _is_crypto_ticker(ticker)
    url = FINNHUB_CRYPTO_NEWS_URL if is_crypto else FINNHUB_COMPANY_NEWS_URL
    today = datetime.now(timezone.utc).date()
    since = today - timedelta(days=lookback_days)
    if is_crypto:
        params = {"category": "crypto", "token": FINNHUB_API_KEY}
    else:
        params = {
            "symbol": ticker,
            "from": since.isoformat(),
            "to": today.isoformat(),
            "token": FINNHUB_API_KEY,
        }

    last_exc: Exception | None = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json() or []
            headlines = []
            for item in payload[:MAX_HEADLINES_PER_TICKER]:
                title = item.get("headline") or item.get("summary") or ""
                published_at = _normalize_timestamp(item.get("datetime"))
                # `/news?category=crypto` no filtra por fecha server-side
                # (a diferencia de `/company-news`) -> recorte client-side
                # con el mismo `lookback_days`, para no mezclar noticias
                # cripto viejas fuera de la ventana esperada.
                if is_crypto and published_at is not None and published_at.date() < since:
                    continue
                if title and published_at is not None:
                    headlines.append({"title": title, "published_at": published_at})
            return headlines
        except Exception as exc:  # noqa: BLE001 — red/HTTP transitorio
            last_exc = exc
            if attempt == _RETRY_ATTEMPTS:
                break
            time.sleep(_RETRY_BACKOFF_SECONDS * attempt)

    logger.warning("Finnhub falló tras %d intentos para %s: %r", _RETRY_ATTEMPTS, ticker, last_exc)
    return []


def fetch_recent_headlines(ticker: str) -> list[dict]:
    """
    Orquesta ambas fuentes nativas (DDGS -> Finnhub) y deduplica por
    título. DDGS es la fuente primaria (no exige API key); Finnhub
    complementa cobertura solo cuando DDGS devuelve pocos titulares.

    CacheManager (kodaquant_core): cachea el resultado ya deduplicado por
    `_HEADLINE_CACHE_TTL_SECONDS` -- evita re-golpear DDGS/Finnhub por
    ticker dentro de la misma ventana corta cuando el radar escanea el
    universo completo. Purga automática local en cada llamada (best-effort,
    nunca bloquea el fetch real si falla).
    """
    try:
        _headline_cache.purge_stale()
    except Exception as exc:  # noqa: BLE001 -- la purga nunca debe tumbar el fetch real
        logger.debug("[CacheManager] purge_stale falló (%r) -- se ignora.", exc)

    cache_key = f"headlines_{ticker}"
    cached = _headline_cache.get_json(cache_key)
    if cached is not None:
        for h in cached:
            h["published_at"] = _normalize_timestamp(h.get("published_at"))
        return [h for h in cached if h.get("published_at") is not None]

    headlines = _fetch_headlines_ddg(ticker)
    if len(headlines) < 5:
        headlines += _fetch_headlines_finnhub(ticker)

    seen_titles: set[str] = set()
    deduped: list[dict] = []
    for h in headlines:
        if h["title"] not in seen_titles:
            seen_titles.add(h["title"])
            deduped.append(h)
    deduped = deduped[:MAX_HEADLINES_PER_TICKER]

    serializable = [
        {"title": h["title"], "published_at": h["published_at"].isoformat()}
        for h in deduped
    ]
    _headline_cache.set_json(cache_key, serializable)
    return deduped


def get_daily_news_sentiment(
    ticker: str,
    target_index: pd.DatetimeIndex,
    scorer_fn: Callable[[str], float] = _score_text,
) -> pd.Series:
    """
    Devuelve `NEWS_SENTIMENT_SCORE` alineado día a día contra `target_index`
    (el mismo índice temporal que ya usan RSI/MACD/ATR/etc.). Cada titular se
    promedia por día de publicación (UTC, normalizado a medianoche); los
    días sin cobertura de prensa se rellenan hacia adelante (`ffill`) y, si
    no hay ningún titular previo disponible, con `NEUTRAL_SCORE` (0.0) — la
    misma convención de "lectura neutra" que SENTIMENT_SCORE ya usa en V4.

    Esta función debe llamarse ANTES de concatenar el tensor final al
    espacio de entrada (batch, time_steps, features), junto al resto de los
    indicadores técnicos en `engineer_asset`.

    FIX RAÍZ V7 (auditoría "señal muerta" — persiste tras el fix de tz V6,
    NEWS_SENTIMENT_SCORE rank 17/17, atribución exacta 0.000000, EN TODOS
    LOS ACTIVOS, no solo cripto): el fix V6 (despojar tz antes de comparar
    fechas) era necesario pero NO suficiente — quedaban dos fallos
    independientes, cualquiera de los dos basta por sí solo para colapsar
    la columna entera a 0.0:

      (a) SCORER: ver `_score_text` — una etiqueta de FinBERT no reconocida
          caía en `.get(label, 0.0)` -> 0.0 silencioso, SIN excepción, para
          el 100% de los titulares, en TODOS los tickers por igual (fix
          separado, ver ese docstring).
      (b) MERGE: `Series.reindex` exige que la fecha-calendario del
          titular calce EXACTA (bit a bit) con una fila del índice de
          trading. Cripto cotiza 24/7 (hay fila para cada día), pero
          equities NO — cualquier titular publicado un fin de semana o
          feriado bursátil no tenía dónde aterrizar y quedaba NaN -> 0.0,
          sin importar qué tan bien alineados estuvieran los tz. Se
          reemplaza por `pd.merge_asof(direction="backward")`: busca el
          titular MÁS RECIENTE publicado EN O ANTES de cada fecha del
          índice (sin exigir igualdad exacta), acotado por
          `_NEWS_MERGE_TOLERANCE_DAYS` para no arrastrar sentimiento añejo
          indefinidamente como hacía el `.ffill()` sin cota anterior.

    Ambos lados de la fecha se normalizan ahora de forma SIMÉTRICA a
    fecha-calendario UTC (`tz_convert("UTC")` real, no un `tz_localize(None)`
    que solo despoja la etiqueta sin convertir el instante) — antes los
    titulares se convertían a UTC pero `target_index` solo se despojaba de
    su tz "tal cual", una asimetría de la que dependía silenciosamente el
    calce.
    """
    headlines = fetch_recent_headlines(ticker)

    if not headlines:
        logger.info("Sin titulares recientes para %s -> NEWS_SENTIMENT_SCORE neutro.", ticker)
        return pd.Series(NEUTRAL_SCORE, index=target_index, name="NEWS_SENTIMENT_SCORE")

    scored = pd.DataFrame(headlines)
    scored["score"] = scored["title"].map(scorer_fn)

    # Diagnóstico (a): aísla un SCORER muerto de un MERGE con mala cobertura
    # temporal — antes ambos colapsaban al mismo mensaje genérico y eran
    # indistinguibles desde el log.
    nonzero_scores = int((scored["score"].abs() > 1e-9).sum())

    scored["date"] = (
        pd.DatetimeIndex(scored["published_at"]).tz_convert("UTC").tz_localize(None).normalize()
    )
    daily = (
        scored.groupby("date")["score"].mean()
        .sort_index()
        .rename("headline_score")
        .reset_index()
    )

    target_dt = pd.DatetimeIndex(target_index)
    if target_dt.tz is not None:
        target_dates = target_dt.tz_convert("UTC").tz_localize(None).normalize()
    else:
        target_dates = target_dt.normalize()

    # FIX: pandas 3.x ya no fuerza datetime64[ns] en todos lados -- conserva
    # la resolución de origen de cada Serie (target_index puede venir en
    # [us] del feed de mercado (Twelve Data/Stooq, ver services/market_data.py),
    # "date" en [s] desde el parseo de published_at).
    # merge_asof exige que ambas claves compartan EXACTAMENTE el mismo
    # dtype, si no: "incompatible merge keys ... must be the same type".
    # Se normalizan ambas a [ns] explícitamente antes de mergear.
    frame = pd.DataFrame({"target_date": target_dates.astype("datetime64[ns]")})
    frame["orig_pos"] = range(len(frame))
    frame_sorted = frame.sort_values("target_date", kind="stable")
    daily["date"] = daily["date"].astype("datetime64[ns]")

    merged = pd.merge_asof(
        frame_sorted,
        daily,
        left_on="target_date",
        right_on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=_NEWS_MERGE_TOLERANCE_DAYS),
    ).sort_values("orig_pos")

    aligned = pd.Series(
        merged["headline_score"].to_numpy(), index=target_index, name="NEWS_SENTIMENT_SCORE",
    )
    aligned = aligned.fillna(NEUTRAL_SCORE).clip(-1.0, 1.0)

    # Canario de cobertura, ahora DESAMBIGUADO: si la columna vuelve a
    # colapsar a puro NEUTRAL_SCORE, el log dice CUÁL de las dos causas
    # posibles es — nunca más un mensaje genérico "revisar fechas o API"
    # cuando el problema real puede estar en el scorer y viceversa.
    if (aligned == NEUTRAL_SCORE).all():
        if nonzero_scores == 0:
            logger.warning(
                "NEWS_SENTIMENT_SCORE 100%% neutro para %s: los %d titular(es) "
                "obtenidos puntuaron EXACTAMENTE 0.0 en el scorer -> revisar "
                "warnings de '_score_text'/etiqueta FinBERT no reconocida "
                "arriba en el log; el merge de fechas es irrelevante, la señal "
                "ya nace muerta antes de tocar cualquier lógica temporal.",
                ticker, len(headlines),
            )
        else:
            logger.warning(
                "NEWS_SENTIMENT_SCORE 100%% neutro para %s pese a %d titular(es) "
                "obtenidos Y %d de ellos con score no-neutro real -> el scorer "
                "funciona, es un problema de COBERTURA TEMPORAL: ninguna fecha "
                "cae dentro de la tolerancia de %d día(s) contra target_index "
                "(¿target_index desactualizado / caché parquet vieja / rango "
                "histórico que no llega hasta fechas recientes?).",
                ticker, len(headlines), nonzero_scores, _NEWS_MERGE_TOLERANCE_DAYS,
            )

    # V8 — suavizado EMA regime-aware (ver constantes arriba): el canario de
    # cobertura recién evaluado corre SIEMPRE sobre la señal cruda (`aligned`
    # sin suavizar) para no enmascarar un scorer/merge realmente muerto tras
    # una EMA; el suavizado se aplica acá, al final, solo sobre lo que
    # efectivamente entra al tensor.
    ema_span = NEWS_SENTIMENT_EMA_SPAN_CRYPTO if _is_crypto_ticker(ticker) else NEWS_SENTIMENT_EMA_SPAN_EQUITY
    if ema_span > 1:
        aligned = aligned.ewm(span=ema_span, adjust=False).mean().rename("NEWS_SENTIMENT_SCORE")

    return aligned