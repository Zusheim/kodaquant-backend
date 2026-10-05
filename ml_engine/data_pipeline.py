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

V10 (auditoría "100% neutro" AMZN/META, post-rollback V23.6 equity):
`DDG_REGION` default venía en `"es-es"` -- sesga `ddgs.news(...)` hacia
resultados LOCALIZADOS a España (con mayor probabilidad de titulares en
español) para consultas de equities 100% US-listed (`"{TICKER} stock"`).
`ProsusAI/finbert` (`_score_text`) es un clasificador ENTRENADO
EXCLUSIVAMENTE en texto financiero en inglés (Financial PhraseBank) -- un
titular en español queda fuera de su distribución de entrenamiento y
tiende a colapsar a la clase "neutral" con score de confianza alto (no es
un titular sin opinión: es el modelo adivinando entre 3 clases sobre texto
que nunca vio en ese idioma). Esto es indistinguible, aguas abajo, del
caso legítimo "titular real sin carga direccional" -- exactamente el
síntoma reportado (9 titulares obtenidos para META, 0 warnings de
etiqueta-no-reconocida, score 100% neutro de todas formas: los titulares
SÍ llegaban y SÍ eran clasificables, solo que en el idioma equivocado para
este scorer). Fix: default cambiado a `"us-en"` (mercado real de todo el
universo de `equity_specialist`: AAPL/MSFT/NVDA/TSLA/GOOGL/AMZN/META/SPY,
todos NYSE/NASDAQ) -- sigue siendo 100% override-able vía env var
`DDG_REGION` si el operador necesita otra región por algún motivo. NO
toca `quanti_engine.py::DDG_REGION` (constante del mismo nombre pero
CONSUMIDOR DISTINTO: alimenta el RAG conversacional de Quanti para
usuarios que preguntan en español, vía `search_financial_web` -> Groq,
nunca pasa por FinBERT -- ahí "es-es" sigue siendo la región correcta y
no se toca).

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

V11 (auditoría rescate 2026-08 -- "esto tiene que quedar perfecto"): 4 puntos
menores detectados en una corrida real de producción, ninguno bloqueante,
todos con fix real:
    a) COLISIÓN DE ENV VAR (el único con impacto real en la señal): este
       módulo y `quanti_engine.py` compartían el nombre `DDG_REGION` para
       DOS propósitos incompatibles (RAG conversacional en español vs.
       scoring FinBERT inglés-únicamente) -- fijar `DDG_REGION=es-es` para
       el chat de Quanti reintroducía en silencio el bug "100% neutro" que
       V10 creía cerrado. Fix: variable dedicada `NEWS_SENTIMENT_DDG_REGION`.
    b) Filtro de idioma `langdetect` (opcional, gratis, sin key) como
       defensa en profundidad DESPUÉS del fix de región -- poda la cola
       larga de titulares fuera de idioma que un agregador global puede
       colar incluso con la región correcta.
    c) Reintento con query más amplia + fallback a `/news?category=general`
       de Finnhub para ETFs (SPY, QQQ...) que nunca tienen perfil de
       compañía en `/company-news` y antes se quedaban sin ninguna señal
       de prensa cuando DDGS tampoco rendía para esa query específica.
    d) `FutureWarning` cosmético de `huggingface_hub` (resume_download)
       silenciado de forma acotada -- no afectaba nada, solo ruido de log.

V12 (auditoría rescate 2026-08 -- crash real en despliegue HF Space):
`ModuleNotFoundError: No module named 'langdetect'` seguido de
`NameError: name 'logger' is not defined`, ambos en import-time de este
módulo (tumbaba todo el proceso, no solo esta feature). Causa raíz: el
bloque `try/except ImportError` de `langdetect` (V11, arriba) llamaba
`logger.warning(...)` en su rama `except`, pero `logger =
logging.getLogger(...)` se definía DEBAJO de ese bloque -- el except SÍ
atrapaba el `ModuleNotFoundError` real (es subclase de `ImportError`),
pero al intentar loguearlo lanzaba un `NameError` sin atrapar, que se
propagaba y abortaba la carga del módulo completo. Fix real: `logger` se
define ANTES de cualquier bloque que pueda loguear en su rama except
(orden, no manejo de excepciones nuevo). `langdetect` además se agrega a
requirements.txt (ver ese archivo) -- puro-Python, gratis, sin key -- para
que la defensa en profundidad V11 quede activa en producción en vez de
degradarse siempre al no-op silencioso.

V13 (auditoría rescate 2026-08-31 -- re-verificación post-reporte de
`ModuleNotFoundError`/`NameError` en este mismo módulo): re-auditado línea
por línea contra el reporte. Ambos síntomas YA estaban resueltos por V12
(arriba) en esta versión del archivo -- `logger` se define en la línea
correcta (antes del primer `except` que lo usa) y `langdetect==1.0.9` ya
está pineado en requirements.txt. Confirmado también con `pyflakes` (cero
warnings) y `ast.parse`/`py_compile` (sintaxis válida) sobre el archivo
completo -- CERO cambios funcionales en este módulo; el crash reportado no
reproduce contra este código fuente. Ver requirements.txt (RONDA 7) para
el crash real de esta corrida (`numpy`/`tensorflow`, módulo distinto).
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

# V11 (auditoría 2026-08) -- `huggingface_hub` emite un FutureWarning
# cosmético ("resume_download is deprecated") en CADA carga de FinBERT vía
# `transformers.pipeline`, sin relación con ningún bug real de este módulo
# ni con la calidad de NEWS_SENTIMENT_SCORE. Acotado por `module=` a la
# librería exacta que lo emite -- nunca un `warnings.filterwarnings("ignore")`
# global, que enmascararía warnings reales de otras dependencias.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"huggingface_hub\..*")

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
logger = logging.getLogger("kodaquant.data_pipeline")

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover
    try:
        from duckduckgo_search import DDGS  # type: ignore[no-redef]
    except ImportError:
        DDGS = None  # type: ignore[assignment,misc]  # se valida en runtime, ver _fetch_headlines_ddg

# V11 (auditoría 2026-08 — colisión de nombre de env var, root cause real
# del "100% neutro" que V10 creyó haber cerrado): V10 cambió el DEFAULT de
# este módulo a "us-en", pero seguía leyendo la MISMA variable de entorno
# `DDG_REGION` que usa `quanti_engine.py::DDG_REGION` para el buscador
# conversacional de Quanti (default "es-es", CORRECTO ahí -- los usuarios
# le hablan en español). Cualquier operador que fije `DDG_REGION=es-es` en
# su .env/entorno (lo natural para un producto en español) pisa SIN
# querer el default de ESTE módulo también -- las dos constantes comparten
# nombre pero NO deberían compartir valor: una alimenta un LLM que lee
# español perfectamente, la otra alimenta FinBERT, que es inglés-únicamente.
# Fix real: variable de entorno dedicada y jamás compartida con
# quanti_engine.py. Quien necesite tocar la región de este módulo usa
# `NEWS_SENTIMENT_DDG_REGION`, nunca `DDG_REGION`.
DDG_REGION = os.getenv("NEWS_SENTIMENT_DDG_REGION", "us-en")
DDG_SAFESEARCH = os.getenv("DDG_SAFESEARCH", "moderate")
MAX_DDG_HEADLINES = 10  # techo por-ticker -- Finnhub complementa si esto no alcanza

# V11 — defensa en profundidad además del fix de región de arriba: agregadores
# de noticias globales pueden devolver un titular fuera de idioma incluso con
# la región correcta (ej. un medio hispano indexado igual bajo "us-en"). Un
# titular no-inglés que llega a FinBERT no lanza NINGÚN warning (ver V10) y
# se pierde como "neutral" falso -- indistinguible de una lectura neutra real.
# `langdetect` es puro-Python, gratuito, sin key, determinístico con seed fija
# -- se usa como filtro PREVIO al scorer, no como reemplazo del fix de región
# (más barato: si la región ya resuelve el 95% de los casos, este filtro solo
# poda la cola larga). Si no está instalado, se degrada a no-op con un único
# warning (nunca bloquea el pipeline).
try:
    from langdetect import DetectorFactory as _LangDetectorFactory, detect_langs as _langdetect_detect_langs
    _LangDetectorFactory.seed = 0  # determinístico -- sin esto, langdetect no es reproducible corrida a corrida
    _LANGDETECT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LANGDETECT_AVAILABLE = False
    logger.warning(
        "langdetect no instalado (`pip install langdetect`) -- se omite el filtro de idioma "
        "de defensa en profundidad; NEWS_SENTIMENT_DDG_REGION sigue siendo la defensa primaria."
    )

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
# RONDA 10 -- sentinela de FALLO cacheado (str del motivo, o None si aún no
# se intentó / tuvo éxito). Antes solo se cacheaba el ÉXITO (`_finbert_pipeline`)
# -- un fallo NUNCA quedaba memorizado, así que CADA titular individual
# repetía el import + intento de carga completo desde cero y volvía a
# producir el traceback entero -- la causa real de las "cientos de líneas
# de error" reportadas, no un crash único. Con esto: se diagnostica UNA
# vez, se loguea el traceback real completo UNA vez, y todo el resto del
# proceso degrada a NEUTRAL_SCORE con un mensaje corto de una línea.
_finbert_unavailable_reason: Optional[str] = None
# Protege la carga perezosa contra llamadas CONCURRENTES: bajo Uvicorn,
# get_daily_news_sentiment se dispara desde varios hilos a la vez (Plan A/Plan B
# en paralelo, radar escaneando el universo completo vía ThreadPoolExecutor).
# Sin lock, el primer request real puede disparar 2+ hilos importando
# torch/transformers simultáneamente -> ImportError transitorio de import
# parcial/circular, indistinguible de "no instalado" sin este fix.
_finbert_init_lock = threading.Lock()
# RONDA 10 -- piso REAL de transformers 5.x, verificado contra la guía de
# migración oficial de la librería (Python 3.10+, PyTorch 2.4+,
# huggingface_hub<2.0,>=1.3.0) -- NO 2.5 como asumía una auditoría previa.
_TRANSFORMERS_MIN_TORCH = (2, 4)

# --- Backend ONNX (camino PRIMARIO, torch-independiente) -------------------
_FINBERT_ONNX_REPO = os.getenv("KODAQUANT_FINBERT_ONNX_REPO", "Xenova/finbert").strip()
# V27 -- default cambiado de _FINBERT_MODEL_NAME (ProsusAI/finbert, sin
# export ONNX -> 404 permanente) a Xenova/finbert (mismo modelo base,
# mismas labels -- config.json id2label idéntico, verificado). Confirmado
# en producción: "[FinBERT/ONNX] sesión activa (Xenova/finbert ...)".
# KODAQUANT_FINBERT_ONNX_REPO sigue pisando esto si se necesita otro repo.
_FINBERT_ONNX_FILE = os.getenv("KODAQUANT_FINBERT_ONNX_FILE", "onnx/model.onnx").strip()
_FINBERT_ONNX_MAXLEN = 256
_finbert_onnx = None                       # tuple(InferenceSession, Tokenizer, input_names)
_finbert_onnx_unavailable: Optional[str] = None
_finbert_onnx_lock = threading.Lock()


def _load_finbert_tokenizer(hf_hub_download):
    """tokenizer.json (fast) con fallback a vocab.txt (WordPiece clásico)."""
    from tokenizers import Tokenizer
    try:
        return Tokenizer.from_file(hf_hub_download(_FINBERT_ONNX_REPO, "tokenizer.json"))
    except Exception:
        from tokenizers import BertWordPieceTokenizer
        vocab = hf_hub_download(_FINBERT_ONNX_REPO, "vocab.txt")
        return BertWordPieceTokenizer(vocab, lowercase=True)


def _get_finbert_onnx():
    """Sesión ONNX + tokenizer (singleton thread-safe). Cero torch, cero TF.

    Esquiva `transformers.pipeline()` por completo -- ahí vive la NameError
    de la compuerta torch>=2.4 diagnosticada en RONDA 10, que NINGÚN kwarg
    de esta capa puede evitar.
    """
    global _finbert_onnx, _finbert_onnx_unavailable
    if _finbert_onnx is not None:
        return _finbert_onnx
    if _finbert_onnx_unavailable is not None:
        raise FinBERTUnavailableError(_finbert_onnx_unavailable)

    with _finbert_onnx_lock:
        if _finbert_onnx is not None:
            return _finbert_onnx
        if _finbert_onnx_unavailable is not None:
            raise FinBERTUnavailableError(_finbert_onnx_unavailable)
        try:
            import numpy as _np
            import onnxruntime as _ort
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            _finbert_onnx_unavailable = (
                f"[FinBERT/ONNX] backend no instalado ({type(exc).__name__}: {exc}). "
                "`pip install onnxruntime tokenizers` en el MISMO venv."
            )
            logger.warning(_finbert_onnx_unavailable)
            raise FinBERTUnavailableError(_finbert_onnx_unavailable) from exc

        try:
            onnx_path = hf_hub_download(_FINBERT_ONNX_REPO, _FINBERT_ONNX_FILE)
            tokenizer = _load_finbert_tokenizer(hf_hub_download)
            opts = _ort.SessionOptions()
            opts.intra_op_num_threads = 1          # ZeroGPU / Uvicorn multihilo
            opts.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = _ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
            input_names = {i.name for i in session.get_inputs()}
            _finbert_onnx = (session, tokenizer, input_names, _np)
            logger.info("[FinBERT/ONNX] sesión activa (%s :: %s, inputs=%s).",
                        _FINBERT_ONNX_REPO, _FINBERT_ONNX_FILE, sorted(input_names))
        except Exception as exc:
            _finbert_onnx_unavailable = (
                f"[FinBERT/ONNX] no se pudo abrir '{_FINBERT_ONNX_REPO}::{_FINBERT_ONNX_FILE}' "
                f"({type(exc).__name__}: {exc}). Si es 404, ese repo no publica export ONNX -- "
                "exportalo una vez y fijá KODAQUANT_FINBERT_ONNX_REPO/_FILE."
            )
            logger.warning(_finbert_onnx_unavailable)
            raise FinBERTUnavailableError(_finbert_onnx_unavailable) from exc
    return _finbert_onnx


def _score_text_onnx(text: str) -> float:
    session, tokenizer, input_names, _np = _get_finbert_onnx()
    enc = tokenizer.encode(text)
    ids = enc.ids[:_FINBERT_ONNX_MAXLEN]
    mask = (enc.attention_mask[:_FINBERT_ONNX_MAXLEN]
            if getattr(enc, "attention_mask", None) else [1] * len(ids))
    feed = {"input_ids": _np.asarray([ids], dtype=_np.int64),
            "attention_mask": _np.asarray([mask], dtype=_np.int64)}
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = _np.zeros((1, len(ids)), dtype=_np.int64)
    logits = _np.asarray(session.run(None, {k: v for k, v in feed.items()
                                            if k in input_names})[0][0], dtype=_np.float64)
    exp = _np.exp(logits - logits.max())
    probs = exp / exp.sum()
    # id2label oficial de ProsusAI/finbert: 0=positive, 1=negative, 2=neutral
    return float(probs[0] - probs[1])


def news_sentiment_available() -> bool:
    """Probe determinista -- ¿hay ALGÚN backend capaz de puntuar sentimiento?

    Consumido por `train_kodaquant_v5.run_regime_pipeline` para decidir, UNA
    vez y antes de construir el tensor, si NEWS_SENTIMENT_SCORE entra o se
    excluye. Inyectar 0.0 constante es peor que excluirla: el feature_scaler
    se ajusta sobre una constante y la red aprende un slot muerto.
    """
    try:
        _get_finbert_onnx()
        return True
    except Exception:
        pass
    try:
        _get_finbert_pipeline()
        return True
    except Exception:
        return False


class FinBERTUnavailableError(ImportError):
    """RONDA 10 -- excepción DEDICADA para el caso 'ya diagnosticado, no
    reintentable en este proceso' (piso de torch no alcanzado / backend TF
    inexistente en transformers 5.x). `_score_text` la distingue de un
    ImportError genérico para loguear corto (sin traceback repetido) en
    vez de con `exc_info=True` en cada titular -- ver ahí."""


def _get_finbert_pipeline():
    """Carga perezosa (singleton, thread-safe) del pipeline FinBERT — CPU-safe para runners sin GPU.

    RONDA 10 (fix REAL, reemplaza el intento fallido de RONDA 6-9) --
    diagnóstico verificado contra el TRACEBACK REAL producido en Mac Intel
    (no solo contra el código fuente): el intento anterior de "fallback a
    backend TensorFlow" era imposible de raíz, no un bug de esta función.
    Evidencia: el segundo intento (`framework="tf"`) reventó con la
    EXACTA MISMA `NameError: name 'torch' is not defined` en la EXACTA
    MISMA línea (`transformers/pipelines/__init__.py::pipeline`, chequeo
    `isinstance(dtype, str) and hasattr(torch, dtype)`) que el primero --
    esa línea corre ANTES de cualquier rama específica de framework, así
    que reintentar con otro framework nunca podía esquivarla. Confirmado
    además contra la guía de migración oficial de transformers 5.x: la
    serie 5.x eliminó el backend TensorFlow POR COMPLETO y sin período de
    gracia (PyTorch es el único backend soportado) -- no había ninguna
    rama TF real a la que caer, el `except NameError: ... framework="tf"`
    de las rondas previas apuntaba a código que ya no existe en la
    librería instalada (`transformers==5.15.0`, ver requirements.txt).
    El piso real es torch>=2.4 (Python 3.10+/huggingface_hub<2.0,>=1.3.0
    son los otros dos). En macOS Intel (x86_64) NINGÚN wheel de `torch`
    >2.2.2 existe en PyPI -- techo real y permanente de esa plataforma, no
    arreglable con más código. Pinear `transformers` a la serie 4.x (que
    SÍ soportaba TF) NO es salida: esa serie exige `huggingface-hub<1.0`
    en TODAS sus versiones (confirmado en vivo, incl. el último release
    4.57.x), lo que choca de frente con `huggingface-hub<2.0,>=1.16.0` que
    ya exige `gradio[oauth,mcp]==6.22.0` en el MISMO requirements.txt --
    `ResolutionImpossible` real, no evitable sin romper el build de HF
    Spaces (`gradio` no está platform-split, no puede estarlo: `sdk_version`
    del README.md debe calzar exacto para TODA plataforma).
    Conclusión honesta: FinBERT vía `transformers.pipeline()` es
    IRRECUPERABLE en Mac Intel local con esta cadena de dependencias --
    no es arreglable sin (a) un Mac Apple Silicon/Linux real, o (b)
    remover `gradio[mcp]` del proyecto para poder bajar a transformers 4.x.
    Producción (HF Space, Linux, `torch==2.11.0`) NO está afectada -- ahí
    `torch_ver >= _TRANSFORMERS_MIN_TORCH` es verdadero y el pipeline
    PyTorch carga y sirve FinBERT completo, sin ninguna rama de este fix
    involucrada. Este fix, entonces, es sobre CALIDAD DE FALLO en Mac
    Intel (un diagnóstico honesto, una sola vez, degrade limpio a
    NEUTRAL_SCORE) -- no sobre "hacer funcionar" algo que la librería ya
    no soporta en esa plataforma.
    """
    global _finbert_pipeline, _finbert_unavailable_reason
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    if _finbert_unavailable_reason is not None:
        raise FinBERTUnavailableError(_finbert_unavailable_reason)

    with _finbert_init_lock:
        if _finbert_pipeline is not None:  # doble check: otro hilo ya lo cargó mientras esperábamos el lock
            return _finbert_pipeline
        if _finbert_unavailable_reason is not None:  # ídem, otro hilo ya lo diagnosticó como inviable
            raise FinBERTUnavailableError(_finbert_unavailable_reason)

        try:
            import torch
            from transformers import pipeline as _hf_pipeline
            from transformers.utils import is_torch_available
        except ModuleNotFoundError as exc:
            # Caso real de "no instalado": el paquete no existe en este intérprete.
            _finbert_unavailable_reason = (
                f"torch/transformers no están instalados en {sys.executable}: {exc}. "
                "Ejecuta `pip install torch transformers` en el MISMO venv que corre Uvicorn."
            )
            raise ImportError(_finbert_unavailable_reason) from exc
        except ImportError as exc:
            # CUALQUIER OTRO ImportError (símbolo nativo, dlopen roto, import
            # circular parcial por carrera entre hilos, etc.) — antes se
            # enmascaraba con el mismo mensaje genérico de "no instalado".
            # Se preserva el tipo y mensaje reales para que el log no mienta.
            _finbert_unavailable_reason = (
                f"torch/transformers están instalados pero fallaron al importar "
                f"({type(exc).__name__}: {exc}). NO es un paquete faltante — revisa "
                "conflictos nativos (runtimes OpenMP duplicados TF/PyTorch en macOS, "
                "ver KMP_DUPLICATE_LIB_OK) o reinstala ambos paquetes emparejados."
            )
            raise ImportError(_finbert_unavailable_reason) from exc

        _ver_match = re.match(r"(\d+)\.(\d+)", torch.__version__)
        torch_ver = tuple(int(p) for p in _ver_match.groups()) if _ver_match else (0, 0)
        if not is_torch_available() or torch_ver < _TRANSFORMERS_MIN_TORCH:
            _finbert_unavailable_reason = (
                f"[FinBERT] no disponible en este proceso: torch {torch.__version__} "
                f"instalado, transformers {__import__('transformers').__version__} exige "
                f"torch>={'.'.join(map(str, _TRANSFORMERS_MIN_TORCH))} y ELIMINÓ el backend "
                "TensorFlow por completo en la serie 5.x (verificado: framework='tf' dispara "
                "la MISMA NameError compartida en pipeline() antes de cualquier rama TF real "
                "-- no hay fallback posible). En macOS Intel ningún wheel de torch >2.2.2 "
                "existe en PyPI (techo real de plataforma). NEWS_SENTIMENT_SCORE degradará a "
                "NEUTRAL_SCORE de forma consistente en este proceso -- producción "
                "(Linux, torch==2.11.0) no está afectada, ahí el piso SÍ se cumple."
            )
            logger.error(_finbert_unavailable_reason)
            raise FinBERTUnavailableError(_finbert_unavailable_reason)

        device = 0 if torch.cuda.is_available() else -1
        try:
            _finbert_pipeline = _hf_pipeline(
                "sentiment-analysis", model=_FINBERT_MODEL_NAME,
                tokenizer=_FINBERT_MODEL_NAME, truncation=True, device=device,
                dtype=None,  # esquiva `isinstance(dtype, str) and hasattr(torch, dtype)` (ver docstring)
            )
        except Exception as exc:
            _finbert_unavailable_reason = (
                f"[FinBERT] pipeline PyTorch falló pese a pasar el check de versión/"
                f"is_torch_available() ({type(exc).__name__}: {exc})."
            )
            logger.error(_finbert_unavailable_reason, exc_info=True)
            raise FinBERTUnavailableError(_finbert_unavailable_reason) from exc
    return _finbert_pipeline

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
# V6 (auditoría "señal muerta" crypto_specialist) — `/company-news` es
# EXCLUSIVO para equities listadas en EE. UU. (doc oficial Finnhub); jamás
# devuelve resultados para tickers cripto. `/news?category=crypto` es el
# endpoint correcto para ese universo — ver `_fetch_headlines_finnhub`.
FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"
FINNHUB_CRYPTO_NEWS_URL = "https://finnhub.io/api/v1/news"
# V11 -- MISMO endpoint que FINNHUB_CRYPTO_NEWS_URL (Finnhub `/news` acepta
# `category` general/forex/crypto/merger); alias legible para el fallback de
# ETFs (`_fetch_headlines_finnhub`, ver auditoría 2026-08), nunca un endpoint nuevo.
FINNHUB_GENERAL_NEWS_URL = FINNHUB_CRYPTO_NEWS_URL

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
    if _finbert_onnx is not None or _finbert_onnx_unavailable is None:
        try:
            return _score_text_onnx(text[:2048])
        except FinBERTUnavailableError:
            pass   # diagnosticado UNA vez en _get_finbert_onnx -> cae a torch
        except Exception as exc:  # noqa: BLE001
            logger.warning("[FinBERT/ONNX] fallo de inferencia (%s: %s) -> fallback torch.",
                           type(exc).__name__, exc)
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
    except FinBERTUnavailableError:
        # RONDA 10 -- caso YA diagnosticado con traceback completo UNA vez
        # dentro de `_get_finbert_pipeline` (logger.error/exc_info=True ahí).
        # Repetir el traceback completo por CADA titular es exactamente la
        # causa real de las "cientos de líneas de error" reportadas -- acá
        # alcanza con un rastro corto, sin traceback, por titular.
        logger.debug("FinBERT no disponible en este proceso (ver diagnóstico previo) -> NEUTRAL_SCORE.")
        return NEUTRAL_SCORE
    except Exception as exc:  # noqa: BLE001 — fallo de inferencia del modelo local, causa NO diagnosticada aún
        # exc_info=True: antes se logueaba solo %r del error (a veces ya
        # reescrito con un mensaje genérico), sin traceback — la causa real
        # (incl. __cause__ encadenado) quedaba invisible en los logs de
        # Uvicorn. Ahora se propaga completa -- SOLO para fallos nuevos/no
        # cacheados (ver FinBERTUnavailableError arriba para el caso ya
        # conocido).
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


# V11 -- umbral calibrado empíricamente (ver auditoría 2026-08): `detect()`
# ingenuo (top-1, sin probabilidad) da FALSOS POSITIVOS reales en titulares
# financieros cortos -- "Tesla stock drops on delivery miss" clasificó como
# sueco con `detect()`. `detect_langs()` expone la distribución de
# probabilidad completa; frases reales en español/inglés (>=20 chars)
# resuelven con >0.999 de confianza al idioma correcto, mientras que el
# ruido de titulares cortos ambiguos rara vez supera 0.97 -- de ahí el
# doble filtro (confianza alta Y longitud mínima) en vez de un simple
# top-1, que habría descartado titulares en inglés legítimos.
_LANGDETECT_MIN_CONFIDENCE = 0.97
_LANGDETECT_MIN_LENGTH = 20


def _is_english_headline(title: str) -> bool:
    """
    Defensa en profundidad (V11) -- filtro de idioma PREVIO al scorer.
    `True` (se deja pasar a FinBERT) salvo que `langdetect` esté MUY seguro
    (`prob >= _LANGDETECT_MIN_CONFIDENCE`) de que el idioma NO es inglés,
    sobre un titular con longitud suficiente para que esa confianza sea
    confiable (`len >= _LANGDETECT_MIN_LENGTH`) -- en cualquier otro caso
    (ambiguo, corto, o `langdetect` no instalado) se deja pasar: este
    filtro poda la cola larga de titulares CLARAMENTE no-ingleses, nunca
    reemplaza el fix de región (`NEWS_SENTIMENT_DDG_REGION`).
    """
    if not _LANGDETECT_AVAILABLE or not title or len(title) < _LANGDETECT_MIN_LENGTH:
        return True
    try:
        top = _langdetect_detect_langs(title)[0]
    except Exception:  # noqa: BLE001 -- langdetect lanza LangDetectException en texto ambiguo
        return True
    return not (top.lang != "en" and top.prob >= _LANGDETECT_MIN_CONFIDENCE)


def _fetch_headlines_ddg(ticker: str, max_results: int = MAX_DDG_HEADLINES) -> list[dict]:
    """
    Fuente primaria — búsqueda de noticias vía DDGS (`ddgs.news`, JSON nativo
    de DuckDuckGo, sin scraping de portales individuales, sin API key).
    Reemplaza 1:1 el rol de `yfinance.Ticker(ticker).news` (retirado, ver
    services/market_data.py): misma firma de retorno (`title`/`published_at`),
    mismo lugar en `fetch_recent_headlines` (primaria, Finnhub complementa).

    V11 (auditoría 2026-08) -- dos mejoras sobre la fuente primaria:
      (a) reintento con query MÁS AMPLIA si la query específica (`"{ticker}
          stock"`) devuelve 0 resultados -- tickers de baja cobertura de
          prensa dedicada (ETFs como SPY, sin "compañía" detrás) rendían
          "No results found" en vez de degradarse a Finnhub con más señal.
      (b) filtro de idioma (`_is_english_headline`) ANTES de devolver los
          titulares -- ver V11 arriba.
    """
    if DDGS is None:
        logger.warning("DDGS no instalado (`pip install ddgs`) -- fuente primaria de noticias deshabilitada; solo Finnhub (si hay API key).")
        return []

    # Cripto: "-USD" es la convención de símbolo de todo el proyecto
    # (REGIME_TICKERS/MACRO_SYMBOL_MAP), no específica de ningún proveedor
    # -- se usa solo el símbolo base (ej. "BTC") para una query de noticias
    # más natural que "BTC-USD".
    is_crypto = _is_crypto_ticker(ticker)
    base_symbol = ticker.split("-")[0]
    queries = [f"{base_symbol} crypto"] if is_crypto else [f"{ticker} stock", f"{ticker} ETF", base_symbol]

    raw: list[dict] = []
    for query in queries:
        try:
            with DDGS() as ddgs:
                raw = list(
                    ddgs.news(query, region=DDG_REGION, safesearch=DDG_SAFESEARCH, max_results=max_results)
                )
        except Exception as exc:  # noqa: BLE001 -- red/HTTP transitorio de DDG
            logger.warning("DDGS.news falló para %s (query=%r): %r", ticker, query, exc)
            continue
        if raw:
            break  # primera query con resultados gana -- no seguimos ampliando innecesariamente

    headlines = []
    dropped_non_english = 0
    for item in raw:
        title = (item.get("title") or "").strip()
        published_at = _normalize_timestamp(item.get("date"))
        if not title or published_at is None:
            continue
        if not _is_english_headline(title):
            dropped_non_english += 1
            continue
        headlines.append({"title": title, "published_at": published_at})
    if dropped_non_english:
        logger.debug("%s: %d titular(es) descartado(s) por filtro de idioma (no-inglés).", ticker, dropped_non_english)
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

    # `filter_clientside_by_date`: True para cualquier llamada al endpoint
    # genérico `/news` (crypto O general), que NO acepta `from`/`to`
    # server-side -- `/company-news` sí los acepta (ver `params` arriba),
    # así que ese caso no necesita el recorte extra.
    headlines = _fetch_finnhub_json(url, params, ticker, filter_clientside_by_date=is_crypto, since=since)

    # V11 (auditoría 2026-08) -- `/company-news` es EXCLUSIVO de compañías
    # con perfil en Finnhub; ETFs (SPY, QQQ, ...) NUNCA tienen uno y
    # devuelven [] SIEMPRE, sin importar cuántos reintentos. Fallback a
    # `/news?category=general` (mercado agregado) filtrado client-side por
    # el símbolo/nombre del ticker en el titular -- mejor que dejar el
    # activo entero sin ninguna señal de prensa cuando DDGS tampoco rindió.
    if not headlines and not is_crypto:
        general_params = {"category": "general", "token": FINNHUB_API_KEY}
        general = _fetch_finnhub_json(FINNHUB_GENERAL_NEWS_URL, general_params, ticker,
                                       filter_clientside_by_date=True, since=since)
        needle = ticker.upper()
        headlines = [h for h in general if needle in h["title"].upper()]

    return headlines


def _fetch_finnhub_json(url: str, params: dict, ticker: str, filter_clientside_by_date: bool, since) -> list[dict]:
    """Request GET con reintentos -- extraído de `_fetch_headlines_finnhub` para
    reutilizarlo también en el fallback `/news?category=general` de ETFs.
    `filter_clientside_by_date`: True para endpoints sin `from`/`to`
    server-side (`/news`, crypto o general) -- recorta del lado del cliente
    contra `since` para no arrastrar noticias arbitrariamente viejas."""
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
                if filter_clientside_by_date and published_at is not None and published_at.date() < since:
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
                "obtenidos puntuaron EXACTAMENTE 0.0 en el scorer -> revisar, "
                "en orden: (1) warnings de '_score_text'/etiqueta FinBERT no "
                "reconocida arriba en el log; (2) idioma real de esos "
                "titulares -- ProsusAI/finbert es inglés-únicamente, un "
                "titular no-inglés puntúa 'neutral' SIN lanzar ningún warning "
                "(ver V10, DDG_REGION); el merge de fechas es irrelevante, la "
                "señal ya nace muerta antes de tocar cualquier lógica temporal.",
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