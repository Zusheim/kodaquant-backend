"""
kodaquant_core.py — KodaQuant V15: Núcleo Compartido Train/Inferencia
========================================================================
Consolidación arquitectónica (auditoría CTO, 2026-08-16). Antes de este
módulo, tres piezas de lógica CRÍTICA vivían duplicadas e independientes
entre el pipeline de entrenamiento (`train_kodaquant_v5.py`) y el de
inferencia (`quanti_engine.py`/`market_data.py`), con implementaciones
DIVERGENTES:

  1. CACHÉ EN DISCO — `market_data.py` (inferencia) cacheaba JSON con TTL
     de horas (`_cache_get`/`_cache_set`); `train_kodaquant_v5.py`
     (entrenamiento) cacheaba Parquet con staleness por antigüedad de
     fecha (`_cache_is_stale`/`download_all`) + un purgado manual separado
     (`purge_data_cache`). Dos formatos, dos políticas de expiración, cero
     purga automática. `CacheManager` unifica ambos modos (JSON de vida
     corta para respuestas de API en vivo, Parquet de vida larga para
     descargas históricas masivas) bajo una sola clase, con purga
     automática local (`purge_stale`) y purga total explícita
     (`purge_all`, reemplazo directo de `purge_data_cache`).

  2. CLIENTE FRED — vivía únicamente en `market_data.py`
     (`_fred_series_daily`), usado SOLO en inferencia. `train_kodaquant_v5.py`
     jamás lo importaba: el entrenamiento resolvía `^TNX` con el
     `yf.download` genérico, mientras que la inferencia ya lo resolvía vía
     FRED (serie oficial `DGS10`) con fallback a Stooq -- exactamente la
     discrepancia "train ve una fuente, infiere con otra" reportada. Se
     porta acá como `FredClient`, reutilizable desde ambos lados.

  3. ALINEACIÓN DE CALENDARIO MACRO (5/7) CONTRA UN ACTIVO 24/7 — el fix
     real ya existía, pero solo del lado de inferencia
     (`market_data._macro_close_ffilled_to_daily_calendar`, fix
     2026-08-15): upsamplear cada serie macro a un calendario diario
     CORRIDO (fin de semana incluido) y hacer forward-fill DENTRO de esa
     serie, ANTES de reindexarla contra el calendario del activo objetivo.
     `engineer_asset()` (entrenamiento) seguía dependiendo de un
     `df.ffill()` ciego sobre el frame YA combinado, que no tiene la misma
     garantía en huecos que no calzan por orden de columnas ni en el
     arranque de la serie. `align_macro_to_calendar` es ahora la ÚNICA
     implementación de esta lógica; train e inferencia la importan por
     igual -- ver `data_pipeline.py`/`train_kodaquant_v5.py`.

DESPLIEGUE EN DOS DIRECTORIOS (sin romper la topología actual del repo):
este archivo debe copiarse tal cual a AMBAS ubicaciones que hoy contienen
copias paralelas de `data_pipeline.py`/`market_data.py`:
    - `services/kodaquant_core.py`   (consumido por quanti_engine.py,
      import de paquete: `from services.kodaquant_core import ...`)
    - <dir. de entrenamiento>/kodaquant_core.py  (consumido por
      train_kodaquant_v5.py/train_pipeline.py, import plano:
      `from kodaquant_core import ...`)
Cero dependencias externas nuevas: solo `pandas`/`numpy`/`requests`, ya
pineados en ambos entornos. `data_pipeline.py`, `train_kodaquant_v5.py` y
`quanti_engine.py` importan este módulo con el mismo patrón dual
try/except ya usado en el repo para `ddgs`/`duckduckgo_search` (ver
`data_pipeline.py`), así que una sola copia de este archivo sirve para
ambas ubicaciones sin ninguna otra modificación.

PENDIENTE FUERA DE ESTE ALCANCE (no es uno de los 4 artefactos
entregados, se deja documentado para no perder el hilo): `market_data.py`
sigue con su propio `_fred_series_daily`/`_cache_get`/`_cache_set`
inline -- migrarlo para que importe `FredClient`/`CacheManager` de acá
es un cambio mecánico de una sola función, recomendado como siguiente
paso para que TODA la superficie del repo, no solo train, comparta el
mismo núcleo.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger("kodaquant.core")


# ===========================================================================
# 1) CACHE MANAGER — disco unificado (JSON TTL-corto + Parquet TTL-largo),
#    thread-safe, con purga automática local.
# ===========================================================================
class CacheManager:
    """
    Caché en disco de dos modos, un solo punto de verdad:

      - `get_json`/`set_json`   -> payloads pequeños de vida corta (Close
        histórico de un símbolo, headlines de noticias). Expira por TTL
        absoluto en segundos (`default_ttl_seconds`), igual semántica que
        el cache JSON que ya usaba `market_data.py`.
      - `get_dataframe`/`set_dataframe` -> descargas masivas (OHLCV
        multi-símbolo de `download_all`). Expira por ANTIGÜEDAD DEL DATO
        (última fecha en el índice vs. `datetime.now()`), no por edad del
        archivo -- misma semántica que `_cache_is_stale` en
        `train_kodaquant_v5.py`: un parquet escrito hace 5 minutos pero
        cuya fila más reciente es de hace 3 días sigue estando "viejo".

    Cada instancia posee su PROPIO `cache_dir` -- entrenamiento e
    inferencia deben instanciar `CacheManager` con directorios distintos
    (no comparten el mismo caché en disco, cada uno con su propia purga),
    pero ambos usan la MISMA clase/lógica.
    """

    def __init__(self, cache_dir: Path | str, default_ttl_seconds: int = 6 * 3600,
                 default_max_age_hours: float = 48.0) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl_seconds = default_ttl_seconds
        self.default_max_age_hours = default_max_age_hours
        self._lock = threading.Lock()

    # -- helpers de path --------------------------------------------------
    def _safe_name(self, key: str) -> str:
        return key.replace("/", "_").replace(":", "_").replace("^", "caret_").replace(" ", "_")

    def _json_path(self, key: str) -> Path:
        return self.cache_dir / f"{self._safe_name(key)}.json"

    def _parquet_path(self, key: str) -> Path:
        return self.cache_dir / f"{self._safe_name(key)}.parquet"

    # -- modo JSON (TTL absoluto) ------------------------------------------
    def get_json(self, key: str, ttl_seconds: Optional[int] = None, ignore_ttl: bool = False):
        """
        Devuelve el payload cacheado (cualquier objeto JSON-serializable,
        incluye DataFrames vía orient="split") o `None` si no existe/expiró.
        `ignore_ttl=True` -- último recurso: devuelve lo último en disco sin
        importar el TTL (ej. Twelve Data Y Stooq caídos a la vez); marca
        `_stale_age_hours` en el resultado si el DataFrame lo soporta.
        """
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        path = self._json_path(key)
        try:
            with self._lock:
                if not path.exists():
                    return None
                payload = json.loads(path.read_text())
            age_s = time.time() - payload.get("_cached_at", 0)
            if not ignore_ttl and age_s > ttl:
                return None
            data = payload.get("data")
            kind = payload.get("_kind", "raw")
            if kind == "dataframe":
                df = pd.read_json(io.StringIO(data), orient="split")
                df.index = pd.to_datetime(df.index)
                if ignore_ttl and age_s > ttl:
                    df.attrs["_stale_age_hours"] = round(age_s / 3600, 1)
                return df
            return data
        except Exception as exc:  # noqa: BLE001 -- caché corrupta jamás tumba el caller
            logger.debug("[CacheManager] JSON ilegible para %s (%r) -- se ignora.", key, exc)
            return None

    def set_json(self, key: str, value) -> None:
        path = self._json_path(key)
        if isinstance(value, pd.DataFrame):
            payload = {"_kind": "dataframe", "data": value.to_json(orient="split", date_format="iso"),
                       "_cached_at": time.time()}
        else:
            payload = {"_kind": "raw", "data": value, "_cached_at": time.time()}
        try:
            with self._lock:
                path.write_text(json.dumps(payload))
        except Exception as exc:  # noqa: BLE001 -- fallo al escribir caché nunca tumba el caller
            logger.debug("[CacheManager] No se pudo escribir caché JSON %s (%r).", key, exc)

    # -- modo DataFrame/Parquet (staleness por fecha del dato) ------------
    def get_dataframe(self, key: str, max_age_hours: Optional[float] = None,
                       required_columns: Optional[set[str]] = None) -> Optional[pd.DataFrame]:
        """
        `None` si no existe, si le faltan columnas requeridas, o si su fila
        más reciente supera `max_age_hours` de antigüedad respecto a AHORA
        (no la edad del archivo) -- ver docstring de la clase.
        """
        max_age = self.default_max_age_hours if max_age_hours is None else max_age_hours
        path = self._parquet_path(key)
        if not path.exists():
            return None
        try:
            with self._lock:
                cached = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CacheManager] Parquet corrupto para %s (%r) -- se ignora.", key, exc)
            return None

        if required_columns and not required_columns.issubset(set(cached.columns)):
            return None
        if cached.empty:
            return None

        max_date = pd.Timestamp(cached.index.max())
        max_date = max_date.tz_localize("UTC") if max_date.tzinfo is None else max_date.tz_convert("UTC")
        age = datetime.now(timezone.utc) - max_date.to_pydatetime()
        if age.total_seconds() > max_age * 3600:
            logger.info("[CacheManager] %s obsoleto -- fecha más reciente %s tiene más de %.0fh -- recarga forzada.",
                        path.name, cached.index.max(), max_age)
            return None
        return cached

    def set_dataframe(self, key: str, df: pd.DataFrame) -> None:
        path = self._parquet_path(key)
        try:
            with self._lock:
                df.to_parquet(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CacheManager] No se pudo escribir parquet %s (%r).", key, exc)

    # -- purga automática local --------------------------------------------
    def purge_all(self) -> int:
        """Vacía TODO el caché en disco (JSON + Parquet) de este `cache_dir`. Reemplazo directo de `purge_data_cache`."""
        removed = list(self.cache_dir.glob("*.json")) + list(self.cache_dir.glob("*.parquet"))
        with self._lock:
            for f in removed:
                f.unlink(missing_ok=True)
        if removed:
            logger.warning("[CacheManager] Purga total -> %d archivo(s) purgado(s) de %s.", len(removed), self.cache_dir)
        return len(removed)

    def purge_stale(self, ttl_seconds: Optional[int] = None, max_age_hours: Optional[float] = None) -> int:
        """
        Purga automática local (Requerimiento 1): recorre `cache_dir` y
        elimina solo las entradas efectivamente vencidas, sin tocar
        entradas todavía válidas -- pensado para invocarse periódicamente
        (ej. al arrancar cada corrida) sin tirar caché sano.
        """
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        max_age = self.default_max_age_hours if max_age_hours is None else max_age_hours
        removed = 0
        now = time.time()
        for path in self.cache_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text())
                if now - payload.get("_cached_at", 0) > ttl:
                    path.unlink(missing_ok=True)
                    removed += 1
            except Exception:  # noqa: BLE001 -- JSON corrupto = candidato directo a purga
                path.unlink(missing_ok=True)
                removed += 1
        for path in self.cache_dir.glob("*.parquet"):
            try:
                cached = pd.read_parquet(path)
                if cached.empty:
                    continue
                max_date = pd.Timestamp(cached.index.max())
                max_date = max_date.tz_localize("UTC") if max_date.tzinfo is None else max_date.tz_convert("UTC")
                age_h = (datetime.now(timezone.utc) - max_date.to_pydatetime()).total_seconds() / 3600
                if age_h > max_age:
                    path.unlink(missing_ok=True)
                    removed += 1
            except Exception:  # noqa: BLE001 -- parquet corrupto = candidato directo a purga
                path.unlink(missing_ok=True)
                removed += 1
        if removed:
            logger.info("[CacheManager] Purga automática -> %d entrada(s) vencida(s) purgada(s) de %s.", removed, self.cache_dir)
        return removed


# ===========================================================================
# 2) CLIENTE FRED — fuente PRIMARIA para series macro sin ETF/futuro
#    equivalente (ej. ^TNX / DGS10). Portado 1:1 de market_data.py, ahora
#    reutilizable desde entrenamiento (antes solo lo veía inferencia).
# ===========================================================================
class FredClient:
    """
    Cliente resiliente para `api.stlouisfed.org/fred/series/observations`.
    100% gratis, requiere `api_key` (opcional -- si no se provee, cada
    llamada devuelve `None` sin tocar la red, igual que antes de RONDA 2 en
    `market_data.py`). Reintenta con backoff exponencial ante 429/timeout.
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str = "", timeout_s: float = 15.0) -> None:
        self.api_key = (api_key or "").strip()
        self.timeout_s = timeout_s
        if not self.api_key:
            logger.warning(
                "[FredClient] Sin api_key -- toda serie FRED (ej. DGS10/^TNX) "
                "devolverá None y caerá al fallback del caller (Stooq)."
            )

    def daily_series(self, series_id: str, tail_days: int, attempts: int = 3) -> Optional[pd.Series]:
        """
        `pd.Series` de Close diario (valor oficial de la serie FRED) o
        `None` si falta `api_key` o la request falla tras `attempts`
        reintentos. FRED marca días sin observación (feriados, fin de
        semana) con `"."` en vez de omitir la fila -- se filtran acá antes
        de devolver, para que el caller nunca vea un NaN string.
        """
        if not self.api_key:
            return None

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": max(tail_days, 30),
        }
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                resp = requests.get(self.BASE_URL, params=params, timeout=self.timeout_s)
                if resp.status_code == 429:
                    raise RuntimeError(f"FRED rate limit (429): {resp.text[:200]}")
                resp.raise_for_status()
                payload = resp.json()
                obs = payload.get("observations", [])
                rows = [(o["date"], o["value"]) for o in obs if o.get("value") not in (None, ".")]
                if not rows:
                    return None
                dates, values = zip(*rows)
                series = pd.Series(
                    [float(v) for v in values],
                    index=pd.to_datetime(dates),
                    name=series_id,
                ).sort_index()
                return series
            except Exception as exc:  # noqa: BLE001 -- red/HTTP transitorio
                last_exc = exc
                if attempt == attempts:
                    break
                time.sleep(1.5 * attempt)
        logger.warning("[FredClient] agotó reintentos para la serie '%s': %r", series_id, last_exc)
        return None


# ===========================================================================
# 3) ALINEACIÓN DE CALENDARIO MACRO — fuente ÚNICA para ambos pipelines.
# ===========================================================================
def normalize_daily_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    tz-aware -> tz-naive + medianoche. Un solo símbolo (target O macro)
    tz-aware colisionando contra el otro tz-naive hace fallar `.reindex()`/
    `.merge()` en TODAS las fechas, no solo fines de semana -- se normaliza
    en el borde de ingesta para que ninguna comparación aguas abajo pueda
    volver a pisar este bug (ver `market_data.py`, fix 2026-08-15).
    """
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def align_macro_to_calendar(macro_close: pd.Series, upto: pd.Timestamp) -> pd.Series:
    """
    CAUSA RAÍZ real de "pérdida masiva de datos cripto en fines de semana":
    los macro tickers (SPY/DGS10/XAU-USD/VIXY/UUP, todos vía bolsas
    tradicionales) cotizan 5/7; un activo cripto cotiza 24/7. Reindexar la
    serie macro DIRECTO contra un calendario de 7 días deja NaN cada
    sábado/domingo -- un `.ffill()` posterior sobre el frame YA combinado
    tapa la mayoría de esos huecos, pero cualquier fila líder anterior a la
    primera fecha del macro ticker (o un hueco interno que el orden de
    columnas no alcance a cubrir de forma determinista) sobrevive como NaN,
    y un `dropna()` sobre el frame completo puede vaciarlo entero.

    Fix: upsamplear la serie macro a un calendario DIARIO CORRIDO (freq="D",
    fin de semana incluido) y forward-fill DENTRO de esa serie -- ANTES de
    reindexarla contra el calendario del activo objetivo. El valor de un
    viernes de bolsa queda disponible como input válido para sábado/domingo
    de cripto, sin depender del orden ni de la suerte del ffill final sobre
    el frame ya combinado. NUNCA se hace `bfill` (relleno hacia atrás) --
    sería fuga de información (un valor FUTURO del macro ticker "explicando"
    un día pasado). Único costo real: los primeros días, anteriores al
    primer dato macro disponible, siguen sin cobertura y se recortan aguas
    abajo por el `dropna()` normal del pipeline, exactamente igual que
    antes de este fix.
    """
    if macro_close.empty:
        return macro_close
    macro_close = macro_close[~macro_close.index.duplicated(keep="last")].sort_index()
    macro_close.index = normalize_daily_index(pd.DatetimeIndex(macro_close.index))
    calendar_end = max(macro_close.index.max(), pd.Timestamp(upto))
    daily_calendar = pd.date_range(macro_close.index.min(), calendar_end, freq="D")
    return macro_close.reindex(daily_calendar).ffill()


def project_macro_frame(macro_closes: dict[str, pd.Series], target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Punto de entrada único: dado `{macro_ticker: pd.Series de Close crudo}`
    y el índice del activo objetivo (24/7 o 5/7, no importa cuál), devuelve
    un DataFrame `out[macro_ticker]` ya alineado con `align_macro_to_calendar`
    -- MISMA llamada desde `engineer_asset` (entrenamiento) y
    `fetch_feature_ohlcv`/`_fetch_feature_window` (inferencia), para que
    ambos pipelines vean el mundo macro exactamente igual (Requerimiento 3).
    """
    target_index = normalize_daily_index(pd.DatetimeIndex(target_index))
    target_last_date = target_index.max() if len(target_index) else pd.Timestamp.now(tz=timezone.utc).normalize()
    out = pd.DataFrame(index=target_index)
    for macro_ticker, macro_close in macro_closes.items():
        aligned = align_macro_to_calendar(macro_close, target_last_date)
        out[macro_ticker] = aligned.reindex(target_index)
    return out