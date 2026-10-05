"""KodaQuant V28 -- alpha gratuito, UNA sola implementacion para train e inferencia.

Copiar IDENTICO a `ml_engine/free_alpha.py` y `services/free_alpha.py`
(train importa de `ml_engine`, el motor de `services`; mismo patron que
`kodaquant_core.py`). Todo feature se calcula aca; ningun lado lo reimplementa,
asi TECH_COLS no puede desincronizarse por construccion.

Causalidad: solo `.rolling()` trailing, `.shift(+n)`, `.diff()`. Cero `center=True`,
cero `shift(-n)`. `_selftest()` lo verifica por invariancia de prefijo.

Fuentes (todas publicas, sin API key):
  * Binance spot klines 1d (host espejo `data-api.binance.vision`, sirve donde
    `api.binance.com` devuelve 451): volumen taker-buy REAL -> imbalance de agresor.
  * Coinbase Exchange candles 1d (`api.exchange.coinbase.com`): prima Coinbase/Binance.
  * Binance USD-M `premiumIndexKlines` 1d: base perpetuo-spot (puede estar geobloqueado).
Cualquier fuente caida -> columna 0.0 (neutra) + warning + `LAST_STATUS`.
"""
from __future__ import annotations

import logging
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("kodaquant.free_alpha")

V28_EQUITY_COLS = [
    "OVERNIGHT_GAP_ZSCORE",      # ln(Open_t / Close_{t-1}) z-scoreado
    "INTRADAY_RET_ZSCORE",       # ln(Close_t / Open_t) z-scoreado
    "OVERNIGHT_MOM_ZSCORE",      # media 10d del gap nocturno, z 120d (persistencia overnight)
    "REL_STRENGTH_SPX_ZSCORE",   # retorno 5d del activo menos retorno 5d de ^GSPC, z-scoreado
]
V28_CRYPTO_COLS = [
    "TAKER_IMBALANCE_ZSCORE",    # (2*takerBuyBase/vol - 1), media 5d, z 60d -- flujo de agresor REAL
    "TRADE_SIZE_ZSCORE",         # log(quoteVol/trades), media 3d, z 60d -- retail vs ballena
    "CB_PREMIUM_ZSCORE",         # ln(close Coinbase USD / close Binance USDT), media 3d, z 60d
    "PERP_BASIS_ZSCORE",         # premium index perpetuo, media 3d, z 60d
]
V28_COLS = V28_EQUITY_COLS + V28_CRYPTO_COLS

INDEX_PROXY_TICKERS = frozenset({"SPY"})  # rel-strength vs ^GSPC seria puro ruido de tracking

Z_WINDOW, Z_MIN, Z_CLIP = 60, 30, 4.0
OVERNIGHT_MOM_DAYS = 10
REL_STRENGTH_DAYS = 5
FLOW_SMOOTH_DAYS = 5
PRICE_SMOOTH_DAYS = 3
GAP_FFILL_LIMIT = 2

CRYPTO_SOURCES = {
    "BTC-USD": {"spot": "BTCUSDT", "cb": "BTC-USD", "perp": "BTCUSDT"},
    "ETH-USD": {"spot": "ETHUSDT", "cb": "ETH-USD", "perp": "ETHUSDT"},
}
_BINANCE_SPOT_URLS = ["https://data-api.binance.vision/api/v3/klines",
                      "https://api.binance.com/api/v3/klines"]
_BINANCE_PREMIUM_URLS = ["https://fapi.binance.com/fapi/v1/premiumIndexKlines"]
_COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"

LAST_STATUS: dict[str, dict[str, str]] = {}
_MEM_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_MEM_TTL_SECONDS = 6 * 3600


# ----------------------------------------------------------------------------
# Primitivas
# ----------------------------------------------------------------------------
def rolling_z(s: pd.Series, window: int = Z_WINDOW, min_periods: int = Z_MIN,
              clip: float = Z_CLIP) -> pd.Series:
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    z = (s - mean) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).clip(-clip, clip).fillna(0.0)


def _naive_daily(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()


# ----------------------------------------------------------------------------
# EQUITY: descomposicion overnight/intraday + fuerza relativa
# ----------------------------------------------------------------------------
def compute_equity_open_features(open_: Optional[pd.Series], close: pd.Series,
                                 spx_logret: Optional[pd.Series] = None,
                                 ticker: str = "") -> pd.DataFrame:
    """Devuelve DataFrame(V28_EQUITY_COLS) alineado a `close.index`.

    `open_` None -> columnas overnight/intraday en 0.0 (neutras).
    `spx_logret` = log-return diario de ^GSPC ya alineado al indice de `close`.
    """
    out = pd.DataFrame(0.0, index=close.index, columns=V28_EQUITY_COLS)
    if open_ is not None:
        gap = np.log(open_ / close.shift(1)).replace([np.inf, -np.inf], np.nan)
        intraday = np.log(close / open_).replace([np.inf, -np.inf], np.nan)
        out["OVERNIGHT_GAP_ZSCORE"] = rolling_z(gap)
        out["INTRADAY_RET_ZSCORE"] = rolling_z(intraday)
        out["OVERNIGHT_MOM_ZSCORE"] = rolling_z(
            gap.rolling(OVERNIGHT_MOM_DAYS, min_periods=OVERNIGHT_MOM_DAYS).mean(),
            window=120, min_periods=60)
    else:
        logger.warning("[V28] '%s': sin columna Open -> features overnight/intraday = 0.0.", ticker)
    if spx_logret is not None and ticker not in INDEX_PROXY_TICKERS:
        asset_lr = np.log(close / close.shift(1))
        rel = (asset_lr.rolling(REL_STRENGTH_DAYS, min_periods=REL_STRENGTH_DAYS).sum()
               - spx_logret.rolling(REL_STRENGTH_DAYS, min_periods=REL_STRENGTH_DAYS).sum())
        out["REL_STRENGTH_SPX_ZSCORE"] = rolling_z(rel)
    return out


# ----------------------------------------------------------------------------
# CRIPTO: fetchers
# ----------------------------------------------------------------------------
def _http_get_json(urls: list[str], params: dict, timeout: int = 15, retries: int = 2):
    import requests
    for url in urls:
        for attempt in range(retries + 1):
            try:
                r = requests.get(url, params=params, timeout=timeout,
                                 headers={"User-Agent": "kodaquant/28"})
                if r.status_code in (403, 451):   # geoblock -> siguiente host
                    break
                if r.status_code in (418, 429):   # rate limit
                    time.sleep(1.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as exc:  # noqa: BLE001 -- proveedor opcional
                logger.debug("GET %s fallo (%r) intento %d", url, exc, attempt + 1)
                time.sleep(0.4 * (attempt + 1))
    return None


def _paginate_binance_klines(urls: list[str], symbol: str, start_ms: int, limit: int) -> list:
    rows: list = []
    cursor, now_ms = start_ms, int(time.time() * 1000)
    for _ in range(40):
        if cursor >= now_ms:
            break
        batch = _http_get_json(urls, {"symbol": symbol, "interval": "1d",
                                      "startTime": cursor, "limit": limit})
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        if last_open < cursor or len(batch) < limit:
            break
        cursor = last_open + 1
        time.sleep(0.12)
    return rows


def fetch_binance_spot_daily(symbol: str, start: str = "2017-08-01") -> Optional[pd.DataFrame]:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows = _paginate_binance_klines(_BINANCE_SPOT_URLS, symbol, start_ms, 1000)
    if not rows:
        return None
    # [0]open_time [4]close [5]volume [7]quote_vol [8]trades [9]taker_buy_base
    df = pd.DataFrame(rows).iloc[:, :10]
    df.columns = ["open_time", "open", "high", "low", "close", "volume",
                  "close_time", "quote_volume", "trades", "taker_buy_base"]
    df["date"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms").dt.normalize()
    for c in ("close", "volume", "quote_volume", "trades", "taker_buy_base"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates("date", keep="last").set_index("date").sort_index()
    return df[["close", "volume", "quote_volume", "trades", "taker_buy_base"]]


def fetch_binance_basis_daily(symbol: str, start: str = "2019-09-01") -> Optional[pd.Series]:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows = _paginate_binance_klines(_BINANCE_PREMIUM_URLS, symbol, start_ms, 1500)
    if not rows:
        return None
    df = pd.DataFrame(rows).iloc[:, :5]
    df.columns = ["open_time", "open", "high", "low", "close"]
    df["date"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms").dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.drop_duplicates("date", keep="last").set_index("date").sort_index()["close"]


def fetch_coinbase_daily(product: str, start: str = "2016-01-01") -> Optional[pd.Series]:
    cursor = pd.Timestamp(start, tz="UTC")
    end_all = pd.Timestamp.now(tz="UTC")
    rows: list = []
    while cursor < end_all:
        chunk_end = min(cursor + pd.Timedelta(days=290), end_all)
        data = _http_get_json([_COINBASE_URL.format(product=product)],
                              {"granularity": 86400, "start": cursor.isoformat(),
                               "end": chunk_end.isoformat()})
        if data is None:
            break
        rows.extend(data)  # [time, low, high, open, close, volume]
        cursor = chunk_end
        time.sleep(0.25)
    if not rows:
        return None
    df = pd.DataFrame(rows).iloc[:, :5]
    df.columns = ["time", "low", "high", "open", "close"]
    df["date"] = pd.to_datetime(pd.to_numeric(df["time"]), unit="s").dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.drop_duplicates("date", keep="last").set_index("date").sort_index()["close"]


def _cached(key: str, fetch, cache=None) -> Optional[pd.DataFrame]:
    now = time.time()
    hit = _MEM_CACHE.get(key)
    if hit and now - hit[0] < _MEM_TTL_SECONDS:
        return hit[1]
    if cache is not None:
        try:
            df = cache.get_dataframe(key)
            if df is not None and not df.empty:
                _MEM_CACHE[key] = (now, df)
                return df
        except Exception:  # noqa: BLE001
            pass
    obj = fetch()
    if obj is None or len(obj) == 0:
        return None
    df = obj.to_frame("v") if isinstance(obj, pd.Series) else obj
    _MEM_CACHE[key] = (now, df)
    if cache is not None:
        try:
            cache.set_dataframe(key, df)
        except Exception:  # noqa: BLE001
            pass
    return df


# ----------------------------------------------------------------------------
# CRIPTO: features (funcion pura, testeable sin red)
# ----------------------------------------------------------------------------
def compute_crypto_flow_features(index, spot: Optional[pd.DataFrame],
                                 coinbase: Optional[pd.Series],
                                 basis: Optional[pd.Series]) -> pd.DataFrame:
    idx = _naive_daily(index)
    out = pd.DataFrame(0.0, index=idx, columns=V28_CRYPTO_COLS)

    def _align(s):
        s = s.copy()
        s.index = _naive_daily(s.index)
        return s[~s.index.duplicated(keep="last")].reindex(idx).ffill(limit=GAP_FFILL_LIMIT)

    if spot is not None and len(spot):
        sp = _align(spot) if isinstance(spot, pd.Series) else spot.copy()
        if isinstance(sp, pd.DataFrame):
            sp.index = _naive_daily(sp.index)
            sp = sp[~sp.index.duplicated(keep="last")].reindex(idx).ffill(limit=GAP_FFILL_LIMIT)
        vol = sp["volume"].replace(0.0, np.nan)
        imb = (2.0 * sp["taker_buy_base"] / vol - 1.0).replace([np.inf, -np.inf], np.nan)
        out["TAKER_IMBALANCE_ZSCORE"] = rolling_z(imb.rolling(FLOW_SMOOTH_DAYS, min_periods=3).mean())
        size = (sp["quote_volume"] / sp["trades"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["TRADE_SIZE_ZSCORE"] = rolling_z(np.log(size).rolling(PRICE_SMOOTH_DAYS, min_periods=2).mean())
        if coinbase is not None and len(coinbase):
            prem = np.log(_align(coinbase) / sp["close"]).replace([np.inf, -np.inf], np.nan)
            out["CB_PREMIUM_ZSCORE"] = rolling_z(prem.rolling(PRICE_SMOOTH_DAYS, min_periods=2).mean())
    if basis is not None and len(basis):
        out["PERP_BASIS_ZSCORE"] = rolling_z(_align(basis).rolling(PRICE_SMOOTH_DAYS, min_periods=2).mean())
    return out


def build_crypto_flow_features(ticker: str, index, cache=None) -> pd.DataFrame:
    """Descarga (con cache) + calcula. Cualquier fuente caida -> su columna queda 0.0."""
    src = CRYPTO_SOURCES.get(ticker)
    if src is None:
        return pd.DataFrame(0.0, index=_naive_daily(index), columns=V28_CRYPTO_COLS)
    st = LAST_STATUS.setdefault(ticker, {})
    spot = _cached(f"v28_spot_{src['spot']}", lambda: fetch_binance_spot_daily(src["spot"]), cache)
    cb = _cached(f"v28_cb_{src['cb']}", lambda: fetch_coinbase_daily(src["cb"]), cache)
    bs = _cached(f"v28_basis_{src['perp']}", lambda: fetch_binance_basis_daily(src["perp"]), cache)
    for name, obj in (("spot", spot), ("coinbase", cb), ("basis", bs)):
        st[name] = "ok" if obj is not None else "DOWN"
        if obj is None:
            logger.warning("[V28] '%s': fuente '%s' no disponible -> sus features quedan en 0.0 "
                           "(riesgo de skew train/serve si difiere del entrenamiento).", ticker, name)
    return compute_crypto_flow_features(
        index,
        spot,
        cb["v"] if cb is not None else None,
        bs["v"] if bs is not None else None,
    )


# ----------------------------------------------------------------------------
# Autotest: causalidad (invariancia de prefijo) + forma
# ----------------------------------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(7)
    n = 400
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    open_ = close.shift(1) * np.exp(rng.normal(0, 0.003, n))
    open_.iloc[0] = close.iloc[0]
    spx = pd.Series(rng.normal(0, 0.008, n), index=idx)

    full = compute_equity_open_features(open_, close, spx, "AAPL")
    cut = 300
    pref = compute_equity_open_features(open_.iloc[:cut], close.iloc[:cut], spx.iloc[:cut], "AAPL")
    np.testing.assert_allclose(full.iloc[:cut].values, pref.values, atol=1e-12,
                               err_msg="LOOKAHEAD en features equity V28")
    assert list(full.columns) == V28_EQUITY_COLS and np.isfinite(full.values).all()
    assert (compute_equity_open_features(open_, close, spx, "SPY")["REL_STRENGTH_SPX_ZSCORE"] == 0).all()
    assert (compute_equity_open_features(None, close, spx, "AAPL")["OVERNIGHT_GAP_ZSCORE"] == 0).all()

    didx = pd.date_range("2019-01-01", periods=n, freq="D")
    vol = pd.Series(rng.uniform(1e3, 2e3, n), index=didx)
    spot = pd.DataFrame({
        "close": 20000 * np.exp(np.cumsum(rng.normal(0, 0.02, n))),
        "volume": vol, "quote_volume": vol * 20000,
        "trades": rng.integers(5e5, 9e5, n).astype(float),
        "taker_buy_base": vol * rng.uniform(0.4, 0.6, n)}, index=didx)
    cb = spot["close"] * np.exp(rng.normal(0, 0.001, n))
    basis = pd.Series(rng.normal(0.0003, 0.0002, n), index=didx)
    f_full = compute_crypto_flow_features(didx, spot, cb, basis)
    f_pref = compute_crypto_flow_features(didx[:300], spot.iloc[:300], cb.iloc[:300], basis.iloc[:300])
    np.testing.assert_allclose(f_full.iloc[:300].values, f_pref.values, atol=1e-12,
                               err_msg="LOOKAHEAD en features cripto V28")
    assert list(f_full.columns) == V28_CRYPTO_COLS and np.isfinite(f_full.values).all()
    assert (compute_crypto_flow_features(didx, None, None, None).values == 0).all()
    print("free_alpha selftest OK (causalidad por prefijo, forma, degradacion a 0.0)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()