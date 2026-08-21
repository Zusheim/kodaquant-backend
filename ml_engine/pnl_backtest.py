"""
pnl_backtest.py -- Traduce directional accuracy en P&L real (V1)
=============================================================================
`walk_forward_eval.py` responde "¿el modelo acierta la dirección más que el
azar?" (sí, medido). Este script responde la pregunta que decide si esto se
opera con capital real: sobre esas MISMAS predicciones walk-forward
out-of-sample, ¿una cartera long/short diaria gana plata después de costos
de transacción? Cero cambios a la matemática/topología/loss de
`train_kodaquant_v5.py` ni a los folds de `walk_forward_eval.py` -- reutiliza
`_build_fold_dataset`/`build_model`/`train_model` tal cual están; lo único
nuevo es capturar `r_hat`/`r_true` crudos (en vez de solo el agregado de
`evaluate_asset`) y simular una cartera simple sobre ellos.

Funciona para CUALQUIER régimen de `REGIMES`, incluido `crypto_specialist`
(BTC-USD/ETH-USD) -- mismo `--regime` que ya acepta `walk_forward_eval.py`,
sin tocar nada.

ESTRATEGIA (deliberadamente simple -- es un piso de referencia honesto, no
un producto):
  - Señal: long si r_hat > 0, short si r_hat < 0 (siempre en mercado).
  - Sizing: igual-ponderado entre los activos evaluables ese día, sin
    apalancamiento (fracción de capital = 1 / n_activos_del_día).
  - Costo: `--cost-bps` se cobra SOLO cuando la posición de un activo
    cambia de signo respecto al día anterior (turnover real, no fijo).

USO:
    python ml_engine/pnl_backtest.py --regime equity_specialist \
        --n-folds 4 --epochs-per-fold 30 --cost-bps 5
    python ml_engine/pnl_backtest.py --regime crypto_specialist \
        --n-folds 4 --epochs-per-fold 30 --cost-bps 10
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- RESOLUCIÓN ABSOLUTA DE IMPORTS (mismo bootstrap que el resto de
# ml_engine/) ----------------------------------------------------------------
def _bootstrap_project_root() -> Path:
    here = Path(__file__).resolve()
    for ancestor in (here.parent, *here.parents):
        if (ancestor / "ml_engine").is_dir() and (ancestor / "services").is_dir():
            return ancestor
    return here.parent.parent


_PROJECT_ROOT = _bootstrap_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Reutiliza walk_forward_eval.py tal cual (mismo `_build_fold_dataset`,
# mismo `tkv5` = train_kodaquant_v5) -- cero lógica de folds duplicada.
import ml_engine.walk_forward_eval as wfe  # noqa: E402

tkv5 = wfe.tkv5

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
logger = logging.getLogger("kodaquant.pnl_backtest")


def collect_fold_predictions(regime_name: str, n_folds: int, initial_train_frac: float,
                              test_frac: float, epochs_per_fold: int, output_dir: Path) -> pd.DataFrame:
    """Idéntico esqueleto que `wfe.run_walk_forward`, pero devuelve r_hat/r_true CRUDOS
    (una fila por fecha x activo x fold) en vez de solo el agregado de `evaluate_asset` --
    es lo que necesita `simulate_portfolio` para calcular PnL día por día."""
    if regime_name not in tkv5.REGIMES:
        raise ValueError(f"Régimen desconocido: '{regime_name}' -- opciones: {list(tkv5.REGIMES)}")

    regime_cfg = tkv5.REGIMES[regime_name]
    tickers = regime_cfg["tickers"]
    asset_to_id = {t: i for i, t in enumerate(tickers)}
    feature_cols = ["LOG_RETURN_1D"] + tkv5.TECH_COLS + tkv5.MACRO_TICKERS
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Descargando/leyendo caché OHLCV+macro para '%s'...", regime_name)
    all_market_data = tkv5.download_all(
        tickers, tkv5.MACRO_TICKERS, tkv5.PERIOD, cache_tag=regime_name, force_refresh=False,
    )
    dfs = {t: tkv5.engineer_asset(all_market_data, t, tkv5.MACRO_TICKERS) for t in tickers}
    n_per_ticker = {t: len(dfs[t]) for t in tickers}
    logger.info("Filas engineered por activo: %s", n_per_ticker)

    tmp_ckpt = output_dir / "_pnl_backtest_tmp.keras"
    rows: list[dict] = []

    for fold_idx in range(n_folds):
        X_train_parts, y_train_parts, aid_train_parts = [], [], []
        fold_test: dict[str, tuple] = {}
        skipped: list[str] = []

        for t in tickers:
            n = n_per_ticker[t]
            train_end = int(n * (initial_train_frac + fold_idx * test_frac))
            test_end = int(n * (initial_train_frac + (fold_idx + 1) * test_frac))
            train_end = max(train_end, tkv5.LOOKBACK + 1)
            test_end = min(test_end, n)
            built = wfe._build_fold_dataset(dfs[t], t, tkv5.LOOKBACK, train_end, test_end, asset_to_id, feature_cols)
            if built is None:
                skipped.append(t)
                continue
            train, test, _f_scaler, t_scaler = built
            fold_test[t] = (test, t_scaler)
            X_train_parts.append(train["X"])
            y_train_parts.append(train["y"])
            aid_train_parts.append(train["asset_id"])

        if not X_train_parts:
            logger.warning("Fold %d/%d: sin datos suficientes -- se salta.", fold_idx + 1, n_folds)
            continue

        X_train = np.concatenate(X_train_parts, axis=0)
        y_train = np.concatenate(y_train_parts, axis=0)
        aid_train = np.concatenate(aid_train_parts, axis=0).reshape(-1, 1)
        n_fit_samples = int(len(X_train) * (1 - tkv5.VALIDATION_SPLIT))
        steps_per_epoch = max(1, n_fit_samples // tkv5.BATCH_SIZE)

        logger.info("Fold %d/%d: X_train=%s | activos evaluables=%d%s",
                    fold_idx + 1, n_folds, X_train.shape, len(fold_test),
                    f" | saltados={skipped}" if skipped else "")

        model, gamma_var = tkv5.build_model(
            n_timesteps=tkv5.LOOKBACK, n_features=tkv5.N_FEATURES, n_assets=len(tickers),
            steps_per_epoch=steps_per_epoch, regime_name=regime_name,
            embed_dim=tkv5.ASSET_EMBED_DIM, huber_delta=tkv5.HUBER_DELTA,
            gamma_initial=tkv5.GAMMA_INITIAL, variance_lambda=tkv5.VARIANCE_LAMBDA,
            variance_cap=tkv5.VARIANCE_CAP, gaussian_nll_beta=tkv5.GAUSSIAN_NLL_BETA,
            log_var_min=tkv5.LOG_VAR_MIN, log_var_max=tkv5.LOG_VAR_MAX,
            log_var_l2_lambda=tkv5.LOG_VAR_L2_LAMBDA, log_var_barrier_lambda=tkv5.LOG_VAR_BARRIER_LAMBDA,
            model_name=f"KodaQuant_{regime_name}_PNL_fold{fold_idx + 1}",
        )
        _history, model = tkv5.train_model(
            model, X_train, aid_train, y_train, checkpoint_path=tmp_ckpt,
            epochs=epochs_per_fold, batch_size=tkv5.BATCH_SIZE, validation_split=tkv5.VALIDATION_SPLIT,
            gamma_variable=gamma_var, gamma_max=tkv5.GAMMA_MAX,
            gamma_warmup_epochs=min(tkv5.GAMMA_WARMUP_EPOCHS, max(1, epochs_per_fold // 3)),
            gamma_schedule=tkv5.GAMMA_SCHEDULE, gamma_sigmoid_steepness=tkv5.GAMMA_SIGMOID_STEEPNESS,
        )

        for t, (test, t_scaler) in fold_test.items():
            asset_id_col = test["asset_id"].reshape(-1, 1)
            y_pred_scaled = model.predict([test["X"], asset_id_col], verbose=0)
            r_hat = t_scaler.inverse_transform(y_pred_scaled[:, 0:1]).flatten()
            r_true = t_scaler.inverse_transform(test["y"]).flatten()
            for d, rh, rt in zip(test["dates"], r_hat, r_true):
                rows.append({"fold": fold_idx + 1, "ticker": t, "date": d,
                             "r_hat": float(rh), "r_true": float(rt)})

        tmp_ckpt.unlink(missing_ok=True)
        try:
            tkv5.keras.backend.clear_session()
        except Exception:  # noqa: BLE001 -- best-effort, nunca debe tumbar la corrida
            pass

    tmp_ckpt.unlink(missing_ok=True)
    return pd.DataFrame(rows)


def simulate_portfolio(preds: pd.DataFrame, cost_bps: float, vol_target_annual: float | None = None,
                        vol_lookback: int = 20, max_leverage: float = 3.0) -> pd.DataFrame:
    """
    Cartera long/short diaria entre los activos evaluables cada día. Costo
    de transacción SOLO cuando la posición de un activo cambia de signo
    respecto al día anterior (turnover real) -- así un activo que el modelo
    mantiene "largo" varios días seguidos no paga comisión de nuevo cada vez.

    `vol_target_annual` (ej. 0.15 = 15% anualizado): si se pasa, cada activo
    se pondera INVERSAMENTE a su volatilidad realizada reciente (rolling
    `vol_lookback` días, `.shift(1)` -- SOLO retornos pasados, cero look-ahead)
    en vez de peso igual fijo. Objetivo: que un activo momentáneamente muy
    volátil (ej. TSLA en un evento) no domine el drawdown del portfolio solo
    por moverse más, no por tener más señal. `max_leverage` limita el tope
    por activo para que un período de vol casi nula no dispare el tamaño a
    infinito. Con `vol_target_annual=None` (default), el comportamiento es
    IDÉNTICO al de antes (igual-ponderado, sin escalar) -- no rompe nada.
    """
    preds = preds.sort_values(["ticker", "date"]).copy()
    preds["signal"] = np.sign(preds["r_hat"])
    preds["prev_signal"] = preds.groupby("ticker")["signal"].shift(1)
    preds["turned_over"] = preds["prev_signal"].notna() & (preds["signal"] != preds["prev_signal"])
    cost = cost_bps / 10_000.0

    if vol_target_annual is not None:
        vol_target_daily = vol_target_annual / np.sqrt(252)
        realized_vol = preds.groupby("ticker")["r_true"].transform(
            lambda s: s.shift(1).rolling(vol_lookback, min_periods=max(2, vol_lookback // 2)).std()
        )
        preds["size"] = (vol_target_daily / realized_vol).clip(upper=max_leverage).fillna(0.0)
    else:
        preds["size"] = 1.0

    preds["pnl_gross"] = preds["signal"] * preds["size"] * preds["r_true"]
    preds["pnl_net"] = preds["pnl_gross"] - np.where(preds["turned_over"], cost * preds["size"], 0.0)

    daily = preds.groupby("date").agg(n_assets=("ticker", "count"), pnl_net=("pnl_net", "mean")).sort_index()
    daily["equity"] = (1.0 + daily["pnl_net"]).cumprod()
    return daily


def summarize_pnl(daily: pd.DataFrame) -> dict:
    r = daily["pnl_net"]
    n = len(r)
    total_return = float(daily["equity"].iloc[-1] - 1.0) if n else 0.0
    ann_return = float((1 + total_return) ** (252 / n) - 1) if n > 0 else 0.0
    ann_vol = float(r.std(ddof=1) * np.sqrt(252)) if n > 1 else 0.0
    sharpe = float((r.mean() * 252) / ann_vol) if ann_vol > 0 else 0.0
    running_max = daily["equity"].cummax()
    drawdown = daily["equity"] / running_max - 1.0
    max_dd = float(drawdown.min()) if n else 0.0
    win_rate = float((r > 0).mean()) if n else 0.0
    return {
        "n_days": n, "total_return_pct": total_return * 100, "annualized_return_pct": ann_return * 100,
        "annualized_vol_pct": ann_vol * 100, "sharpe": sharpe, "max_drawdown_pct": max_dd * 100,
        "win_rate_pct": win_rate * 100,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest de P&L sobre predicciones walk-forward.")
    parser.add_argument("--regime", default="equity_specialist", choices=list(tkv5.REGIMES))
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--initial-train-frac", type=float, default=0.60)
    parser.add_argument("--test-frac", type=float, default=0.08)
    parser.add_argument("--epochs-per-fold", type=int, default=30)
    parser.add_argument("--cost-bps", type=float, default=5.0,
                         help="Costo por operación (turnover), en basis points. Ajustá según tu bróker/exchange "
                              "real -- 5bps es un piso optimista para equities líquidas; cripto suele ser más caro.")
    parser.add_argument("--vol-target-annual", type=float, default=None,
                         help="Ej. 0.15 = objetivo de 15%% de volatilidad anualizada. Si se omite, sizing "
                              "igual-ponderado (comportamiento original). Recomendado para controlar drawdown.")
    parser.add_argument("--vol-lookback", type=int, default=20)
    parser.add_argument("--max-leverage", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=tkv5.BASE_DIR / "walk_forward_reports")
    args = parser.parse_args()

    preds_df = collect_fold_predictions(
        regime_name=args.regime, n_folds=args.n_folds,
        initial_train_frac=args.initial_train_frac, test_frac=args.test_frac,
        epochs_per_fold=args.epochs_per_fold, output_dir=args.output_dir,
    )

    if preds_df.empty:
        logger.error("Ningún fold produjo predicciones -- revisá --n-folds/--test-frac.")
        raise SystemExit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preds_csv = args.output_dir / f"pnl_predictions_{args.regime}.csv"
    preds_df.to_csv(preds_csv, index=False)

    daily_pnl = simulate_portfolio(preds_df, cost_bps=args.cost_bps, vol_target_annual=args.vol_target_annual,
                                    vol_lookback=args.vol_lookback, max_leverage=args.max_leverage)
    daily_csv = args.output_dir / f"pnl_daily_{args.regime}.csv"
    daily_pnl.to_csv(daily_csv)

    stats = summarize_pnl(daily_pnl)
    logger.info("=" * 78)
    logger.info("P&L -- régimen '%s' | %d día(s) de test out-of-sample | costo=%.1fbps por turnover",
                args.regime, stats["n_days"], args.cost_bps)
    logger.info("  Retorno total:        %+.2f%%", stats["total_return_pct"])
    logger.info("  Retorno anualizado:   %+.2f%%", stats["annualized_return_pct"])
    logger.info("  Volatilidad anual.:   %.2f%%", stats["annualized_vol_pct"])
    logger.info("  Sharpe (sin rf):      %.2f", stats["sharpe"])
    logger.info("  Max drawdown:         %.2f%%", stats["max_drawdown_pct"])
    logger.info("  Win rate (días):      %.1f%%", stats["win_rate_pct"])
    logger.info("=" * 78)
    logger.info("Predicciones -> %s", preds_csv)
    logger.info("Equity diaria -> %s", daily_csv)

    if stats["sharpe"] <= 0 or stats["total_return_pct"] <= 0:
        logger.warning("Sharpe/retorno <= 0 con este costo asumido: el edge direccional medido en "
                        "walk_forward_eval.py NO alcanza para cubrir %.1fbps de costo por turnover con este "
                        "sizing igual-ponderado. Antes de operar, probá con costos reales de tu bróker/exchange "
                        "y/o un sizing menos ingenuo (ej. Kelly fraccional, filtrar señales débiles).", args.cost_bps)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(daily_pnl.index, daily_pnl["equity"], linewidth=1.5)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Break-even")
        ax.set_ylabel("Equity (base = 1.0)")
        ax.set_title(f"Curva de equity walk-forward -- {args.regime} (costo={args.cost_bps}bps)")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        chart_path = args.output_dir / f"pnl_equity_curve_{args.regime}.png"
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        logger.info("Chart -> %s", chart_path)
    except Exception as exc:  # noqa: BLE001 -- el chart es un extra, nunca debe tumbar el resumen
        logger.debug("No se pudo generar el chart (%r) -- no bloqueante.", exc)