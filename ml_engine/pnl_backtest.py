"""
pnl_backtest.py -- Traduce la señal del modelo en P&L real (V1/V22)
=============================================================================
`walk_forward_eval.py` responde "¿el modelo acierta más que el azar?" (sí,
medido). Este script responde la pregunta que decide si esto se opera con
capital real: sobre esas MISMAS predicciones walk-forward out-of-sample,
¿una cartera basada en la señal del modelo gana plata después de costos de
transacción? Cero cambios a la matemática/topología/loss de
`train_kodaquant_v5.py` ni a los folds de `walk_forward_eval.py` -- reutiliza
`_build_fold_dataset`/`build_model`/`train_model` tal cual están.

Funciona para CUALQUIER régimen de `REGIMES`, incluido `crypto_specialist`
-- mismo `--regime` que ya acepta `walk_forward_eval.py`, sin tocar nada.
`target_mode` (leído de `tkv5.REGIMES[regime]`) bifurca la lógica de
señal/P&L, ver `collect_fold_predictions`/`simulate_portfolio*` abajo.

ESTRATEGIA `target_mode="regression"` (equity_specialist, SIN CAMBIOS V1):
  - Señal: long si r_hat > 0, short si r_hat < 0 (siempre en mercado).
  - Sizing: igual-ponderado entre los activos evaluables ese día, sin
    apalancamiento (fracción de capital = 1 / n_activos_del_día), o
    vol-target opcional (`--vol-target-annual`).
  - Costo: `--cost-bps` se cobra SOLO cuando la posición de un activo
    cambia de signo respecto al día anterior (turnover real, no fijo).

ESTRATEGIA `target_mode="triple_barrier"` (V22, crypto_specialist):
  - Señal: long si la clase predicha es "TP", short si es "SL", SIN
    OPERAR (meta-labeling: se descarta la apuesta) si es "TIMEOUT".
  - P&L: cada fila YA es un trade discreto -- `realized_return` es el
    log-return REAL logrado durante toda la ventana de
    `max_holding_days` hasta que se resolvió la barrera (o venció la
    vertical), ver `train_kodaquant_v5.compute_triple_barrier_labels`.
    Costo se cobra UNA VEZ por trade (entry+exit), no por turnover diario
    -- acá no hay "mantener posición día a día", ver
    `simulate_portfolio_triple_barrier`.
  - LIMITACIÓN CONOCIDA (documentada, no oculta): trades con
    `max_holding_days > 1` pueden solaparse en el tiempo; este V1 NO
    neteo/promedia posiciones concurrentes (López de Prado, cap. 4,
    "Sample Weights" trata esto rigurosamente) -- piso de referencia
    honesto, no un motor de sizing de cartera de producción.

USO:
    python ml_engine/pnl_backtest.py --regime equity_specialist \
        --n-folds 4 --cost-bps 5
    python ml_engine/pnl_backtest.py --regime crypto_specialist \
        --n-folds 4 --cost-bps 10
    (sin --epochs-per-fold: usa el presupuesto de producción del régimen,
    ver AUDITORÍA RESCATE 2026-08 más abajo / walk_forward_eval.py.)
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
    """
    Idéntico esqueleto que `wfe.run_walk_forward`, pero devuelve las
    predicciones CRUDAS (una fila por fecha x activo x fold) en vez de
    solo el agregado de `evaluate_asset`/`evaluate_asset_triple_barrier`
    -- es lo que necesitan `simulate_portfolio`/
    `simulate_portfolio_triple_barrier` para calcular P&L trade a trade.

    V22: bifurca en `target_mode` (leído de `tkv5.REGIMES[regime_name]`,
    igual que `wfe.run_walk_forward`) -- columnas devueltas:
        regression:      fold, ticker, date, r_hat, r_true
        triple_barrier:  fold, ticker, date, pred_class_idx, true_class_idx,
                          p_sl, p_timeout, p_tp, realized_return
    """
    if regime_name not in tkv5.REGIMES:
        raise ValueError(f"Régimen desconocido: '{regime_name}' -- opciones: {list(tkv5.REGIMES)}")

    regime_cfg = tkv5.REGIMES[regime_name]
    tickers = regime_cfg["tickers"]
    asset_to_id = {t: i for i, t in enumerate(tickers)}
    feature_cols = ["LOG_RETURN_1D"] + tkv5.get_tech_cols(regime_name) + tkv5.MACRO_TICKERS
    output_dir.mkdir(parents=True, exist_ok=True)

    target_mode = regime_cfg.get("target_mode", "regression")
    # V23.5 -- default POR TICKER (fallback SOLO si CRYPTO_CALIBRATE_
    # BARRIERS=False, ver `wfe.run_walk_forward`/`tkv5.run_regime_pipeline`);
    # si está en True (default de producción), cada fold recalibra por
    # activo vía `wfe._calibrate_fold_tb_params` -- mismo criterio train-only
    # de ESE fold, cero lógica de grid search duplicada.
    _default_tb_params: dict | None = None
    if target_mode == "triple_barrier":
        _default_tb_params = {
            "pt_mult": regime_cfg.get("pt_mult", tkv5.TRIPLE_BARRIER_PT_MULT),
            "sl_mult": regime_cfg.get("sl_mult", tkv5.TRIPLE_BARRIER_SL_MULT),
            "max_holding_days": regime_cfg.get("max_holding_days", tkv5.TRIPLE_BARRIER_MAX_HOLDING_DAYS),
            "vol_lookback": regime_cfg.get("vol_lookback", tkv5.TRIPLE_BARRIER_VOL_LOOKBACK),
        }

    logger.info("Descargando/leyendo caché OHLCV+macro para '%s'...", regime_name)
    all_market_data = tkv5.download_all(
        tickers, tkv5.MACRO_TICKERS, tkv5.PERIOD, cache_tag=regime_name, force_refresh=False,
    )
    dfs = {t: tkv5.engineer_asset(all_market_data, t, tkv5.MACRO_TICKERS) for t in tickers}
    n_per_ticker = {t: len(dfs[t]) for t in tickers}
    logger.info("Filas engineered por activo: %s", n_per_ticker)

    tmp_ckpt = output_dir / "_pnl_backtest_tmp.keras"
    rows: list[dict] = []
    monitor_metric = "val_actionable_accuracy" if target_mode == "triple_barrier" else "val_directional_accuracy_metric"

    for fold_idx in range(n_folds):
        # V23.5 -- fronteras POR TICKER primero (la calibración por-fold las
        # necesita antes de construir datasets), mismo orden que
        # `wfe.run_walk_forward`.
        train_end_by_ticker: dict[str, int] = {}
        test_end_by_ticker: dict[str, int] = {}
        for t in tickers:
            n = n_per_ticker[t]
            train_end = int(n * (initial_train_frac + fold_idx * test_frac))
            test_end = int(n * (initial_train_frac + (fold_idx + 1) * test_frac))
            train_end_by_ticker[t] = max(train_end, tkv5.LOOKBACK + 1)
            test_end_by_ticker[t] = min(test_end, n)

        tb_params_fold: dict[str, dict] | None = None
        if target_mode == "triple_barrier":
            if tkv5.CRYPTO_CALIBRATE_BARRIERS:
                tb_params_fold = wfe._calibrate_fold_tb_params(dfs, tickers, train_end_by_ticker)
            else:
                tb_params_fold = {t: dict(_default_tb_params) for t in tickers}
            for t in tickers:
                tbt = tb_params_fold[t]
                logger.info("Fold %d/%d [%s]: pt_mult=%.2f sl_mult=%.2f vol_lookback=%d max_holding_days=%d",
                            fold_idx + 1, n_folds, t, tbt["pt_mult"], tbt["sl_mult"],
                            tbt["vol_lookback"], tbt["max_holding_days"])

        # ROLLBACK V23.6 -- CONGRUENCIA con walk_forward_eval.py/tkv5: el
        # sample_weight equity por volatilidad fue purgado en los tres
        # archivos (ver changelog de `train_kodaquant_v5.py`). Este backtest
        # ya entrena equity_specialist con gradiente plano/uniforme, igual
        # que walk_forward_eval.py y run_regime_pipeline.
        X_train_parts, y_train_parts, aid_train_parts, sw_train_parts = [], [], [], []
        fold_test: dict[str, tuple] = {}
        skipped: list[str] = []

        for t in tickers:
            train_end = train_end_by_ticker[t]
            test_end = test_end_by_ticker[t]
            tb_params_t = tb_params_fold[t] if tb_params_fold is not None else None
            built = wfe._build_fold_dataset(dfs[t], t, tkv5.LOOKBACK, train_end, test_end, asset_to_id,
                                             feature_cols, target_mode=target_mode, tb_params=tb_params_t)
            if built is None:
                skipped.append(t)
                continue
            train, test, _f_scaler, t_scaler = built
            fold_test[t] = (test, t_scaler)
            X_train_parts.append(train["X"])
            y_train_parts.append(train["y"])
            aid_train_parts.append(train["asset_id"])
            if target_mode == "triple_barrier":
                sw_train_parts.append(train["sample_weight"])
            # ROLLBACK V23.6: rama "regression" (equity) purgada, ver arriba.

        if not X_train_parts:
            logger.warning("Fold %d/%d: sin datos suficientes -- se salta.", fold_idx + 1, n_folds)
            continue

        X_train = np.concatenate(X_train_parts, axis=0)
        y_train = np.concatenate(y_train_parts, axis=0)
        aid_train = np.concatenate(aid_train_parts, axis=0).reshape(-1, 1)
        # ROLLBACK V23.6: solo triple_barrier (cripto) pesa muestras.
        _uses_sample_weight = target_mode == "triple_barrier"
        sw_train = np.concatenate(sw_train_parts, axis=0) if _uses_sample_weight else None
        n_fit_samples = int(len(X_train) * (1 - tkv5.VALIDATION_SPLIT))
        steps_per_epoch = max(1, n_fit_samples // tkv5.BATCH_SIZE)

        logger.info("Fold %d/%d: X_train=%s | activos evaluables=%d%s",
                    fold_idx + 1, n_folds, X_train.shape, len(fold_test),
                    f" | saltados={skipped}" if skipped else "")

        class_alpha = None
        if target_mode == "triple_barrier":
            train_class_idx = np.argmax(y_train, axis=-1)
            class_counts = np.maximum(np.bincount(train_class_idx, minlength=3).astype(np.float64), 1.0)
            class_alpha = (len(train_class_idx) / (3.0 * class_counts)).tolist()

        model, gamma_var = tkv5.build_model(
            n_timesteps=tkv5.LOOKBACK, n_features=tkv5.get_n_features(regime_name), n_assets=len(tickers),
            steps_per_epoch=steps_per_epoch, regime_name=regime_name,
            embed_dim=tkv5.ASSET_EMBED_DIM, huber_delta=tkv5.HUBER_DELTA,
            gamma_initial=tkv5.GAMMA_INITIAL, variance_lambda=tkv5.VARIANCE_LAMBDA,
            variance_cap=tkv5.VARIANCE_CAP, gaussian_nll_beta=tkv5.GAUSSIAN_NLL_BETA,
            log_var_min=tkv5.LOG_VAR_MIN, log_var_max=tkv5.LOG_VAR_MAX,
            log_var_l2_lambda=tkv5.LOG_VAR_L2_LAMBDA, log_var_barrier_lambda=tkv5.LOG_VAR_BARRIER_LAMBDA,
            target_mode=target_mode, class_alpha=class_alpha,
            focal_gamma=tkv5.TRIPLE_BARRIER_FOCAL_GAMMA, label_smoothing=tkv5.TRIPLE_BARRIER_LABEL_SMOOTHING,
            model_name=f"KodaQuant_{regime_name}_PNL_fold{fold_idx + 1}",
        )
        _history, model = tkv5.train_model(
            model, X_train, aid_train, y_train, checkpoint_path=tmp_ckpt,
            epochs=epochs_per_fold, batch_size=tkv5.BATCH_SIZE, validation_split=tkv5.VALIDATION_SPLIT,
            gamma_variable=gamma_var, gamma_max=tkv5.GAMMA_MAX,
            gamma_warmup_epochs=min(tkv5.GAMMA_WARMUP_EPOCHS, max(1, epochs_per_fold // 3)),
            gamma_schedule=tkv5.GAMMA_SCHEDULE, gamma_sigmoid_steepness=tkv5.GAMMA_SIGMOID_STEEPNESS,
            sample_weight=sw_train, monitor_metric=monitor_metric,
        )

        for t, (test, t_scaler) in fold_test.items():
            asset_id_col = test["asset_id"].reshape(-1, 1)
            if target_mode == "triple_barrier":
                y_pred_proba = model.predict([test["X"], asset_id_col], verbose=0)
                pred_class_idx = np.argmax(y_pred_proba, axis=-1)
                true_class_idx = np.argmax(test["y"], axis=-1)
                realized_return = test["y_return"]
                for d, pc, tc, proba, rret in zip(test["dates"], pred_class_idx, true_class_idx,
                                                    y_pred_proba, realized_return):
                    rows.append({
                        "fold": fold_idx + 1, "ticker": t, "date": d,
                        "pred_class_idx": int(pc), "true_class_idx": int(tc),
                        "p_sl": float(proba[0]), "p_timeout": float(proba[1]), "p_tp": float(proba[2]),
                        "realized_return": float(rret),
                    })
            else:
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
    df = pd.DataFrame(rows)
    if not df.empty:
        df.attrs["target_mode"] = target_mode
    return df


def simulate_portfolio(preds: pd.DataFrame, cost_bps: float, vol_target_annual: float | None = None,
                        vol_lookback: int = 20, max_leverage: float = 3.0) -> pd.DataFrame:
    """
    `target_mode="regression"` (equity_specialist, SIN CAMBIOS V1) --
    cartera long/short diaria entre los activos evaluables cada día. Costo
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


def simulate_portfolio_triple_barrier(preds: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """
    V22 -- simulación de cartera para `target_mode="triple_barrier"`. A
    diferencia de `simulate_portfolio` (regresión, un retorno DIARIO por
    fila), acá cada fila YA ES un trade completo -- `realized_return` es
    el log-return REAL logrado durante toda la ventana de
    `max_holding_days` hasta que se resolvió la barrera (o venció la
    vertical), ver `train_kodaquant_v5.compute_triple_barrier_labels`.

    Señal:
        long  (+1) si la clase predicha es "TP"
        short (-1) si la clase predicha es "SL"
        SIN OPERAR (se descarta la fila entera, cero costo/cero P&L) si la
            clase predicha es "TIMEOUT" -- meta-labeling: "sin convicción".

    `realized_return` ya es simétrico respecto a la dirección real del
    precio (positivo si subió, negativo si bajó, cualquiera sea la clase
    verdadera) -- multiplicarlo por `signal` da el P&L correcto tanto si
    el trade fue GANADOR (predicción == clase real) como si fue PERDEDOR
    (predicción != clase real, incluida una TP/SL cruzada).

    Costo se cobra UNA VEZ por trade abierto (entry+exit) -- acá no hay
    noción de "mantener posición día a día": cada fila es un trade
    discreto autocontenido, a diferencia del turnover diario de
    `simulate_portfolio`.

    LIMITACIÓN CONOCIDA (documentada, no oculta): con `max_holding_days`
    > 1 los trades pueden SOLAPARSE en el tiempo (una entrada el día t y
    otra el día t+3 conviven simultáneamente) -- este V1 NO neteo/promedia
    posiciones concurrentes (López de Prado, cap. 4, "Sample Weights", es
    el tratamiento riguroso de bets concurrentes) y compone la curva de
    equity POR FECHA DE ENTRADA, igual-ponderado entre los trades abiertos
    ese día -- un piso de referencia honesto sobre si la señal tiene edge
    económicamente significativo neto de costos, no un motor de sizing de
    cartera de producción.
    """
    preds = preds.copy()
    cost = cost_bps / 10_000.0

    tp_idx = tkv5.TB_LABEL_TO_CLASS_IDX[1]
    sl_idx = tkv5.TB_LABEL_TO_CLASS_IDX[-1]
    preds["signal"] = 0
    preds.loc[preds["pred_class_idx"] == tp_idx, "signal"] = 1
    preds.loc[preds["pred_class_idx"] == sl_idx, "signal"] = -1

    traded = preds[preds["signal"] != 0].copy()
    if traded.empty:
        return pd.DataFrame(columns=["n_trades", "pnl_net", "equity"])

    traded["pnl_gross"] = traded["signal"] * traded["realized_return"]
    traded["pnl_net"] = traded["pnl_gross"] - cost  # un costo por trade (entry+exit), no por turnover

    daily = traded.groupby("date").agg(n_trades=("ticker", "count"), pnl_net=("pnl_net", "mean")).sort_index()
    daily["equity"] = (1.0 + daily["pnl_net"]).cumprod()
    return daily


def summarize_pnl(daily: pd.DataFrame) -> dict:
    """
    Genérico entre ambos `target_mode` -- solo lee las columnas
    `pnl_net`/`equity`, presentes en el resultado de `simulate_portfolio`
    (regresión) y de `simulate_portfolio_triple_barrier` (V22) por igual.
    La anualización (factor 252) asume eventos de frecuencia diaria en el
    índice -- válida sin matices en el régimen de regresión (un retorno
    por día de mercado); en `triple_barrier` es una aproximación (el
    índice son fechas de ENTRADA, no de resolución -- ver limitación
    documentada en `simulate_portfolio_triple_barrier`), suficiente para
    un veredicto de "V1, piso de referencia honesto".
    """
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
    parser.add_argument("--epochs-per-fold", type=int, default=None,
                         help="AUDITORÍA RESCATE 2026-08: default REAL ahora es el presupuesto de producción "
                              "del régimen (tkv5.REGIMES[regime]['epochs'], ver walk_forward_eval.py) en vez "
                              "de 30 fijo -- el default anterior subentrenaba el modelo sistemáticamente.")
    parser.add_argument("--cost-bps", type=float, default=5.0,
                         help="Costo por operación, en basis points. Ajustá según tu bróker/exchange real -- "
                              "5bps es un piso optimista para equities líquidas; cripto suele ser más caro. En "
                              "target_mode='regression' se cobra por turnover diario; en 'triple_barrier' se "
                              "cobra una vez por trade (entry+exit), ver simulate_portfolio_triple_barrier.")
    parser.add_argument("--vol-target-annual", type=float, default=None,
                         help="SOLO target_mode='regression'. Ej. 0.15 = objetivo de 15%% de volatilidad "
                              "anualizada. Si se omite, sizing igual-ponderado (comportamiento original).")
    parser.add_argument("--vol-lookback", type=int, default=20)
    parser.add_argument("--max-leverage", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=tkv5.BASE_DIR / "walk_forward_reports")
    args = parser.parse_args()

    target_mode = tkv5.REGIMES[args.regime].get("target_mode", "regression")

    # AUDITORÍA RESCATE 2026-08: mismo fix que walk_forward_eval.py -- sin
    # --epochs-per-fold explícito, usar el presupuesto de PRODUCCIÓN del
    # régimen, nunca un default de iteración rápida hardcodeado.
    effective_epochs_per_fold = args.epochs_per_fold
    if effective_epochs_per_fold is None:
        effective_epochs_per_fold = tkv5.REGIMES[args.regime].get("epochs", tkv5.EPOCHS)
        logger.info(
            "--epochs-per-fold no especificado -> usando presupuesto de producción de '%s': %d épocas.",
            args.regime, effective_epochs_per_fold,
        )

    preds_df = collect_fold_predictions(
        regime_name=args.regime, n_folds=args.n_folds,
        initial_train_frac=args.initial_train_frac, test_frac=args.test_frac,
        epochs_per_fold=effective_epochs_per_fold, output_dir=args.output_dir,
    )

    if preds_df.empty:
        logger.error("Ningún fold produjo predicciones -- revisá --n-folds/--test-frac.")
        raise SystemExit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preds_csv = args.output_dir / f"pnl_predictions_{args.regime}.csv"
    preds_df.to_csv(preds_csv, index=False)

    if target_mode == "triple_barrier":
        if args.vol_target_annual is not None:
            logger.warning("--vol-target-annual se ignora en target_mode='triple_barrier' "
                            "(ver simulate_portfolio_triple_barrier).")
        daily_pnl = simulate_portfolio_triple_barrier(preds_df, cost_bps=args.cost_bps)
        cost_desc = "por trade (entry+exit)"
    else:
        daily_pnl = simulate_portfolio(preds_df, cost_bps=args.cost_bps, vol_target_annual=args.vol_target_annual,
                                        vol_lookback=args.vol_lookback, max_leverage=args.max_leverage)
        cost_desc = "por turnover"

    if daily_pnl.empty:
        logger.error("Ninguna fila operable (target_mode='triple_barrier': todas las predicciones fueron "
                      "'TIMEOUT') -- no hay P&L que reportar. Esto es en sí un resultado: el modelo no "
                      "encontró convicción suficiente en este split; probá con --n-folds/--epochs-per-fold "
                      "distintos o revisá la distribución de clases logueada durante el entrenamiento.")
        raise SystemExit(1)

    daily_csv = args.output_dir / f"pnl_daily_{args.regime}.csv"
    daily_pnl.to_csv(daily_csv)

    stats = summarize_pnl(daily_pnl)
    logger.info("=" * 78)
    logger.info("P&L -- régimen '%s' (target_mode=%s) | %d día(s) de test out-of-sample | costo=%.1fbps %s",
                args.regime, target_mode, stats["n_days"], args.cost_bps, cost_desc)
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
        logger.warning("Sharpe/retorno <= 0 con este costo asumido: el edge medido en walk_forward_eval.py NO "
                        "alcanza para cubrir %.1fbps de costo %s. Antes de operar, probá con costos reales de "
                        "tu bróker/exchange y/o (target_mode='triple_barrier') pt_mult/sl_mult/max_holding_days "
                        "distintos, o (target_mode='regression') un sizing menos ingenuo.", args.cost_bps, cost_desc)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(daily_pnl.index, daily_pnl["equity"], linewidth=1.5)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Break-even")
        ax.set_ylabel("Equity (base = 1.0)")
        ax.set_title(f"Curva de equity walk-forward -- {args.regime} (costo={args.cost_bps}bps, {target_mode})")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        chart_path = args.output_dir / f"pnl_equity_curve_{args.regime}.png"
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        logger.info("Chart -> %s", chart_path)
    except Exception as exc:  # noqa: BLE001 -- el chart es un extra, nunca debe tumbar el resumen
        logger.debug("No se pudo generar el chart (%r) -- no bloqueante.", exc)