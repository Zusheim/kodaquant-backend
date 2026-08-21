"""
walk_forward_eval.py -- MEDICIÓN HONESTA de acertividad direccional (V1)
=============================================================================
Un solo split 80/20 (`train_kodaquant_v5.run_regime_pipeline`) da UN número
de directional accuracy por activo. Ese número puede ser una racha de suerte
del régimen de mercado que le tocó a ese 20% final, o al revés. Este script
NO reemplaza al entrenamiento de producción -- lo AUDITA: repite el ciclo
completo (fit de scalers -> build_model -> train_model -> evaluate_asset)
sobre múltiples ventanas walk-forward (expanding window, estrictamente
cronológicas, sin fuga de información futura) y reporta la DISTRIBUCIÓN de
directional accuracy, no un único punto.

Cero cambios a la matemática/topología/loss de `train_kodaquant_v5.py`: este
script IMPORTA `build_model`/`train_model`/`evaluate_asset`/`engineer_asset`
tal cual están (mismo grafo, mismos hiperparámetros, misma
DirectionalGaussianNLL) -- lo único nuevo acá es la ORQUESTACIÓN de folds y
la agregación estadística. La única lógica duplicada es la generalización de
`build_asset_dataset` (fija a un único `train_ratio`) a fronteras explícitas
de fold -- ver `_build_fold_dataset`, misma fórmula exacta (mismo
`MinMaxScaler`/`StandardScaler(with_mean=False)`/target de log-return/
construcción de ventanas), solo parametrizada distinto.

USO:
    python ml_engine/walk_forward_eval.py --regime equity_specialist \
        --n-folds 4 --epochs-per-fold 30

    (con 4 folds x hasta 100 épocas c/u, esto es CARO -- cada fold reentrena
    la red desde cero. Bajá --epochs-per-fold para iterar rápido; subilo a
    EPOCHS (100) para la medición final "de verdad".)

COSTO: cada fold = un entrenamiento completo nuevo (misma red que
`run_regime_pipeline`, EarlyStopping propio). Con `--n-folds 4
--epochs-per-fold 30` esperá ~4x el tiempo de un `train_kodaquant_v5.py`
recortado a 30 épocas -- correlo offline, no interactivo.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- RESOLUCIÓN ABSOLUTA DE IMPORTS (mismo bootstrap que el resto de
# ml_engine/ -- ver train_kodaquant_v5.py/data_pipeline.py) -----------------
def _bootstrap_project_root() -> Path:
    here = Path(__file__).resolve()
    for ancestor in (here.parent, *here.parents):
        if (ancestor / "ml_engine").is_dir() and (ancestor / "services").is_dir():
            return ancestor
    return here.parent.parent


_PROJECT_ROOT = _bootstrap_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import como MÓDULO (no `python -c`): el bloque `if __name__ == "__main__"`
# de train_kodaquant_v5.py NO se ejecuta acá (ese guard existe justamente
# para que sea seguro importarlo), así que esto NO dispara un entrenamiento
# completo -- solo trae funciones/clases/constantes ya definidas y probadas.
import ml_engine.train_kodaquant_v5 as tkv5  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
logger = logging.getLogger("kodaquant.walk_forward")


def _build_fold_dataset(df_asset: pd.DataFrame, ticker: str, lookback: int,
                         train_end: int, test_end: int, asset_to_id: dict[str, int],
                         feature_cols: list[str]):
    """
    Misma matemática EXACTA que `tkv5.build_asset_dataset` -- mismo
    `MinMaxScaler(feature_range=(0,1))` fiteado solo en train, mismo target
    `StandardScaler(with_mean=False)` (preserva signo del retorno), misma
    construcción de ventanas -- generalizada a fronteras EXPLÍCITAS
    (`train_end`/`test_end`, posiciones enteras) en vez de un único
    `train_ratio` fijo, para poder generar múltiples folds walk-forward sin
    tocar `train_kodaquant_v5.py`. Devuelve `None` si el fold no tiene
    suficiente historia para este activo (se salta, no se rellena con
    basura).
    """
    prices = df_asset["RAW_CLOSE"].values
    features = df_asset[feature_cols].values
    log_returns = np.diff(np.log(prices))
    n = len(df_asset)
    test_end = min(test_end, n)

    if train_end <= lookback or test_end <= train_end:
        return None

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(features[:train_end])
    features_scaled = feature_scaler.transform(features)

    X, y_raw, last_price, dates = [], [], [], []
    for t in range(lookback, test_end):
        X.append(features_scaled[t - lookback: t])
        y_raw.append(log_returns[t - 1])
        last_price.append(prices[t - 1])
        dates.append(df_asset.index[t])

    X = np.array(X, dtype=np.float32)
    y_raw = np.array(y_raw, dtype=np.float32).reshape(-1, 1)
    last_price = np.array(last_price, dtype=np.float32)
    dates = pd.DatetimeIndex(dates)

    window_split = train_end - lookback
    if window_split <= 0 or window_split >= len(y_raw):
        return None

    target_scaler = StandardScaler(with_mean=False)
    target_scaler.fit(y_raw[:window_split])
    y_scaled = target_scaler.transform(y_raw).astype(np.float32)

    asset_id = np.full((len(X),), asset_to_id[ticker], dtype=np.int32)

    train = dict(X=X[:window_split], y=y_scaled[:window_split], asset_id=asset_id[:window_split],
                 last_price=last_price[:window_split], dates=dates[:window_split])
    test = dict(X=X[window_split:], y=y_scaled[window_split:], asset_id=asset_id[window_split:],
                last_price=last_price[window_split:], dates=dates[window_split:])
    if len(train["X"]) == 0 or len(test["X"]) == 0:
        return None
    return train, test, feature_scaler, target_scaler


def run_walk_forward(regime_name: str, n_folds: int, initial_train_frac: float,
                      test_frac: float, epochs_per_fold: int, output_dir: Path) -> pd.DataFrame:
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

    logger.info("Ingeniería de features (una sola vez, se reparte en folds por posición)...")
    dfs = {t: tkv5.engineer_asset(all_market_data, t, tkv5.MACRO_TICKERS) for t in tickers}
    n_per_ticker = {t: len(dfs[t]) for t in tickers}
    logger.info("Filas engineered por activo: %s", n_per_ticker)

    tmp_ckpt = output_dir / "_walk_forward_tmp.keras"
    records: list[dict] = []

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
            built = _build_fold_dataset(dfs[t], t, tkv5.LOOKBACK, train_end, test_end, asset_to_id, feature_cols)
            if built is None:
                skipped.append(t)
                continue
            train, test, _f_scaler, t_scaler = built
            fold_test[t] = (test, t_scaler)
            X_train_parts.append(train["X"])
            y_train_parts.append(train["y"])
            aid_train_parts.append(train["asset_id"])

        if not X_train_parts:
            logger.warning("Fold %d/%d: sin datos suficientes para ningún activo -- se salta.", fold_idx + 1, n_folds)
            continue

        X_train = np.concatenate(X_train_parts, axis=0)
        y_train = np.concatenate(y_train_parts, axis=0)
        aid_train = np.concatenate(aid_train_parts, axis=0).reshape(-1, 1)

        n_fit_samples = int(len(X_train) * (1 - tkv5.VALIDATION_SPLIT))
        steps_per_epoch = max(1, n_fit_samples // tkv5.BATCH_SIZE)

        logger.info("Fold %d/%d: X_train=%s | activos evaluables=%d%s",
                    fold_idx + 1, n_folds, X_train.shape, len(fold_test),
                    f" | saltados={skipped}" if skipped else "")

        # Red NUEVA por fold (mismos hiperparámetros/arquitectura que
        # producción vía `regime_name`) -- nunca se reusan pesos entre
        # folds, cada uno mide generalización desde cero.
        model, gamma_var = tkv5.build_model(
            n_timesteps=tkv5.LOOKBACK, n_features=tkv5.N_FEATURES, n_assets=len(tickers),
            steps_per_epoch=steps_per_epoch, regime_name=regime_name,
            embed_dim=tkv5.ASSET_EMBED_DIM, huber_delta=tkv5.HUBER_DELTA,
            gamma_initial=tkv5.GAMMA_INITIAL, variance_lambda=tkv5.VARIANCE_LAMBDA,
            variance_cap=tkv5.VARIANCE_CAP, gaussian_nll_beta=tkv5.GAUSSIAN_NLL_BETA,
            log_var_min=tkv5.LOG_VAR_MIN, log_var_max=tkv5.LOG_VAR_MAX,
            log_var_l2_lambda=tkv5.LOG_VAR_L2_LAMBDA, log_var_barrier_lambda=tkv5.LOG_VAR_BARRIER_LAMBDA,
            model_name=f"KodaQuant_{regime_name}_WF_fold{fold_idx + 1}",
        )
        _history, model = tkv5.train_model(
            model, X_train, aid_train, y_train, checkpoint_path=tmp_ckpt,
            epochs=epochs_per_fold, batch_size=tkv5.BATCH_SIZE, validation_split=tkv5.VALIDATION_SPLIT,
            gamma_variable=gamma_var, gamma_max=tkv5.GAMMA_MAX,
            gamma_warmup_epochs=min(tkv5.GAMMA_WARMUP_EPOCHS, max(1, epochs_per_fold // 3)),
            gamma_schedule=tkv5.GAMMA_SCHEDULE, gamma_sigmoid_steepness=tkv5.GAMMA_SIGMOID_STEEPNESS,
        )

        for t, (test, t_scaler) in fold_test.items():
            r = tkv5.evaluate_asset(model, test, t, t_scaler)
            records.append({
                "fold": fold_idx + 1, "ticker": t, "n_test": len(test["X"]),
                "directional_accuracy": r["directional_accuracy"],
                "mae_price": r["mae_price"], "rmse_price": r["rmse_price"],
            })
            logger.info("  Fold %d  %-10s  n_test=%3d  DirAcc=%.1f%%",
                        fold_idx + 1, t, len(test["X"]), r["directional_accuracy"] * 100)

        tmp_ckpt.unlink(missing_ok=True)
        try:
            tkv5.keras.backend.clear_session()  # libera memoria -- red nueva cada fold
        except Exception:  # noqa: BLE001 -- best-effort, nunca debe tumbar la corrida
            pass

    tmp_ckpt.unlink(missing_ok=True)
    return pd.DataFrame.from_records(records)


def summarize(df: pd.DataFrame, output_dir: Path, seed: int) -> None:
    if df.empty:
        logger.error("Ningún fold produjo resultados -- revisá --n-folds/--test-frac vs. el historial disponible.")
        return

    csv_path = output_dir / f"walk_forward_results_seed{seed}.csv"
    df.to_csv(csv_path, index=False)

    logger.info("=" * 78)
    logger.info("RESUMEN POR ACTIVO (across %d fold(s))", df["fold"].nunique())
    for ticker, g in df.groupby("ticker"):
        logger.info("  %-10s  DirAcc media=%.1f%%  std=%.1f pp  min=%.1f%%  max=%.1f%%  (n_folds=%d)",
                    ticker, g["directional_accuracy"].mean() * 100, g["directional_accuracy"].std(ddof=0) * 100,
                    g["directional_accuracy"].min() * 100, g["directional_accuracy"].max() * 100, len(g))

    acc = df["directional_accuracy"].to_numpy()
    mean_acc = float(acc.mean())
    std_acc = float(acc.std(ddof=1)) if len(acc) > 1 else 0.0
    n = len(acc)

    logger.info("-" * 78)
    logger.info("VEREDICTO GLOBAL (%d observaciones ticker x fold, walk-forward, sin fuga de datos):", n)
    logger.info("  DirAcc media = %.2f%%   desvío = %.2f pp", mean_acc * 100, std_acc * 100)

    try:
        from scipy import stats
        if n > 1 and std_acc > 0:
            t_stat, p_value = stats.ttest_1samp(acc, popmean=0.5)
            ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean_acc, scale=std_acc / np.sqrt(n))
            logger.info("  IC 95%% = [%.2f%%, %.2f%%]   H0: media=50%% -> p=%.4f (%s)",
                        ci_low * 100, ci_high * 100, p_value,
                        "se rechaza H0, hay evidencia de edge real" if p_value < 0.05
                        else "NO se rechaza H0 -- no hay evidencia estadística de superar 50%")
        else:
            logger.info("  Muy pocas observaciones para un test estadístico confiable -- corré más folds.")
    except ImportError:
        se = std_acc / np.sqrt(n) if n > 0 else 0.0
        ci_low, ci_high = mean_acc - 1.96 * se, mean_acc + 1.96 * se
        logger.info("  (scipy no disponible -- IC 95%% aproximado por normal: [%.2f%%, %.2f%%])",
                    ci_low * 100, ci_high * 100)
        logger.info("  Si el IC completo queda por debajo/incluye 50%%, NO hay evidencia sólida de edge real todavía.")

    logger.info("=" * 78)
    logger.info("CSV completo (por fold x activo) -> %s", csv_path)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        order = sorted(df["ticker"].unique())
        data = [df.loc[df["ticker"] == t, "directional_accuracy"] * 100 for t in order]
        ax.boxplot(data, tick_labels=order, showmeans=True)
        ax.axhline(50.0, color="red", linestyle="--", linewidth=1, label="Coinflip (50%)")
        ax.set_ylabel("Directional Accuracy (%)")
        ax.set_title(f"Walk-forward directional accuracy -- {df['fold'].nunique()} fold(s)")
        ax.legend()
        fig.tight_layout()
        chart_path = output_dir / f"walk_forward_chart_seed{seed}.png"
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        logger.info("Chart -> %s", chart_path)
    except Exception as exc:  # noqa: BLE001 -- el chart es un extra, nunca debe tumbar el resumen
        logger.debug("No se pudo generar el chart (%r) -- no bloqueante.", exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validación walk-forward honesta de directional accuracy.")
    parser.add_argument("--regime", default="equity_specialist", choices=list(tkv5.REGIMES))
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--initial-train-frac", type=float, default=0.60,
                         help="Fracción inicial de historia usada como train antes del primer fold de test.")
    parser.add_argument("--test-frac", type=float, default=0.08,
                         help="Tamaño de cada bloque de test walk-forward, como fracción de la historia total.")
    parser.add_argument("--epochs-per-fold", type=int, default=30,
                         help="EPOCHS del train_kodaquant_v5.py de producción es 100 -- bajalo acá para iterar "
                              "rápido; subilo (hasta 100) para la medición final. EarlyStopping sigue activo.")
    parser.add_argument("--output-dir", type=Path, default=tkv5.BASE_DIR / "walk_forward_reports")
    parser.add_argument("--seed", type=int, default=None,
                         help="Sobrescribe SEED=42 (fijo en train_kodaquant_v5.py -- sin esto, toda corrida "
                              "es idéntica bit a bit). Correr 3-5 veces con semillas distintas y promediar "
                              "'DirAcc media' separa señal real de varianza de inicialización -- especialmente "
                              "relevante en crypto_specialist, con std de 3.8-6.2pp entre folds.")
    args = parser.parse_args()

    if args.seed is not None:
        tkv5.keras.utils.set_random_seed(args.seed)

    results_df = run_walk_forward(
        regime_name=args.regime, n_folds=args.n_folds,
        initial_train_frac=args.initial_train_frac, test_frac=args.test_frac,
        epochs_per_fold=args.epochs_per_fold, output_dir=args.output_dir,
    )
    summarize(results_df, args.output_dir, seed=args.seed if args.seed is not None else tkv5.SEED)