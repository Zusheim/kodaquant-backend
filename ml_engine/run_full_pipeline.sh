#!/usr/bin/env bash
# run_full_pipeline.sh -- KodaQuant, ciclo completo post-auditoría 2026-08
# Entrenamiento (épocas de producción) + walk-forward + backtest de P&L,
# equity_specialist y crypto_specialist. Corré esto DESDE ml_engine/.
set -euo pipefail

cd "$(dirname "$0")"

echo "=== [1/6] Full retrain -- equity_specialist + crypto_specialist (EPOCHS=100/180) ==="
python train_kodaquant_v5.py

echo "=== [2/6] Walk-forward EQUITY (sin --epochs-per-fold -> usa EPOCHS=100 de producción) ==="
python walk_forward_eval.py --regime equity_specialist --n-folds 4

echo "=== [3/6] Walk-forward CRYPTO (sin --epochs-per-fold -> usa epochs=180 de producción) ==="
python walk_forward_eval.py --regime crypto_specialist --n-folds 4

echo "=== [4/6] PnL Backtest EQUITY (5bps, sizing igual-ponderado) ==="
python pnl_backtest.py --regime equity_specialist --n-folds 4 --cost-bps 5

echo "=== [5/6] PnL Backtest CRYPTO (10bps, target_mode=triple_barrier) ==="
python pnl_backtest.py --regime crypto_specialist --n-folds 4 --cost-bps 10

echo "=== [6/6] Fine-tuning incremental MLOps (Champion vs Challenger, ambos regímenes) ==="
python train_pipeline.py

echo "=== LISTO -- reportes en ml_engine/walk_forward_reports/ ==="