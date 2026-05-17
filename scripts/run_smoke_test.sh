#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" scripts/make_toy_data.py --output data/toy_affect
mkdir -p outputs/smoke/checkpoints outputs/smoke/metrics

"$PYTHON_BIN" src/dekt/main.py \
  --seed 7 \
  --data-root data/toy_affect \
  --epochs 1 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --state-dim 16 \
  --emotion-buckets 100 \
  --q-gamma 0.0 \
  --lr 0.001 \
  --dropout 0.1 \
  --scheduler step \
  --affect-loss-weight 0.1 \
  --stability-weight 0.1 \
  --stability-perturbation mixed \
  --stability-rate 0.4 \
  --stability-noise-std 0.1 \
  --patience 0 \
  --save-path outputs/smoke/checkpoints/dekt_toy.pt \
  --metrics-path outputs/smoke/metrics/clean.json

"$PYTHON_BIN" src/dekt/main.py \
  --seed 7 \
  --data-root data/toy_affect \
  --skip-train \
  --batch-size 4 \
  --eval-batch-size 4 \
  --state-dim 16 \
  --emotion-buckets 100 \
  --q-gamma 0.0 \
  --load-path outputs/smoke/checkpoints/dekt_toy.pt \
  --save-path outputs/smoke/checkpoints/dekt_toy.pt \
  --eval-perturbation mask \
  --eval-perturbation-rate 0.4 \
  --metrics-path outputs/smoke/metrics/mask_0p4.json

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

for name in ["clean", "mask_0p4"]:
    path = Path("outputs/smoke/metrics") / f"{name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    auc = payload["test_summary"]["auc"]
    assert 0.0 <= auc <= 1.0, (name, auc)
    print(f"{name}: auc={auc:.4f}, accuracy={payload['test_summary']['accuracy']:.4f}")
print("smoke test passed")
PY
