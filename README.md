# Stable Affective Knowledge Tracing under Unreliable Affect Observations

This repository contains the minimal release code for the paper draft on stable affective knowledge tracing under unreliable affect observations.

The code is intentionally small. It keeps only the repaired DEKT-style affective knowledge tracing model, the stability-training additions, perturbation evaluation utilities, paper-facing result tables, and a toy smoke test. Large datasets, checkpoints, server logs, and historical exploratory reports are not included.

## What is included

- `src/dekt/DEKTNet.py`: repaired dual-state affective knowledge tracing model.
- `src/dekt/dekt_train.py`: training, evaluation, affect perturbation, and stability-loss logic.
- `src/dekt/main.py`: command-line entry point for training and evaluation.
- `src/dekt/load_data.py`: sequence-data loader.
- `tools/reliability_kt/`: aggregation utilities used to produce paper-facing robustness tables.
- `results/tables/`: CSV tables used in the manuscript.
- `results/figures/`: rendered paper figures.
- `scripts/make_toy_data.py`: creates a tiny synthetic sequence dataset for smoke testing.
- `scripts/run_smoke_test.sh`: runs a short end-to-end training and perturbation evaluation test.

## Method summary

Let \(e_t\) be an observed affect vector and \(\tilde e_t^{(k)}\) be a perturbed affect vector under perturbation type \(k\). The same affective knowledge tracing model \(f_\theta\) is evaluated under normal and perturbed affect:

\[
p_t^0 = f_\theta(H_t, q_t, c_t, e_t),
\quad
p_t^k = f_\theta(H_t, q_t, c_t, \tilde e_t^{(k)}).
\]

The stability-trained model optimizes:

\[
\mathcal L = \mathcal L_{\mathrm{pred}} + \lambda \mathcal L_{\mathrm{stab}},
\quad
\mathcal L_{\mathrm{stab}} =
\frac{1}{|\Omega|}\sum_{(i,t)\in\Omega}(p_{i,t}^0-p_{i,t}^k)^2.
\]

Only affect observations are perturbed. Questions, concepts, time features, and response labels are unchanged.

## Data

The paper experiments use ASSISTments Challenge and ASSISTments 2012 with Affect. The full datasets are not redistributed here. Please obtain them from their original sources and convert them into the 16-line sequence format expected by `src/dekt/load_data.py`.

For testing the code path without external data, use the synthetic toy data generator:

```bash
python scripts/make_toy_data.py --output data/toy_affect
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already have a PyTorch environment, installing the remaining requirements is enough.

## Smoke test

```bash
bash scripts/run_smoke_test.sh
```

The smoke test creates a tiny synthetic dataset, trains for one epoch, evaluates clean affect and one perturbed-affect setting, and writes JSON metrics under `outputs/smoke/`.

## Reproducing the paper evidence

After preparing the full sequence datasets, use:

```bash
python tools/reliability_kt/run_reliability_kt_plan.py \
  --output-root outputs/challenge_formal \
  --datasets challenge \
  --models dekt,dekt_generic_robust_selected,dekt_stability,dekt_combined_robust \
  --seeds 545194,545195,545196 \
  --epochs 30 \
  --lr 0.001 \
  --dropout 0.2 \
  --scheduler step \
  --stability-weight 0.2 \
  --stability-perturbation mixed \
  --stability-rate 0.4 \
  --stability-noise-std 0.1 \
  --train-perturbation mask \
  --train-perturbation-rate 0.3 \
  --include-perturbations \
  --python-bin python
```

You may need to edit dataset paths in `tools/reliability_kt/run_reliability_kt_plan.py` or pass dataset-specific roots depending on where you place the downloaded data.

## Main reported results

The key paper tables are stored in `results/tables/`.

On the main dataset, the original affective KT model has normal AUC \(0.843228 \pm 0.000604\) and maximum perturbation drop \(0.162636 \pm 0.009772\). The stability-trained model has normal AUC \(0.844509 \pm 0.000439\) and maximum perturbation drop \(0.079296 \pm 0.002754\). The combined robust model further lowers the maximum perturbation drop to \(0.060253 \pm 0.000787\), with a small normal-AUC trade-off.

## Repository scope

This is a paper-release repository, not the full research workspace. It excludes:

- raw ASSISTments datasets;
- trained checkpoints;
- server logs;
- personal notes;
- intermediate failed experiments;
- manuscript drafts and local submission packages.

## Citation

Citation information will be added after the manuscript is submitted or accepted.

