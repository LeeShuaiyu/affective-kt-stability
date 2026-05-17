# Reproducibility Notes

This file records the experiment protocol used for the manuscript results.

## Dataset Splits

The experiments use learner-level splits. A learner sequence appears in only one of training, validation, or test.

The main dataset is ASSISTments Challenge. The processed split contains:

| Split | Learners | Interactions |
|---|---:|---:|
| Training | 1,093 | 582,510 |
| Validation | 274 | 135,773 |
| Test | 342 | 171,606 |

The boundary dataset is ASSISTments 2012--2013 with Affect. The processed full source contains:

| Split | Learners | Interactions |
|---|---:|---:|
| Training | 29,662 | 3,816,003 |
| Validation | 7,416 | 1,003,561 |
| Test | 9,270 | 1,224,782 |

The manuscript boundary experiment uses a fixed learner-level subset sampled from the processed source with seed `545194`: 1,000 training learners, 250 validation learners, and 300 test learners.

## Main Hyperparameters

| Item | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Learning-rate schedule | Step schedule |
| Batch size | 32 |
| Evaluation batch size | 32 |
| Maximum epochs | 30 |
| Early stopping patience | 8 |
| Latent state dimension | 128 |
| Dropout | 0.20 |
| Affect buckets | 5,000 |
| Auxiliary affect loss weight | 2.0 |
| Stability weight search | 0.05, 0.10, 0.20 |
| Final stability weight | 0.20 |
| Input masking training rate | 0.30 |
| Stability masking rate | 0.40 |
| Stability mismatch rate | 0.40 |
| Stability Gaussian noise standard deviation | 0.10 |
| Random seeds | 545194, 545195, 545196 |

## Evaluation Perturbations

| Perturbation | Evaluation strengths |
|---|---|
| Affect masking | 20%, 40%, 60% |
| Affect noise | Gaussian noise standard deviations 0.05, 0.10, 0.20 |
| Affect mismatch | 20%, 40%, 60% |

The robust score is

\[
R = A_0 - S,
\qquad
S=\max_k \max(0,A_0-A_k),
\]

where \(A_0\) is clean AUC and \(A_k\) is AUC under the \(k\)-th affect perturbation.

## Minimal Verification

Use the toy smoke test to verify that the software path works without external datasets:

```bash
bash scripts/run_smoke_test.sh
```

This does not reproduce the paper results; it verifies that training, checkpointing, clean evaluation, and perturbation evaluation execute end to end.
