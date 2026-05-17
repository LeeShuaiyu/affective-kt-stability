from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path

import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


METRICS = ["auc", "accuracy", "nll", "brier", "ece"]
PERTURBATION_LABELS = {
    "mask": "mask",
    "noise": "noise",
    "mismatch": "mismatch",
}


def mark_main_method(frame: pd.DataFrame, main_model: str) -> pd.DataFrame:
    frame = frame.copy()
    if "model" in frame and "paper_name" in frame:
        mask = frame["model"] == main_model
        frame.loc[mask, "paper_name"] = frame.loc[mask, "paper_name"].astype(str).map(
            lambda text: text if "本文方法" in text else f"{text} (本文方法)"
        )
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create publication evidence summaries and paired tests.")
    parser.add_argument("--metric-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-model", default="dekt")
    parser.add_argument("--main-model", default="dekt_combined_robust")
    return parser.parse_args()


def mean_std_summary(frame: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    summary = frame.groupby(group_cols, as_index=False)[value_cols].agg(["mean", "std"])
    summary.columns = ["_".join(item).strip("_") for item in summary.columns.to_flat_index()]
    return summary


def paired_tests(rows: pd.DataFrame, baseline_model: str, compare_models: list[str]) -> pd.DataFrame:
    outputs = []
    for dataset in sorted(rows["dataset"].dropna().unique()):
        data = rows.loc[rows["dataset"] == dataset]
        for value in ("clean_auc", "max_delta_auc", "robust_score"):
            base = data.loc[data["model"] == baseline_model, ["seed", value]].dropna()
            for model in compare_models:
                other = data.loc[data["model"] == model, ["seed", value]].dropna()
                merged = base.merge(other, on="seed", suffixes=("_baseline", "_model"))
                if merged.empty:
                    continue
                diff = merged[f"{value}_model"] - merged[f"{value}_baseline"]
                record = {
                    "dataset": dataset,
                    "comparison": f"{model} vs {baseline_model}",
                    "metric": value,
                    "n": int(len(diff)),
                    "mean_difference": float(diff.mean()),
                    "std_difference": float(diff.std(ddof=1)) if len(diff) > 1 else math.nan,
                }
                if stats is not None and len(diff) >= 2:
                    try:
                        record["paired_t_p"] = float(stats.ttest_rel(merged[f"{value}_model"], merged[f"{value}_baseline"]).pvalue)
                    except Exception:
                        record["paired_t_p"] = math.nan
                    try:
                        record["wilcoxon_p"] = float(stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
                    except Exception:
                        record["wilcoxon_p"] = math.nan
                outputs.append(record)
    return pd.DataFrame(outputs)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = mark_main_method(pd.read_csv(args.metric_rows), args.main_model)
    clean = metrics.loc[metrics["perturbation"] == "clean"].copy()
    clean = clean.rename(columns={"auc": "clean_auc"})
    clean_cols = ["dataset", "model", "paper_name", "seed", "clean_auc", "accuracy", "nll", "brier", "ece"]
    clean[clean_cols].to_csv(output_dir / "clean_rows.csv", index=False)
    mean_std_summary(clean, ["dataset", "model", "paper_name"], ["clean_auc", "accuracy", "nll", "brier", "ece"]).to_csv(
        output_dir / "clean_summary.csv",
        index=False,
    )

    clean_auc = clean.set_index(["dataset", "model", "seed"])["clean_auc"].to_dict()
    perturb = metrics.loc[metrics["perturbation"] != "clean"].copy()
    perturb = perturb.loc[perturb["perturbation"].isin(PERTURBATION_LABELS)].copy()
    if not perturb.empty:
        perturb["clean_auc"] = [
            clean_auc.get((row.dataset, row.model, row.seed), math.nan)
            for row in perturb.itertuples(index=False)
        ]
        perturb["delta_auc"] = perturb["clean_auc"] - perturb["auc"]
        perturb.to_csv(output_dir / "perturbation_delta_rows.csv", index=False)
        mean_std_summary(
            perturb,
            ["dataset", "model", "paper_name", "perturbation", "perturbation_rate", "perturbation_noise_std"],
            ["delta_auc"],
        ).to_csv(output_dir / "perturbation_delta_summary.csv", index=False)

    robust_rows = []
    for (dataset, model, seed), group in perturb.groupby(["dataset", "model", "seed"]):
        clean_row = clean.loc[(clean["dataset"] == dataset) & (clean["model"] == model) & (clean["seed"] == seed)]
        if clean_row.empty:
            continue
        max_delta = float(group["delta_auc"].max())
        clean_value = float(clean_row.iloc[0]["clean_auc"])
        robust_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "paper_name": clean_row.iloc[0]["paper_name"],
                "seed": int(seed),
                "clean_auc": clean_value,
                "max_delta_auc": max_delta,
                "robust_score": clean_value - max_delta,
            }
        )
    robust = pd.DataFrame(robust_rows)
    robust.to_csv(output_dir / "robust_score_rows.csv", index=False)
    if not robust.empty:
        mean_std_summary(robust, ["dataset", "model", "paper_name"], ["clean_auc", "max_delta_auc", "robust_score"]).to_csv(
            output_dir / "robust_score_summary.csv",
            index=False,
        )
        compare_models = [model for model in sorted(robust["model"].unique()) if model != args.baseline_model]
        paired_tests(robust, args.baseline_model, compare_models).to_csv(output_dir / "paired_tests.csv", index=False)

    if args.main_model in robust["model"].unique():
        thresholds = []
        for threshold in (0.002, 0.005, 0.010, 0.020):
            for dataset, data in robust.groupby("dataset"):
                max_sensitivity = data.loc[data["model"] == args.baseline_model, "max_delta_auc"]
                if max_sensitivity.empty:
                    continue
                thresholds.append(
                    {
                        "dataset": dataset,
                        "tau": threshold,
                        "baseline_mean_max_delta_auc": float(max_sensitivity.mean()),
                        "diagnosis": "affect_sensitive" if float(max_sensitivity.mean()) >= threshold else "low_affect_reliance",
                    }
                )
        pd.DataFrame(thresholds).to_csv(output_dir / "threshold_sensitivity.csv", index=False)

    print(output_dir)


if __name__ == "__main__":
    main()
