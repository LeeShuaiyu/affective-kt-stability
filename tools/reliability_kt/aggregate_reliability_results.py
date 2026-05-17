from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = ["auc", "accuracy", "nll", "brier", "ece"]


def paper_name(model: object, fallback: object) -> object:
    if model == "dekt_combined_robust":
        text = str(fallback)
        return text if "本文方法" in text else f"{text} (本文方法)"
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate reliability KT experiment metrics into paper-facing tables.")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_metric_row(row: pd.Series) -> dict[str, object] | None:
    metrics_path = row.get("metrics_path")
    if not isinstance(metrics_path, str) or not metrics_path:
        return None
    path = Path(metrics_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    test = payload["test_summary"]
    config = payload["config"]
    record = {
        "dataset": row.get("dataset"),
        "model": row.get("model"),
        "paper_name": paper_name(row.get("model"), row.get("paper_name")),
        "seed": int(row.get("seed")),
        "stage": row.get("stage"),
        "perturbation": config.get("eval_perturbation", "clean"),
        "perturbation_rate": float(config.get("eval_perturbation_rate", 0.0)),
        "perturbation_noise_std": float(config.get("eval_perturbation_noise_std", 0.0)),
        "best_epoch": payload["train_summary"].get("best_epoch"),
        "checkpoint_path": payload.get("checkpoint_path") or config.get("save_path"),
    }
    for metric in METRICS:
        record[metric] = float(test[metric])
        valid = payload.get("valid_eval_summary")
        if valid is not None and metric in valid:
            record[f"valid_{metric}"] = float(valid[metric])
    perturbation_info = test.get("perturbation_info", {})
    for key in ("selected", "same_skill", "fallback", "same_skill_rate", "fallback_rate"):
        if key in perturbation_info:
            record[f"perturbation_{key}"] = float(perturbation_info[key])
    record["stability_weight"] = float(config.get("stability_weight", 0.0))
    record["train_perturbation"] = config.get("train_perturbation", "clean")
    record["train_perturbation_rate"] = float(config.get("train_perturbation_rate", 0.0))
    return record


def load_public_baseline_rows(row: pd.Series) -> list[dict[str, object]]:
    if row.get("model") != "dkt":
        return []
    output_dir = row.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        return []
    summaries = list(Path(output_dir).glob("*_summary.json"))
    if not summaries:
        return []
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    records = []
    for run in payload.get("runs", []):
        test = run.get("test", {})
        record = {
            "dataset": row.get("dataset"),
            "model": "dkt",
            "paper_name": paper_name("dkt", row.get("paper_name", "DKT")),
            "seed": int(run.get("seed", 0)),
            "stage": "clean",
            "perturbation": "clean",
            "perturbation_rate": 0.0,
            "perturbation_noise_std": 0.0,
            "best_epoch": run.get("best_epoch"),
            "checkpoint_path": run.get("checkpoint_path"),
            "stability_weight": 0.0,
            "train_perturbation": "not_applicable",
            "train_perturbation_rate": 0.0,
        }
        for metric in METRICS:
            if metric in test:
                record[metric] = float(test[metric])
        records.append(record)
    return records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest_csv)
    rows = []
    for _, row in manifest.iterrows():
        metric_row = load_metric_row(row)
        if metric_row is not None:
            rows.append(metric_row)
        rows.extend(load_public_baseline_rows(row))
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError("no metrics could be loaded from manifest")
    metrics.to_csv(output_dir / "all_metric_rows.csv", index=False)

    clean = metrics.loc[metrics["perturbation"] == "clean"].copy()
    clean.to_csv(output_dir / "clean_main_table_rows.csv", index=False)
    clean_summary = (
        clean.groupby(["dataset", "model", "paper_name"], as_index=False)[METRICS]
        .agg(["mean", "std"])
    )
    clean_summary.columns = ["_".join(item).strip("_") for item in clean_summary.columns.to_flat_index()]
    clean_summary.to_csv(output_dir / "clean_main_table_summary.csv", index=False)

    clean_key = clean.set_index(["dataset", "model", "seed"])["auc"].to_dict()
    perturb = metrics.loc[metrics["perturbation"] != "clean"].copy()
    if not perturb.empty:
        perturb["clean_auc"] = [
            clean_key.get((row.dataset, row.model, row.seed), float("nan"))
            for row in perturb.itertuples(index=False)
        ]
        perturb["delta_auc"] = perturb["clean_auc"] - perturb["auc"]
        perturb.to_csv(output_dir / "perturbation_delta_rows.csv", index=False)
        delta_summary = (
            perturb.groupby(["dataset", "model", "paper_name", "perturbation", "perturbation_rate", "perturbation_noise_std"], as_index=False)["delta_auc"]
            .agg(["mean", "std"])
        )
        delta_summary.columns = ["_".join(item).strip("_") for item in delta_summary.columns.to_flat_index()]
        delta_summary.to_csv(output_dir / "perturbation_delta_summary.csv", index=False)

    print(output_dir)


if __name__ == "__main__":
    main()
