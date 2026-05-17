from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import deque
from contextlib import nullcontext
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "src" / "dekt"
MAIN_PY = CODE_ROOT / "main.py"
PYKT_ROOT = REPO_ROOT / "external" / "pykt-toolkit"
PUBLIC_DATA_ROOT = REPO_ROOT / "data" / "public_baselines_pykt"
DEFAULT_PYTHON = sys.executable


DATASETS = {
    "challenge": {
        "data_root": REPO_ROOT / "data" / "challenge",
        "public_root": PUBLIC_DATA_ROOT / "assistchall_public",
        "batch_size": 32,
    },
    "assist2012": {
        "data_root": REPO_ROOT / "data" / "assist2012",
        "public_root": PUBLIC_DATA_ROOT / "assist2012_public",
        "batch_size": 32,
    },
}


MODEL_CONFIGS = {
    "dekt": {
        "paper_name": "DEKT",
        "reliability_mode": "off",
        "stability_weight": 0.0,
        "train_perturbation": "clean",
        "train_perturbation_rate": 0.0,
    },
    "dekt_stability": {
        "paper_name": "DEKT with stability objective",
        "reliability_mode": "off",
        "stability_weight": None,
        "train_perturbation": "clean",
        "train_perturbation_rate": 0.0,
    },
    "dekt_generic_robust": {
        "paper_name": "DEKT with input affect masking",
        "reliability_mode": "off",
        "stability_weight": 0.0,
        "train_perturbation": "mask",
        "train_perturbation_rate": 0.4,
    },
    "dekt_generic_robust_selected": {
        "paper_name": "DEKT with input affect masking (selected rate)",
        "reliability_mode": "off",
        "stability_weight": 0.0,
        "train_perturbation": "mask",
        "train_perturbation_rate": None,
    },
    "dekt_combined_robust": {
        "paper_name": "DEKT with input affect masking and stability objective (本文方法)",
        "reliability_mode": "off",
        "stability_weight": None,
        "train_perturbation": "mask",
        "train_perturbation_rate": None,
    },
    "trusted_affect": {
        "paper_name": "Reliability-Calibrated Affective Knowledge Tracing",
        "reliability_mode": "learned",
        "stability_weight": None,
    },
    "fixed_half": {
        "paper_name": "Fixed reliability control",
        "reliability_mode": "fixed_half",
        "stability_weight": None,
    },
    "trusted_no_stability": {
        "paper_name": "Reliability gate without stability objective",
        "reliability_mode": "learned",
        "stability_weight": 0.0,
    },
    "dkt": {
        "paper_name": "DKT",
        "public_baseline": True,
    },
}


PERTURBATION_GRID = [
    ("mask", 0.2, 0.0),
    ("mask", 0.4, 0.0),
    ("mask", 0.6, 0.0),
    ("noise", 0.0, 0.05),
    ("noise", 0.0, 0.10),
    ("noise", 0.0, 0.20),
    ("mismatch", 0.2, 0.0),
    ("mismatch", 0.4, 0.0),
    ("mismatch", 0.6, 0.0),
]

MEDIUM_PERTURBATION_GRID = [
    ("mask", 0.4, 0.0),
    ("noise", 0.0, 0.10),
    ("mismatch", 0.4, 0.0),
]


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reliability-calibrated affective knowledge tracing plan.")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "reports" / "reliability_kt_20260503"))
    parser.add_argument("--datasets", default="challenge")
    parser.add_argument("--models", default="dekt,dekt_stability,dekt_generic_robust,dkt")
    parser.add_argument("--seeds", default="545194")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--assist2012-epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alt-lr", type=float, default=0.0015)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--emotion-buckets", type=int, default=5000)
    parser.add_argument("--q-gamma", type=float, default=0.03)
    parser.add_argument("--affect-loss-weight", type=float, default=2.0)
    parser.add_argument("--stability-weight", type=float, default=0.1)
    parser.add_argument("--stability-perturbation", choices=("mixed", "mask", "noise", "mismatch"), default="mixed")
    parser.add_argument("--stability-rate", type=float, default=0.4)
    parser.add_argument("--stability-noise-std", type=float, default=0.1)
    parser.add_argument("--train-perturbation", choices=("clean", "mask", "noise", "mismatch"), default="clean")
    parser.add_argument("--train-perturbation-rate", type=float, default=0.0)
    parser.add_argument("--train-perturbation-noise-std", type=float, default=0.0)
    parser.add_argument("--train-perturbation-seed", type=int, default=8191)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--scheduler", choices=("step", "cosine"), default="step")
    parser.add_argument("--perturbation-seed", type=int, default=2718)
    parser.add_argument("--medium-only", action="store_true")
    parser.add_argument("--include-perturbations", action="store_true")
    parser.add_argument("--include-dkt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--pykt-root", default=str(PYKT_ROOT))
    return parser.parse_args()


def sanitize(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_").replace(".", "p")


def run_command(command: list[str], dry_run: bool, log_path: Path | None = None) -> str:
    if dry_run:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("dry_run\n" + " ".join(command) + "\n", encoding="utf-8")
        return "dry_run"
    tail: deque[str] = deque(maxlen=120)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    with (log_path.open("w", encoding="utf-8") if log_path is not None else nullcontext(None)) as log_handle:
        if log_path is not None:
            log_handle.write("$ " + " ".join(command) + "\n")
            log_handle.flush()
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            tail.append(line)
            if log_path is not None:
                log_handle.write(line)
                log_handle.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command, output="".join(tail))
    return "".join(tail)[-4000:]


def write_manifest(rows: list[dict[str, object]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "manifest.json"
    csv_path = output_root / "manifest.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def model_value(model_config: dict[str, object], key: str, default: object) -> object:
    value = model_config.get(key, default)
    return default if value is None else value


def main_command(
    args: argparse.Namespace,
    dataset_name: str,
    model_name: str,
    seed: int,
    checkpoint_path: Path,
    metrics_path: Path,
    *,
    skip_train: bool,
    perturbation: str = "clean",
    perturbation_rate: float = 0.0,
    perturbation_noise_std: float = 0.0,
    rho_record_path: Path | None = None,
) -> list[str]:
    dataset = DATASETS[dataset_name]
    model_config = MODEL_CONFIGS[model_name]
    stability_weight = model_config["stability_weight"]
    if stability_weight is None:
        stability_weight = args.stability_weight
    train_perturbation = model_value(model_config, "train_perturbation", args.train_perturbation)
    train_perturbation_rate = model_value(model_config, "train_perturbation_rate", args.train_perturbation_rate)
    train_perturbation_noise_std = model_value(model_config, "train_perturbation_noise_std", args.train_perturbation_noise_std)
    epochs = args.assist2012_epochs if dataset_name == "assist2012" else args.epochs
    command = [
        args.python_bin,
        str(MAIN_PY),
        "--seed",
        str(seed),
        "--data-root",
        str(dataset["data_root"]),
        "--epochs",
        str(epochs),
        "--lr",
        str(args.lr),
        "--scheduler",
        args.scheduler,
        "--batch-size",
        str(dataset["batch_size"]),
        "--eval-batch-size",
        str(dataset["batch_size"]),
        "--dropout",
        str(args.dropout),
        "--state-dim",
        str(args.state_dim),
        "--emotion-buckets",
        str(args.emotion_buckets),
        "--q-gamma",
        str(args.q_gamma),
        "--graft-mode",
        "off",
        "--reliability-mode",
        str(model_config["reliability_mode"]),
        "--affect-loss-weight",
        str(args.affect_loss_weight),
        "--stability-weight",
        str(stability_weight),
        "--stability-perturbation",
        args.stability_perturbation,
        "--stability-rate",
        str(args.stability_rate),
        "--stability-noise-std",
        str(args.stability_noise_std),
        "--train-perturbation",
        str(train_perturbation),
        "--train-perturbation-rate",
        str(train_perturbation_rate),
        "--train-perturbation-noise-std",
        str(train_perturbation_noise_std),
        "--train-perturbation-seed",
        str(args.train_perturbation_seed),
        "--patience",
        str(args.patience),
        "--save-path",
        str(checkpoint_path),
        "--metrics-path",
        str(metrics_path),
        "--eval-perturbation",
        perturbation,
        "--eval-perturbation-rate",
        str(perturbation_rate),
        "--eval-perturbation-noise-std",
        str(perturbation_noise_std),
        "--eval-perturbation-seed",
        str(args.perturbation_seed),
    ]
    if skip_train:
        command.extend(["--skip-train", "--load-path", str(checkpoint_path)])
    if rho_record_path is not None:
        command.extend(["--rho-record-path", str(rho_record_path)])
    return command


def run_dkt(args: argparse.Namespace, dataset_name: str, seed_text: str, output_root: Path) -> dict[str, object]:
    runner = REPO_ROOT / "tools" / "public_baselines" / "run_public_baseline_pykt.py"
    if not runner.exists():
        raise FileNotFoundError(
            "DKT baseline execution requires a separate pyKT runner. "
            "Place it at tools/public_baselines/run_public_baseline_pykt.py "
            "or run DKT with your own pyKT installation."
        )
    dataset = DATASETS[dataset_name]
    output_dir = output_root / "dkt" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "logs" / f"{dataset_name}_dkt.log"
    command = [
        args.python_bin,
        str(runner),
        "--pykt-root",
        args.pykt_root,
        "--dataset-root",
        str(dataset["public_root"]),
        "--model-name",
        "dkt",
        "--output-dir",
        str(output_dir),
        "--seeds",
        seed_text,
        "--num-epochs",
        str(args.assist2012_epochs if dataset_name == "assist2012" else args.epochs),
        "--batch-size",
        "64",
        "--eval-batch-size",
        "256",
        "--learning-rate",
        str(args.lr),
    ]
    summaries = list(output_dir.glob("*_summary.json"))
    if args.resume and summaries:
        stdout_tail = "skipped_existing"
        status = "skipped_existing"
    else:
        stdout_tail = run_command(command, args.dry_run, log_path)
        status = "planned" if args.dry_run else "completed"
    return {
        "dataset": dataset_name,
        "model": "dkt",
        "paper_name": MODEL_CONFIGS["dkt"]["paper_name"],
        "stage": "public_baseline",
        "command": " ".join(command),
        "status": status,
        "stdout_tail": stdout_tail,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    (output_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)
    (output_root / "rho_records").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    datasets = parse_list(args.datasets)
    models = parse_list(args.models)
    seeds = [int(seed) for seed in parse_list(args.seeds)]
    perturbations = MEDIUM_PERTURBATION_GRID if args.medium_only else PERTURBATION_GRID
    rows: list[dict[str, object]] = []

    for dataset_name in datasets:
        if dataset_name not in DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        for model_name in models:
            if model_name not in MODEL_CONFIGS:
                raise ValueError(f"Unsupported model: {model_name}")
            if model_name == "dkt":
                if args.include_dkt:
                    rows.append(run_dkt(args, dataset_name, args.seeds, output_root))
                continue
            for seed in seeds:
                tag = sanitize(f"{dataset_name}_{model_name}_seed{seed}")
                checkpoint_path = output_root / "checkpoints" / f"{tag}.pt"
                metrics_path = output_root / "metrics" / f"{tag}_clean.json"
                log_path = output_root / "logs" / f"{tag}_clean.log"
                rho_record_path = output_root / "rho_records" / f"{tag}_clean.csv" if model_name == "trusted_affect" else None
                command = main_command(
                    args,
                    dataset_name,
                    model_name,
                    seed,
                    checkpoint_path,
                    metrics_path,
                    skip_train=False,
                    rho_record_path=rho_record_path,
                )
                if args.resume and metrics_path.exists() and checkpoint_path.exists():
                    stdout_tail = "skipped_existing"
                    status = "skipped_existing"
                else:
                    stdout_tail = run_command(command, args.dry_run, log_path)
                    status = "planned" if args.dry_run else "completed"
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "paper_name": MODEL_CONFIGS[model_name]["paper_name"],
                        "stage": "clean",
                        "seed": seed,
                        "checkpoint_path": str(checkpoint_path),
                        "metrics_path": str(metrics_path),
                        "rho_record_path": str(rho_record_path) if rho_record_path else "",
                        "command": " ".join(command),
                        "status": status,
                        "stdout_tail": stdout_tail,
                        "log_path": str(log_path),
                    }
                )
                write_manifest(rows, output_root)
                if args.include_perturbations:
                    for perturbation, rate, noise_std in perturbations:
                        perturb_tag = sanitize(f"{tag}_{perturbation}_r{rate}_n{noise_std}")
                        perturb_metrics_path = output_root / "metrics" / f"{perturb_tag}.json"
                        perturb_log_path = output_root / "logs" / f"{perturb_tag}.log"
                        command = main_command(
                            args,
                            dataset_name,
                            model_name,
                            seed,
                            checkpoint_path,
                            perturb_metrics_path,
                            skip_train=True,
                            perturbation=perturbation,
                            perturbation_rate=rate,
                            perturbation_noise_std=noise_std,
                        )
                        if args.resume and perturb_metrics_path.exists():
                            stdout_tail = "skipped_existing"
                            status = "skipped_existing"
                        else:
                            stdout_tail = run_command(command, args.dry_run, perturb_log_path)
                            status = "planned" if args.dry_run else "completed"
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "model": model_name,
                                "paper_name": MODEL_CONFIGS[model_name]["paper_name"],
                                "stage": perturbation,
                                "perturbation_rate": rate,
                                "perturbation_noise_std": noise_std,
                                "seed": seed,
                                "checkpoint_path": str(checkpoint_path),
                                "metrics_path": str(perturb_metrics_path),
                                "command": " ".join(command),
                                "status": status,
                                "stdout_tail": stdout_tail,
                                "log_path": str(perturb_log_path),
                            }
                        )
                        write_manifest(rows, output_root)
        write_manifest(rows, output_root)


if __name__ == "__main__":
    main()
