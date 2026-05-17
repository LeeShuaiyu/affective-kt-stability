import argparse
import ast
import csv
import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from dekt_train import DEKT
from load_data import DATA


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_problem_to_skill(path: Path) -> dict[int, int]:
    payload = path.read_text(encoding="utf-8").strip()
    return ast.literal_eval(payload)


def generate_q_matrix(path: Path, n_skill: int, n_problem: int, gamma: float = 0.0) -> np.ndarray:
    problem2skill = load_problem_to_skill(path)
    q_matrix = np.zeros((n_problem, n_skill), dtype=np.float32) + gamma
    for problem_id, skill_id in problem2skill.items():
        q_matrix[problem_id][skill_id] = 1.0
    return q_matrix


def infer_problem_skill_counts(problem2skill_path: Path) -> tuple[int, int]:
    mapping = load_problem_to_skill(problem2skill_path)
    return max(mapping.keys()) + 1, max(mapping.values()) + 1


def build_problem_to_skill_vector(problem2skill_path: Path, n_problem: int) -> np.ndarray:
    mapping = load_problem_to_skill(problem2skill_path)
    vector = np.zeros((n_problem,), dtype=np.int64)
    for problem_id, skill_id in mapping.items():
        vector[problem_id] = skill_id
    return vector


def infer_feature_cardinalities(*data_tuples: tuple[np.ndarray, ...]) -> dict[str, int]:
    maxima = {
        "n_at": 0,
        "n_it": 0,
        "n_tp": 0,
        "n_att": 0,
        "n_qd": 0,
        "n_sd": 0,
    }
    for data_tuple in data_tuples:
        _, _, _, it_data, at_data, _, _, _, _, qd_data, sd_data, tp_data, _, _, att_data = data_tuple
        maxima["n_at"] = max(maxima["n_at"], int(at_data.max()) + 1)
        maxima["n_it"] = max(maxima["n_it"], int(it_data.max()) + 1)
        maxima["n_tp"] = max(maxima["n_tp"], int(tp_data.max()) + 1)
        maxima["n_att"] = max(maxima["n_att"], int(att_data.max()) + 1)
        maxima["n_qd"] = max(maxima["n_qd"], int(qd_data.max()) + 1)
        maxima["n_sd"] = max(maxima["n_sd"], int(sd_data.max()) + 1)
    return maxima


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate repaired or enhanced DEKT on a sequence dataset.")
    parser.add_argument("--seed", type=int, default=545194)
    parser.add_argument("--data-root", default="../data/anonymized_full_release_competition_dataset")
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--valid-split", default=None)
    parser.add_argument("--test-split", default=None)
    parser.add_argument("--q-matrix-path", default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--lr-decay-step", type=int, default=10)
    parser.add_argument("--lr-decay-rate", type=float, default=0.5)
    parser.add_argument("--scheduler", choices=("step", "cosine"), default="step")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--emotion-buckets", type=int, default=5000)
    parser.add_argument("--q-gamma", type=float, default=0.03)
    parser.add_argument("--graft-mode", choices=("off", "affect_blend", "both"), default="off")
    parser.add_argument("--reliability-mode", choices=("off", "learned", "fixed_one", "fixed_half"), default="off")
    parser.add_argument("--init-rho-bias", type=float, default=3.0)
    parser.add_argument("--affect-loss-weight", type=float, default=2.0)
    parser.add_argument("--stability-weight", type=float, default=0.0)
    parser.add_argument("--stability-perturbation", choices=("mixed", "mask", "noise", "mismatch"), default="mixed")
    parser.add_argument("--stability-rate", type=float, default=0.4)
    parser.add_argument("--stability-noise-std", type=float, default=0.1)
    parser.add_argument("--stability-seed", type=int, default=1729)
    parser.add_argument("--train-perturbation", choices=("clean", "mask", "noise", "mismatch"), default="clean")
    parser.add_argument("--train-perturbation-rate", type=float, default=0.0)
    parser.add_argument("--train-perturbation-noise-std", type=float, default=0.0)
    parser.add_argument("--train-perturbation-seed", type=int, default=8191)
    parser.add_argument("--eval-perturbation", choices=("clean", "mask", "noise", "mismatch"), default="clean")
    parser.add_argument("--eval-perturbation-rate", type=float, default=0.0)
    parser.add_argument("--eval-perturbation-noise-std", type=float, default=0.0)
    parser.add_argument("--eval-perturbation-seed", type=int, default=2718)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--save-path", default="params/dekt.params")
    parser.add_argument("--load-path", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--rho-record-path", default=None)
    return parser.parse_args()


def resolve_split_path(data_root: Path, explicit_path: Optional[str], filename: str) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return data_root / filename


def write_record_csv(records: dict[str, list], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records.keys())
    row_count = len(records[fieldnames[0]])
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(row_count):
            writer.writerow({field: records[field][idx] for field in fieldnames})


def main():
    args = parse_args()
    set_random_seed(args.seed)

    data_root = Path(args.data_root)
    train_split = resolve_split_path(data_root, args.train_split, "train0.txt")
    valid_split = resolve_split_path(data_root, args.valid_split, "valid0.txt")
    test_split = resolve_split_path(data_root, args.test_split, "test.txt")
    q_matrix_path = Path(args.q_matrix_path) if args.q_matrix_path else data_root / "problem2skill"
    eval_batch_size = args.eval_batch_size or args.batch_size

    dat = DATA(seqlen=500, separate_char=",")
    train_data = dat.load_data(str(train_split))
    valid_data = dat.load_data(str(valid_split))
    test_data = dat.load_data(str(test_split))

    n_exercise, n_question = infer_problem_skill_counts(q_matrix_path)
    feature_sizes = infer_feature_cardinalities(train_data, valid_data, test_data)
    problem_to_skill = build_problem_to_skill_vector(q_matrix_path, n_exercise)

    logging.getLogger().setLevel(logging.INFO)

    dekt = DEKT(
        n_at=feature_sizes["n_at"],
        n_it=feature_sizes["n_it"],
        n_exercise=n_exercise,
        n_question=n_question,
        d_a=50,
        d_e=args.state_dim,
        d_k=args.state_dim,
        d_m=args.emotion_buckets,
        q_matrix=None,
        n_qd=feature_sizes["n_qd"],
        n_sd=feature_sizes["n_sd"],
        n_tp=feature_sizes["n_tp"],
        n_att=feature_sizes["n_att"],
        batch_size=args.batch_size,
        dropout=args.dropout,
        eval_batch_size=eval_batch_size,
        graft_mode=args.graft_mode,
        problem_to_skill=problem_to_skill,
        q_gamma=args.q_gamma,
        reliability_mode=args.reliability_mode,
        init_rho_bias=args.init_rho_bias,
    )
    if args.skip_train:
        train_summary = {"skipped": True, "save_path": args.save_path}
    else:
        train_summary = dekt.train(
            train_data,
            valid_data,
            epoch=args.epochs,
            lr=args.lr,
            lr_decay_step=args.lr_decay_step,
            lr_decay_rate=args.lr_decay_rate,
            save_path=args.save_path,
            scheduler_name=args.scheduler,
            min_lr=args.min_lr,
            early_stop_patience=args.patience or None,
            affect_loss_weight=args.affect_loss_weight,
            stability_weight=args.stability_weight,
            stability_perturbation=args.stability_perturbation,
            stability_rate=args.stability_rate,
            stability_noise_std=args.stability_noise_std,
            stability_seed=args.stability_seed,
            train_perturbation=args.train_perturbation,
            train_perturbation_rate=args.train_perturbation_rate,
            train_perturbation_noise_std=args.train_perturbation_noise_std,
            train_perturbation_seed=args.train_perturbation_seed,
            eval_batch_size=eval_batch_size,
            return_summary=True,
        )

    load_path = args.load_path or args.save_path
    dekt.load(load_path)
    test_summary = dekt.eval(
        test_data,
        batch_size=eval_batch_size,
        return_summary=True,
        perturbation=args.eval_perturbation,
        perturbation_rate=args.eval_perturbation_rate,
        perturbation_noise_std=args.eval_perturbation_noise_std,
        perturbation_seed=args.eval_perturbation_seed,
        return_records=bool(args.rho_record_path),
    )
    valid_eval_summary = dekt.eval(
        valid_data,
        batch_size=eval_batch_size,
        return_summary=True,
        perturbation=args.eval_perturbation,
        perturbation_rate=args.eval_perturbation_rate,
        perturbation_noise_std=args.eval_perturbation_noise_std,
        perturbation_seed=args.eval_perturbation_seed,
        return_records=False,
    )
    rho_records = test_summary.pop("records", None)
    print(
        "seed: %d, graft_mode: %s, reliability_mode: %s, perturbation: %s, auc: %.6f, accuracy: %.6f, rmse: %.6f, ece: %.6f"
        % (
            args.seed,
            args.graft_mode,
            args.reliability_mode,
            args.eval_perturbation,
            test_summary["auc"],
            test_summary["accuracy"],
            test_summary["rmse"],
            test_summary["ece"],
        )
    )

    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": args.seed,
            "data_root": str(data_root),
            "train_split": str(train_split),
            "valid_split": str(valid_split),
            "test_split": str(test_split),
            "q_matrix_path": str(q_matrix_path),
            "config": {
                "epochs": args.epochs,
                "lr": args.lr,
                "lr_decay_step": args.lr_decay_step,
                "lr_decay_rate": args.lr_decay_rate,
                "scheduler": args.scheduler,
                "min_lr": args.min_lr,
                "batch_size": args.batch_size,
                "eval_batch_size": eval_batch_size,
                "dropout": args.dropout,
                "state_dim": args.state_dim,
                "emotion_buckets": args.emotion_buckets,
                "q_gamma": args.q_gamma,
                "graft_mode": args.graft_mode,
                "reliability_mode": args.reliability_mode,
                "init_rho_bias": args.init_rho_bias,
                "affect_loss_weight": args.affect_loss_weight,
                "stability_weight": args.stability_weight,
                "stability_perturbation": args.stability_perturbation,
                "stability_rate": args.stability_rate,
                "stability_noise_std": args.stability_noise_std,
                "stability_seed": args.stability_seed,
                "train_perturbation": args.train_perturbation,
                "train_perturbation_rate": args.train_perturbation_rate,
                "train_perturbation_noise_std": args.train_perturbation_noise_std,
                "train_perturbation_seed": args.train_perturbation_seed,
                "eval_perturbation": args.eval_perturbation,
                "eval_perturbation_rate": args.eval_perturbation_rate,
                "eval_perturbation_noise_std": args.eval_perturbation_noise_std,
                "eval_perturbation_seed": args.eval_perturbation_seed,
                "patience": args.patience,
                "skip_train": args.skip_train,
            },
            "cardinalities": {
                "n_exercise": n_exercise,
                "n_question": n_question,
                **feature_sizes,
            },
            "train_summary": train_summary,
            "valid_eval_summary": valid_eval_summary,
            "test_summary": test_summary,
        }
        metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.rho_record_path:
        write_record_csv(rho_records or {}, Path(args.rho_record_path))


if __name__ == "__main__":
    main()
