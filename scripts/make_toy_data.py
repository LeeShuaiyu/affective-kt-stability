#!/usr/bin/env python3
"""Create a tiny synthetic affective KT sequence dataset for smoke tests."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def write_split(path: Path, num_sequences: int, seed: int) -> None:
    rng = random.Random(seed)
    lines: list[str] = []
    for student in range(1, num_sequences + 1):
        length = rng.randint(8, 12)
        skills = [rng.randint(1, 4) for _ in range(length)]
        answers = [rng.randint(0, 1) for _ in range(length)]
        exercises = [skill + rng.randint(0, 1) * 4 for skill in skills]
        intervals = [rng.randint(1, 5) for _ in range(length)]
        answer_times = [rng.randint(1, 6) for _ in range(length)]
        boredom = [round(rng.uniform(0.05, 0.70), 4) for _ in range(length)]
        concentration = [round(rng.uniform(0.30, 0.95), 4) for _ in range(length)]
        confusion = [round(rng.uniform(0.00, 0.70), 4) for _ in range(length)]
        frustration = [round(rng.uniform(0.00, 0.70), 4) for _ in range(length)]
        qd = [rng.randint(1, 5) for _ in range(length)]
        sd = [rng.randint(1, 5) for _ in range(length)]
        tp = [rng.randint(1, 3) for _ in range(length)]
        stu = [student for _ in range(length)]
        pre = [0 for _ in range(length)]
        att = [rng.randint(1, 5) for _ in range(length)]

        fields = [
            [length],
            skills,
            answers,
            exercises,
            intervals,
            answer_times,
            boredom,
            concentration,
            confusion,
            frustration,
            qd,
            sd,
            tp,
            stu,
            pre,
            att,
        ]
        for field in fields:
            lines.append(",".join(str(x) for x in field))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/toy_affect")
    args = parser.parse_args()

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    write_split(root / "train0.txt", num_sequences=10, seed=11)
    write_split(root / "valid0.txt", num_sequences=6, seed=17)
    write_split(root / "test.txt", num_sequences=6, seed=23)
    mapping = {idx: max(1, ((idx - 1) % 4) + 1) for idx in range(0, 9)}
    mapping[0] = 0
    (root / "problem2skill").write_text(str(mapping), encoding="utf-8")
    print(f"wrote toy data to {root}")


if __name__ == "__main__":
    main()

