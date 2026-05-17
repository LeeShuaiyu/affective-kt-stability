from pathlib import Path


def test_release_has_core_files():
    root = Path(__file__).resolve().parents[1]
    required = [
        "src/dekt/DEKTNet.py",
        "src/dekt/dekt_train.py",
        "src/dekt/load_data.py",
        "src/dekt/main.py",
        "scripts/make_toy_data.py",
        "scripts/run_smoke_test.sh",
        "results/tables/table1_clean_prediction_main.csv",
    ]
    for rel in required:
        assert (root / rel).exists(), rel


def test_no_large_raw_data_files_in_release():
    root = Path(__file__).resolve().parents[1]
    forbidden_suffixes = {".zip", ".tar", ".gz", ".pt", ".pth", ".params"}
    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_file():
            assert path.suffix not in forbidden_suffixes, path
            assert path.stat().st_size < 20 * 1024 * 1024, path

