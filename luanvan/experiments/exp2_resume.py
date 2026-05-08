"""
Experiment 2 Resume - Tu dong check va chay cac experiments con thieu.

Script nay se:
1. Doc config tu configs/exp2_config.json
2. Check ket qua nao da co trong results/exp2_personalized_accuracy/
3. Chi chay nhung combination con thieu
4. Bo qua nhung cai da chay xong

Usage:
    python exp2_resume.py                   # Check + chay missing
    python exp2_resume.py --dry-run         # Chi xem status, khong chay
    python exp2_resume.py --dataset fmnist  # Chi check fmnist
    python exp2_resume.py --force           # Chay lai tat ca
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple, Optional

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "exp2_config.json"
EXP2_RESULTS_DIR   = Path(__file__).parent / "results" / "exp2_personalized_accuracy"

METHODS       = ["fedavg", "fedprox", "fed_m3", "fed_dgd"]
METHOD_LABELS = {"fedavg": "FedAvg", "fedprox": "FedProx", "fed_m3": "Fed-M3", "fed_dgd": "Fed-DGD"}
DATASETS      = ["fmnist", "cifar10"]
ALPHAS        = [0.1, 0.5, 1.0]


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_completed(method: str, dataset: str, alpha: float) -> bool:
    """Kiem tra combination da chay xong chua."""
    files = list(EXP2_RESULTS_DIR.glob(f"personalized_results_*.json"))
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        key = f"{method}_{dataset}_a{alpha}"
        if key in data:
            return True
    return False


def print_status(methods: List[str], alphas: List[float], datasets: List[str]):
    done = []
    missing = []

    print(f"\n{'='*65}")
    print("STATUS CHECK - EXP2 PERSONALIZED ACCURACY")
    print(f"{'='*65}")

    for dataset in datasets:
        print(f"\n  {dataset.upper()}:")
        for method in methods:
            row_parts = []
            for alpha in alphas:
                if is_completed(method, dataset, alpha):
                    row_parts.append(f"a={alpha}:DONE")
                    done.append((method, dataset, alpha))
                else:
                    row_parts.append(f"a={alpha}:MISS")
                    missing.append((method, dataset, alpha))
            print(f"    {METHOD_LABELS[method]:<10}: {' | '.join(row_parts)}")

    total = len(done) + len(missing)
    print(f"\n  Done:    {len(done)}/{total}")
    print(f"  Missing: {len(missing)}/{total}")

    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2 Resume - Chay cac experiments con thieu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python exp2_resume.py                    # Check + chay missing
  python exp2_resume.py --dry-run          # Chi xem status
  python exp2_resume.py --dataset fmnist   # Chi fmnist
  python exp2_resume.py --force            # Chay lai tat ca
        """
    )

    parser.add_argument("--config",     type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset",    type=str, choices=["cifar10", "fmnist"])
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--force",      action="store_true")
    parser.add_argument("--no-confirm", action="store_true")

    args = parser.parse_args()

    config   = load_config(args.config)
    datasets = [args.dataset] if args.dataset else config.get("datasets", DATASETS)
    methods  = config.get("methods", METHODS)
    alphas   = config.get("alphas", ALPHAS)

    missing = print_status(methods, alphas, datasets)

    if args.dry_run:
        print("\n[DRY RUN] Bo --dry-run de chay thuc.")
        return

    to_run = list(product(methods, datasets, alphas)) if args.force else [
        (m, d, a) for m, d, a in product(methods, datasets, alphas)
        if (m, d, a) in [(x[0], x[1], x[2]) for x in missing]
    ]

    if not to_run:
        print("\nTat ca experiments da chay xong!")
        return

    print(f"\nSe chay {len(to_run)} combinations.")

    if not args.no_confirm:
        try:
            input("Press Enter de bat dau (Ctrl+C de huy)...")
        except KeyboardInterrupt:
            print("\nHuy.")
            return

    # Chay exp2 cho tung dataset
    for dataset in datasets:
        combos = [(m, a) for m, d, a in to_run if d == dataset]
        if not combos:
            continue

        method_args  = list(dict.fromkeys(m for m, _ in combos))
        alpha_args   = list(dict.fromkeys(str(a) for _, a in combos))

        cmd = [
            sys.executable, "exp2_personalized_accuracy.py",
            "--config", args.config,
            "--dataset", dataset,
            "--methods", *method_args,
            "--alphas", *alpha_args,
            "--no-confirm",
        ]

        print(f"\nRunning: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, cwd=Path(__file__).parent, check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] Failed for dataset={dataset}")


if __name__ == "__main__":
    main()