"""
Experiment 3 Resume - Tự động check và chạy các experiments còn thiếu.

Script này sẽ:
1. Đọc config từ configs/exp3_config.json
2. Check kết quả nào đã có trong results/exp3_num_clients/
3. Chỉ chạy những combination còn thiếu
4. Bỏ qua những cái đã chạy xong

Usage:
    python exp3_resume.py                   # Check + chạy missing
    python exp3_resume.py --dry-run         # Chỉ xem status, không chạy
    python exp3_resume.py --dataset fmnist  # Chỉ check fmnist
    python exp3_resume.py --force           # Chạy lại tất cả
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "exp3_config.json"
EXP3_RESULTS_DIR   = Path(__file__).parent / "results" / "exp3_num_clients"

METHODS       = ["fedavg", "fedprox", "fed_m3", "fed_dgd"]
METHOD_LABELS = {"fedavg": "FedAvg", "fedprox": "FedProx", "fed_m3": "Fed-M3", "fed_dgd": "Fed-DGD"}
DATASETS      = ["fmnist", "cifar10"]
ALPHAS        = [0.1, 0.5]
NUM_CLIENTS_LIST = [10, 20, 30, 50]


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_completed(method: str, dataset: str, alpha: float, num_clients: int) -> bool:
    """Kiểm tra xem combination đã chạy xong chưa."""
    # Check trong folder: method/dataset_dirichlet_aX.X_cXX/
    folder = EXP3_RESULTS_DIR / method / f"{dataset}_dirichlet_a{alpha}_c{num_clients}"

    if not folder.exists():
        return False

    # Check if metrics file exists
    metrics_files = list(folder.glob("metrics_*.json"))
    if not metrics_files:
        return False

    # Verify metrics file has valid data
    try:
        metrics_file = max(metrics_files, key=lambda x: x.stat().st_mtime)
        with open(metrics_file) as f:
            data = json.load(f)
        rounds = data.get('rounds', [])
        return len(rounds) > 0
    except:
        return False


def print_status(methods: List[str], datasets: List[str], alphas: List[float], num_clients_list: List[int]):
    """Print status matrix."""
    done = []
    missing = []

    print(f"\n{'='*80}")
    print("STATUS CHECK - EXP3: EFFECT OF NUMBER OF CLIENTS")
    print(f"{'='*80}")

    for dataset in datasets:
        print(f"\n  {dataset.upper()}:")
        for alpha in alphas:
            print(f"    Alpha = {alpha}:")
            for method in methods:
                row_parts = []
                for num_clients in num_clients_list:
                    if is_completed(method, dataset, alpha, num_clients):
                        row_parts.append(f"N={num_clients}:✓")
                        done.append((method, dataset, alpha, num_clients))
                    else:
                        row_parts.append(f"N={num_clients}:✗")
                        missing.append((method, dataset, alpha, num_clients))
                print(f"      {METHOD_LABELS[method]:<10}: {' '.join(row_parts)}")

    total = len(done) + len(missing)
    print(f"\n  Done:    {len(done)}/{total}")
    print(f"  Missing: {len(missing)}/{total}")

    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3 Resume - Chạy các experiments còn thiếu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python exp3_resume.py                    # Check + chạy missing
  python exp3_resume.py --dry-run          # Chỉ xem status
  python exp3_resume.py --dataset fmnist   # Chỉ fmnist
  python exp3_resume.py --force            # Chạy lại tất cả
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
    num_clients_list = config.get("num_clients_list", NUM_CLIENTS_LIST)

    missing = print_status(methods, datasets, alphas, num_clients_list)

    if args.dry_run:
        print("\n[DRY RUN] Bỏ --dry-run để chạy thực.")
        return

    to_run = list(product(methods, datasets, alphas, num_clients_list)) if args.force else missing

    if not to_run:
        print("\nTất cả experiments đã chạy xong!")
        return

    print(f"\nSẽ chạy {len(to_run)} combinations.")

    if not args.no_confirm:
        try:
            input("Press Enter để bắt đầu (Ctrl+C để hủy)...")
        except KeyboardInterrupt:
            print("\nHủy.")
            return

    # Run missing experiments
    for method, dataset, alpha, num_clients in to_run:
        print(f"\n{'='*70}")
        print(f"Running: {method.upper()} | {dataset.upper()} | α={alpha} | N={num_clients}")
        print(f"{'='*70}")

        cmd = [
            sys.executable, "exp3_num_clients.py",
            "--config", args.config,
            "--methods", method,
            "--datasets", dataset,
            "--alphas", str(alpha),
            "--num-clients", str(num_clients),
            "--no-confirm",
        ]

        try:
            subprocess.run(cmd, cwd=Path(__file__).parent, check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] Failed: {method} {dataset} α={alpha} N={num_clients}")

    print(f"\n{'='*70}")
    print("RESUME COMPLETED")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
