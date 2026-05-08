"""
Experiment 2 Report - In Table 2: Personalized Accuracy.

Format bang:
  Dataset  α   FedAvg          FedProx         Fed-M3          Fed-DGD
           |   Min(%) Mean(%)  Min(%) Mean(%)  Min(%) Mean(%)  Min(%) Mean(%)
  FMNIST   0.1  –      –        –      –        –      –        –      –
  ...

Usage:
    python exp2_report.py               # In bang tu results mac dinh
    python exp2_report.py --save-csv    # Luu ra CSV
    python exp2_report.py --results-dir ./results/exp2_personalized_accuracy
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional

METHODS       = ["fedavg", "fedprox", "fed_m3", "fed_dgd"]
METHOD_LABELS = {"fedavg": "FedAvg", "fedprox": "FedProx", "fed_m3": "Fed-M3", "fed_dgd": "Fed-DGD"}
DATASETS      = ["fmnist", "cifar10"]
ALPHAS        = [0.1, 0.5, 1.0]

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results" / "exp2_personalized_accuracy"


# ============================================================================
# LOAD
# ============================================================================

def load_all_results(results_dir: Path) -> Dict:
    """Doc tat ca file JSON trong results_dir, merge lai thanh 1 dict."""
    merged = {}
    for f in sorted(results_dir.glob("personalized_results_*.json")):
        with open(f) as fp:
            data = json.load(fp)
        # File moi ghi de len file cu (neu chay lai)
        merged.update(data)
    return merged


def get_result(data: Dict, method: str, dataset: str, alpha: float) -> Optional[Dict]:
    key = f"{method}_{dataset}_a{alpha}"
    return data.get(key)


# ============================================================================
# TABLE 2
# ============================================================================

def print_table2(data: Dict, datasets: List[str]):
    print(f"\n{'='*90}")
    print("TABLE 2. Personalized accuracy after local fine-tuning on 10% of each client's local data.")
    print(f"{'='*90}")

    # Header row 1
    col_w = 16
    header1 = f"{'Dataset':<10} {'α':<5} " + "".join([f"{METHOD_LABELS[m]:^{col_w}}" for m in METHODS])
    print(header1)

    # Header row 2
    sub = "Min (%)  Mean (%)"
    header2 = f"{'':10} {'':5} " + "".join([f"{sub:^{col_w}}" for _ in METHODS])
    print(header2)
    print("-" * len(header1))

    for dataset in datasets:
        first_row = True
        for alpha in ALPHAS:
            row = f"{dataset.upper() if first_row else '':<10} {alpha:<5} "
            for method in METHODS:
                r = get_result(data, method, dataset, alpha)
                if r is None:
                    row += f"{'–':>6}   {'–':>6}  "
                else:
                    row += f"{r['min_acc']:>6.2f}   {r['mean_acc']:>6.2f}  "
            print(row)
            first_row = False
        print()


def save_table2_csv(data: Dict, datasets: List[str], output_path: str):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ["Dataset", "Alpha"]
        for method in METHODS:
            label = METHOD_LABELS[method]
            header += [f"{label} Min (%)", f"{label} Mean (%)"]
        writer.writerow(header)

        for dataset in datasets:
            for alpha in ALPHAS:
                row = [dataset.upper(), alpha]
                for method in METHODS:
                    r = get_result(data, method, dataset, alpha)
                    if r is None:
                        row += ["–", "–"]
                    else:
                        row += [f"{r['min_acc']:.2f}", f"{r['mean_acc']:.2f}"]
                writer.writerow(row)

    print(f"  CSV saved: {output_path}")


# ============================================================================
# STATUS
# ============================================================================

def print_status(data: Dict, datasets: List[str]):
    print(f"\n{'='*65}")
    print("STATUS: Ket qua exp2 hien co")
    print(f"{'='*65}")

    total = done = 0
    for dataset in datasets:
        print(f"\n  {dataset.upper()}:")
        for method in METHODS:
            parts = []
            for alpha in ALPHAS:
                total += 1
                r = get_result(data, method, dataset, alpha)
                if r is not None:
                    done += 1
                    parts.append(f"a={alpha}:DONE")
                else:
                    parts.append(f"a={alpha}:MISS")
            print(f"    {METHOD_LABELS[method]:<10}: {' | '.join(parts)}")

    print(f"\n  Tong: {done}/{total}")
    if done < total:
        print(f"  Con thieu {total-done} runs - chay 'python exp2_resume.py' de tiep tuc")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2 Report - In Table 2: Personalized Accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python exp2_report.py                            # In bang
  python exp2_report.py --save-csv                 # Luu CSV
  python exp2_report.py --dataset cifar10          # Chi CIFAR-10
  python exp2_report.py --results-dir ./my_results # Dung folder khac
        """
    )

    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--dataset",     type=str, choices=["cifar10", "fmnist"])
    parser.add_argument("--save-csv",    action="store_true")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    datasets    = [args.dataset] if args.dataset else DATASETS

    if not results_dir.exists():
        print(f"[ERROR] Results dir khong ton tai: {results_dir}")
        print("Chay exp2_personalized_accuracy.py truoc.")
        return

    print(f"\nLoading results from: {results_dir}")
    data = load_all_results(results_dir)

    if not data:
        print("[ERROR] Khong co ket qua nao. Chay exp2_personalized_accuracy.py truoc.")
        return

    # Status
    print_status(data, datasets)

    # Table 2
    print_table2(data, datasets)

    # Save CSV
    if args.save_csv:
        csv_path = str(results_dir / "table2_personalized_accuracy.csv")
        save_table2_csv(data, datasets, csv_path)


if __name__ == "__main__":
    main()