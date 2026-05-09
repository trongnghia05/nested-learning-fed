"""
Experiment 3 Report - Table 4: Effect of Number of Clients on Global Accuracy.

Format bảng:
  Dataset  α   Num Clients   FedAvg(%)  FedProx(%)  Fed-M3(%)  Fed-DGD(%)
  FMNIST   0.1      10         –          –           –          –
                    20         –          –           –          –
                    30         –          –           –          –
                    50         –          –           –          –
  ...

Usage:
    python exp3_report.py               # In bảng từ results mặc định
    python exp3_report.py --save-csv    # Lưu ra CSV
    python exp3_report.py --results-dir ./results/exp3_num_clients
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional

METHODS       = ["fedavg", "fedprox", "fed_m3", "fed_dgd"]
METHOD_LABELS = {"fedavg": "FedAvg", "fedprox": "FedProx", "fed_m3": "Fed-M3", "fed_dgd": "Fed-DGD"}
DATASETS      = ["fmnist", "cifar10"]
ALPHAS        = [0.1, 0.5]
NUM_CLIENTS_LIST = [10, 20, 30, 50]

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results" / "exp3_num_clients"


# ============================================================================
# LOAD RESULTS
# ============================================================================

def load_all_results(results_dir: Path) -> Dict:
    """
    Đọc tất cả metrics files từ results_dir.

    Structure:
        results_dir/
        ├── method/
        │   └── dataset_dirichlet_aX.X_cXX/
        │       └── metrics_*.json
    """
    results = {}

    for method in METHODS:
        method_dir = results_dir / method
        if not method_dir.exists():
            continue

        # Scan all subdirectories
        for run_dir in method_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Parse directory name: dataset_dirichlet_a0.5_c10
            dir_name = run_dir.name
            if "_dirichlet_a" not in dir_name:
                continue

            try:
                # Extract dataset, alpha, num_clients
                parts = dir_name.split('_')
                dataset = parts[0]
                alpha_str = parts[2]  # a0.5
                alpha = float(alpha_str[1:])  # 0.5

                # Extract num_clients from directory name (if exists)
                # Format: dataset_dirichlet_aX.X_cXX
                if len(parts) >= 4 and parts[3].startswith('c'):
                    num_clients = int(parts[3][1:])  # c10 -> 10
                else:
                    # Fallback: infer from metrics file
                    metrics_files = list(run_dir.glob("metrics_*.json"))
                    if not metrics_files:
                        continue
                    # Use most recent metrics file
                    metrics_file = max(metrics_files, key=lambda x: x.stat().st_mtime)
                    with open(metrics_file) as f:
                        data = json.load(f)
                    # Infer num_clients from first round client data
                    rounds = data.get('rounds', [])
                    if rounds and 'clients' in rounds[0]:
                        num_clients = len(rounds[0]['clients'])
                    else:
                        continue

                # Find best accuracy
                metrics_files = list(run_dir.glob("metrics_*.json"))
                if not metrics_files:
                    continue

                best_acc = 0.0
                for metrics_file in metrics_files:
                    with open(metrics_file) as f:
                        data = json.load(f)
                    rounds = data.get('rounds', [])
                    if rounds:
                        run_best = max(r['server']['test_acc'] for r in rounds)
                        best_acc = max(best_acc, run_best)

                # Store result
                key = (method, dataset, alpha, num_clients)
                if key not in results or results[key] < best_acc:
                    results[key] = best_acc

            except Exception as e:
                print(f"Warning: Could not parse {run_dir}: {e}")
                continue

    return results


def get_result(data: Dict, method: str, dataset: str, alpha: float, num_clients: int) -> Optional[float]:
    """Get accuracy for a specific configuration."""
    key = (method, dataset, alpha, num_clients)
    return data.get(key)


# ============================================================================
# TABLE 4
# ============================================================================

def print_table4(data: Dict, datasets: List[str]):
    print(f"\n{'='*100}")
    print("TABLE 4: Effect of Number of Clients on Global Accuracy")
    print(f"{'='*100}")
    print("(Server Test Accuracy - Best across all rounds)")
    print()

    # Header
    col_w = 12
    header = f"{'Dataset':<10} {'α':<5} {'Clients':<10} " + "".join([f"{METHOD_LABELS[m]:>{col_w}}" for m in METHODS])
    print(header)
    print("-" * len(header))

    for dataset in datasets:
        first_dataset = True
        for alpha in ALPHAS:
            first_alpha = True
            for num_clients in NUM_CLIENTS_LIST:
                # Build row
                dataset_str = dataset.upper() if first_dataset else ""
                alpha_str = str(alpha) if first_alpha else ""

                row = f"{dataset_str:<10} {alpha_str:<5} {num_clients:<10} "

                for method in METHODS:
                    acc = get_result(data, method, dataset, alpha, num_clients)
                    if acc is None:
                        row += f"{'–':>{col_w}}"
                    else:
                        row += f"{acc*100:>{col_w}.2f}"

                print(row)
                first_alpha = False
                first_dataset = False

            # Blank line after each alpha
            print()

    print(f"{'='*100}\n")


def save_table4_csv(data: Dict, datasets: List[str], output_path: str):
    """Save Table 4 as CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ["Dataset", "Alpha", "Num_Clients"]
        for method in METHODS:
            header.append(f"{METHOD_LABELS[method]} (%)")
        writer.writerow(header)

        # Data rows
        for dataset in datasets:
            for alpha in ALPHAS:
                for num_clients in NUM_CLIENTS_LIST:
                    row = [dataset.upper(), alpha, num_clients]
                    for method in METHODS:
                        acc = get_result(data, method, dataset, alpha, num_clients)
                        if acc is None:
                            row.append("")
                        else:
                            row.append(f"{acc*100:.2f}")
                    writer.writerow(row)

    print(f"CSV saved to: {output_path}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3 Report - Effect of Number of Clients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--results-dir", type=str,
                       default=str(DEFAULT_RESULTS_DIR),
                       help="Results directory")
    parser.add_argument("--datasets", nargs="+",
                       default=DATASETS,
                       choices=DATASETS,
                       help="Datasets to include in report")
    parser.add_argument("--save-csv", action="store_true",
                       help="Save table as CSV")
    parser.add_argument("--output", type=str,
                       help="Output CSV path (default: auto-generated)")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print(f"\nRun experiments first:")
        print(f"  python exp3_num_clients.py")
        return

    print(f"Loading results from: {results_dir}")
    data = load_all_results(results_dir)

    if not data:
        print("No results found!")
        print("\nRun experiments first:")
        print("  python exp3_num_clients.py")
        return

    print(f"Loaded {len(data)} result(s)")

    # Print table
    print_table4(data, args.datasets)

    # Save CSV if requested
    if args.save_csv:
        if args.output:
            csv_path = args.output
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = results_dir / f"table4_num_clients_{timestamp}.csv"

        save_table4_csv(data, args.datasets, str(csv_path))


if __name__ == "__main__":
    main()
