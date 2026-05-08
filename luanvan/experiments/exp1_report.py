"""
Experiment 1 Report - Tong hop ket qua va in cac bang trong paper.

In ra:
  - Table 2: Global Accuracy (Final & Best)
  - Table 3: Convergence Speed (so rounds de dat target accuracy)
  - Plot: Accuracy curves theo rounds

Usage:
    python exp1_report.py                   # In tat ca bang
    python exp1_report.py --no-plot         # Chi in bang, khong ve do thi
    python exp1_report.py --save-csv        # Luu bang ra file CSV
    python exp1_report.py --dataset cifar10 # Chi report 1 dataset
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# CONFIG
# ============================================================================

METHODS = ["fedavg", "fed_m3", "fed_dgd", "fedprox"]
METHOD_LABELS = {
    "fedavg":  "FedAvg",
    "fed_m3":  "Fed-M3",
    "fed_dgd": "Fed-DGD",
    "fedprox": "FedProx",
}
DATASETS = ["fmnist", "cifar10"]
ALPHAS = [0.1, 0.5, 1.0]

# Target accuracy cho Table 3 (Convergence Speed)
CONVERGENCE_TARGETS = {
    "cifar10": 0.70,   # 70%
    "fmnist":  0.85,   # 85%
}

RESULTS_DIR = Path(__file__).parent / "results" / "exp1_global_accuracy"


# ============================================================================
# DATA LOADING
# ============================================================================

def find_metrics_file(method: str, dataset: str, alpha: float, results_dir: Optional[Path] = None) -> Optional[Path]:
    base = results_dir or RESULTS_DIR
    files = list((base / method / f"{dataset}_dirichlet_a{alpha}").glob("metrics_*.json"))
    if not files:
        return None
    return max(files, key=lambda x: x.stat().st_mtime)


def load_metrics(path: Path) -> List[Dict]:
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('rounds', [])


def get_final_acc(rounds: List[Dict]) -> float:
    return rounds[-1]['server']['test_acc']


def get_best_acc(rounds: List[Dict]) -> float:
    return max(r['server']['test_acc'] for r in rounds)


def get_convergence_round(rounds: List[Dict], target: float) -> Optional[int]:
    for r in rounds:
        if r['server']['test_acc'] >= target:
            return r['round']
    return None


def load_all_results(datasets: List[str], results_dir: Optional[Path] = None) -> Dict:
    """Load tat ca metrics files. Tra ve dict[dataset][method][alpha] = rounds."""
    base = results_dir or RESULTS_DIR
    results = {}
    for dataset in datasets:
        results[dataset] = {}
        for method in METHODS:
            results[dataset][method] = {}
            for alpha in ALPHAS:
                path = find_metrics_file(method, dataset, alpha, base)
                if path:
                    results[dataset][method][alpha] = load_metrics(path)
                else:
                    results[dataset][method][alpha] = None
    return results


# ============================================================================
# TABLE 2: GLOBAL ACCURACY
# ============================================================================

def print_table2(results: Dict, datasets: List[str], use_best: bool = False):
    acc_type = "Best" if use_best else "Final"
    get_acc = get_best_acc if use_best else get_final_acc

    print(f"\n{'='*75}")
    print(f"TABLE 2: GLOBAL ACCURACY ({acc_type} round) - Server Test Accuracy")
    print(f"{'='*75}")

    header = f"{'Dataset':<10} {'α':<5} " + "".join([f"{METHOD_LABELS[m]:>10}" for m in METHODS])
    print(header)
    print("-" * len(header))

    for dataset in datasets:
        first_row = True
        for alpha in ALPHAS:
            row = f"{dataset.upper() if first_row else '':<10} {alpha:<5} "
            for method in METHODS:
                rounds = results[dataset][method][alpha]
                if rounds is None:
                    row += f"{'–':>10}"
                else:
                    acc = get_acc(rounds)
                    row += f"{acc*100:>9.2f}%"
            print(row)
            first_row = False
        print()

    print(f"  Ghi chu: '{acc_type}' = accuracy tai round {'cao nhat' if use_best else 'cuoi'}")


def save_table2_csv(results: Dict, datasets: List[str], output_path: str, use_best: bool = False):
    get_acc = get_best_acc if use_best else get_final_acc
    acc_type = "best" if use_best else "final"

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Alpha"] + [METHOD_LABELS[m] for m in METHODS])
        for dataset in datasets:
            for alpha in ALPHAS:
                row = [dataset.upper(), alpha]
                for method in METHODS:
                    rounds = results[dataset][method][alpha]
                    if rounds is None:
                        row.append("–")
                    else:
                        row.append(f"{get_acc(rounds)*100:.2f}")
                writer.writerow(row)

    print(f"  Saved: {output_path}")


# ============================================================================
# TABLE 3: CONVERGENCE SPEED
# ============================================================================

def print_table3(results: Dict, datasets: List[str]):
    print(f"\n{'='*75}")
    print("TABLE 3: CONVERGENCE SPEED (so rounds de dat target accuracy)")
    print(f"{'='*75}")

    header = f"{'Dataset':<10} {'α':<5} {'Target':>7} " + "".join([f"{METHOD_LABELS[m]:>10}" for m in METHODS])
    print(header)
    print("-" * len(header))

    for dataset in datasets:
        target = CONVERGENCE_TARGETS[dataset]
        first_row = True
        for alpha in ALPHAS:
            row = f"{dataset.upper() if first_row else '':<10} {alpha:<5} {target*100:>6.0f}% "
            for method in METHODS:
                rounds = results[dataset][method][alpha]
                if rounds is None:
                    row += f"{'–':>10}"
                else:
                    conv = get_convergence_round(rounds, target)
                    if conv is None:
                        row += f"{'N/A':>10}"
                    else:
                        row += f"{conv:>9}R"
            print(row)
            first_row = False
        print()

    print(f"  Ghi chu: 'R' = round, 'N/A' = khong dat target trong 100 rounds")


def save_table3_csv(results: Dict, datasets: List[str], output_path: str):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Alpha", "Target (%)"] + [METHOD_LABELS[m] for m in METHODS])
        for dataset in datasets:
            target = CONVERGENCE_TARGETS[dataset]
            for alpha in ALPHAS:
                row = [dataset.upper(), alpha, f"{target*100:.0f}"]
                for method in METHODS:
                    rounds = results[dataset][method][alpha]
                    if rounds is None:
                        row.append("–")
                    else:
                        conv = get_convergence_round(rounds, target)
                        row.append(str(conv) if conv else "N/A")
                writer.writerow(row)

    print(f"  Saved: {output_path}")


# ============================================================================
# STATUS CHECK
# ============================================================================

def print_status(results: Dict, datasets: List[str]):
    print(f"\n{'='*75}")
    print("STATUS: Ket qua hien co")
    print(f"{'='*75}")

    total = done = 0
    for dataset in datasets:
        print(f"\n  {dataset.upper()}:")
        for method in METHODS:
            row_parts = []
            for alpha in ALPHAS:
                total += 1
                if results[dataset][method][alpha] is not None:
                    done += 1
                    row_parts.append(f"a={alpha}:DONE")
                else:
                    row_parts.append(f"a={alpha}:MISS")
            status = " | ".join(row_parts)
            print(f"    {METHOD_LABELS[method]:<10}: {status}")

    print(f"\n  Tong: {done}/{total} ({done/total*100:.0f}%)")
    if done < total:
        print(f"  Con thieu {total-done} runs - chay 'python exp1_resume.py' de tiep tuc")


# ============================================================================
# PLOTS
# ============================================================================

def plot_accuracy_curves(results: Dict, datasets: List[str], save_dir: Optional[str] = None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[WARNING] matplotlib chua cai, bo qua plot.")
        return

    colors = {"fedavg": "#2196F3", "fed_m3": "#F44336", "fed_dgd": "#4CAF50", "fedprox": "#FF9800"}
    linestyles = {"fedavg": "-", "fed_m3": "-", "fed_dgd": "--", "fedprox": "--"}

    for dataset in datasets:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        fig.suptitle(f"Global Accuracy - {dataset.upper()}", fontsize=13, fontweight='bold')

        for idx, alpha in enumerate(ALPHAS):
            ax = axes[idx]
            ax.set_title(f"α = {alpha}", fontsize=11)
            ax.set_xlabel("Round")
            ax.set_ylabel("Test Accuracy (%)" if idx == 0 else "")
            ax.grid(True, alpha=0.3)

            # Target line
            target = CONVERGENCE_TARGETS[dataset]
            ax.axhline(y=target * 100, color='gray', linestyle=':', linewidth=1, label=f"Target {target*100:.0f}%")

            for method in METHODS:
                rounds_data = results[dataset][method][alpha]
                if rounds_data is None:
                    continue
                xs = [r['round'] for r in rounds_data]
                ys = [r['server']['test_acc'] * 100 for r in rounds_data]
                ax.plot(xs, ys,
                        label=METHOD_LABELS[method],
                        color=colors[method],
                        linestyle=linestyles[method],
                        linewidth=1.8)

            ax.legend(fontsize=8)

        plt.tight_layout()

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            out = Path(save_dir) / f"accuracy_curves_{dataset}.png"
            plt.savefig(out, dpi=150, bbox_inches='tight')
            print(f"  Plot saved: {out}")
        else:
            plt.show()

        plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1 Report - Tong hop ket qua",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python exp1_report.py                    # In tat ca bang + plot
  python exp1_report.py --no-plot          # Chi in bang
  python exp1_report.py --dataset cifar10  # Chi report CIFAR-10
  python exp1_report.py --save-csv         # Luu bang ra CSV
  python exp1_report.py --use-best         # Dung best acc thay vi final acc
  python exp1_report.py --save-plots ./figs  # Luu plot vao folder
        """
    )

    parser.add_argument("--dataset", type=str, choices=["cifar10", "fmnist"],
                       help="Chi report 1 dataset (mac dinh: ca 2)")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR),
                       help=f"Thu muc chua ket qua (mac dinh: {RESULTS_DIR})")
    parser.add_argument("--no-plot", action="store_true",
                       help="Khong ve do thi")
    parser.add_argument("--save-plots", type=str, metavar="DIR",
                       help="Luu plot vao thu muc chi dinh")
    parser.add_argument("--save-csv", action="store_true",
                       help="Luu bang ra CSV")
    parser.add_argument("--use-best", action="store_true",
                       help="Dung best accuracy thay vi final accuracy")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    datasets = [args.dataset] if args.dataset else DATASETS

    print(f"\nLoading results from: {results_dir}")
    results = load_all_results(datasets, results_dir)

    # Status
    print_status(results, datasets)

    # Table 2: Global Accuracy
    print_table2(results, datasets, use_best=args.use_best)
    if args.use_best:
        print_table2(results, datasets, use_best=False)

    # Table 3: Convergence Speed
    print_table3(results, datasets)

    # Save CSV
    if args.save_csv:
        save_table2_csv(results, datasets, str(results_dir / "table2_global_accuracy.csv"), use_best=args.use_best)
        save_table3_csv(results, datasets, str(results_dir / "table3_convergence_speed.csv"))
        print(f"\nCSV files saved to: {results_dir}")

    # Plot
    if not args.no_plot:
        plot_accuracy_curves(results, datasets, save_dir=args.save_plots)


if __name__ == "__main__":
    main()