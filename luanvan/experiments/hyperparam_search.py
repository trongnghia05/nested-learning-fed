"""
Hyperparameter Grid Search for FL Methods.

Tim bo tham so tot nhat cho moi method bang cach chay grid search.

Usage:
    python hyperparam_search.py                           # Chay voi config mac dinh
    python hyperparam_search.py --config my_config.json   # Dung config khac
    python hyperparam_search.py --method fed_m3           # Chi search 1 method
    python hyperparam_search.py --dry-run                 # Xem truoc, khong chay
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from itertools import product
from typing import Dict, Any, List, Tuple
import numpy as np


DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "hyperparam_search.json"


def load_config(config_path: str) -> Dict:
    """Load config from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_param_combinations(grid: Dict[str, List]) -> List[Dict]:
    """Generate all combinations of parameters from a grid."""
    if not grid:
        return [{}]

    keys = list(grid.keys())
    values = list(grid.values())

    combinations = []
    for combo in product(*values):
        param_dict = dict(zip(keys, combo))
        combinations.append(param_dict)

    return combinations


def build_command(method: str, base_config: Dict, params: Dict, save_dir: str) -> List[str]:
    """Build command line arguments for run_experiment.py."""

    cmd = [
        sys.executable, "run_experiment.py",
        "--method", method,
        "--dataset", base_config["dataset"],
        "--alpha", str(base_config["alpha"]),
        "--num-clients", str(base_config["num_clients"]),
        "--num-rounds", str(base_config["num_rounds"]),
        "--local-epochs", str(base_config["local_epochs"]),
        "--batch-size", str(base_config["batch_size"]),
        "--lr", str(base_config["lr"]),
        "--seed", str(base_config["seed"]),
        "--save-dir", save_dir,
    ]

    # Method-specific parameters
    if method == "fed_m3":
        if "beta1" in params:
            cmd.extend(["--beta1", str(params["beta1"])])
        if "beta3" in params:
            cmd.extend(["--beta3", str(params["beta3"])])
        if "lam" in params:
            cmd.extend(["--lam", str(params["lam"])])
        if "beta2" in params:
            cmd.extend(["--beta2", str(params["beta2"])])

    elif method == "fed_dgd":
        if "decay_strength" in params:
            cmd.extend(["--dgd-decay-strength", str(params["decay_strength"])])
        if "alpha" in params:
            cmd.extend(["--dgd-alpha", str(params["alpha"])])

    elif method == "fedprox":
        if "mu" in params:
            cmd.extend(["--fedprox-mu", str(params["mu"])])

    return cmd


def find_latest_metrics_file(save_dir: str, method: str, dataset: str, alpha: float) -> str:
    """Find the most recent metrics JSON file."""
    pattern = Path(save_dir) / method / f"{dataset}_dirichlet_a{alpha}" / "metrics_*.json"
    files = list(Path(save_dir).glob(f"{method}/{dataset}_dirichlet_a{alpha}/metrics_*.json"))
    if not files:
        return None
    return str(max(files, key=lambda x: x.stat().st_mtime))


def extract_metrics_from_file(metrics_path: str) -> Dict[str, float]:
    """Extract key metrics from metrics JSON file."""
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)

        if not data.get('rounds'):
            return {}

        final_round = data['rounds'][-1]

        # Server metrics (Final)
        server_test_acc = final_round['server']['test_acc']

        # Client aggregated metrics (Final)
        client_agg = final_round.get('client_aggregated', {})
        client_train_acc_mean = client_agg.get('train_acc', {}).get('mean', 0)
        client_test_acc_mean = client_agg.get('test_acc', {}).get('mean', 0)

        # Best values across all rounds
        best_server_acc = max(r['server']['test_acc'] for r in data['rounds'])
        best_client_train_acc = max(
            r.get('client_aggregated', {}).get('train_acc', {}).get('mean', 0)
            for r in data['rounds']
        )
        best_client_test_acc = max(
            r.get('client_aggregated', {}).get('test_acc', {}).get('mean', 0)
            for r in data['rounds']
        )

        return {
            # Final (round cuoi)
            'server_test_acc': server_test_acc,
            'client_train_acc_mean': client_train_acc_mean,
            'client_test_acc_mean': client_test_acc_mean,
            # Best (cao nhat qua cac rounds)
            'best_server_acc': best_server_acc,
            'best_client_train_acc': best_client_train_acc,
            'best_client_test_acc': best_client_test_acc,
        }
    except Exception as e:
        print(f"[WARNING] Failed to parse metrics file: {e}")
        return {}


def run_single_experiment(
    method: str,
    base_config: Dict,
    params: Dict,
    save_dir: str,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Run a single experiment with given parameters."""

    cmd = build_command(method, base_config, params, save_dir)
    param_str = ", ".join([f"{k}={v}" for k, v in params.items()])

    print(f"\n{'='*70}")
    print(f"Running: {method.upper()} with {param_str}")
    print(f"{'='*70}")

    if dry_run:
        print(f"[DRY RUN] Command:")
        print(f"  {' '.join(cmd)}")
        return {
            "status": "dry_run",
            "method": method,
            "params": params,
        }

    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            check=True,
            capture_output=False,  # Show output in real-time
        )

        # Find and parse the metrics file
        metrics_path = find_latest_metrics_file(
            save_dir, method,
            base_config["dataset"],
            base_config["alpha"]
        )

        metrics = {}
        if metrics_path:
            metrics = extract_metrics_from_file(metrics_path)
            print(f"  Metrics from: {metrics_path}")

        return {
            "status": "success",
            "method": method,
            "params": params,
            "metrics_path": metrics_path,
            **metrics,
        }

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Experiment failed")
        return {
            "status": "failed",
            "method": method,
            "params": params,
            "error": str(e),
        }


def find_best_params(results: List[Dict]) -> Dict[str, Any]:
    """Find the best parameters based on best_server_acc."""
    successful = [r for r in results if r["status"] == "success" and r.get("best_server_acc")]
    if not successful:
        return None

    best = max(successful, key=lambda x: x["best_server_acc"])
    return best


def print_results_table(results: List[Dict], method: str):
    """Print results in a table format."""
    successful = [r for r in results if r["status"] == "success"]

    if not successful:
        print(f"No successful runs for {method}")
        return

    # Sort by best_server_acc descending
    successful.sort(key=lambda x: x.get("best_server_acc", 0), reverse=True)

    print(f"\n{'='*140}")
    print(f"RESULTS FOR {method.upper()}")
    print(f"{'='*140}")

    # Get param keys
    param_keys = list(successful[0]["params"].keys())

    # Header
    metric_headers = [
        "Server Acc (Final)", "Server Acc (Best)",
        "Mean Local Acc (Final)", "Mean Local Acc (Best)",
        "Mean Global Acc (Final)", "Mean Global Acc (Best)"
    ]
    header = " | ".join([f"{k:>12}" for k in param_keys] + [f"{h:>22}" for h in metric_headers])
    print(header)
    print("-" * len(header))

    # Rows
    for r in successful:
        row = " | ".join(
            [f"{r['params'].get(k, '-'):>12}" for k in param_keys] +
            [
                f"{r.get('server_test_acc', 0)*100:>21.2f}%",
                f"{r.get('best_server_acc', 0)*100:>21.2f}%",
                f"{r.get('client_train_acc_mean', 0)*100:>21.2f}%",
                f"{r.get('best_client_train_acc', 0)*100:>21.2f}%",
                f"{r.get('client_test_acc_mean', 0)*100:>21.2f}%",
                f"{r.get('best_client_test_acc', 0)*100:>21.2f}%",
            ]
        )
        print(row)

    # Best params
    best = successful[0]
    print(f"\nBest params (by Server Best): {best['params']}")
    print(f"  Server:     Final={best.get('server_test_acc', 0)*100:.2f}%  Best={best.get('best_server_acc', 0)*100:.2f}%")
    print(f"  Local Acc:  Final={best.get('client_train_acc_mean', 0)*100:.2f}%  Best={best.get('best_client_train_acc', 0)*100:.2f}%")
    print(f"  Global Acc: Final={best.get('client_test_acc_mean', 0)*100:.2f}%  Best={best.get('best_client_test_acc', 0)*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Grid Search for FL Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH),
                       help="Config file path")
    parser.add_argument("--method", type=str, nargs="+",
                       choices=["fed_m3", "fed_dgd", "fedprox"],
                       help="Methods to search (default: all in config)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview commands without running")
    parser.add_argument("--no-confirm", action="store_true",
                       help="Skip confirmation prompt")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    base_config = config["base_config"]
    grid = config["grid"]
    save_dir = config["output"]["save_dir"]

    # Determine methods to search
    methods = args.method if args.method else list(grid.keys())

    # Calculate total runs
    total_runs = sum(len(generate_param_combinations(grid.get(m, {}))) for m in methods)

    print(f"\n{'='*70}")
    print("HYPERPARAMETER GRID SEARCH")
    print(f"{'='*70}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Total runs: {total_runs}")
    print(f"Base config:")
    for k, v in base_config.items():
        print(f"  {k}: {v}")
    print(f"\nGrids:")
    for method in methods:
        combos = generate_param_combinations(grid.get(method, {}))
        print(f"  {method}: {len(combos)} combinations")
        for k, v in grid.get(method, {}).items():
            print(f"    {k}: {v}")

    # Confirm
    if not args.dry_run and not args.no_confirm:
        try:
            input("\nPress Enter to start (Ctrl+C to cancel)...")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return

    # Run experiments
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for method in methods:
        combinations = generate_param_combinations(grid.get(method, {}))
        results = []

        for params in combinations:
            result = run_single_experiment(
                method=method,
                base_config=base_config,
                params=params,
                save_dir=f"{save_dir}/{method}",
                dry_run=args.dry_run,
            )
            results.append(result)

        all_results[method] = results

        # Print table for this method
        if not args.dry_run:
            print_results_table(results, method)

    # Save summary
    if not args.dry_run:
        summary_path = Path(save_dir) / f"summary_{timestamp}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "timestamp": timestamp,
            "base_config": base_config,
            "grid": grid,
            "results": all_results,
            "best_params": {
                method: find_best_params(results)
                for method, results in all_results.items()
            }
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*120}")
        print("SUMMARY")
        print(f"{'='*120}")
        print(f"Results saved to: {summary_path}")
        print(f"\nBest parameters for each method (ranked by Server Best Acc):")
        for method, best in summary["best_params"].items():
            if best:
                print(f"\n  {method.upper()}:")
                print(f"    Params: {best['params']}")
                print(f"    Server:     Final={best.get('server_test_acc', 0)*100:.2f}%  Best={best.get('best_server_acc', 0)*100:.2f}%")
                print(f"    Local Acc:  Final={best.get('client_train_acc_mean', 0)*100:.2f}%  Best={best.get('best_client_train_acc', 0)*100:.2f}%")
                print(f"    Global Acc: Final={best.get('client_test_acc_mean', 0)*100:.2f}%  Best={best.get('best_client_test_acc', 0)*100:.2f}%")
            else:
                print(f"\n  {method.upper()}: No successful runs")


if __name__ == "__main__":
    main()
