"""
Experiment 1: Global Accuracy Comparison

So sanh Global Accuracy cua cac methods:
- FedAvg (baseline)
- Fed-M3 (proposed)
- Fed-DGD (proposed)
- FedProx (baseline)

Voi cac muc do non-IID khac nhau:
- alpha = 0.1 (severe non-IID)
- alpha = 0.5 (moderate non-IID)
- alpha = 1.0 (mild non-IID)

Usage:
    python exp1_global_accuracy.py                          # Chay voi config mac dinh
    python exp1_global_accuracy.py --config my_config.json  # Dung config khac
    python exp1_global_accuracy.py --methods fedavg fed_m3  # Override methods
    python exp1_global_accuracy.py --dry-run                # Xem config, khong chay
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from itertools import product
from typing import Dict, Any, List


# ============================================================================
# DEFAULT CONFIG PATH
# ============================================================================

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "exp1_config.json"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def deep_update(base: Dict, update: Dict) -> Dict:
    """Recursively update nested dict."""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str) -> Dict:
    """Load config from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(config: Dict, save_dir: str, timestamp: str) -> str:
    """Save config to JSON file."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    config_file = save_path / f"config_{timestamp}.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return str(config_file)


def build_command(method: str, alpha: float, config: Dict) -> List[str]:
    """Build command line arguments for run_experiment.py."""

    fl = config["fl"]
    training = config["training"]
    output = config["output"]

    cmd = [
        sys.executable, "run_experiment.py",
        "--method", method,
        "--dataset", config["dataset"]["name"],
        "--alpha", str(alpha),
        "--num-clients", str(fl["num_clients"]),
        "--num-rounds", str(fl["num_rounds"]),
        "--local-epochs", str(fl["local_epochs"]),
        "--batch-size", str(fl["batch_size"]),
        "--lr", str(fl["lr"]),
        "--non-iid", fl["non_iid_type"],
        "--seed", str(training["seed"]),
        "--device", training["device"],
        "--save-dir", output["save_dir"],
    ]

    if training.get("debug", False):
        cmd.append("--debug")

    # Method-specific hyperparameters
    if method == "fed_m3":
        params = config.get("fed_m3", {})
        cmd.extend(["--beta1", str(params.get("beta1", 0.9))])
        cmd.extend(["--beta2", str(params.get("beta2", 0.999))])
        cmd.extend(["--beta3", str(params.get("beta3", 0.9))])
        cmd.extend(["--lam", str(params.get("lam", 0.3))])
        cmd.extend(["--ns-steps", str(params.get("ns_steps", 5))])
        cmd.extend(["--v-init", str(params.get("v_init", 1.0))])

    elif method == "fed_dgd":
        params = config.get("fed_dgd", {})
        cmd.extend(["--dgd-alpha", str(params.get("alpha", 1.0))])
        cmd.extend(["--dgd-decay-strength", str(params.get("decay_strength", 0.1))])

    elif method == "fedprox":
        params = config.get("fedprox", {})
        cmd.extend(["--fedprox-mu", str(params.get("mu", 0.01))])

    return cmd


def run_single_experiment(
    method: str,
    alpha: float,
    config: Dict,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Run a single experiment configuration."""

    cmd = build_command(method, alpha, config)

    print(f"\n{'='*70}")
    print(f"Running: {method.upper()} with alpha={alpha}")
    print(f"{'='*70}")

    if dry_run:
        print(f"[DRY RUN] Command:")
        print(f"  {' '.join(cmd)}")
        return {"status": "dry_run", "method": method, "alpha": alpha}

    try:
        subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            check=True,
            capture_output=False,
        )
        return {"status": "success", "method": method, "alpha": alpha}
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Experiment failed: {method} alpha={alpha}")
        return {"status": "failed", "method": method, "alpha": alpha, "error": str(e)}


def print_config(config: Dict, methods: List[str], alphas: List[float]):
    """Print experiment configuration."""

    fl = config["fl"]
    training = config["training"]
    total_runs = len(methods) * len(alphas)

    print(f"""
{'='*70}
EXPERIMENT 1: GLOBAL ACCURACY COMPARISON
{'='*70}

Experiment: {config['experiment']['name']}
Description: {config['experiment']['description']}

{'─'*70}
DATASET & MODEL
{'─'*70}
  Dataset:        {config['dataset']['name'].upper()}
  Num classes:    {config['dataset']['num_classes']}
  Model type:     {config['model']['type']}

{'─'*70}
FL SETTINGS
{'─'*70}
  Clients:        {fl['num_clients']}
  Rounds:         {fl['num_rounds']}
  Local epochs:   {fl['local_epochs']}
  Batch size:     {fl['batch_size']}
  Learning rate:  {fl['lr']}
  Non-IID type:   {fl['non_iid_type']}

{'─'*70}
METHODS & HYPERPARAMETERS
{'─'*70}
  Methods:        {', '.join(methods)}
  Alpha values:   {', '.join(map(str, alphas))}
  Total runs:     {total_runs}
""")

    if "fed_m3" in methods:
        m3 = config.get('fed_m3', {})
        print(f"""  Fed-M3:
    beta1={m3.get('beta1', 0.9)}, beta2={m3.get('beta2', 0.999)}, beta3={m3.get('beta3', 0.9)}
    lam={m3.get('lam', 0.3)}, v_init={m3.get('v_init', 1.0)}
""")

    if "fed_dgd" in methods:
        dgd = config.get('fed_dgd', {})
        print(f"""  Fed-DGD:
    alpha={dgd.get('alpha', 1.0)}, decay_strength={dgd.get('decay_strength', 0.1)}
""")

    if "fedprox" in methods:
        prox = config.get('fedprox', {})
        print(f"""  FedProx:
    mu={prox.get('mu', 0.01)}
""")

    print(f"""{'─'*70}
TRAINING
{'─'*70}
  Seed:           {training['seed']}
  Device:         {training['device']}
  Debug:          {training.get('debug', False)}

{'─'*70}
OUTPUT
{'─'*70}
  Save dir:       {config['output']['save_dir']}

{'='*70}

Experiment Matrix:
""")

    # Print matrix
    header = f"{'Method':<12} | " + " | ".join([f"a={a}" for a in alphas])
    print(header)
    print("-" * len(header))
    for method in methods:
        row = f"{method:<12} | " + " | ".join(["  Run " for _ in alphas])
        print(row)

    print()


def print_summary(results: List[Dict], config: Dict):
    """Print experiment summary."""

    print(f"\n{'='*70}")
    print("EXPERIMENT 1 SUMMARY")
    print(f"{'='*70}")

    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    dry_run = sum(1 for r in results if r["status"] == "dry_run")

    print(f"Total runs:  {len(results)}")
    print(f"Successful:  {success}")
    print(f"Failed:      {failed}")
    if dry_run > 0:
        print(f"Dry run:     {dry_run}")

    if failed > 0:
        print("\nFailed experiments:")
        for r in results:
            if r["status"] == "failed":
                print(f"  - {r['method']} alpha={r['alpha']}: {r.get('error', 'Unknown')}")

    print(f"\nResults saved to: {config['output']['save_dir']}")
    print(f"{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1: Global Accuracy Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python exp1_global_accuracy.py                          # Run with default config
  python exp1_global_accuracy.py --config my_config.json  # Use custom config
  python exp1_global_accuracy.py --methods fedavg fed_m3  # Override methods
  python exp1_global_accuracy.py --alphas 0.5             # Override alphas
  python exp1_global_accuracy.py --dry-run                # Preview only

  # Quick test
  python exp1_global_accuracy.py --num-rounds 10 --methods fedavg --alphas 0.5

  # Override hyperparameters
  python exp1_global_accuracy.py --fed-m3-lam 0.5 --fedprox-mu 0.1

  # Export config for editing
  python exp1_global_accuracy.py --export-config my_config.json
        """
    )

    # Config file
    parser.add_argument("--config", type=str,
                       default=str(DEFAULT_CONFIG_PATH),
                       help=f"Config file path (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--export-config", type=str,
                       help="Export current config to JSON file and exit")

    # Override options
    parser.add_argument("--methods", nargs="+",
                       choices=["fedavg", "fed_m3", "fed_dgd", "fedprox"],
                       help="Methods to run")
    parser.add_argument("--alphas", nargs="+", type=float,
                       help="Alpha values to test")

    # Dataset settings
    parser.add_argument("--dataset", type=str, choices=["cifar10", "fmnist"])

    # FL settings
    parser.add_argument("--num-clients", type=int)
    parser.add_argument("--num-rounds", type=int)
    parser.add_argument("--local-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--non-iid", type=str, choices=["dirichlet", "quantity", "iid"])

    # Fed-M3 settings
    parser.add_argument("--fed-m3-beta1", type=float)
    parser.add_argument("--fed-m3-beta2", type=float)
    parser.add_argument("--fed-m3-beta3", type=float)
    parser.add_argument("--fed-m3-lam", type=float)
    parser.add_argument("--fed-m3-v-init", type=float)

    # Fed-DGD settings
    parser.add_argument("--fed-dgd-alpha", type=float)
    parser.add_argument("--fed-dgd-decay", type=float)

    # FedProx settings
    parser.add_argument("--fedprox-mu", type=float)

    # Training settings
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--debug", action="store_true")

    # Output settings
    parser.add_argument("--save-dir", type=str)

    # Control
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview commands without running")
    parser.add_argument("--no-confirm", action="store_true",
                       help="Skip confirmation prompt")

    args = parser.parse_args()

    # Load config from file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        print(f"Create one using: python exp1_global_accuracy.py --export-config {config_path}")
        sys.exit(1)

    print(f"Loading config from: {config_path}")
    config = load_config(str(config_path))

    # Apply command line overrides
    if args.methods:
        config["methods"] = args.methods
    if args.alphas:
        config["alphas"] = args.alphas
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.num_clients:
        config["fl"]["num_clients"] = args.num_clients
    if args.num_rounds:
        config["fl"]["num_rounds"] = args.num_rounds
    if args.local_epochs:
        config["fl"]["local_epochs"] = args.local_epochs
    if args.batch_size:
        config["fl"]["batch_size"] = args.batch_size
    if args.lr:
        config["fl"]["lr"] = args.lr
    if args.non_iid:
        config["fl"]["non_iid_type"] = args.non_iid
    if args.seed:
        config["training"]["seed"] = args.seed
    if args.device:
        config["training"]["device"] = args.device
    if args.debug:
        config["training"]["debug"] = True
    if args.save_dir:
        config["output"]["save_dir"] = args.save_dir

    # Fed-M3 overrides
    if args.fed_m3_beta1:
        config.setdefault("fed_m3", {})["beta1"] = args.fed_m3_beta1
    if args.fed_m3_beta2:
        config.setdefault("fed_m3", {})["beta2"] = args.fed_m3_beta2
    if args.fed_m3_beta3:
        config.setdefault("fed_m3", {})["beta3"] = args.fed_m3_beta3
    if args.fed_m3_lam:
        config.setdefault("fed_m3", {})["lam"] = args.fed_m3_lam
    if args.fed_m3_v_init:
        config.setdefault("fed_m3", {})["v_init"] = args.fed_m3_v_init

    # Fed-DGD overrides
    if args.fed_dgd_alpha:
        config.setdefault("fed_dgd", {})["alpha"] = args.fed_dgd_alpha
    if args.fed_dgd_decay:
        config.setdefault("fed_dgd", {})["decay_strength"] = args.fed_dgd_decay

    # FedProx overrides
    if args.fedprox_mu:
        config.setdefault("fedprox", {})["mu"] = args.fedprox_mu

    # Export config if requested
    if args.export_config:
        with open(args.export_config, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Config exported to: {args.export_config}")
        return

    # Get methods and alphas
    methods = config.get("methods", ["fedavg"])
    alphas = config.get("alphas", [0.5])

    # Print configuration
    print_config(config, methods, alphas)

    # Confirm before running
    if not args.dry_run and not args.no_confirm:
        try:
            input("Press Enter to start experiments (Ctrl+C to cancel)...")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return

    # Save config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config["output"].get("save_config", True) and not args.dry_run:
        config_file = save_config(config, config["output"]["save_dir"], timestamp)
        print(f"Config saved to: {config_file}")

    # Run experiments
    results = []
    for method, alpha in product(methods, alphas):
        result = run_single_experiment(method, alpha, config, args.dry_run)
        results.append(result)

    # Save results summary
    if not args.dry_run:
        summary_path = Path(config["output"]["save_dir"])
        summary_path.mkdir(parents=True, exist_ok=True)
        summary_file = summary_path / f"summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "config": config,
                "results": results,
                "timestamp": timestamp,
            }, f, indent=2, ensure_ascii=False)

    # Print summary
    print_summary(results, config)


if __name__ == "__main__":
    main()
