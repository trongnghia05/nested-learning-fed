"""
Experiment 1 - Resume: Tu dong check va chay cac experiments con thieu.

Script nay se:
1. Doc config tu exp1_config.json
2. Check ket qua nao da co trong results/
3. Chi chay nhung cai con thieu
4. Bo qua nhung cai da chay xong

Usage:
    python exp1_resume.py                   # Check + chay missing
    python exp1_resume.py --dry-run         # Chi xem status, khong chay
    python exp1_resume.py --dataset fmnist  # Chi check fmnist
    python exp1_resume.py --force           # Chay lai tat ca (ke ca da co)
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "exp1_config.json"


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_completed(method: str, dataset: str, alpha: float, save_dir: str) -> bool:
    """Kiem tra experiment da chay xong chua bang cach tim metrics file."""
    pattern = Path(save_dir) / method / f"{dataset}_dirichlet_a{alpha}" / "metrics_*.json"
    files = list(Path(save_dir).glob(f"{method}/{dataset}_dirichlet_a{alpha}/metrics_*.json"))
    return len(files) > 0


def get_metrics_path(method: str, dataset: str, alpha: float, save_dir: str) -> str:
    """Lay duong dan den metrics file neu da ton tai."""
    files = list(Path(save_dir).glob(f"{method}/{dataset}_dirichlet_a{alpha}/metrics_*.json"))
    if files:
        return str(max(files, key=lambda x: x.stat().st_mtime))
    return None


def build_command(method: str, alpha: float, config: Dict) -> List[str]:
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

    if method == "fed_m3":
        params = config.get("fed_m3", {})
        cmd.extend(["--beta1", str(params.get("beta1", 0.9))])
        cmd.extend(["--beta2", str(params.get("beta2", 0.999))])
        cmd.extend(["--beta3", str(params.get("beta3", 0.5))])
        cmd.extend(["--lam", str(params.get("lam", 0.5))])
        cmd.extend(["--ns-steps", str(params.get("ns_steps", 5))])
        cmd.extend(["--v-init", str(params.get("v_init", 1.0))])
    elif method == "fed_dgd":
        params = config.get("fed_dgd", {})
        cmd.extend(["--dgd-alpha", str(params.get("alpha", 1.0))])
        cmd.extend(["--dgd-decay-strength", str(params.get("decay_strength", 0.05))])
    elif method == "fedprox":
        params = config.get("fedprox", {})
        cmd.extend(["--fedprox-mu", str(params.get("mu", 0.01))])

    return cmd


def print_status(methods: List[str], alphas: List[float], dataset: str, save_dir: str):
    """In bang trang thai: done/missing cho tung combination."""
    done = []
    missing = []

    print(f"\n{'='*65}")
    print(f"STATUS CHECK - {dataset.upper()}")
    print(f"{'='*65}")

    header = f"{'Method':<12} | " + " | ".join([f"a={a:<4}" for a in alphas])
    print(header)
    print("-" * len(header))

    for method in methods:
        row_parts = []
        for alpha in alphas:
            if is_completed(method, dataset, alpha, save_dir):
                row_parts.append(f"{'DONE':<6}")
                done.append((method, alpha))
            else:
                row_parts.append(f"{'MISS':<6}")
                missing.append((method, alpha))
        print(f"{method:<12} | " + " | ".join(row_parts))

    print(f"\nDone:    {len(done)}/{len(done)+len(missing)}")
    print(f"Missing: {len(missing)}/{len(done)+len(missing)}")

    if missing:
        print(f"\nCan chay:")
        for method, alpha in missing:
            print(f"  - {method} alpha={alpha}")

    return missing


def run_single(method: str, alpha: float, config: Dict) -> Dict:
    cmd = build_command(method, alpha, config)
    dataset = config["dataset"]["name"]

    print(f"\n{'='*65}")
    print(f"Running: {method.upper()} | {dataset.upper()} | alpha={alpha}")
    print(f"{'='*65}")

    try:
        subprocess.run(cmd, cwd=Path(__file__).parent, check=True, capture_output=False)
        return {"status": "success", "method": method, "alpha": alpha}
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed: {method} alpha={alpha}")
        return {"status": "failed", "method": method, "alpha": alpha, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1 Resume - Tu dong chay cac experiments con thieu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python exp1_resume.py                    # Check + chay tat ca missing
  python exp1_resume.py --dry-run          # Chi xem status, khong chay
  python exp1_resume.py --dataset fmnist   # Chi check/chay fmnist
  python exp1_resume.py --dataset cifar10  # Chi check/chay cifar10
  python exp1_resume.py --force            # Chay lai tat ca ke ca da xong
        """
    )

    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset", type=str, choices=["cifar10", "fmnist"],
                       help="Chi check 1 dataset (mac dinh: theo config)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Chi xem status, khong chay")
    parser.add_argument("--force", action="store_true",
                       help="Chay lai tat ca, ke ca da co ket qua")
    parser.add_argument("--no-confirm", action="store_true",
                       help="Khong hoi xac nhan")

    args = parser.parse_args()

    config = load_config(args.config)
    methods = config.get("methods", ["fedavg"])
    alphas = config.get("alphas", [0.5])
    save_dir = config["output"]["save_dir"]

    if args.dataset:
        config["dataset"]["name"] = args.dataset

    dataset = config["dataset"]["name"]

    # Check status
    missing = print_status(methods, alphas, dataset, save_dir)

    if args.dry_run:
        print("\n[DRY RUN] Khong chay. Bo --dry-run de chay thuc.")
        return

    # Determine what to run
    to_run = list(product(methods, alphas)) if args.force else missing

    if not to_run:
        print("\nTat ca experiments da chay xong! Khong can chay them.")
        return

    print(f"\nSe chay {len(to_run)} experiments.")

    if not args.no_confirm:
        try:
            input("Press Enter de bat dau (Ctrl+C de huy)...")
        except KeyboardInterrupt:
            print("\nHuy.")
            return

    # Run missing experiments
    results = []
    for method, alpha in to_run:
        result = run_single(method, alpha, config)
        results.append(result)

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")

    print(f"\n{'='*65}")
    print("RESUME SUMMARY")
    print(f"{'='*65}")
    print(f"Chay them:  {len(results)}")
    print(f"Thanh cong: {success}")
    print(f"That bai:   {failed}")

    if failed > 0:
        print("\nThat bai:")
        for r in results:
            if r["status"] == "failed":
                print(f"  - {r['method']} alpha={r['alpha']}")

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path(save_dir)
    summary_path.mkdir(parents=True, exist_ok=True)
    with open(summary_path / f"resume_summary_{timestamp}.json", 'w') as f:
        json.dump({"timestamp": timestamp, "dataset": dataset, "results": results}, f, indent=2)

    print(f"\nKet qua luu tai: {save_dir}")


if __name__ == "__main__":
    main()