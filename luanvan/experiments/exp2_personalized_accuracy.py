"""
Experiment 2: Personalized Accuracy after Local Fine-tuning.

Quy trinh:
  1. Load global model da train xong tu exp1 (model_*.pt)
  2. Voi moi client: chia local data thanh 10% fine-tune / 90% test
  3. Fine-tune global model tren 10% local data (vai epochs)
  4. Evaluate personalized model tren 90% local data con lai
  5. Bao cao: mean, min, max accuracy across clients

Usage:
    python exp2_personalized_accuracy.py                     # Chay tat ca
    python exp2_personalized_accuracy.py --dataset cifar10   # Chi CIFAR-10
    python exp2_personalized_accuracy.py --dry-run           # Xem config
    python exp2_personalized_accuracy.py --finetune-epochs 5 # Tune so epochs
"""

import argparse
import json
import sys
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from models import get_model
from fl.data_split import dirichlet_split, iid_split
from utils.seed import set_seed

# ============================================================================
# CONFIG
# ============================================================================

METHODS      = ["fedavg", "fed_m3", "fed_dgd", "fedprox"]
METHOD_LABELS = {"fedavg": "FedAvg", "fed_m3": "Fed-M3", "fed_dgd": "Fed-DGD", "fedprox": "FedProx"}
DATASETS     = ["fmnist", "cifar10"]
ALPHAS       = [0.1, 0.5, 1.0]

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "exp2_config.json"

RESULTS_DIR = Path(__file__).parent / "results" / "exp1_global_accuracy"
EXP2_DIR    = Path(__file__).parent / "results" / "exp2_personalized_accuracy"


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def config_to_cfg(config: Dict) -> Dict:
    """Chuyen config JSON thanh flat dict de dung trong code."""
    return {
        "num_clients":     config["fl"]["num_clients"],
        "seed":            config["fl"]["seed"],
        "finetune_ratio":  config["finetune"]["ratio"],
        "finetune_epochs": config["finetune"]["epochs"],
        "finetune_lr":     config["finetune"]["lr"],
        "batch_size":      config["finetune"]["batch_size"],
        "non_iid_type":    config["fl"]["non_iid_type"],
    }


# ============================================================================
# HELPERS
# ============================================================================

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def find_model_file(method: str, dataset: str, alpha: float) -> Optional[Path]:
    folder = RESULTS_DIR / method / f"{dataset}_dirichlet_a{alpha}"
    files = list(folder.glob("model_*.pt"))
    if not files:
        return None
    return max(files, key=lambda x: x.stat().st_mtime)


def load_dataset(dataset: str, seed: int):
    """Load train dataset (same split as exp1 de dam bao cung data)."""
    from torchvision import datasets, transforms
    from torch.utils.data import random_split

    data_dir = Path(__file__).parent / "data"

    if dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        full = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        model_type = "cnn_medium"
    elif dataset == "fmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        full = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        model_type = "cnn_small"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    total = len(full)
    train_size = int(total * 0.8)
    val_size   = int(total * 0.1)
    test_size  = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_data, _, _ = random_split(full, [train_size, val_size, test_size], generator=generator)
    return train_data, model_type


def split_client_data(client_dataset: Subset, finetune_ratio: float, seed: int
                      ) -> Tuple[Subset, Subset]:
    """Chia local data cua 1 client: finetune_ratio% fine-tune, phan con lai test."""
    n = len(client_dataset)
    n_finetune = max(1, int(n * finetune_ratio))
    n_test     = n - n_finetune

    generator = torch.Generator().manual_seed(seed)
    finetune_data, test_data = random_split(client_dataset, [n_finetune, n_test], generator=generator)
    return finetune_data, test_data


def finetune_model(
    model: nn.Module,
    finetune_data,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> nn.Module:
    """Fine-tune model tren local data cua 1 client."""
    personal_model = copy.deepcopy(model)
    personal_model.to(device)
    personal_model.train()

    if len(finetune_data) == 0:
        return personal_model

    loader = DataLoader(finetune_data, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.SGD(personal_model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(personal_model(x), y)
            loss.backward()
            optimizer.step()

    return personal_model


def evaluate_model(model: nn.Module, test_data, batch_size: int, device: torch.device) -> float:
    """Evaluate model accuracy tren test data."""
    model.eval()
    if len(test_data) == 0:
        return 0.0

    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0


# ============================================================================
# CORE: PERSONALIZED ACCURACY FOR 1 RUN
# ============================================================================

def run_personalized(
    method: str,
    dataset: str,
    alpha: float,
    cfg: Dict,
    device: torch.device,
) -> Optional[Dict]:
    """
    Tinh personalized accuracy cho 1 combination (method, dataset, alpha).
    Tra ve dict voi mean/min/max/std accuracy, hoac None neu khong co model.
    """
    model_path = find_model_file(method, dataset, alpha)
    if model_path is None:
        return None

    print(f"  Loading model: {model_path.name}")

    # Load data (cung seed voi exp1 de dam bao cung data split)
    set_seed(cfg["seed"])
    train_data, model_type = load_dataset(dataset, cfg["seed"])

    # Split data cho clients (cung config voi exp1)
    if cfg["non_iid_type"] == "dirichlet":
        client_datasets = dirichlet_split(train_data, cfg["num_clients"], alpha, cfg["seed"])
    else:
        client_datasets = iid_split(train_data, cfg["num_clients"], cfg["seed"])

    # Load global model
    model = get_model(model_type)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Fine-tune va evaluate moi client
    client_accs = []
    for client_id, client_data in enumerate(client_datasets):
        finetune_data, test_data = split_client_data(
            client_data, cfg["finetune_ratio"], seed=cfg["seed"] + client_id
        )

        personal_model = finetune_model(
            model, finetune_data,
            epochs=cfg["finetune_epochs"],
            lr=cfg["finetune_lr"],
            batch_size=cfg["batch_size"],
            device=device,
        )

        acc = evaluate_model(personal_model, test_data, cfg["batch_size"], device)
        client_accs.append(acc)

        print(f"    Client {client_id:2d}: finetune={len(finetune_data):4d} | "
              f"test={len(test_data):4d} | acc={acc*100:.2f}%")

    import numpy as np
    accs = [a * 100 for a in client_accs]
    return {
        "method":      method,
        "dataset":     dataset,
        "alpha":       alpha,
        "num_clients": len(client_accs),
        "mean_acc":    float(np.mean(accs)),
        "min_acc":     float(np.min(accs)),
        "max_acc":     float(np.max(accs)),
        "std_acc":     float(np.std(accs)),
        "client_accs": accs,
    }


# ============================================================================
# TABLE OUTPUT
# ============================================================================

def print_table(all_results: Dict, datasets: List[str]):
    print(f"\n{'='*80}")
    print("TABLE 2: PERSONALIZED ACCURACY (Mean / Min) after local fine-tuning (10% local data)")
    print(f"{'='*80}")

    header = f"{'Dataset':<10} {'α':<5} " + "".join([f"{METHOD_LABELS[m]:>16}" for m in METHODS])
    print(header)
    print("-" * len(header))

    for dataset in datasets:
        first_row = True
        for alpha in ALPHAS:
            row = f"{dataset.upper() if first_row else '':<10} {alpha:<5} "
            for method in METHODS:
                r = all_results.get((method, dataset, alpha))
                if r is None:
                    row += f"{'–':>16}"
                else:
                    row += f"{r['mean_acc']:>7.2f}% / {r['min_acc']:>6.2f}%"
            print(row)
            first_row = False
        print()

    print("  Format: Mean% / Min%")


def save_results(all_results: Dict, datasets: List[str]):
    EXP2_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON chi tiet
    data = {k: v for k, v in all_results.items() if v is not None}
    serializable = {f"{k[0]}_{k[1]}_a{k[2]}": v for k, v in data.items()}
    json_path = EXP2_DIR / f"personalized_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  JSON saved: {json_path}")

    # CSV
    import csv
    csv_path = EXP2_DIR / f"table2_personalized_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Alpha", "Method", "Mean Acc (%)", "Min Acc (%)", "Max Acc (%)", "Std (%)"])
        for dataset in datasets:
            for alpha in ALPHAS:
                for method in METHODS:
                    r = all_results.get((method, dataset, alpha))
                    if r:
                        writer.writerow([dataset.upper(), alpha, METHOD_LABELS[method],
                                         f"{r['mean_acc']:.2f}", f"{r['min_acc']:.2f}",
                                         f"{r['max_acc']:.2f}", f"{r['std_acc']:.2f}"])
    print(f"  CSV saved:  {csv_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: Personalized Accuracy after Local Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python exp2_personalized_accuracy.py                      # Chay tat ca
  python exp2_personalized_accuracy.py --dry-run            # Xem config
  python exp2_personalized_accuracy.py --dataset cifar10    # Chi CIFAR-10
  python exp2_personalized_accuracy.py --finetune-epochs 10 # Fine-tune nhieu hon
  python exp2_personalized_accuracy.py --finetune-lr 0.005  # Doi learning rate
        """
    )

    parser.add_argument("--config",           type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset",          type=str, choices=["cifar10", "fmnist"])
    parser.add_argument("--methods",          type=str, nargs="+", choices=METHODS)
    parser.add_argument("--alphas",           type=float, nargs="+")
    parser.add_argument("--finetune-epochs",  type=int)
    parser.add_argument("--finetune-lr",      type=float)
    parser.add_argument("--finetune-ratio",   type=float)
    parser.add_argument("--num-clients",      type=int)
    parser.add_argument("--batch-size",       type=int)
    parser.add_argument("--seed",             type=int)
    parser.add_argument("--device",           type=str)
    parser.add_argument("--results-dir",      type=str)
    parser.add_argument("--dry-run",          action="store_true")
    parser.add_argument("--no-confirm",       action="store_true")

    args = parser.parse_args()

    # Load config file
    config = load_config(args.config)
    cfg = config_to_cfg(config)

    # Apply command line overrides
    if args.finetune_epochs: cfg["finetune_epochs"] = args.finetune_epochs
    if args.finetune_lr:     cfg["finetune_lr"]     = args.finetune_lr
    if args.finetune_ratio:  cfg["finetune_ratio"]  = args.finetune_ratio
    if args.num_clients:     cfg["num_clients"]      = args.num_clients
    if args.batch_size:      cfg["batch_size"]       = args.batch_size
    if args.seed:            cfg["seed"]             = args.seed

    device_str  = args.device or config["training"]["device"]
    results_dir = Path(args.results_dir) if args.results_dir else Path(config["input"]["exp1_results_dir"])
    exp2_dir    = Path(config["output"]["save_dir"])

    datasets = [args.dataset] if args.dataset else config.get("datasets", DATASETS)
    methods  = args.methods   or config.get("methods", METHODS)
    alphas   = args.alphas    or config.get("alphas", ALPHAS)

    total = len(methods) * len(datasets) * len(alphas)

    print(f"\n{'='*65}")
    print("EXPERIMENT 2: PERSONALIZED ACCURACY")
    print(f"{'='*65}")
    print(f"  Fine-tune ratio:  {cfg['finetune_ratio']*100:.0f}% local data")
    print(f"  Fine-tune epochs: {cfg['finetune_epochs']}")
    print(f"  Fine-tune LR:     {cfg['finetune_lr']}")
    print(f"  Methods:          {', '.join(methods)}")
    print(f"  Datasets:         {', '.join(datasets)}")
    print(f"  Alphas:           {alphas}")
    print(f"  Total runs:       {total}")
    print(f"  Results dir:      {results_dir}")

    # Override global paths
    global RESULTS_DIR, EXP2_DIR
    RESULTS_DIR = results_dir
    EXP2_DIR    = exp2_dir

    # Check available models
    print(f"\n  Model availability:")
    missing = []
    for dataset in datasets:
        for method in methods:
            for alpha in alphas:
                path = find_model_file(method, dataset, alpha)
                status = "OK" if path else "MISS"
                print(f"    [{status}] {method} | {dataset} | a={alpha}")
                if not path:
                    missing.append((method, dataset, alpha))

    if missing:
        print(f"\n  WARN: {len(missing)} models chua co - se bo qua cac runs nay.")
        print(f"  Chay exp1 truoc de co du models.")

    if args.dry_run:
        print("\n[DRY RUN] Bo --dry-run de chay thuc.")
        return

    if not args.no_confirm:
        try:
            input("\nPress Enter de bat dau (Ctrl+C de huy)...")
        except KeyboardInterrupt:
            print("\nHuy.")
            return

    device = get_device(device_str)
    print(f"\nUsing device: {device}")

    all_results = {}
    for dataset in datasets:
        for method in methods:
            for alpha in alphas:
                print(f"\n{'─'*65}")
                print(f"Running: {METHOD_LABELS[method]} | {dataset.upper()} | alpha={alpha}")
                print(f"{'─'*65}")

                result = run_personalized(method, dataset, alpha, cfg, device)
                all_results[(method, dataset, alpha)] = result

                if result:
                    print(f"  Mean={result['mean_acc']:.2f}% | "
                          f"Min={result['min_acc']:.2f}% | "
                          f"Max={result['max_acc']:.2f}%")
                else:
                    print(f"  [SKIP] Khong tim thay model.")

    # Print table
    print_table(all_results, datasets)

    # Save results
    save_results(all_results, datasets)


if __name__ == "__main__":
    main()