"""
Script to compare multiple FL methods on the same data split.

Usage:
    python run_comparison.py --dataset cifar10 --alpha 0.5
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent))

from run_experiment import run_fl_experiment, get_dataset
from fl import dirichlet_split, quantity_skew_split, iid_split
from fl.data_split import print_data_distribution, get_client_labels
from utils import set_seed, plot_results, plot_client_distribution


def run_comparison(
    dataset_name: str = 'cifar10',
    methods: list = ['fedavg', 'fed_m3'],
    num_clients: int = 10,
    num_rounds: int = 100,
    local_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.01,
    alpha: float = 0.5,
    non_iid_type: str = 'dirichlet',
    seed: int = 42,
    device: str = 'auto',
    save_dir: str = './results',
):
    """
    Compare multiple FL methods on the SAME data split.

    This ensures fair comparison - only the algorithm differs.
    """
    print("=" * 70)
    print("FL METHOD COMPARISON")
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Non-IID: {non_iid_type} (alpha={alpha})")
    print(f"Methods: {methods}")
    print(f"Clients: {num_clients}, Rounds: {num_rounds}, Local Epochs: {local_epochs}")
    print("=" * 70 + "\n")

    # Setup
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset ONCE
    print("Loading dataset...")
    train_data, test_data, model_type = get_dataset(dataset_name)

    # Create data split ONCE (same for all methods)
    print(f"Creating {non_iid_type} split with seed={seed}...")
    set_seed(seed)

    if non_iid_type == 'dirichlet':
        client_datasets = dirichlet_split(train_data, num_clients, alpha, seed)
    elif non_iid_type == 'quantity':
        client_datasets = quantity_skew_split(train_data, num_clients, 'power_law', seed)
    else:
        client_datasets = iid_split(train_data, num_clients, seed)

    print_data_distribution(client_datasets)

    # Plot and save data distribution
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(save_dir) / 'comparison' / f"{dataset_name}_{non_iid_type}_a{alpha}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    client_labels = get_client_labels(client_datasets)
    plot_client_distribution(
        client_labels,
        num_classes=10,
        save_path=str(result_dir / 'data_distribution.png')
    )

    # Run each method
    all_results = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running {method.upper()}...")
        print(f"{'='*60}")

        result = run_fl_experiment(
            method=method,
            dataset_name=dataset_name,
            num_clients=num_clients,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            alpha=alpha,
            non_iid_type=non_iid_type,
            seed=seed,  # Same seed for model init
            device=device,
            save_dir=str(result_dir),
        )

        all_results[method] = result

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print(f"\n{'Method':<15} {'Final Acc':<12} {'Best Acc':<12} {'Conv@50%':<12} {'Conv@60%':<12}")
    print("-" * 63)

    for method, result in all_results.items():
        history = result['history']
        final_acc = result['final_accuracy'] * 100
        best_acc = result['best_accuracy'] * 100

        # Convergence rounds
        conv_50 = None
        conv_60 = None
        for i, acc in enumerate(history['test_acc']):
            if conv_50 is None and acc >= 0.5:
                conv_50 = history['round'][i]
            if conv_60 is None and acc >= 0.6:
                conv_60 = history['round'][i]

        conv_50_str = str(conv_50) if conv_50 else "N/A"
        conv_60_str = str(conv_60) if conv_60 else "N/A"

        print(f"{method:<15} {final_acc:<12.2f} {best_acc:<12.2f} {conv_50_str:<12} {conv_60_str:<12}")

    print("=" * 70)

    # Plot comparison
    histories = {method: result['history'] for method, result in all_results.items()}
    plot_results(
        histories,
        save_path=str(result_dir / 'comparison_plot.png'),
        title=f"{dataset_name.upper()} - {non_iid_type} (alpha={alpha})"
    )

    # Save summary
    summary_path = result_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("FL METHOD COMPARISON SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Non-IID: {non_iid_type} (alpha={alpha})\n")
        f.write(f"Clients: {num_clients}\n")
        f.write(f"Rounds: {num_rounds}\n")
        f.write(f"Local Epochs: {local_epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Seed: {seed}\n\n")

        f.write("Results:\n")
        f.write("-" * 50 + "\n")
        for method, result in all_results.items():
            f.write(f"\n{method}:\n")
            f.write(f"  Final Accuracy: {result['final_accuracy']*100:.2f}%\n")
            f.write(f"  Best Accuracy: {result['best_accuracy']*100:.2f}%\n")

    print(f"\nResults saved to: {result_dir}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple FL methods on the SAME data split (fair comparison)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare FedAvg vs Fed-M3 on CIFAR-10
  python run_comparison.py --dataset cifar10 --methods fedavg fed_m3 --alpha 0.5

  # Compare on severe non-IID
  python run_comparison.py --dataset cifar10 --methods fedavg fed_m3 --alpha 0.1

  # Quick test
  python run_comparison.py --dataset fmnist --num-rounds 20

IMPORTANT: This script ensures FAIR COMPARISON by:
  - Using SAME data split for all methods (same seed)
  - Using SAME model initialization
  - Only the optimizer/algorithm differs

Output files:
  results/comparison/<dataset>_<non_iid>_a<alpha>_<timestamp>/
  ├── data_distribution.png   # Data distribution visualization
  ├── comparison_plot.png     # Accuracy curves comparison
  ├── summary.txt             # Results summary
  └── <method>/               # Per-method results
        """
    )

    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'fmnist'],
                       help='Dataset: cifar10 or fmnist (default: cifar10)')
    parser.add_argument('--methods', nargs='+', default=['fedavg', 'fed_m3'],
                       help='Methods to compare, e.g., --methods fedavg fed_m3 (default: fedavg fed_m3)')
    parser.add_argument('--num-clients', type=int, default=10,
                       help='Number of FL clients (default: 10)')
    parser.add_argument('--num-rounds', type=int, default=100,
                       help='Number of communication rounds (default: 100)')
    parser.add_argument('--local-epochs', type=int, default=5,
                       help='Local training epochs per round (default: 5)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Local batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--non-iid', type=str, default='dirichlet',
                       choices=['dirichlet', 'quantity', 'iid'],
                       help='Non-IID type: dirichlet, quantity, or iid (default: dirichlet)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet alpha - smaller = more non-IID (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed - SAME seed ensures SAME data split (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--save-dir', type=str, default='./results',
                       help='Directory to save results (default: ./results)')

    args = parser.parse_args()

    run_comparison(
        dataset_name=args.dataset,
        methods=args.methods,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        non_iid_type=args.non_iid,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
    )


if __name__ == '__main__':
    main()
