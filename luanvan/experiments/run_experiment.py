"""
Main script to run FL experiments.

Usage:
    python run_experiment.py --method fedavg --dataset cifar10 --alpha 0.5
    python run_experiment.py --method fed_m3 --dataset cifar10 --alpha 0.5
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable

import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from fl import FLClient, FLServer, dirichlet_split, quantity_skew_split, iid_split
from fl.data_split import print_data_distribution
from fl.aggregators import fedavg_aggregate
from optimizers import fed_m3_optimizer_fn, fed_m3_aggregate
from utils import set_seed, MetricsTracker


def get_dataset(
    name: str,
    data_dir: str = './data',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Load dataset with train/val/test split (80-10-10 by default).

    Args:
        name: Dataset name ('cifar10' or 'fmnist')
        data_dir: Directory to store data
        train_ratio: Ratio for training (default: 0.8 = 80%)
        val_ratio: Ratio for validation (default: 0.1 = 10%)
        test_ratio: Ratio for testing (default: 0.1 = 10%)
        seed: Random seed for reproducible split

    Returns:
        train_data: Training dataset (to be split among clients)
        val_data: Validation dataset (for tuning)
        test_data: Test dataset (for final evaluation)
        model_type: Model architecture to use

    Split strategy (80-10-10):
        CIFAR-10:  50,000 total → 40,000 train + 5,000 val + 5,000 test
        FMNIST:    60,000 total → 48,000 train + 6,000 val + 6,000 test
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    if name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
        # Note: Dùng transform_train cho cả val/test để đơn giản
        # Trong thực tế có thể dùng transform riêng (không augment)
        full_data = datasets.CIFAR10(data_dir, train=True, download=True,
                                     transform=transform_train)
        model_type = 'cnn_medium'

    elif name == 'fmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        full_data = datasets.FashionMNIST(data_dir, train=True, download=True,
                                          transform=transform)
        model_type = 'cnn_small'

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Calculate split sizes
    total = len(full_data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size  # Remainder goes to test

    # Use generator for reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_data, val_data, test_data = random_split(
        full_data,
        [train_size, val_size, test_size],
        generator=generator
    )

    print(f"\nDataset: {name.upper()}")
    print(f"  Total:  {total} samples")
    print(f"  ├── Train: {len(train_data)} samples ({train_ratio*100:.0f}%) → to be split among clients")
    print(f"  ├── Val:   {len(val_data)} samples ({val_ratio*100:.0f}%) → for hyperparameter tuning")
    print(f"  └── Test:  {len(test_data)} samples ({test_ratio*100:.0f}%) → for final evaluation")

    return train_data, val_data, test_data, model_type


def run_fl_experiment(
    method: str,
    dataset_name: str,
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
    # Fed-M3 specific
    beta1: float = 0.9,
    beta2: float = 0.999,
    beta3: float = 0.9,
    lam: float = 0.3,
    ns_steps: int = 5,
    v_init: float = 1.0,
    # Debug mode
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run a complete FL experiment.

    Args:
        method: 'fedavg' or 'fed_m3'
        dataset_name: 'cifar10' or 'fmnist'
        num_clients: Number of FL clients
        num_rounds: Number of communication rounds
        local_epochs: Local training epochs per round
        batch_size: Local batch size
        lr: Learning rate
        alpha: Dirichlet alpha for non-IID (smaller = more non-IID)
        non_iid_type: 'dirichlet', 'quantity', or 'iid'
        seed: Random seed
        device: 'cuda', 'cpu', or 'auto'
        save_dir: Directory to save results
        beta1, beta2, beta3, lam, ns_steps, v_init: Fed-M3 hyperparameters

    Returns:
        Dict with experiment results
    """
    # Setup device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")

    # Set seed
    set_seed(seed)

    # Load dataset
    print(f"\nLoading {dataset_name} dataset...")
    train_data, val_data, test_data, model_type = get_dataset(
        dataset_name, val_ratio=0.1, seed=seed
    )

    # Create data splits
    print(f"Creating {non_iid_type} split (alpha={alpha})...")
    if non_iid_type == 'dirichlet':
        client_datasets = dirichlet_split(train_data, num_clients, alpha, seed)
    elif non_iid_type == 'quantity':
        client_datasets = quantity_skew_split(train_data, num_clients, 'power_law', seed)
    else:  # iid
        client_datasets = iid_split(train_data, num_clients, seed)

    print_data_distribution(client_datasets)

    # Create model
    print(f"Creating model: {model_type}")
    model = create_model(model_type, num_classes=10)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create server
    # Note: val_data được giữ riêng cho hyperparameter tuning (không dùng trong training loop)
    # Test data dùng cho final evaluation
    server = FLServer(model, device, test_data)

    # Create clients
    clients = []
    for i in range(num_clients):
        client = FLClient(
            client_id=i,
            dataset=client_datasets[i],
            model=model,
            device=device,
            batch_size=batch_size,
        )
        clients.append(client)

    # Setup method-specific components
    if method == 'fedavg':
        optimizer_fn = None
        aggregator_fn = fedavg_aggregate
    elif method == 'fed_m3':
        def optimizer_fn(model, lr, extra_state):
            return fed_m3_optimizer_fn(
                model, lr, extra_state,
                beta1=beta1, beta2=beta2, lam=lam, ns_steps=ns_steps,
                v_init=v_init, debug=debug
            )
        def aggregator_fn(global_model, client_results, server_state):
            return fed_m3_aggregate(
                global_model, client_results, server_state,
                beta3=beta3, ns_steps=ns_steps
            )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Metrics tracker
    metrics = MetricsTracker()

    # Training loop
    print(f"\n{'='*70}")
    print(f"Starting {method.upper()} training")
    print(f"Rounds: {num_rounds}, Local epochs: {local_epochs}, LR: {lr}")
    print(f"Clients: {num_clients}, Non-IID: {non_iid_type} (alpha={alpha})")
    if debug:
        print(f"DEBUG MODE: ON")
        if method == 'fed_m3':
            print(f"Fed-M3 params: beta1={beta1}, beta2={beta2}, beta3={beta3}, lam={lam}, ns_steps={ns_steps}, v_init={v_init}")
    print(f"{'='*70}\n")

    for round_num in tqdm(range(1, num_rounds + 1), desc="FL Rounds", disable=debug):
        # Get global parameters
        global_params = server.get_global_params()

        # Get extra state for Fed-M3 (global direction)
        extra_state = {}
        if method == 'fed_m3' and 'global_momentum' in server.server_state:
            extra_state['global_momentum'] = server.server_state['global_momentum']

        # Client training
        client_results = []
        client_train_losses = []
        for client in clients:
            result = client.local_train(
                global_params=global_params,
                local_epochs=local_epochs,
                lr=lr,
                optimizer_fn=optimizer_fn,
                extra_state=extra_state,
            )
            client_results.append(result)
            client_train_losses.append(result['train_loss'])

            # Debug: Print per-client training info
            if debug:
                print(f"  [DEBUG] Client {client.client_id}: "
                      f"loss={result['train_loss']:.4f}, "
                      f"samples={result['num_samples']}")

                # Print Fed-M3 optimizer stats if available
                if method == 'fed_m3':
                    extra = result.get('extra', {})
                    get_extra_fn = extra.get('get_extra_fn')
                    if get_extra_fn:
                        extra_data = get_extra_fn()
                        debug_stats = extra_data.get('debug_stats', [])
                        if debug_stats:
                            print(f"    [DEBUG] Optimizer stats (first step):")
                            for stat in debug_stats[:3]:
                                print(f"      {stat['param']}: grad={stat['grad_norm']:.4f}, "
                                      f"m1={stat['m1_norm']:.4f}, m2={stat['m2_norm']:.4f}, "
                                      f"update={stat['update_norm']:.4f}")

        # Evaluate each client on their LOCAL data (before aggregation)
        client_local_accs = []
        for client in clients:
            client_eval = client.evaluate()
            client_local_accs.append(client_eval['accuracy'])

        # Server aggregation
        agg_result = server.aggregate(client_results, aggregator_fn)

        # Debug: Print aggregation info
        if debug:
            print(f"\n  [DEBUG] Round {round_num} Aggregation:")
            print(f"    Avg train loss: {agg_result['train_loss']:.4f}")
            print(f"    Total samples: {agg_result.get('total_samples', 'N/A')}")

            # Check for NaN/Inf in model parameters
            global_params_after = server.get_global_params()
            has_nan = False
            has_inf = False
            param_stats = []
            for name, param in global_params_after.items():
                # Skip non-floating point tensors (e.g., num_batches_tracked)
                if not param.is_floating_point():
                    continue
                if torch.isnan(param).any():
                    has_nan = True
                    print(f"    WARNING: NaN in {name}")
                if torch.isinf(param).any():
                    has_inf = True
                    print(f"    WARNING: Inf in {name}")
                param_stats.append({
                    'name': name,
                    'mean': param.mean().item(),
                    'std': param.std().item() if param.numel() > 1 else 0.0,
                    'min': param.min().item(),
                    'max': param.max().item(),
                })

            if not has_nan and not has_inf:
                print(f"    Model params: OK (no NaN/Inf)")

            # Print first few param stats
            print(f"    Sample param stats:")
            for stat in param_stats[:3]:
                print(f"      {stat['name']}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, "
                      f"range=[{stat['min']:.4f}, {stat['max']:.4f}]")

            # Fed-M3 specific debug
            if method == 'fed_m3' and 'global_momentum' in agg_result:
                gd = agg_result['global_momentum']
                print(f"    Global momentum (m2) keys: {len(gd)}")
                for key in list(gd.keys())[:2]:
                    gd_tensor = gd[key]
                    print(f"      {key}: norm={torch.norm(gd_tensor).item():.4f}")

        # Evaluate GLOBAL model on TEST data
        eval_result = server.evaluate()

        # Evaluate each client with GLOBAL model on their local data
        # (This shows how well global model works for each client)
        client_global_accs = []
        global_state = server.get_global_params()
        for client in clients:
            client.set_model_params(global_state)
            client_eval = client.evaluate()
            client_global_accs.append(client_eval['accuracy'])

        # Log metrics (including per-client)
        metrics.log(
            round_num=round_num,
            train_loss=agg_result['train_loss'],
            test_acc=eval_result['accuracy'],
            test_loss=eval_result['loss'],
            client_accs=client_global_accs,  # Per-client accuracy with global model
        )

        # Print progress every 10 rounds (or first round, or every round in debug mode)
        if round_num % 10 == 0 or round_num == 1 or debug:
            print(f"\n{'─'*70}")
            print(f"Round {round_num}")
            print(f"{'─'*70}")

            # Global metrics
            print(f"  Global Test Acc: {eval_result['accuracy']*100:.2f}% | "
                  f"Test Loss: {eval_result['loss']:.4f} | "
                  f"Avg Train Loss: {agg_result['train_loss']:.4f}")

            # Per-client metrics
            print(f"\n  Per-Client Results (with global model on local data):")
            print(f"  {'Client':<8} {'Samples':<10} {'Train Loss':<12} {'Local Acc':<12} {'Global Acc':<12}")
            print(f"  {'-'*54}")
            for i in range(num_clients):
                print(f"  {i:<8} {client_results[i]['num_samples']:<10} "
                      f"{client_train_losses[i]:<12.4f} "
                      f"{client_local_accs[i]*100:<12.2f} "
                      f"{client_global_accs[i]*100:<12.2f}")

            # Summary statistics
            print(f"\n  Summary:")
            print(f"    Acc Range: {min(client_global_accs)*100:.2f}% - {max(client_global_accs)*100:.2f}%")
            print(f"    Acc Std:   {np.std(client_global_accs)*100:.2f}%")
            print(f"    Loss Range: {min(client_train_losses):.4f} - {max(client_train_losses):.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(save_dir) / method / f"{dataset_name}_{non_iid_type}_a{alpha}"
    result_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = result_dir / f"metrics_{timestamp}.json"
    metrics.save(str(metrics_path))
    print(f"\nMetrics saved to: {metrics_path}")

    # Save checkpoint
    checkpoint_path = result_dir / f"model_{timestamp}.pt"
    server.save_checkpoint(str(checkpoint_path), num_rounds, metrics.history)
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Method:       {method}")
    print(f"Dataset:      {dataset_name}")
    print(f"Non-IID:      {non_iid_type} (alpha={alpha})")
    print(f"Clients:      {num_clients}")
    print(f"Rounds:       {num_rounds}")
    print(f"{'─'*70}")
    print(f"Final Global Test Accuracy:  {metrics.get_final_accuracy()*100:.2f}%")
    print(f"Best Global Test Accuracy:   {metrics.get_best_accuracy()*100:.2f}%")
    print(f"Convergence to 50%:          Round {metrics.get_convergence_round(0.5)}")
    print(f"Convergence to 60%:          Round {metrics.get_convergence_round(0.6)}")

    # Final per-client statistics
    if metrics.history['client_accs'] and metrics.history['client_accs'][-1]:
        final_client_accs = metrics.history['client_accs'][-1]
        print(f"{'─'*70}")
        print("Final Per-Client Accuracy (global model on local data):")
        print(f"  Mean:   {np.mean(final_client_accs)*100:.2f}%")
        print(f"  Std:    {np.std(final_client_accs)*100:.2f}%")
        print(f"  Min:    {min(final_client_accs)*100:.2f}%")
        print(f"  Max:    {max(final_client_accs)*100:.2f}%")
        print(f"  Range:  {(max(final_client_accs)-min(final_client_accs))*100:.2f}%")

    print(f"{'='*70}\n")

    return {
        'method': method,
        'dataset': dataset_name,
        'non_iid_type': non_iid_type,
        'alpha': alpha,
        'final_accuracy': metrics.get_final_accuracy(),
        'best_accuracy': metrics.get_best_accuracy(),
        'history': metrics.history,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run FL Experiment (FedAvg or Fed-M3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FedAvg on CIFAR-10 with moderate non-IID
  python run_experiment.py --method fedavg --dataset cifar10 --alpha 0.5

  # Fed-M3 on CIFAR-10 with severe non-IID
  python run_experiment.py --method fed_m3 --dataset cifar10 --alpha 0.1

  # Quick test (few rounds)
  python run_experiment.py --method fedavg --num-rounds 10 --local-epochs 1

Non-IID Alpha values:
  0.1  = Very non-IID (each client has 1-2 dominant classes)
  0.5  = Moderate non-IID (recommended for main experiments)
  1.0  = Mild non-IID
  10.0 = Near IID

Fed-M3 Lambda values:
  0.0 = Only local direction
  0.3 = Default (70% local + 30% global)
  0.5 = Balanced
  1.0 = Prefer global direction
        """
    )

    # Method and dataset
    parser.add_argument('--method', type=str, default='fedavg',
                       choices=['fedavg', 'fed_m3'],
                       help='FL method: fedavg or fed_m3 (default: fedavg)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'fmnist'],
                       help='Dataset: cifar10 or fmnist (default: cifar10)')

    # FL settings
    parser.add_argument('--num-clients', type=int, default=5,
                       help='Number of FL clients (default: 10)')
    parser.add_argument('--num-rounds', type=int, default=10,
                       help='Number of communication rounds (default: 100)')
    parser.add_argument('--local-epochs', type=int, default=5,
                       help='Local training epochs per round (default: 5)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Local batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')

    # Non-IID settings
    parser.add_argument('--non-iid', type=str, default='dirichlet',
                       choices=['dirichlet', 'quantity', 'iid'],
                       help='Non-IID type: dirichlet, quantity, or iid (default: dirichlet)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet alpha - smaller = more non-IID (default: 0.5)')

    # Fed-M3 settings
    parser.add_argument('--beta1', type=float, default=0.9,
                       help='Fed-M3: fast momentum coefficient (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999,
                       help='Fed-M3: second moment coefficient (default: 0.999)')
    parser.add_argument('--beta3', type=float, default=0.9,
                       help='Fed-M3: slow momentum coefficient at server (default: 0.9)')
    parser.add_argument('--lam', type=float, default=0.3,
                       help='Fed-M3: lambda - balance local vs global, 0=local only (default: 0.3)')
    parser.add_argument('--ns-steps', type=int, default=5,
                       help='Newton-Schulz iterations (default: 5)')
    parser.add_argument('--v-init', type=float, default=1.0,
                       help='Fed-M3: initial value for v (default: 1.0, prevents explosion)')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto (default: auto)')
    parser.add_argument('--save-dir', type=str, default='./results',
                       help='Directory to save results (default: ./results)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with detailed logging')

    args = parser.parse_args()

    # Run experiment
    run_fl_experiment(
        method=args.method,
        dataset_name=args.dataset,
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
        beta1=args.beta1,
        beta2=args.beta2,
        beta3=args.beta3,
        lam=args.lam,
        ns_steps=args.ns_steps,
        v_init=args.v_init,
        debug=args.debug,
    )


if __name__ == '__main__':
    main()
