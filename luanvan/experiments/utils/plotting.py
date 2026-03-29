"""Plotting utilities for FL experiments."""

from typing import Dict, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_results(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "FL Experiment Results"
) -> None:
    """
    Plot comparison of multiple methods.

    Args:
        results: Dict mapping method name to MetricsTracker.history
        save_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Test Accuracy
    ax1 = axes[0]
    for method_name, history in results.items():
        rounds = history['round']
        acc = history['test_acc']
        ax1.plot(rounds, acc, label=method_name, linewidth=2)

    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title(f'{title} - Test Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training Loss
    ax2 = axes[1]
    for method_name, history in results.items():
        rounds = history['round']
        loss = history['train_loss']
        ax2.plot(rounds, loss, label=method_name, linewidth=2)

    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Training Loss')
    ax2.set_title(f'{title} - Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_client_distribution(
    client_labels: List[List[int]],
    num_classes: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Plot class distribution for each client.

    Args:
        client_labels: List of label lists for each client
        num_classes: Total number of classes
        save_path: Path to save figure
    """
    num_clients = len(client_labels)

    # Determine grid size
    cols = min(5, num_clients)
    rows = (num_clients + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if num_clients == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for client_id, labels in enumerate(client_labels):
        ax = axes[client_id]
        counts = np.bincount(labels, minlength=num_classes)
        ax.bar(range(num_classes), counts, color='steelblue')
        ax.set_title(f'Client {client_id}')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_xticks(range(num_classes))

    # Hide empty subplots
    for i in range(num_clients, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Data Distribution per Client', fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_convergence_comparison(
    results: Dict[str, Dict],
    target_accuracies: List[float] = [0.5, 0.6, 0.7],
    save_path: Optional[str] = None
) -> None:
    """
    Plot bar chart comparing rounds to reach target accuracies.

    Args:
        results: Dict mapping method name to MetricsTracker.history
        target_accuracies: List of target accuracy thresholds
        save_path: Path to save figure
    """
    methods = list(results.keys())
    x = np.arange(len(target_accuracies))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(methods):
        history = results[method]
        rounds_to_target = []

        for target in target_accuracies:
            found = None
            for j, acc in enumerate(history['test_acc']):
                if acc >= target:
                    found = history['round'][j]
                    break
            rounds_to_target.append(found if found else history['round'][-1])

        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, rounds_to_target, width, label=method)

    ax.set_xlabel('Target Accuracy')
    ax.set_ylabel('Rounds to Reach')
    ax.set_title('Convergence Speed Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(t*100)}%' for t in target_accuracies])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
