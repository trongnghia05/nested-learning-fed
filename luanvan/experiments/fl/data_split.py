"""Data splitting utilities for non-IID FL experiments."""

from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def iid_split(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42
) -> List[Subset]:
    """
    Split dataset IID (uniformly) across clients.

    Args:
        dataset: Full training dataset
        num_clients: Number of clients
        seed: Random seed for reproducibility

    Returns:
        List of client datasets (Subset objects)
    """
    np.random.seed(seed)

    n = len(dataset)
    indices = np.random.permutation(n)

    # Split evenly
    split_size = n // num_clients
    client_datasets = []

    for i in range(num_clients):
        start = i * split_size
        end = start + split_size if i < num_clients - 1 else n
        client_indices = indices[start:end].tolist()
        client_datasets.append(Subset(dataset, client_indices))

    return client_datasets


def dirichlet_split(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42
) -> List[Subset]:
    """
    Split dataset using Dirichlet distribution for label skew.

    Args:
        dataset: Full training dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (smaller = more non-IID)
               - alpha = 0.1: Very non-IID (each client dominated by 1-2 classes)
               - alpha = 0.5: Moderate non-IID
               - alpha = 1.0: Mild non-IID
               - alpha = 10.0: Near IID
        seed: Random seed for reproducibility

    Returns:
        List of client datasets (Subset objects)
    """
    np.random.seed(seed)

    # Get labels
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))

    # Sample from Dirichlet for each class
    # proportions[class_id][client_id] = proportion of class for client
    proportions = np.random.dirichlet(
        [alpha] * num_clients,
        size=num_classes
    )  # Shape: (num_classes, num_clients)

    # Assign indices to clients
    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)

        # Split according to proportions
        props = proportions[class_id]
        props = props / props.sum()  # Normalize

        split_points = (np.cumsum(props) * len(class_indices)).astype(int)

        start = 0
        for client_id in range(num_clients):
            end = split_points[client_id]
            client_indices[client_id].extend(class_indices[start:end].tolist())
            start = end

    # Create subsets
    client_datasets = []
    for indices in client_indices:
        if len(indices) > 0:
            client_datasets.append(Subset(dataset, indices))
        else:
            # Handle empty client (shouldn't happen with reasonable alpha)
            # Give at least one sample
            client_datasets.append(Subset(dataset, [0]))

    return client_datasets


def quantity_skew_split(
    dataset: Dataset,
    num_clients: int,
    distribution: str = 'power_law',
    seed: int = 42
) -> List[Subset]:
    """
    Split dataset with different quantities per client.

    Args:
        dataset: Full training dataset
        num_clients: Number of clients
        distribution: 'power_law', 'exponential', or 'uniform'
        seed: Random seed for reproducibility

    Returns:
        List of client datasets (Subset objects)
    """
    np.random.seed(seed)

    n = len(dataset)
    indices = np.random.permutation(n).tolist()

    if distribution == 'power_law':
        # Power law: few clients have many samples (Zipf-like)
        weights = 1 / np.arange(1, num_clients + 1)
    elif distribution == 'exponential':
        # Exponential decay
        weights = np.exp(-np.arange(num_clients) / (num_clients / 3))
    else:  # uniform
        weights = np.ones(num_clients)

    weights = weights / weights.sum()

    # Compute split points
    split_points = (np.cumsum(weights) * n).astype(int)

    # Create subsets
    client_datasets = []
    start = 0
    for i in range(num_clients):
        end = split_points[i]
        client_datasets.append(Subset(dataset, indices[start:end]))
        start = end

    return client_datasets


def get_client_labels(client_datasets: List[Subset]) -> List[List[int]]:
    """
    Extract labels from client datasets for visualization.

    Args:
        client_datasets: List of client Subset objects

    Returns:
        List of label lists for each client
    """
    client_labels = []
    for ds in client_datasets:
        labels = [ds.dataset[idx][1] for idx in ds.indices]
        client_labels.append(labels)
    return client_labels


def print_data_distribution(client_datasets: List[Subset], num_classes: int = 10) -> None:
    """Print data distribution summary for each client."""
    print("\n" + "=" * 60)
    print("DATA DISTRIBUTION SUMMARY")
    print("=" * 60)

    total_samples = sum(len(ds) for ds in client_datasets)

    for i, ds in enumerate(client_datasets):
        labels = [ds.dataset[idx][1] for idx in ds.indices]
        counts = np.bincount(labels, minlength=num_classes)
        dominant_class = np.argmax(counts)
        dominant_pct = counts[dominant_class] / len(labels) * 100 if labels else 0

        print(f"Client {i:2d}: {len(ds):5d} samples ({len(ds)/total_samples*100:5.1f}%) | "
              f"Dominant class: {dominant_class} ({dominant_pct:.1f}%) | "
              f"Classes: {np.sum(counts > 0)}/10")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test with CIFAR-10
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)

    print("Testing IID split...")
    iid_datasets = iid_split(train_data, num_clients=10)
    print_data_distribution(iid_datasets)

    print("Testing Dirichlet split (alpha=0.5)...")
    dir_datasets = dirichlet_split(train_data, num_clients=10, alpha=0.5)
    print_data_distribution(dir_datasets)

    print("Testing Quantity skew split...")
    qty_datasets = quantity_skew_split(train_data, num_clients=10)
    print_data_distribution(qty_datasets)
