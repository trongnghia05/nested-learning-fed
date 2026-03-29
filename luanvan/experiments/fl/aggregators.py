"""Aggregation functions for FL."""

from typing import Dict, List, Any
import torch
import torch.nn as nn


def fedavg_aggregate(
    global_model: nn.Module,
    client_results: List[Dict[str, Any]],
    server_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    FedAvg aggregation: weighted average by number of samples.

    Args:
        global_model: Global model (will be updated in-place)
        client_results: Results from client training
        server_state: Server state dict (unused for FedAvg)

    Returns:
        Dict with aggregation info
    """
    total_samples = sum(r['num_samples'] for r in client_results)

    # Initialize aggregated params
    aggregated_params = {}
    for key in client_results[0]['params']:
        aggregated_params[key] = torch.zeros_like(
            client_results[0]['params'][key],
            dtype=torch.float32
        )

    # Weighted average
    for result in client_results:
        weight = result['num_samples'] / total_samples
        for key in aggregated_params:
            aggregated_params[key] += weight * result['params'][key].float()

    # Update global model
    global_model.load_state_dict(aggregated_params)

    # Compute average training loss
    avg_train_loss = sum(
        r['train_loss'] * r['num_samples'] for r in client_results
    ) / total_samples

    return {
        'train_loss': avg_train_loss,
        'total_samples': total_samples,
    }


def weighted_aggregate(
    global_model: nn.Module,
    client_results: List[Dict[str, Any]],
    server_state: Dict[str, Any],
    weights: List[float] = None,
) -> Dict[str, Any]:
    """
    Weighted aggregation with custom weights.

    Args:
        global_model: Global model (will be updated in-place)
        client_results: Results from client training
        server_state: Server state dict
        weights: Custom weights for each client (default: uniform)

    Returns:
        Dict with aggregation info
    """
    num_clients = len(client_results)

    if weights is None:
        weights = [1.0 / num_clients] * num_clients
    else:
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

    # Initialize aggregated params
    aggregated_params = {}
    for key in client_results[0]['params']:
        aggregated_params[key] = torch.zeros_like(
            client_results[0]['params'][key],
            dtype=torch.float32
        )

    # Weighted average
    for result, weight in zip(client_results, weights):
        for key in aggregated_params:
            aggregated_params[key] += weight * result['params'][key].float()

    # Update global model
    global_model.load_state_dict(aggregated_params)

    # Compute average training loss
    avg_train_loss = sum(
        r['train_loss'] * w for r, w in zip(client_results, weights)
    )

    return {
        'train_loss': avg_train_loss,
        'total_samples': sum(r['num_samples'] for r in client_results),
    }
