"""
FedProx: Federated Optimization with Proximal Term

Paper: "Federated Optimization in Heterogeneous Networks" (Li et al., 2020)

Core idea:
    Add proximal term to local objective to keep local model close to global.

    Local objective:
        min_θ L(θ; D_local) + (μ/2) * ||θ - θ_global||²

    Update rule:
        θ = θ - lr * (grad + μ * (θ - θ_global))

    μ (mu) controls the strength of the proximal term:
        - μ = 0: Same as FedAvg (no regularization)
        - μ > 0: Penalize drift from global model
        - Larger μ: Stronger regularization, less client drift
"""

import copy
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer


class FedProxOptimizer(Optimizer):
    """
    FedProx: SGD with Proximal Term for Federated Learning.

    Update rule:
        θ = θ - lr * (grad + μ * (θ - θ_global))

    This is equivalent to:
        θ = θ - lr * grad - lr * μ * (θ - θ_global)
        = (1 - lr*μ) * θ + lr*μ * θ_global - lr * grad

    Args:
        params: Model parameters
        lr: Learning rate
        mu: Proximal term coefficient (default: 0.01)
        global_params: Global model parameters (θ_global)
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        mu: float = 0.01,
        global_params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        defaults = dict(lr=lr, mu=mu)
        super().__init__(params, defaults)

        # Store global params for proximal term
        self.global_params = global_params or {}

        # Debug mode
        self.debug = False
        self.debug_stats = []

    def set_global_params(self, global_params: Dict[str, torch.Tensor]) -> None:
        """Set global parameters for proximal term."""
        self.global_params = {k: v.clone() for k, v in global_params.items()}

    def set_debug(self, debug: bool) -> None:
        """Enable or disable debug mode."""
        self.debug = debug
        if debug:
            self.debug_stats = []

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                param_name = self._get_param_name(p)

                # Get or initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0

                state['step'] += 1

                # ========================================
                # FedProx UPDATE
                # θ = θ - lr * grad - lr * μ * (θ - θ_global)
                # ========================================

                # Compute proximal term BEFORE gradient step
                # (using original θ, not θ - lr*grad)
                prox_term = None
                if mu > 0 and param_name in self.global_params:
                    global_p = self.global_params[param_name].to(p.device)
                    prox_term = p.data - global_p  # θ - θ_global

                # 1. Gradient step: - lr * grad
                p.data.add_(grad, alpha=-lr)

                # 2. Proximal term: - lr * μ * (θ - θ_global)
                if prox_term is not None:
                    p.data.add_(prox_term, alpha=-lr * mu)

                # Debug stats
                if self.debug and state['step'] == 1:
                    diff_norm = 0.0
                    if param_name in self.global_params:
                        global_p = self.global_params[param_name].to(p.device)
                        diff_norm = torch.norm(p.data - global_p).item()

                    self.debug_stats.append({
                        'param': param_name[:30],
                        'grad_norm': torch.norm(grad).item(),
                        'diff_from_global': diff_norm,
                        'param_norm': torch.norm(p.data).item(),
                    })

        return loss

    def get_debug_stats(self):
        """Return debug statistics."""
        return self.debug_stats

    def clear_debug_stats(self):
        """Clear debug statistics."""
        self.debug_stats = []

    def _get_param_name(self, param: torch.Tensor) -> str:
        """Get a unique name for the parameter."""
        if hasattr(self, 'param_names') and id(param) in self.param_names:
            return self.param_names[id(param)]
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p is param:
                    return f"param_{i}"
        return f"param_unknown"

    def set_param_names(self, named_params):
        """Set parameter names from model.named_parameters()."""
        self.param_names = {id(p): name for name, p in named_params}


def fedprox_optimizer_fn(
    model: nn.Module,
    lr: float = 0.01,
    extra_state: Optional[Dict[str, Any]] = None,
    mu: float = 0.01,
    debug: bool = False,
) -> Tuple[FedProxOptimizer, Dict[str, Any]]:
    """
    Factory function to create FedProx optimizer for a client.

    Args:
        model: Neural network model
        lr: Learning rate
        extra_state: Contains 'global_params' for proximal term
        mu: Proximal term coefficient
        debug: Enable debug mode

    Returns:
        Tuple of (optimizer, extra_data_dict)
    """
    # Get global params for proximal term
    global_params = {}
    if extra_state and 'global_params' in extra_state:
        global_params = extra_state['global_params']

    optimizer = FedProxOptimizer(
        model.parameters(),
        lr=lr,
        mu=mu,
        global_params=global_params,
    )

    # Set parameter names
    optimizer.set_param_names(list(model.named_parameters()))

    if debug:
        optimizer.set_debug(True)

    def get_extra():
        result = {}
        if debug:
            result['debug_stats'] = optimizer.get_debug_stats()
        return result

    return optimizer, {'get_extra_fn': get_extra}


def fedprox_aggregate(
    global_model: nn.Module,
    client_results: List[Dict[str, Any]],
    server_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    FedProx aggregation: Same as FedAvg (weighted average).

    The proximal term only affects local training, not aggregation.

    Args:
        global_model: Global model (updated in-place)
        client_results: Results from client training
        server_state: Server state

    Returns:
        Dict with 'train_loss'
    """
    total_samples = sum(r['num_samples'] for r in client_results)

    # FedAvg: Aggregate model parameters
    aggregated_params = {}
    for key in client_results[0]['params']:
        aggregated_params[key] = torch.zeros_like(
            client_results[0]['params'][key],
            dtype=torch.float32
        )

    for result in client_results:
        weight = result['num_samples'] / total_samples
        for key in aggregated_params:
            aggregated_params[key] += weight * result['params'][key].float()

    global_model.load_state_dict(aggregated_params)

    # Compute average training loss
    avg_train_loss = sum(
        r['train_loss'] * r['num_samples'] for r in client_results
    ) / total_samples

    return {
        'train_loss': avg_train_loss,
        'total_samples': total_samples,
    }
