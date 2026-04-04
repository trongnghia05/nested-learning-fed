"""
Fed-DGD: Federated Delta Gradient Descent

Core idea from Nested Learning/TITAN paper:
- Preconditioner P = α*I - η*(k ⊗ k) to selectively "forget" old information
- k = gradient direction (normalized accumulated gradient)
- Decay along the direction of local gradient before aggregation

Simplified version (to avoid O(d²) memory for k⊗k):
- Per-parameter gradient direction k
- Decay component: project weights onto k direction and decay
- Update: θ = θ - η*g - η*decay_strength*(k·θ)*k
  where the last term decays weights along gradient direction

Why this helps FL:
- Client trains on biased local data → weights drift in local gradient direction
- Before aggregation, decay along this direction to reduce local bias
- Aggregated model is less affected by individual client drift
"""

import copy
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer


class FedDGDOptimizer(Optimizer):
    """
    Fed-DGD: Simplified Delta Gradient Descent for Federated Learning.

    Update rule (per parameter):
        g = gradient
        k = drift direction = normalize(θ_current - θ_global)

        # Decay component (reduce drift along k)
        decay = decay_strength * dot(k, theta) * k

        # Update
        theta = theta - lr * g - lr * decay

    Idea: k = drift direction captures "how much client drifted from global"
    Decaying along k = "reduce local bias, stay closer to global"

    Args:
        params: Model parameters
        lr: Learning rate (η in paper)
        alpha: Uniform decay factor (default: 1.0 = no uniform decay)
        decay_strength: Strength of directional decay along drift (default: 0.1)
        global_params: Global model parameters (to compute drift)
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        alpha: float = 1.0,  # No uniform decay by default
        decay_strength: float = 0.1,
        global_params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        defaults = dict(
            lr=lr,
            alpha=alpha,
            decay_strength=decay_strength,
        )
        super().__init__(params, defaults)

        # Store global params to compute drift
        self.global_params = global_params or {}

        # Drift direction k (normalized drift from global)
        self.drift_direction: Dict[str, torch.Tensor] = {}

        # Debug mode
        self.debug = False
        self.debug_stats = []

    def set_global_params(self, global_params: Dict[str, torch.Tensor]) -> None:
        """Set global parameters to compute drift direction."""
        self.global_params = {k: v.clone() for k, v in global_params.items()}

    def set_debug(self, debug: bool) -> None:
        """Enable or disable debug mode."""
        self.debug = debug
        if debug:
            self.debug_stats = []

    def get_drift_direction(self) -> Dict[str, torch.Tensor]:
        """Return drift directions (k) to send to server."""
        return {k: v.clone() for k, v in self.drift_direction.items()}

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            decay_strength = group['decay_strength']

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
                # COMPUTE DRIFT DIRECTION k
                # k = normalize(θ_current - θ_global)
                # ========================================
                if param_name in self.global_params:
                    global_p = self.global_params[param_name].to(p.device)
                    drift = p.data - global_p
                    drift_norm = torch.norm(drift)
                    if drift_norm > 1e-8:
                        k = drift / drift_norm  # Unit vector in drift direction
                    else:
                        k = torch.zeros_like(p.data)
                else:
                    k = torch.zeros_like(p.data)

                # Save drift direction for server
                self.drift_direction[param_name] = k.clone()

                # ========================================
                # DGD UPDATE (Drift-based)
                # ========================================
                # k = drift direction (θ - θ_global)
                # Decay along k = reduce drift, stay closer to global

                # 1. Uniform decay: α * θ (skip if α = 1.0)
                if alpha < 1.0:
                    p.data.mul_(alpha)

                # 2. Gradient step: - η * g
                p.data.add_(grad, alpha=-lr)

                # 3. Directional decay: - η * decay_strength * (k·θ) * k
                # This decays weights along DRIFT direction (reduces client drift)
                if decay_strength > 0 and torch.norm(k) > 1e-8:
                    # Flatten for dot product
                    p_flat = p.data.view(-1)
                    k_flat = k.view(-1)

                    # Projection of θ onto k (drift direction)
                    proj = torch.dot(p_flat, k_flat)

                    # Decay along drift direction
                    decay_term = decay_strength * proj * k
                    p.data.add_(decay_term, alpha=-lr)

                # Debug stats
                if self.debug and state['step'] == 1:
                    self.debug_stats.append({
                        'param': param_name[:30],
                        'grad_norm': torch.norm(grad).item(),
                        'k_norm': torch.norm(k).item(),
                        'drift_norm': drift_norm.item() if param_name in self.global_params else 0.0,
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


def fed_dgd_optimizer_fn(
    model: nn.Module,
    lr: float = 0.01,
    extra_state: Optional[Dict[str, Any]] = None,
    alpha: float = 1.0,  # No uniform decay by default
    decay_strength: float = 0.1,
    debug: bool = False,
) -> Tuple[FedDGDOptimizer, Dict[str, Any]]:
    """
    Factory function to create Fed-DGD optimizer for a client.

    Args:
        model: Neural network model
        lr: Learning rate
        extra_state: Contains 'global_params' for computing drift
        alpha: Uniform decay factor
        decay_strength: Strength of directional decay along drift
        debug: Enable debug mode

    Returns:
        Tuple of (optimizer, extra_data_dict)
    """
    # Get global params for drift computation
    global_params = {}
    if extra_state and 'global_params' in extra_state:
        global_params = extra_state['global_params']

    optimizer = FedDGDOptimizer(
        model.parameters(),
        lr=lr,
        alpha=alpha,
        decay_strength=decay_strength,
        global_params=global_params,
    )

    # Set parameter names
    optimizer.set_param_names(list(model.named_parameters()))

    if debug:
        optimizer.set_debug(True)

    def get_extra():
        result = {'drift_direction': optimizer.get_drift_direction()}
        if debug:
            result['debug_stats'] = optimizer.get_debug_stats()
        return result

    return optimizer, {'get_extra_fn': get_extra}


def fed_dgd_aggregate(
    global_model: nn.Module,
    client_results: List[Dict[str, Any]],
    server_state: Dict[str, Any],
    alpha_global: float = 1.0,  # No global decay by default
) -> Dict[str, Any]:
    """
    Fed-DGD aggregation: FedAvg + optional global decay.

    Args:
        global_model: Global model (updated in-place)
        client_results: Results from client training
        server_state: Server state
        alpha_global: Global decay factor (not used currently)

    Returns:
        Dict with 'train_loss', 'drift_direction'
    """
    total_samples = sum(r['num_samples'] for r in client_results)

    # ==========================================
    # 1. FedAvg: Aggregate model parameters
    # ==========================================
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

    # ==========================================
    # 2. Aggregate drift directions (k)
    # ==========================================
    aggregated_k = {}

    for result in client_results:
        extra = result.get('extra', {})
        get_extra_fn = extra.get('get_extra_fn')
        if get_extra_fn:
            extra_data = get_extra_fn()
            if 'drift_direction' in extra_data:
                k_dict = extra_data['drift_direction']
                weight = result['num_samples'] / total_samples
                for key, k in k_dict.items():
                    if key not in aggregated_k:
                        aggregated_k[key] = torch.zeros_like(k)
                    aggregated_k[key] += weight * k

    # Normalize aggregated k
    global_k = {}
    for key, k in aggregated_k.items():
        norm = torch.norm(k)
        if norm > 1e-8:
            global_k[key] = k / norm
        else:
            global_k[key] = k.clone()

    server_state['drift_direction'] = global_k

    # ==========================================
    # 3. Apply global decay (optional)
    # ==========================================
    # Simple version: just use aggregated params
    # Advanced version: decay global model along aggregated k direction

    # For now, just use FedAvg result
    global_model.load_state_dict(aggregated_params)

    # ==========================================
    # 4. Compute average training loss
    # ==========================================
    avg_train_loss = sum(
        r['train_loss'] * r['num_samples'] for r in client_results
    ) / total_samples

    return {
        'train_loss': avg_train_loss,
        'total_samples': total_samples,
        'drift_direction': global_k,
    }
