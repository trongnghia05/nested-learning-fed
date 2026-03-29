"""
Fed-M3 Lite: Federated Multi-scale Momentum (without Newton-Schulz)

Core idea from Nested Learning: Multi-scale optimization
- Fast momentum (m1): Client-side, adapts quickly to local data
- Slow momentum (m2): Server-side, preserves long-term global direction

Key formulas:
    # Client (fast scale)
    m1 = beta1 * m1 + grad        # Fast momentum (EMA style, bounded)
    update = m1 + lam * m2_global # Combine local + global direction
    theta = theta - lr * update

    # Server (slow scale)
    m2 = beta3 * m2 + agg_buffer  # Slow momentum (EMA style, bounded)

Why no Newton-Schulz?
    - NS orthogonalizes updates to fixed magnitude
    - This breaks FedAvg aggregation (clients contribute equally regardless of gradient size)
    - Multi-scale momentum is the key insight, not orthogonalization
"""

import copy
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer


class FedM3LiteOptimizer(Optimizer):
    """
    Fed-M3 Lite: Multi-scale Momentum without Newton-Schulz.

    Update rule:
        m1 = beta1 * m1 + grad        # Fast momentum (EMA, bounded)
        update = m1 + lam * m2_global # Combine with slow momentum (from server)
        theta = theta - lr * update

    Args:
        params: Model parameters
        lr: Learning rate (default: 0.01)
        beta1: Fast momentum coefficient (default: 0.9)
        lam: Balance factor for local vs global direction (default: 0.3)
        global_momentum: Slow momentum (m2) from server
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        beta1: float = 0.9,
        lam: float = 0.3,
        global_momentum: Optional[Dict[str, torch.Tensor]] = None,
    ):
        defaults = dict(
            lr=lr,
            beta1=beta1,
            lam=lam,
        )
        super().__init__(params, defaults)

        # Store global momentum (m2) from server
        self.global_momentum = global_momentum or {}

        # Gradient buffer to send to server
        self.gradient_buffer: Dict[str, torch.Tensor] = {}

        # Debug mode
        self.debug = False
        self.debug_stats = []

    def set_global_momentum(self, global_momentum: Dict[str, torch.Tensor]) -> None:
        """Set global momentum (m2) received from server."""
        self.global_momentum = global_momentum

    def set_debug(self, debug: bool) -> None:
        """Enable or disable debug mode."""
        self.debug = debug
        if debug:
            self.debug_stats = []

    def get_gradient_buffer(self) -> Dict[str, torch.Tensor]:
        """Return accumulated gradients to send to server."""
        return copy.deepcopy(self.gradient_buffer)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            lam = group['lam']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                param_name = self._get_param_name(p)

                # Get or initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['m1'] = torch.zeros_like(p.data)  # Fast momentum
                    state['step'] = 0

                m1 = state['m1']
                state['step'] += 1

                # ========================================
                # MULTI-SCALE MOMENTUM (no Newton-Schulz)
                # ========================================

                # Fast momentum: m1 = beta1 * m1 + grad (EMA style, bounded)
                m1.mul_(beta1).add_(grad)

                # Accumulate gradient for server
                if param_name not in self.gradient_buffer:
                    self.gradient_buffer[param_name] = torch.zeros_like(grad)
                self.gradient_buffer[param_name].add_(grad)

                # Get global momentum m2 (slow, from server)
                if param_name in self.global_momentum:
                    m2 = self.global_momentum[param_name].to(p.device)
                else:
                    m2 = torch.zeros_like(m1)

                # Combine: update = m1 + lam * m2
                update = m1 + lam * m2

                # Apply update
                p.data.add_(update, alpha=-lr)

                # Debug stats
                if self.debug and state['step'] == 1:
                    self.debug_stats.append({
                        'param': param_name[:30],
                        'grad_norm': torch.norm(grad).item(),
                        'm1_norm': torch.norm(m1).item(),
                        'm2_norm': torch.norm(m2).item(),
                        'update_norm': torch.norm(update).item(),
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


def fed_m3_optimizer_fn(
    model: nn.Module,
    lr: float = 0.01,
    extra_state: Optional[Dict[str, Any]] = None,
    beta1: float = 0.9,
    lam: float = 0.3,
    debug: bool = False,
    # Legacy parameters (ignored, kept for compatibility)
    beta2: float = 0.999,
    ns_steps: int = 5,
    v_init: float = 1.0,
) -> Tuple[FedM3LiteOptimizer, Dict[str, Any]]:
    """
    Factory function to create Fed-M3 Lite optimizer for a client.

    Args:
        model: Neural network model
        lr: Learning rate
        extra_state: Contains 'global_momentum' (m2) from server
        beta1: Fast momentum coefficient
        lam: Balance local vs global momentum
        debug: Enable debug mode

    Returns:
        Tuple of (optimizer, extra_data_dict)
    """
    global_momentum = {}
    if extra_state and 'global_momentum' in extra_state:
        global_momentum = extra_state['global_momentum']

    optimizer = FedM3LiteOptimizer(
        model.parameters(),
        lr=lr,
        beta1=beta1,
        lam=lam,
        global_momentum=global_momentum,
    )

    # Set parameter names
    optimizer.set_param_names(list(model.named_parameters()))

    if debug:
        optimizer.set_debug(True)

    def get_extra():
        result = {'gradient_buffer': optimizer.get_gradient_buffer()}
        if debug:
            result['debug_stats'] = optimizer.get_debug_stats()
        return result

    return optimizer, {'get_extra_fn': get_extra}


def fed_m3_aggregate(
    global_model: nn.Module,
    client_results: List[Dict[str, Any]],
    server_state: Dict[str, Any],
    beta3: float = 0.9,
    # Legacy parameters (ignored)
    ns_steps: int = 5,
) -> Dict[str, Any]:
    """
    Fed-M3 Lite aggregation: FedAvg + slow momentum update.

    Args:
        global_model: Global model (updated in-place)
        client_results: Results from client training
        server_state: Server state (contains 'm2' slow momentum)
        beta3: Slow momentum coefficient

    Returns:
        Dict with 'train_loss', 'global_momentum'
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

    global_model.load_state_dict(aggregated_params)

    # ==========================================
    # 2. Aggregate gradient buffers from clients
    # ==========================================
    aggregated_buffer = {}

    for result in client_results:
        extra = result.get('extra', {})
        get_extra_fn = extra.get('get_extra_fn')
        if get_extra_fn:
            buffer = get_extra_fn()['gradient_buffer']
            weight = result['num_samples'] / total_samples
            for key, grad in buffer.items():
                if key not in aggregated_buffer:
                    aggregated_buffer[key] = torch.zeros_like(grad)
                aggregated_buffer[key] += weight * grad

    # ==========================================
    # 3. Update slow momentum m2 (NO Newton-Schulz)
    # ==========================================
    if 'm2' not in server_state:
        server_state['m2'] = {}

    m2 = server_state['m2']

    # m2 = beta3 * m2 + buffer (EMA style, bounded)
    for key, buffer in aggregated_buffer.items():
        if key not in m2:
            m2[key] = torch.zeros_like(buffer)
        m2[key].mul_(beta3).add_(buffer)

    # Normalize m2 to prevent unbounded growth
    # Use simple normalization: m2_normalized = m2 / ||m2|| * scale
    global_momentum = {}
    for key, momentum in m2.items():
        norm = torch.norm(momentum)
        if norm > 1e-6:
            # Scale to have reasonable magnitude (similar to gradient scale)
            global_momentum[key] = momentum / norm * 0.1
        else:
            global_momentum[key] = momentum.clone()

    server_state['global_momentum'] = global_momentum

    # ==========================================
    # 4. Compute average training loss
    # ==========================================
    avg_train_loss = sum(
        r['train_loss'] * r['num_samples'] for r in client_results
    ) / total_samples

    return {
        'train_loss': avg_train_loss,
        'total_samples': total_samples,
        'global_momentum': global_momentum,
    }
