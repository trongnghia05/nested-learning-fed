"""Newton-Schulz orthogonalization for momentum tensors."""

import torch
from typing import Union


def newton_schulz_orthogonalize(
    M: torch.Tensor,
    num_steps: int = 5,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Orthogonalize tensor using Newton-Schulz iteration.

    For 2D tensors (matrices): X_{k+1} = 0.5 * X_k @ (3I - X_k^T @ X_k)
    For 1D tensors (vectors): Simply normalize to unit vector.

    Args:
        M: Input tensor to orthogonalize (1D or 2D)
        num_steps: Number of Newton-Schulz iterations (for 2D)
        eps: Small constant for numerical stability

    Returns:
        Orthogonalized tensor with same shape as input
    """
    if M.dim() == 1:
        # 1D tensor: just normalize
        norm = torch.norm(M) + eps
        return M / norm

    elif M.dim() == 2:
        # 2D tensor: Newton-Schulz iteration
        # Normalize first
        norm = torch.norm(M) + eps
        X = M / norm

        # Get identity matrix
        rows, cols = X.shape
        if rows <= cols:
            I = torch.eye(rows, device=M.device, dtype=M.dtype)
            for _ in range(num_steps):
                X = 0.5 * X @ (3 * torch.eye(cols, device=M.device, dtype=M.dtype)
                              - X.T @ X)
        else:
            I = torch.eye(cols, device=M.device, dtype=M.dtype)
            for _ in range(num_steps):
                X = 0.5 * (3 * torch.eye(rows, device=M.device, dtype=M.dtype)
                          - X @ X.T) @ X

        return X

    elif M.dim() == 4:
        # 4D tensor (conv weights): reshape to 2D, orthogonalize, reshape back
        # Shape: (out_channels, in_channels, H, W)
        out_c, in_c, h, w = M.shape
        M_2d = M.view(out_c, -1)  # (out_channels, in_channels * H * W)
        M_orth = newton_schulz_orthogonalize(M_2d, num_steps, eps)
        return M_orth.view(out_c, in_c, h, w)

    else:
        # Other dimensions: just normalize
        norm = torch.norm(M) + eps
        return M / norm


def orthogonalize_state_dict(
    state_dict: dict,
    num_steps: int = 5,
    eps: float = 1e-6
) -> dict:
    """
    Orthogonalize all tensors in a state dict (e.g., momentum buffers).

    Args:
        state_dict: Dictionary of tensors
        num_steps: Number of Newton-Schulz iterations
        eps: Numerical stability constant

    Returns:
        Dictionary with orthogonalized tensors
    """
    orthogonalized = {}
    for key, tensor in state_dict.items():
        if tensor.numel() > 1:  # Skip scalars
            orthogonalized[key] = newton_schulz_orthogonalize(
                tensor, num_steps, eps
            )
        else:
            orthogonalized[key] = tensor
    return orthogonalized


if __name__ == "__main__":
    # Test Newton-Schulz
    print("Testing Newton-Schulz orthogonalization...")

    # Test 1D
    v = torch.randn(10)
    v_orth = newton_schulz_orthogonalize(v)
    print(f"1D: ||v|| = {torch.norm(v):.4f}, ||v_orth|| = {torch.norm(v_orth):.4f}")

    # Test 2D
    M = torch.randn(5, 8)
    M_orth = newton_schulz_orthogonalize(M)
    print(f"2D: shape {M.shape} -> {M_orth.shape}")
    # Check orthogonality: M @ M^T should be close to identity
    product = M_orth @ M_orth.T
    print(f"    M_orth @ M_orth^T - I (should be ~0):")
    print(f"    max error = {(product - torch.eye(5)).abs().max():.6f}")

    # Test 4D (conv weights)
    C = torch.randn(32, 16, 3, 3)
    C_orth = newton_schulz_orthogonalize(C)
    print(f"4D: shape {C.shape} -> {C_orth.shape}")
