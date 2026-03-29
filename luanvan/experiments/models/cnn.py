"""CNN models for image classification in FL experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class CNNSmall(nn.Module):
    """
    Small CNN for Fashion-MNIST (28x28 grayscale images).

    Architecture:
    - Conv2d(1, 32, 3) -> ReLU -> MaxPool
    - Conv2d(32, 64, 3) -> ReLU -> MaxPool
    - FC(64*7*7, 128) -> ReLU
    - FC(128, num_classes)

    Total params: ~100K
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 7, 7)
        x = x.view(-1, 64 * 7 * 7)            # (batch, 3136)
        x = F.relu(self.fc1(x))               # (batch, 128)
        x = self.fc2(x)                       # (batch, num_classes)
        return x


class CNNMedium(nn.Module):
    """
    Medium CNN for CIFAR-10 (32x32 RGB images).

    Architecture:
    - Conv2d(3, 64, 3) -> ReLU -> Conv2d(64, 64, 3) -> ReLU -> MaxPool
    - Conv2d(64, 128, 3) -> ReLU -> Conv2d(128, 128, 3) -> ReLU -> MaxPool
    - FC(128*8*8, 256) -> ReLU -> Dropout
    - FC(256, num_classes)

    Total params: ~1.2M
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch, 3, 32, 32)
        x = self.features(x)          # (batch, 128, 8, 8)
        x = x.view(x.size(0), -1)     # (batch, 8192)
        x = self.classifier(x)        # (batch, num_classes)
        return x


def create_model(
    model_type: Literal['cnn_small', 'cnn_medium'] = 'cnn_medium',
    num_classes: int = 10
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: Type of model ('cnn_small' for FMNIST, 'cnn_medium' for CIFAR-10)
        num_classes: Number of output classes

    Returns:
        Neural network model
    """
    if model_type == 'cnn_small':
        return CNNSmall(num_classes=num_classes)
    elif model_type == 'cnn_medium':
        return CNNMedium(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("Testing CNNSmall (for FMNIST)...")
    model_small = CNNSmall()
    x_fmnist = torch.randn(2, 1, 28, 28)
    out = model_small(x_fmnist)
    print(f"  Input: {x_fmnist.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Parameters: {count_parameters(model_small):,}")

    print("\nTesting CNNMedium (for CIFAR-10)...")
    model_medium = CNNMedium()
    x_cifar = torch.randn(2, 3, 32, 32)
    out = model_medium(x_cifar)
    print(f"  Input: {x_cifar.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Parameters: {count_parameters(model_medium):,}")
