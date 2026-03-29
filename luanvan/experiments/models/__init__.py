"""Neural network models for FL experiments."""

from .cnn import CNNSmall, CNNMedium, create_model

__all__ = [
    'CNNSmall',
    'CNNMedium',
    'create_model',
]
