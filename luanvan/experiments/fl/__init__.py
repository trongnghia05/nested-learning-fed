"""Federated Learning framework components."""

from .data_split import dirichlet_split, quantity_skew_split, iid_split
from .client import FLClient
from .server import FLServer
from .aggregators import fedavg_aggregate, weighted_aggregate

__all__ = [
    'dirichlet_split',
    'quantity_skew_split',
    'iid_split',
    'FLClient',
    'FLServer',
    'fedavg_aggregate',
    'weighted_aggregate',
]
