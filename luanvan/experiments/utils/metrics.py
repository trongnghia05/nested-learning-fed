"""Metrics tracking for FL experiments."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


class MetricsTracker:
    """
    Track and store experiment metrics with structured format.

    Structure:
    {
        "rounds": [
            {
                "round": 1,
                "server": {
                    "test_acc": 0.50,
                    "test_loss": 1.25
                },
                "clients": [
                    {
                        "client_id": 0,
                        "num_samples": 1200,
                        "train_loss": 0.82,
                        "train_acc": 0.58,
                        "test_acc": 0.52,
                        "test_loss": 0.95,
                        "epoch_metrics": [
                            {"epoch": 1, "loss": 1.2, "acc": 0.35},
                            ...
                        ]
                    },
                    ...
                ],
                "client_aggregated": {
                    "train_acc": {"mean": 0.55, "median": 0.53},
                    "train_loss": {"mean": 0.85, "median": 0.82},
                    "test_acc": {"mean": 0.50, "median": 0.48},
                    "test_loss": {"mean": 1.25, "median": 1.20}
                }
            },
            ...
        ]
    }
    """

    def __init__(self):
        self.rounds: List[Dict[str, Any]] = []

    def _compute_client_aggregated(self, clients: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Compute client_aggregated statistics (mean and median) from client data."""
        if not clients:
            return {
                'train_acc': {'mean': 0.0, 'median': 0.0},
                'train_loss': {'mean': 0.0, 'median': 0.0},
                'test_acc': {'mean': 0.0, 'median': 0.0},
                'test_loss': {'mean': 0.0, 'median': 0.0},
            }

        # Extract values
        train_accs = [c['train_acc'] for c in clients]
        train_losses = [c['train_loss'] for c in clients]
        test_accs = [c['test_acc'] for c in clients]
        test_losses = [c['test_loss'] for c in clients]

        # Simple mean (no weights)
        def mean(values):
            if not values:
                return 0.0
            return float(np.mean(values))

        # Median (no weights)
        def median(values):
            if not values:
                return 0.0
            return float(np.median(values))

        return {
            'train_acc': {
                'mean': mean(train_accs),
                'median': median(train_accs),
            },
            'train_loss': {
                'mean': mean(train_losses),
                'median': median(train_losses),
            },
            'test_acc': {
                'mean': mean(test_accs),
                'median': median(test_accs),
            },
            'test_loss': {
                'mean': mean(test_losses),
                'median': median(test_losses),
            },
        }

    def log(
        self,
        round_num: int,
        server_test_acc: float,
        server_test_loss: float,
        clients: List[Dict[str, Any]],
    ) -> None:
        """
        Log metrics for a round.

        Args:
            round_num: Current round number
            server_test_acc: Global model accuracy on test set
            server_test_loss: Global model loss on test set
            clients: List of client metrics, each containing:
                {
                    'client_id': int,
                    'num_samples': int,
                    'train_loss': float,
                    'train_acc': float,
                    'test_acc': float,      # Global model on client's local data
                    'test_loss': float,     # Global model on client's local data
                    'epoch_metrics': [      # Per-epoch details
                        {'epoch': 1, 'loss': float, 'acc': float},
                        ...
                    ]
                }
        """
        # Compute client_aggregated stats
        client_aggregated = self._compute_client_aggregated(clients)

        # Build round data
        round_data = {
            'round': round_num,
            'server': {
                'test_acc': server_test_acc,
                'test_loss': server_test_loss,
            },
            'clients': clients,
            'client_aggregated': client_aggregated,
        }

        self.rounds.append(round_data)

    def get_best_accuracy(self) -> float:
        """Get best server test accuracy achieved."""
        if not self.rounds:
            return 0.0
        return max(r['server']['test_acc'] for r in self.rounds)

    def get_final_accuracy(self) -> float:
        """Get final server test accuracy."""
        if not self.rounds:
            return 0.0
        return self.rounds[-1]['server']['test_acc']

    def get_convergence_round(self, target_acc: float) -> Optional[int]:
        """Get first round that reached target accuracy."""
        for r in self.rounds:
            if r['server']['test_acc'] >= target_acc:
                return r['round']
        return None

    def get_client_variance(self) -> List[float]:
        """Get variance of client test accuracies per round."""
        variances = []
        for r in self.rounds:
            accs = [c['test_acc'] for c in r['clients']]
            if accs:
                variances.append(float(np.var(accs)))
            else:
                variances.append(0.0)
        return variances

    def get_server_metrics(self) -> Dict[str, List[float]]:
        """Get server metrics across all rounds."""
        return {
            'round': [r['round'] for r in self.rounds],
            'test_acc': [r['server']['test_acc'] for r in self.rounds],
            'test_loss': [r['server']['test_loss'] for r in self.rounds],
        }

    def get_client_aggregated_metrics(self) -> Dict[str, Dict[str, List[float]]]:
        """Get client_aggregated metrics across all rounds."""
        return {
            'train_acc': {
                'mean': [r['client_aggregated']['train_acc']['mean'] for r in self.rounds],
                'median': [r['client_aggregated']['train_acc']['median'] for r in self.rounds],
            },
            'train_loss': {
                'mean': [r['client_aggregated']['train_loss']['mean'] for r in self.rounds],
                'median': [r['client_aggregated']['train_loss']['median'] for r in self.rounds],
            },
            'test_acc': {
                'mean': [r['client_aggregated']['test_acc']['mean'] for r in self.rounds],
                'median': [r['client_aggregated']['test_acc']['median'] for r in self.rounds],
            },
            'test_loss': {
                'mean': [r['client_aggregated']['test_loss']['mean'] for r in self.rounds],
                'median': [r['client_aggregated']['test_loss']['median'] for r in self.rounds],
            },
        }

    def save(self, path: str) -> None:
        """Save metrics to JSON file."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({'rounds': self.rounds}, f, indent=2)

    def load(self, path: str) -> None:
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
            self.rounds = data.get('rounds', [])

    # Legacy compatibility properties
    @property
    def history(self) -> Dict[str, List]:
        """Legacy property for backward compatibility."""
        return {
            'round': [r['round'] for r in self.rounds],
            'train_loss': [r['client_aggregated']['train_loss']['mean'] for r in self.rounds],
            'train_acc': [r['client_aggregated']['train_acc']['mean'] for r in self.rounds],
            'test_acc': [r['server']['test_acc'] for r in self.rounds],
            'test_loss': [r['server']['test_loss'] for r in self.rounds],
            'client_accs': [[c['test_acc'] for c in r['clients']] for r in self.rounds],
        }
