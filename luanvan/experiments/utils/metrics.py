"""Metrics tracking for FL experiments."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class MetricsTracker:
    """Track and store experiment metrics."""

    def __init__(self):
        self.history: Dict[str, List] = {
            'round': [],
            'train_loss': [],
            'test_acc': [],
            'test_loss': [],
            'client_accs': [],
        }

    def log(
        self,
        round_num: int,
        train_loss: float,
        test_acc: float,
        test_loss: float,
        client_accs: Optional[List[float]] = None
    ) -> None:
        """Log metrics for a round."""
        self.history['round'].append(round_num)
        self.history['train_loss'].append(train_loss)
        self.history['test_acc'].append(test_acc)
        self.history['test_loss'].append(test_loss)
        self.history['client_accs'].append(client_accs or [])

    def get_best_accuracy(self) -> float:
        """Get best test accuracy achieved."""
        if not self.history['test_acc']:
            return 0.0
        return max(self.history['test_acc'])

    def get_final_accuracy(self) -> float:
        """Get final test accuracy."""
        if not self.history['test_acc']:
            return 0.0
        return self.history['test_acc'][-1]

    def get_convergence_round(self, target_acc: float) -> Optional[int]:
        """Get first round that reached target accuracy."""
        for i, acc in enumerate(self.history['test_acc']):
            if acc >= target_acc:
                return self.history['round'][i]
        return None

    def get_client_variance(self) -> List[float]:
        """Get variance of client accuracies per round."""
        variances = []
        for accs in self.history['client_accs']:
            if accs:
                variances.append(np.var(accs))
            else:
                variances.append(0.0)
        return variances

    def save(self, path: str) -> None:
        """Save metrics to JSON file."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load(self, path: str) -> None:
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            self.history = json.load(f)
