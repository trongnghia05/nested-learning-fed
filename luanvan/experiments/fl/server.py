"""FL Server implementation."""

import copy
from typing import Dict, List, Any, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class FLServer:
    """
    Federated Learning Server.

    Manages global model and coordinates training.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        test_dataset: Optional[Dataset] = None,
        test_batch_size: int = 128,
    ):
        """
        Initialize FL Server.

        Args:
            model: Global neural network model
            device: Device for evaluation
            test_dataset: Test dataset for global evaluation
            test_batch_size: Batch size for evaluation
        """
        self.device = device
        self.global_model = model.to(device)
        self.test_dataset = test_dataset

        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=False
            )
        else:
            self.test_loader = None

        # Server state (for Fed-M3 slow momentum, etc.)
        self.server_state: Dict[str, Any] = {}

    def get_global_params(self) -> Dict[str, torch.Tensor]:
        """Return global model parameters."""
        return copy.deepcopy(self.global_model.state_dict())

    def set_global_params(self, params: Dict[str, torch.Tensor]) -> None:
        """Set global model parameters."""
        self.global_model.load_state_dict(params)

    def aggregate(
        self,
        client_results: List[Dict[str, Any]],
        aggregator_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate client updates.

        Args:
            client_results: List of results from client.local_train()
            aggregator_fn: Custom aggregation function
                          If None, uses FedAvg (weighted average)

        Returns:
            Dict with aggregation info
        """
        if aggregator_fn is not None:
            return aggregator_fn(
                self.global_model,
                client_results,
                self.server_state
            )

        # Default: FedAvg (weighted average by num_samples)
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
        self.set_global_params(aggregated_params)

        # Compute average training loss
        avg_train_loss = sum(
            r['train_loss'] * r['num_samples'] for r in client_results
        ) / total_samples

        return {
            'train_loss': avg_train_loss,
            'total_samples': total_samples,
        }

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate global model on test dataset.

        Returns:
            Dict with 'accuracy' and 'loss'
        """
        if self.test_loader is None:
            return {'accuracy': 0.0, 'loss': 0.0}

        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.global_model(batch_x)
                loss = criterion(outputs, batch_y)

                total_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'loss': total_loss / total if total > 0 else 0.0,
        }

    def save_checkpoint(self, path: str, round_num: int, metrics: Dict) -> None:
        """Save server checkpoint."""
        torch.save({
            'round': round_num,
            'model_state_dict': self.global_model.state_dict(),
            'server_state': self.server_state,
            'metrics': metrics,
        }, path)

    def load_checkpoint(self, path: str) -> Dict:
        """Load server checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.server_state = checkpoint.get('server_state', {})
        return {
            'round': checkpoint['round'],
            'metrics': checkpoint.get('metrics', {}),
        }
