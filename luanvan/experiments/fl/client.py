"""FL Client implementation."""

import copy
from typing import Dict, Any, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


class FLClient:
    """
    Federated Learning Client.

    Handles local training on client's private data.
    """

    def __init__(
        self,
        client_id: int,
        dataset: Subset,
        model: nn.Module,
        device: torch.device,
        batch_size: int = 32,
    ):
        """
        Initialize FL Client.

        Args:
            client_id: Unique client identifier
            dataset: Client's local dataset (Subset)
            model: Neural network model (will be copied)
            device: Device to train on
            batch_size: Local batch size
        """
        self.client_id = client_id
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

        # Create local model (copy of global)
        self.model = copy.deepcopy(model).to(device)

        # Create data loader
        self.train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )

    def get_num_samples(self) -> int:
        """Return number of local samples."""
        return len(self.dataset)

    def set_model_params(self, global_params: Dict[str, torch.Tensor]) -> None:
        """
        Load global model parameters.

        Args:
            global_params: State dict from global model
        """
        self.model.load_state_dict(global_params)

    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """Return current model parameters."""
        return copy.deepcopy(self.model.state_dict())

    def local_train(
        self,
        global_params: Dict[str, torch.Tensor],
        local_epochs: int,
        lr: float,
        optimizer_fn: Optional[Callable] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform local training.

        Args:
            global_params: Parameters from global model
            local_epochs: Number of local training epochs
            lr: Learning rate
            optimizer_fn: Custom optimizer factory (for Fed-M3, Fed-DGD)
                         If None, uses SGD
            extra_state: Extra state for custom optimizers (e.g., global momentum)

        Returns:
            Dict containing:
                - 'params': Updated local model parameters
                - 'num_samples': Number of training samples
                - 'train_loss': Average training loss
                - 'extra': Any extra data to send to server
        """
        # Load global parameters
        self.model.load_state_dict(global_params)
        self.model.train()

        # Create optimizer
        if optimizer_fn is not None:
            optimizer, extra_data = optimizer_fn(
                self.model,
                lr=lr,
                extra_state=extra_state
            )
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
            extra_data = {}

        criterion = nn.CrossEntropyLoss()

        # Training loop
        total_loss = 0.0
        total_samples = 0

        for epoch in range(local_epochs):
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        return {
            'params': self.get_model_params(),
            'num_samples': self.get_num_samples(),
            'train_loss': avg_loss,
            'extra': extra_data,
        }

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on local data.

        Returns:
            Dict with 'accuracy' and 'loss'
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                total_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'loss': total_loss / total if total > 0 else 0.0,
        }
