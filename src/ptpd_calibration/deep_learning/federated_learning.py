"""
Federated Community Learning for Platinum-Palladium printing.

Enables privacy-preserving collaborative model training across multiple
practitioners without sharing raw data. Supports differential privacy,
gradient compression, and multiple aggregation strategies.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np

from ptpd_calibration.deep_learning.config import FederatedLearningSettings
from ptpd_calibration.deep_learning.models import (
    FederatedRoundResult,
    FederatedUpdate,
)
from ptpd_calibration.deep_learning.types import (
    AggregationStrategy,
    PrivacyLevel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Differential Privacy
# =============================================================================


class DifferentialPrivacy:
    """
    Differential privacy mechanisms for federated learning.

    Implements Gaussian mechanism for privacy-preserving gradient updates.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
    ):
        """
        Initialize differential privacy mechanism.

        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Privacy parameter
            clip_norm: Gradient clipping norm
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.privacy_accountant = {"epsilon_used": 0.0, "rounds": 0}

    def clip_gradients(self, gradients: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Clip gradients to bound sensitivity.

        Args:
            gradients: Dictionary of gradient arrays

        Returns:
            dict: Clipped gradients
        """
        # Compute global gradient norm
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += np.sum(grad**2)
        total_norm = np.sqrt(total_norm)

        # Clip if necessary
        if total_norm > self.clip_norm:
            clip_factor = self.clip_norm / (total_norm + 1e-10)
            clipped = {name: grad * clip_factor for name, grad in gradients.items()}
        else:
            clipped = gradients

        return clipped

    def add_noise(
        self, gradients: dict[str, np.ndarray], sensitivity: float | None = None
    ) -> dict[str, np.ndarray]:
        """
        Add Gaussian noise for differential privacy.

        Args:
            gradients: Gradients to add noise to
            sensitivity: Sensitivity of the mechanism (uses clip_norm if None)

        Returns:
            dict: Noisy gradients
        """
        sens = sensitivity or self.clip_norm

        # Gaussian noise calibrated to privacy budget
        noise_scale = self._compute_noise_scale(sens)

        noisy_gradients = {}
        for name, grad in gradients.items():
            noise = np.random.normal(loc=0.0, scale=noise_scale, size=grad.shape)
            noisy_gradients[name] = grad + noise

        # Update privacy accountant
        self.privacy_accountant["epsilon_used"] += self._compute_privacy_cost()
        self.privacy_accountant["rounds"] += 1

        return noisy_gradients

    def privatize_update(self, gradients: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Apply full DP mechanism: clip then add noise.

        Args:
            gradients: Raw gradients

        Returns:
            dict: Privatized gradients
        """
        clipped = self.clip_gradients(gradients)
        noisy = self.add_noise(clipped)
        return noisy

    def _compute_noise_scale(self, sensitivity: float) -> float:
        """Compute the scale of Gaussian noise."""
        # Gaussian mechanism: σ = (sensitivity * sqrt(2 * ln(1.25/δ))) / ε
        return (sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))) / self.epsilon

    def _compute_privacy_cost(self) -> float:
        """Compute privacy cost of one update."""
        # Simplified privacy accounting
        return self.epsilon / 100  # Assuming composition over 100 rounds

    def get_privacy_spent(self) -> dict[str, float]:
        """Get total privacy budget spent."""
        return {
            "epsilon_used": self.privacy_accountant["epsilon_used"],
            "epsilon_remaining": max(0, self.epsilon - self.privacy_accountant["epsilon_used"]),
            "rounds": self.privacy_accountant["rounds"],
        }

    def reset_accountant(self):
        """Reset privacy accountant."""
        self.privacy_accountant = {"epsilon_used": 0.0, "rounds": 0}


# =============================================================================
# Gradient Compression
# =============================================================================


class GradientCompressor:
    """
    Gradient compression for communication efficiency.

    Implements top-k sparsification and quantization.
    """

    def __init__(
        self,
        compression_ratio: float = 0.1,
        quantization_bits: int = 8,
    ):
        """
        Initialize gradient compressor.

        Args:
            compression_ratio: Fraction of gradients to keep (0.0-1.0)
            quantization_bits: Number of bits for quantization
        """
        self.compression_ratio = compression_ratio
        self.quantization_bits = quantization_bits

    def compress(self, gradients: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Compress gradients using top-k sparsification.

        Args:
            gradients: Gradients to compress

        Returns:
            tuple: (compressed_gradients, metadata_for_decompression)
        """
        compressed = {}
        metadata = {}

        for name, grad in gradients.items():
            # Flatten gradient
            flat_grad = grad.flatten()
            original_shape = grad.shape

            # Top-k sparsification
            k = max(1, int(len(flat_grad) * self.compression_ratio))
            top_k_indices = np.argpartition(np.abs(flat_grad), -k)[-k:]
            top_k_values = flat_grad[top_k_indices]

            # Quantize values
            quantized_values = self._quantize(top_k_values)

            compressed[name] = {
                "indices": top_k_indices.tolist(),
                "values": quantized_values.tolist(),
            }
            metadata[name] = {
                "shape": original_shape,
                "size": len(flat_grad),
                "k": k,
            }

        return compressed, metadata

    def decompress(
        self, compressed: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        """
        Decompress gradients.

        Args:
            compressed: Compressed gradient data
            metadata: Decompression metadata

        Returns:
            dict: Decompressed gradients
        """
        decompressed = {}

        for name, comp_data in compressed.items():
            meta = metadata[name]

            # Reconstruct sparse gradient
            flat_grad = np.zeros(meta["size"])
            indices = np.array(comp_data["indices"])
            values = self._dequantize(np.array(comp_data["values"]))

            flat_grad[indices] = values

            # Reshape to original shape
            decompressed[name] = flat_grad.reshape(meta["shape"])

        return decompressed

    def _quantize(self, values: np.ndarray) -> np.ndarray:
        """Quantize values to reduce precision."""
        if self.quantization_bits >= 32:
            return values

        # Min-max quantization
        v_min, v_max = values.min(), values.max()
        if v_max - v_min < 1e-10:
            return values

        # Scale to [0, 2^bits - 1]
        max_val = 2**self.quantization_bits - 1
        quantized = ((values - v_min) / (v_max - v_min) * max_val).astype(int)

        # Store min/max for dequantization (simplified - in practice would store separately)
        return quantized.astype(np.float32)

    def _dequantize(self, quantized: np.ndarray) -> np.ndarray:
        """Dequantize values."""
        # Simplified dequantization
        return quantized


# =============================================================================
# Federated Client
# =============================================================================


class FederatedClient:
    """
    Federated learning client.

    Performs local training and sends updates to the server.
    """

    def __init__(
        self,
        client_id: str,
        settings: FederatedLearningSettings,
        local_data: Any | None = None,
    ):
        """
        Initialize federated client.

        Args:
            client_id: Unique client identifier
            settings: Federated learning settings
            local_data: Local training data
        """
        self.client_id = client_id
        self.settings = settings
        self.local_data = local_data
        self.global_model_params: dict[str, np.ndarray] | None = None
        self.local_model_params: dict[str, np.ndarray] | None = None

        # Initialize privacy and compression
        if settings.privacy_level == PrivacyLevel.DIFFERENTIAL:
            self.dp_mechanism = DifferentialPrivacy(
                epsilon=settings.differential_privacy_epsilon,
                delta=settings.differential_privacy_delta,
                clip_norm=settings.clip_norm,
            )
        else:
            self.dp_mechanism = None

        if settings.gradient_compression:
            self.compressor = GradientCompressor(compression_ratio=settings.compression_ratio)
        else:
            self.compressor = None

    async def local_train(
        self,
        epochs: int | None = None,
        batch_size: int | None = None,
    ) -> FederatedUpdate:
        """
        Perform local training.

        Args:
            epochs: Number of local epochs (uses settings if None)
            batch_size: Batch size (uses settings if None)

        Returns:
            FederatedUpdate: Training update
        """
        start_time = datetime.utcnow()

        epochs = epochs or self.settings.local_epochs
        num_samples = self._get_num_local_samples()

        # Simulate local training (placeholder)
        # In production, this would:
        # 1. Load global model parameters
        # 2. Train on local data for specified epochs
        # 3. Compute gradients or model updates

        await asyncio.sleep(0.1)  # Simulate training time

        # Generate mock gradients
        gradients = self._compute_mock_gradients()

        # Apply differential privacy if enabled
        if self.dp_mechanism:
            gradients = self.dp_mechanism.privatize_update(gradients)
            logger.info(
                f"Client {self.client_id}: Applied DP, "
                f"privacy spent: {self.dp_mechanism.get_privacy_spent()}"
            )

        # Compress gradients if enabled
        if self.compressor:
            compressed, metadata = self.compressor.compress(gradients)
            # Store for sending
            self._compressed_update = compressed
            self._compression_metadata = metadata

        # Compute loss (mock)
        local_loss = np.random.uniform(0.1, 0.5)

        training_time = (datetime.utcnow() - start_time).total_seconds()

        return FederatedUpdate(
            client_id=self.client_id,
            round_number=0,  # Will be set by server
            num_samples=num_samples,
            local_loss=local_loss,
            local_accuracy=None,
            training_time_seconds=training_time,
        )

    async def send_update(self, server_address: str, update: FederatedUpdate) -> bool:
        """
        Send update to the federated server.

        Args:
            server_address: Server address
            update: Federated update to send

        Returns:
            bool: Whether send was successful
        """
        # Placeholder for network communication
        # In production, would use gRPC, HTTP, or WebSocket

        try:
            logger.info(f"Client {self.client_id}: Sending update to {server_address}")
            await asyncio.sleep(0.05)  # Simulate network delay
            return True
        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to send update: {e}")
            return False

    async def receive_global_model(self, server_address: str) -> dict[str, np.ndarray] | None:
        """
        Receive global model from server.

        Args:
            server_address: Server address

        Returns:
            Optional[dict]: Global model parameters
        """
        try:
            logger.info(f"Client {self.client_id}: Receiving global model from {server_address}")
            await asyncio.sleep(0.05)  # Simulate network delay

            # Placeholder - would receive actual model
            self.global_model_params = self._get_mock_global_model()
            return self.global_model_params

        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to receive global model: {e}")
            return None

    def _get_num_local_samples(self) -> int:
        """Get number of local training samples."""
        if self.local_data is not None:
            # In production, would get actual data size
            return len(self.local_data) if hasattr(self.local_data, "__len__") else 100

        # Mock value
        return np.random.randint(
            self.settings.min_local_samples,
            self.settings.min_local_samples * 10,
        )

    def _compute_mock_gradients(self) -> dict[str, np.ndarray]:
        """Generate mock gradients for demonstration."""
        # In production, these would be actual model gradients
        return {
            "layer1.weight": np.random.randn(128, 64).astype(np.float32),
            "layer1.bias": np.random.randn(128).astype(np.float32),
            "layer2.weight": np.random.randn(64, 32).astype(np.float32),
            "layer2.bias": np.random.randn(64).astype(np.float32),
        }

    def _get_mock_global_model(self) -> dict[str, np.ndarray]:
        """Get mock global model parameters."""
        return {
            "layer1.weight": np.random.randn(128, 64).astype(np.float32) * 0.01,
            "layer1.bias": np.zeros(128, dtype=np.float32),
            "layer2.weight": np.random.randn(64, 32).astype(np.float32) * 0.01,
            "layer2.bias": np.zeros(64, dtype=np.float32),
        }


# =============================================================================
# Federated Server
# =============================================================================


class FederatedServer:
    """
    Federated learning server.

    Aggregates client updates and broadcasts global model.
    """

    def __init__(self, settings: FederatedLearningSettings):
        """
        Initialize federated server.

        Args:
            settings: Federated learning settings
        """
        self.settings = settings
        self.global_model_params: dict[str, np.ndarray] | None = None
        self.round_number = 0
        self.client_updates: dict[str, tuple[FederatedUpdate, dict[str, np.ndarray]]] = {}

    async def aggregate(
        self,
        client_updates: list[tuple[FederatedUpdate, dict[str, np.ndarray]]],
        strategy: AggregationStrategy | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Aggregate client updates into global model.

        Args:
            client_updates: List of (update, gradients) tuples
            strategy: Aggregation strategy (uses settings if None)

        Returns:
            dict: Updated global model parameters
        """
        strategy = strategy or self.settings.aggregation_strategy

        if strategy == AggregationStrategy.FEDAVG:
            return await self._federated_averaging(client_updates)
        elif strategy == AggregationStrategy.FEDPROX:
            return await self._federated_prox(client_updates)
        elif strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            return await self._weighted_averaging(client_updates)
        else:
            logger.warning(f"Unknown strategy {strategy}, using FedAvg")
            return await self._federated_averaging(client_updates)

    async def _federated_averaging(
        self,
        client_updates: list[tuple[FederatedUpdate, dict[str, np.ndarray]]],
    ) -> dict[str, np.ndarray]:
        """
        Standard Federated Averaging (FedAvg).

        Weighted average based on number of local samples.
        """
        if not client_updates:
            return self.global_model_params or {}

        # Compute total samples
        total_samples = sum(update.num_samples for update, _ in client_updates)

        # Initialize aggregated model
        aggregated = {}

        # Get parameter names from first client
        param_names = list(client_updates[0][1].keys())

        for param_name in param_names:
            weighted_sum = None

            for update, gradients in client_updates:
                weight = update.num_samples / total_samples
                param_update = gradients[param_name] * weight

                if weighted_sum is None:
                    weighted_sum = param_update
                else:
                    weighted_sum += param_update

            aggregated[param_name] = weighted_sum

        # Update global model
        if self.global_model_params is None:
            self.global_model_params = aggregated
        else:
            # Apply updates to existing model
            for param_name, update in aggregated.items():
                self.global_model_params[param_name] = (
                    self.global_model_params.get(param_name, 0) - update
                )

        return self.global_model_params

    async def _federated_prox(
        self,
        client_updates: list[tuple[FederatedUpdate, dict[str, np.ndarray]]],
        mu: float = 0.01,
    ) -> dict[str, np.ndarray]:
        """
        Federated Proximal (FedProx).

        Adds proximal term to handle heterogeneous data.
        """
        # Similar to FedAvg but with proximal regularization
        # For simplicity, using FedAvg here; in production would add proximal term
        result = await self._federated_averaging(client_updates)

        # Apply proximal term (simplified)
        if self.global_model_params is not None:
            for param_name in result:
                result[param_name] = (1 - mu) * result[
                    param_name
                ] + mu * self.global_model_params.get(param_name, 0)

        return result

    async def _weighted_averaging(
        self,
        client_updates: list[tuple[FederatedUpdate, dict[str, np.ndarray]]],
    ) -> dict[str, np.ndarray]:
        """
        Weighted averaging with loss-based weights.

        Gives more weight to clients with lower loss.
        """
        if not client_updates:
            return self.global_model_params or {}

        # Compute inverse loss weights
        losses = np.array([update.local_loss for update, _ in client_updates])
        # Inverse loss (lower loss = higher weight)
        inv_losses = 1.0 / (losses + 1e-10)
        weights = inv_losses / np.sum(inv_losses)

        # Initialize aggregated model
        aggregated = {}
        param_names = list(client_updates[0][1].keys())

        for param_name in param_names:
            weighted_sum = None

            for (_update, gradients), weight in zip(client_updates, weights, strict=True):
                param_update = gradients[param_name] * weight

                if weighted_sum is None:
                    weighted_sum = param_update
                else:
                    weighted_sum += param_update

            aggregated[param_name] = weighted_sum

        # Update global model
        if self.global_model_params is None:
            self.global_model_params = aggregated
        else:
            for param_name, update in aggregated.items():
                self.global_model_params[param_name] = (
                    self.global_model_params.get(param_name, 0) - update
                )

        return self.global_model_params

    async def broadcast(
        self, model_params: dict[str, np.ndarray], client_ids: list[str]
    ) -> dict[str, bool]:
        """
        Broadcast global model to clients.

        Args:
            model_params: Global model parameters
            client_ids: List of client IDs to broadcast to

        Returns:
            dict: Success status for each client
        """
        results = {}

        for client_id in client_ids:
            try:
                # Simulate broadcast
                await asyncio.sleep(0.01)
                results[client_id] = True
                logger.info(f"Broadcasted model to client {client_id}")
            except Exception as e:
                logger.error(f"Failed to broadcast to {client_id}: {e}")
                results[client_id] = False

        return results

    async def coordinate_round(
        self,
        participating_clients: list[str],
    ) -> FederatedRoundResult:
        """
        Coordinate a complete federated learning round.

        Args:
            participating_clients: List of participating client IDs

        Returns:
            FederatedRoundResult: Round result
        """
        start_time = datetime.utcnow()
        self.round_number += 1

        logger.info(f"Starting round {self.round_number} with {len(participating_clients)} clients")

        # Collect updates (simulated)
        client_updates_list = []
        update_objects = []

        for client_id in participating_clients:
            # Simulate receiving update
            await asyncio.sleep(0.01)

            # Mock update
            update = FederatedUpdate(
                client_id=client_id,
                round_number=self.round_number,
                num_samples=np.random.randint(50, 500),
                local_loss=np.random.uniform(0.1, 0.5),
                local_accuracy=np.random.uniform(0.7, 0.95),
                training_time_seconds=np.random.uniform(1.0, 10.0),
            )

            gradients = {
                "layer1.weight": np.random.randn(128, 64).astype(np.float32) * 0.01,
                "layer1.bias": np.random.randn(128).astype(np.float32) * 0.01,
                "layer2.weight": np.random.randn(64, 32).astype(np.float32) * 0.01,
                "layer2.bias": np.random.randn(64).astype(np.float32) * 0.01,
            }

            client_updates_list.append((update, gradients))
            update_objects.append(update)

        # Aggregate updates
        aggregated_model = await self.aggregate(client_updates_list)

        # Broadcast to clients
        await self.broadcast(aggregated_model, participating_clients)

        # Compute metrics
        global_loss = np.mean([u.local_loss for u in update_objects])
        global_accuracy = np.mean(
            [u.local_accuracy for u in update_objects if u.local_accuracy is not None]
        )

        inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return FederatedRoundResult(
            round_number=self.round_number,
            num_participants=len(participating_clients),
            aggregation_strategy=self.settings.aggregation_strategy.value,
            global_loss=global_loss,
            global_accuracy=global_accuracy if global_accuracy else None,
            participant_updates=update_objects,
            privacy_level=self.settings.privacy_level.value,
            noise_added=self.settings.privacy_level == PrivacyLevel.DIFFERENTIAL,
            loss_improvement=0.0,  # Would track from previous round
            bytes_communicated=0,  # Would track actual bytes
            compression_ratio=self.settings.compression_ratio
            if self.settings.gradient_compression
            else None,
            inference_time_ms=inference_time,
            device_used="cpu",
            model_version="federated-1.0.0",
        )


# =============================================================================
# Federated Learning Manager
# =============================================================================


class FederatedLearningManager:
    """
    Main manager for federated learning system.

    Orchestrates clients and server for complete federated training.
    """

    def __init__(self, settings: FederatedLearningSettings | None = None):
        """
        Initialize federated learning manager.

        Args:
            settings: Federated learning settings
        """
        self.settings = settings or FederatedLearningSettings()

        if not self.settings.enabled:
            logger.info("Federated learning is disabled in settings")

        self.server: FederatedServer | None = None
        self.clients: dict[str, FederatedClient] = {}
        self.round_results: list[FederatedRoundResult] = []

    def initialize_server(self) -> FederatedServer:
        """
        Initialize the federated server.

        Returns:
            FederatedServer: Initialized server
        """
        self.server = FederatedServer(self.settings)
        logger.info("Federated server initialized")
        return self.server

    def register_client(
        self, client_id: str | None = None, local_data: Any | None = None
    ) -> FederatedClient:
        """
        Register a new federated client.

        Args:
            client_id: Client identifier (generates if None)
            local_data: Local training data

        Returns:
            FederatedClient: Registered client
        """
        if client_id is None:
            client_id = str(uuid4())

        client = FederatedClient(
            client_id=client_id,
            settings=self.settings,
            local_data=local_data,
        )

        self.clients[client_id] = client
        logger.info(f"Registered client {client_id}")

        return client

    async def run_federated_training(
        self,
        num_rounds: int | None = None,
        min_clients: int | None = None,
        max_clients: int | None = None,
    ) -> list[FederatedRoundResult]:
        """
        Run complete federated training.

        Args:
            num_rounds: Number of communication rounds
            min_clients: Minimum clients per round
            max_clients: Maximum clients per round

        Returns:
            list: Results from all rounds
        """
        if not self.settings.enabled:
            logger.warning("Federated learning is disabled")
            return []

        num_rounds = num_rounds or self.settings.communication_rounds
        min_clients = min_clients or self.settings.min_clients_per_round
        max_clients = max_clients or self.settings.max_clients_per_round

        # Initialize server if needed
        if self.server is None:
            self.initialize_server()

        logger.info(
            f"Starting federated training: {num_rounds} rounds, "
            f"{min_clients}-{max_clients} clients per round"
        )

        self.round_results = []

        for round_num in range(num_rounds):
            # Select participating clients
            num_participating = min(max_clients, max(min_clients, len(self.clients)))

            if len(self.clients) < min_clients:
                logger.warning(
                    f"Not enough clients ({len(self.clients)} < {min_clients}), "
                    f"skipping round {round_num + 1}"
                )
                continue

            participating = np.random.choice(
                list(self.clients.keys()),
                size=num_participating,
                replace=False,
            ).tolist()

            # Coordinate round
            round_result = await self.server.coordinate_round(participating)
            self.round_results.append(round_result)

            logger.info(
                f"Round {round_num + 1}/{num_rounds} completed: "
                f"loss={round_result.global_loss:.4f}, "
                f"accuracy={round_result.global_accuracy:.4f if round_result.global_accuracy else 'N/A'}"
            )

            # Optional: Early stopping based on convergence
            if self._check_convergence():
                logger.info(f"Converged after {round_num + 1} rounds")
                break

        logger.info(f"Federated training completed: {len(self.round_results)} rounds")
        return self.round_results

    def _check_convergence(self, window: int = 5, threshold: float = 0.001) -> bool:
        """
        Check if training has converged.

        Args:
            window: Number of recent rounds to check
            threshold: Convergence threshold

        Returns:
            bool: Whether training has converged
        """
        if len(self.round_results) < window:
            return False

        recent_losses = [r.global_loss for r in self.round_results[-window:]]
        loss_variance = np.var(recent_losses)

        return loss_variance < threshold

    def get_training_summary(self) -> dict[str, Any]:
        """
        Get summary of federated training.

        Returns:
            dict: Training summary
        """
        if not self.round_results:
            return {"status": "no_training_completed"}

        final_result = self.round_results[-1]

        return {
            "total_rounds": len(self.round_results),
            "num_clients": len(self.clients),
            "final_loss": final_result.global_loss,
            "final_accuracy": final_result.global_accuracy,
            "aggregation_strategy": self.settings.aggregation_strategy.value,
            "privacy_level": self.settings.privacy_level.value,
            "gradient_compression": self.settings.gradient_compression,
            "average_participants_per_round": np.mean(
                [r.num_participants for r in self.round_results]
            ),
        }

    def save_global_model(self, path: str) -> bool:
        """
        Save the global model to disk.

        Args:
            path: Path to save model

        Returns:
            bool: Whether save was successful
        """
        if self.server is None or self.server.global_model_params is None:
            logger.error("No global model to save")
            return False

        try:
            # In production, would use proper model serialization
            # e.g., torch.save() or tf.saved_model.save()
            np.savez(path, **self.server.global_model_params)
            logger.info(f"Saved global model to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_global_model(self, path: str) -> bool:
        """
        Load a global model from disk.

        Args:
            path: Path to load model from

        Returns:
            bool: Whether load was successful
        """
        try:
            loaded = np.load(path)
            model_params = {key: loaded[key] for key in loaded.files}

            if self.server is None:
                self.initialize_server()

            self.server.global_model_params = model_params
            logger.info(f"Loaded global model from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
