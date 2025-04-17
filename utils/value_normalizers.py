import torch
import numpy as np

class WelfordValueNormalizer:
    """
    Normalizes value function targets using Welford's algorithm for running statistics.
    Optimized for MAPPO use case with consistent tensor handling.
    """
    def __init__(self, device=torch.device("cpu"), epsilon=1e-8, min_var=1e-2):
        """
        Initialize the value normalizer.
        Optimized for MAPPO implementation by flattening tensors.

        Args:
            device: Device to store running statistics
            epsilon: Small constant for numerical stability
            min_var: Minimum variance threshold
        """
        self.device = device
        self.epsilon = epsilon
        self.min_var = min_var

        # Running statistics
        self.running_mean = torch.zeros(1, device=device)
        self.running_var = torch.ones(1, device=device)
        self.count = torch.zeros(1, device=device)

    def _get_stats(self):
        """Get current mean and variance statistics."""
        mean = self.running_mean
        var = self.running_var.clamp(min=self.min_var)  # Use minimum variance threshold
        return mean, var

    def update(self, values):
        """
        Update running statistics with new values.
        Optimized for MAPPO by flattening tensors.

        Args:
            values: Tensor of values, typically with shape [batch_size, n_agents, 1]
        """
        # Convert numpy arrays to tensors if needed
        if type(values) is np.ndarray:
            values = torch.from_numpy(values).to(self.device).float()

        # Flatten tensor to (batch_size) for consistent statistics
        # Since our running statistics are scalars, we just need the mean across all elements
        flat_values = values.reshape(-1)

        # Compute batch statistics
        batch_mean = flat_values.mean()
        batch_var = flat_values.var(unbiased=False)
        batch_count = flat_values.shape[0]

        # Update running statistics using Welford's online algorithm
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count

        self.running_mean = self.running_mean + delta * batch_count / total_count

        # Update running variance
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.running_var = M2 / total_count

        self.count = total_count

    def normalize(self, values, update=True):
        """
        Normalize values using running statistics.
        Optimized for MAPPO by preserving input shape.

        Args:
            values: Tensor of values, typically with shape [batch_size, n_agents, 1]
            update: Whether to update running statistics

        Returns:
            Normalized values with same shape as input
        """
        # Convert numpy arrays to tensors if needed
        if type(values) is np.ndarray:
            values = torch.from_numpy(values).to(self.device).float()

        # Update statistics if needed
        if update:
            self.update(values)

        # Get current statistics
        mean, var = self._get_stats()

        # Normalize without changing shape
        # Since mean and var are scalars, broadcasting applies normalization to all elements
        return (values - mean) / torch.sqrt(var + self.epsilon)

    def denormalize(self, normalized_values):
        """
        Convert normalized values back to original scale.
        Optimized for MAPPO by preserving input shape.

        Args:
            normalized_values: Normalized values

        Returns:
            Values in original scale with same shape as input
        """
        # Convert numpy arrays to tensors if needed
        was_numpy = type(normalized_values) is np.ndarray
        if was_numpy:
            normalized_values = torch.from_numpy(normalized_values).to(self.device).float()

        # Get current statistics
        mean, var = self._get_stats()

        # Denormalize without changing shape
        # Since mean and var are scalars, broadcasting applies denormalization to all elements
        denormalized_values = normalized_values * torch.sqrt(var + self.epsilon) + mean

        # Convert back to numpy if input was numpy
        if was_numpy:
            denormalized_values = denormalized_values.cpu().numpy()

        return denormalized_values


class EMAValueNormalizer:
    """
    Normalizes value function targets using exponential moving average.
    Based on the official MAPPO implementation, optimized for MAPPO use case.
    """
    def __init__(self, device=torch.device("cpu"), beta=0.99999, epsilon=1e-5, min_var=1e-2):
        """
        Initialize the value normalizer.
        Optimized for MAPPO implementation by flattening tensors.

        Args:
            device: Device to store running statistics
            beta: EMA coefficient (close to 1 for slow updates)
            epsilon: Small constant for numerical stability
            min_var: Minimum variance threshold
        """
        self.device = device
        self.beta = beta
        self.epsilon = epsilon
        self.min_var = min_var

        # Running statistics
        self.running_mean = torch.zeros(1, device=device)
        self.running_mean_sq = torch.zeros(1, device=device)
        self.debiasing_term = torch.zeros(1, device=device)

    def _get_stats(self):
        """Get current mean and variance statistics with debiasing."""
        mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        var = (self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)) - (mean ** 2)
        var = var.clamp(min=self.min_var)  # Use minimum variance threshold
        return mean, var

    def update(self, values):
        """
        Update running statistics with new values.
        Optimized for MAPPO by flattening tensors.

        Args:
            values: Tensor of values, typically with shape [batch_size, n_agents, 1]
        """
        # Convert numpy arrays to tensors if needed
        if type(values) is np.ndarray:
            values = torch.from_numpy(values).to(self.device).float()

        # Flatten tensor to (batch_size)
        # Since our running statistics are scalars, we just need the mean across all elements
        flat_values = values.reshape(-1)

        # Compute mean and mean squared
        batch_mean = flat_values.mean()
        batch_mean_sq = (flat_values ** 2).mean()

        # Update running stats with EMA
        self.running_mean = self.beta * self.running_mean + (1.0 - self.beta) * batch_mean
        self.running_mean_sq = self.beta * self.running_mean_sq + (1.0 - self.beta) * batch_mean_sq
        self.debiasing_term = self.beta * self.debiasing_term + (1.0 - self.beta)

    def normalize(self, values, update=True):
        """
        Normalize values using running statistics.
        Optimized for MAPPO by preserving input shape.

        Args:
            values: Tensor of values, typically with shape [batch_size, n_agents, 1]
            update: Whether to update running statistics

        Returns:
            Normalized values with same shape as input
        """
        # Convert numpy arrays to tensors if needed
        if type(values) is np.ndarray:
            values = torch.from_numpy(values).to(self.device).float()

        # Update statistics if needed
        if update:
            self.update(values)

        # Get current statistics
        mean, var = self._get_stats()

        # Normalize without changing shape
        # Since mean and var are scalars, broadcasting applies normalization to all elements
        return (values - mean) / torch.sqrt(var)

    def denormalize(self, normalized_values):
        """
        Convert normalized values back to original scale.
        Optimized for MAPPO by preserving input shape.

        Args:
            normalized_values: Normalized values

        Returns:
            Values in original scale with same shape as input
        """
        # Convert numpy arrays to tensors if needed
        was_numpy = type(normalized_values) is np.ndarray
        if was_numpy:
            normalized_values = torch.from_numpy(normalized_values).to(self.device).float()

        # Get current statistics
        mean, var = self._get_stats()

        # Denormalize without changing shape
        # Since mean and var are scalars, broadcasting applies denormalization to all elements
        denormalized_values = normalized_values * torch.sqrt(var) + mean

        # Convert back to numpy if input was numpy
        if was_numpy:
            denormalized_values = denormalized_values.cpu().numpy()

        return denormalized_values


# Factory function to create the appropriate normalizer
def create_value_normalizer(normalizer_type="ema", **kwargs):
    """
    Create a value normalizer of the specified type.

    Args:
        normalizer_type: Type of normalizer ('welford' or 'ema')
        **kwargs: Additional arguments to pass to the normalizer

    Returns:
        A value normalizer instance
    """
    if normalizer_type.lower() == "welford":
        return WelfordValueNormalizer(**kwargs)
    elif normalizer_type.lower() == "ema":
        return EMAValueNormalizer(**kwargs)
    else:
        raise ValueError(f"Unknown normalizer type: {normalizer_type}. Choose 'welford' or 'ema'.")
