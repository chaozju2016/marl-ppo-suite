"""Various reward normalizers."""
import numpy as np

class RunningMeanStd:
    """Dynamically calculate mean and std using Welford's algorithm."""
    def __init__(self, shape=()):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)  # Sum of squared differences
        self.std = np.sqrt(self.S)

    def reset(self):
        """Reset the statistics."""
        self.n = 0
        self.mean = np.zeros_like(self.mean)
        self.S = np.zeros_like(self.S)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

class EfficientStandardNormalizer:
    """A highly optimized standard normalizer using Welford's algorithm."""
    def __init__(self, epsilon=1e-8):
        # Initialize statistics directly (no separate class)
        self.n = 0
        self.mean = 0.0
        self.S = 0.0  # Sum of squared differences
        self.std = 0.0
        self.epsilon = epsilon

    def reset(self):
        """Reset the normalizer."""
        self.n = 0
        self.mean = 0.0
        self.S = 0.0
        self.std = 0.0

    def normalize(self, x, update=True):
        # Direct type check for float (fastest path)
        if type(x) is float:
            # Update statistics using Welford's algorithm
            if update:
                self.n += 1
                delta = x - self.mean
                self.mean += delta / self.n
                delta2 = x - self.mean
                self.S += delta * delta2
                self.std = (self.S / self.n) ** 0.5

            # Normalize using current statistics
            return (x - self.mean) / (self.std + self.epsilon)

        # Fast conversion for numpy arrays
        if type(x) is np.ndarray and x.size == 1:
            return self.normalize(float(x.item()), update)

        # Handle lists
        if type(x) is list and len(x) == 1:
            return self.normalize(float(x[0]), update)

        # Return unchanged for other types
        return x

class EMANormalizer:
    """An ultra-fast reward normalizer using EMA for statistics tracking."""
    def __init__(self, decay=0.99999, epsilon=1e-5, min_var=1e-2):
        # Store parameters
        self.decay = decay  # EMA decay factor
        self.epsilon = epsilon
        self.min_var = min_var

        # Initialize statistics
        self.mean = 0.0
        self.var = 1.0

        # Cache frequently used values
        self._one_minus_decay = 1.0 - decay
        self._min_std = (min_var + epsilon) ** 0.5

    def reset(self):
        """Reset the normalizer."""
        self.mean = 0.0
        self.var = 1.0

    def normalize(self, x, update=True):
        # Direct type check for float (fastest path)
        if type(x) is float:
            if update:
                # Update mean with single-pass EMA
                delta = x - self.mean
                self.mean += self._one_minus_decay * delta

                # Update variance with optimized EMA formula
                self.var = self.decay * self.var + self._one_minus_decay * delta * delta

            # Fast path: if variance is above minimum, use it directly
            if self.var > self.min_var:
                # Inline sqrt calculation (faster than np.sqrt for single values)
                std = (self.var + self.epsilon) ** 0.5
                return (x - self.mean) / std
            else:
                # Use pre-computed minimum std
                return (x - self.mean) / self._min_std

        # Fast conversion for numpy arrays
        if type(x) is np.ndarray and x.size == 1:
            return self.normalize(float(x.item()), update)

        # Handle lists
        if type(x) is list and len(x) == 1:
            return self.normalize(float(x[0]), update)

        # Return unchanged for other types
        return x

# TODO: Try again above one compare learning
# class FastRewardNormalizer:
#     """A faster reward normalizer that uses EMA for statistics tracking."""
#     def __init__(self, decay=0.99999, epsilon=1e-5, min_var=1e-2):
#         self.decay = decay  # EMA decay factor (higher = more history weight)
#         self.epsilon = epsilon
#         self.min_var = min_var
#         self.running_mean = 0.0
#         self.running_var = 1.0
#         self.debiasing_term = 0.0  # Add debiasing term

#     def update(self, reward):
#         """Update running statistics with new reward."""
#         # Convert reward to float if needed
#         if isinstance(reward, (list, np.ndarray)):
#             reward = float(reward[0])

#         # Update debiasing term
#         self.debiasing_term = self.decay * self.debiasing_term + (1.0 - self.decay)

#         # Update mean
#         delta = reward - self.running_mean
#         self.running_mean = self.running_mean + (1 - self.decay) * delta

#         # Update variance
#         self.running_var = self.decay * self.running_var + (1 - self.decay) * (reward - self.running_mean)**2

#     def normalize(self, reward, update=True):
#         """Normalize reward using exponential moving average statistics."""
#         # Convert reward to float if it's not already
#         if isinstance(reward, (list, np.ndarray)):
#             reward = float(reward[0])

#         # Update statistics if needed
#         if update:
#             self.update(reward)

#         # Get debiased statistics
#         mean = self.running_mean / (self.debiasing_term + self.epsilon)
#         var = self.running_var / (self.debiasing_term + self.epsilon)
#         var = max(var, self.min_var)  # Apply minimum variance threshold

#         # Normalize reward
#         return (reward - mean) / np.sqrt(var + self.epsilon)

#     def denormalize(self, normalized_reward):
#         """Convert normalized reward back to original scale."""
#         # Get debiased statistics
#         mean = self.running_mean / (self.debiasing_term + self.epsilon)
#         var = self.running_var / (self.debiasing_term + self.epsilon)
#         var = max(var, self.min_var)  # Apply minimum variance threshold

#         # Denormalize
#         return normalized_reward * np.sqrt(var + self.epsilon) + mean


#     def reset(self):
#         """No need to reset."""
#         pass


