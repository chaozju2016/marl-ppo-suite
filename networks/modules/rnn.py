import torch
import torch.nn as nn


class GRUModule(nn.Module):
    """Reusable GRU module for sequence processing with masking."""

    def __init__(self, input_dim, hidden_dim, num_layers = 1):
        """
        Initialize the reusable GRU module.

        Args:
            input_dim (int): Dimension of the input features
            hidden_dim (int): Dimension of the hidden states
            num_layers (int, optional): Number of layers in the GRU. Defaults to 1.
        """
        super(GRUModule, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initialize GRU
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=False# (seq_len, batch, input_size)
            )

        # Layer norm after GRU
        self.gru_layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize the weights with orthogonal initialization with higher gain
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=1.4)  # Higher gain for better gradient flow
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

        # Initialize layer norms
        nn.init.constant_(self.gru_layer_norm.weight, 1.0)
        nn.init.constant_(self.gru_layer_norm.bias, 0.0)

    def forward(self, x, rnn_states, masks):
        """
        Forward pass through the GRU module with masking.
        B is batch and its size is n_agents * n_rollout_threads 

        Args:
            x: (T, B, input_dim) or (B, input_dim) for single step
            rnn_states: (B, num_layers, hidden_dim)
            masks: (T, B, 1) or (B, 1) for single step
        """
        # Single step case: (B, input_dim) -> (1, B, input_dim)
        is_single_step = x.dim() == 2
        if is_single_step:
            if __debug__:
                # Single step case: (B, input_dim)
                assert x.size(1) == self.input_dim, f"Expected input dimension {self.input_dim}, got {x.size(1)}"
            x = x.view(1, -1, self.input_dim)

        if __debug__:
            if not is_single_step:
                # Multi-step case: (T, B, input_dim)
                assert x.dim() == 3, f"Expected 3D input (T, B, input_dim), got {x.dim()}D"
                assert x.size(2) == self.input_dim, f"Expected input dimension {self.input_dim}, got {x.size(2)}"
            # Assert RNN states shape
            assert rnn_states.size(1) == self.num_layers, f"Expected {self.num_layers} RNN layers, got {rnn_states.size(1)}"
            assert rnn_states.size(2) == self.hidden_dim, f"Expected hidden dimension {self.hidden_dim}, got {rnn_states.size(2)}"
            # Assert batch sizes match
            batch_size = x.size(1)
            assert rnn_states.size(0) == batch_size, f"Batch size mismatch: x has {batch_size}, rnn_states has {rnn_states.size(0)}"

        rnn_states = rnn_states.transpose(0, 1) # (B, num_layers, hidden_dim) -> (num_layers, B, hidden_dim)

        # Handle single timestep (evaluation/rollout) case
        if is_single_step:
            # Apply mask to RNN states
            temp_states = (rnn_states * masks.view(1, -1, 1)).contiguous() # (num_layers, batch_size, hidden_size)
            x, rnn_states = self.gru(x, temp_states)
            x = x.squeeze(0) # (B, hidden_size) 
            rnn_states = rnn_states.transpose(0, 1) # (num_layers, B, hidden_dim) -> (B, num_layers, hidden_dim)
            return self.gru_layer_norm(x), rnn_states

        ##################################
        # Handle Multi-step case (training)
        ##################################
        
        seq_len, batch_size = x.shape[:2]
        # Find sequences of zeros in masks for efficient processing
        masks = masks.view(seq_len, batch_size).contiguous() # (T, B, 1) -> (T, B)

        # Using trick from iKostrikov to process sequences in chunks.
        #
        # The trick works by:
        # 1. Finding timesteps where masks contain zeros (episode boundaries)
        # 2. Processing all timesteps between zeros as a single chunk
        # 3. Resetting RNN states at episode boundaries (where masks=0)
        # This avoids unnecessary sequential processing of each timestep
        has_zeros = ((masks[1:] == 0.0)
             .any(dim=-1) # (T-1)
             .nonzero() # (num_true, 1)
             .squeeze()) # (num_true)
        has_zeros = [has_zeros.item() + 1] if has_zeros.dim() == 0 else [(idx + 1) for idx in has_zeros.tolist()]
        has_zeros = [0] + has_zeros + [seq_len]

        # Process sequences between zero masks
        outputs = []
        for i in range(len(has_zeros) - 1):
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]

            # Apply mask to current RNN states (num_layers, batch, hidden_dim)
            temp_states = (rnn_states * masks[start_idx].view(1, -1, 1)).contiguous()

            # Process current sequence
            out, rnn_states = self.gru(x[start_idx:end_idx], temp_states)
            outputs.append(out)

        # Combine outputs and apply layer norm
        x = torch.cat(outputs, dim=0) # (T, B, hidden_dim)
        x = self.gru_layer_norm(x)

        # Transpose back to (B, num_layers, hidden_dim)
        rnn_states = rnn_states.transpose(0, 1)

        return x, rnn_states
