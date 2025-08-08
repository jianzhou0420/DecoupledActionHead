import torch
import torch.nn as nn
import torch.nn.functional as F

# A simple FiLM modulation layer to apply the affine transformation


class FiLMLayer(nn.Module):
    """
    Applies Feature-wise Linear Modulation (FiLM) to an input tensor.
    The modulation is an affine transformation: output = x * gamma + beta.

    Args:
        feature_dim (int): The dimension of the features to be modulated.
    """

    def __init__(self, feature_dim: int):
        super(FiLMLayer, self).__init__()
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor to be modulated.
            gamma (torch.Tensor): The scaling parameter, expected to be broadcastable to x.
            beta (torch.Tensor): The shifting parameter, expected to be broadcastable to x.

        Returns:
            torch.Tensor: The modulated tensor.
        """
        # We apply the modulation directly. In the FiLM paper, they often use
        # gamma as a scaling factor, but the term is sometimes simplified.
        # Here we use x * gamma + beta, which is the standard affine transform.
        return x * gamma + beta


class FiLMTransformerDecoderLayer(nn.Module):
    """
    A custom Transformer Decoder Layer that accepts FiLM parameters to modulate
    the output of its sub-layers (self-attention, cross-attention, FFN).

    This class is inspired by the PyTorch nn.TransformerDecoderLayer.

    Args:
        d_model (int): The dimension of the model's features.
        nhead (int): The number of attention heads.
        dim_feedforward (int): The dimension of the inner layer of the FFN.
        dropout (float): The dropout rate.
        activation (str): The activation function for the FFN ("relu" or "gelu").
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu"):
        super(FiLMTransformerDecoderLayer, self).__init__()

        # Standard transformer components
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

        # FiLM layers for each modulation point. We create these to handle the
        # affine transformation logic, but the actual gamma and beta parameters
        # are passed to the forward method.
        self.film_self_attn = FiLMLayer(d_model)
        self.film_cross_attn = FiLMLayer(d_model)
        self.film_ffn = FiLMLayer(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None, memory_key_padding_mask: torch.Tensor = None,
                gamma_self_attn: torch.Tensor = None, beta_self_attn: torch.Tensor = None,
                gamma_cross_attn: torch.Tensor = None, beta_cross_attn: torch.Tensor = None,
                gamma_ffn: torch.Tensor = None, beta_ffn: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            tgt (torch.Tensor): The sequence from the decoder's input (required).
            memory (torch.Tensor): The sequence from the encoder's output (required).

            ... other standard transformer arguments ...

            gamma_self_attn (torch.Tensor): FiLM scaling for self-attention.
            beta_self_attn (torch.Tensor): FiLM shifting for self-attention.
            gamma_cross_attn (torch.Tensor): FiLM scaling for cross-attention.
            beta_cross_attn (torch.Tensor): FiLM shifting for cross-attention.
            gamma_ffn (torch.Tensor): FiLM scaling for the feed-forward network.
            beta_ffn (torch.Tensor): FiLM shifting for the feed-forward network.

        Returns:
            torch.Tensor: The output of the decoder layer.
        """
        # 1. Self-Attention Block
        # The first part is standard self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]

        # Apply FiLM modulation if parameters are provided
        if gamma_self_attn is not None and beta_self_attn is not None:
            tgt2 = self.film_self_attn(tgt2, gamma_self_attn, beta_self_attn)

        # Add residual connection and layer normalization
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2. Cross-Attention Block
        # Standard cross-attention
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]

        # Apply FiLM modulation if parameters are provided
        if gamma_cross_attn is not None and beta_cross_attn is not None:
            tgt2 = self.film_cross_attn(tgt2, gamma_cross_attn, beta_cross_attn)

        # Add residual connection and layer normalization
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 3. Feed-Forward Network Block
        # Standard FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        # Apply FiLM modulation if parameters are provided
        if gamma_ffn is not None and beta_ffn is not None:
            tgt2 = self.film_ffn(tgt2, gamma_ffn, beta_ffn)

        # Add residual connection and layer normalization
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

# --- Example Usage ---


# Define dimensions
d_model = 512
nhead = 8
dim_feedforward = 2048
batch_size = 2
seq_len_tgt = 10
seq_len_memory = 20

# Create dummy input data
tgt_input = torch.rand(batch_size, seq_len_tgt, d_model)  # Target sequence
memory_input = torch.rand(batch_size, seq_len_memory, d_model)  # Encoder output

# Create a sample FiLM parameter generator (this is a simple example;
# a real model would have a more complex generator)


class FiLMParamGenerator(nn.Module):
    def __init__(self, context_dim, d_model):
        super(FiLMParamGenerator, self).__init__()
        self.gamma_layer = nn.Linear(context_dim, d_model)
        self.beta_layer = nn.Linear(context_dim, d_model)

    def forward(self, context_tensor):
        # We need to broadcast the context to match the sequence length
        # In a real scenario, the context might be a per-token embedding
        gamma = self.gamma_layer(context_tensor).unsqueeze(1).expand(-1, 10, -1)
        beta = self.beta_layer(context_tensor).unsqueeze(1).expand(-1, 10, -1)
        return gamma, beta


# Create the custom decoder layer
film_decoder_layer = FiLMTransformerDecoderLayer(d_model, nhead, dim_feedforward)

# Create a sample context for FiLM parameters
context_tensor = torch.rand(batch_size, d_model)
film_generator = FiLMParamGenerator(d_model, d_model)

# Generate FiLM parameters for each sub-layer
gamma_sa, beta_sa = film_generator(context_tensor)
gamma_ca, beta_ca = film_generator(context_tensor)
gamma_ffn, beta_ffn = film_generator(context_tensor)

# Pass the parameters to the decoder layer's forward method
output_sequence = film_decoder_layer(
    tgt=tgt_input,
    memory=memory_input,
    gamma_self_attn=gamma_sa,
    beta_self_attn=beta_sa,
    gamma_cross_attn=gamma_ca,
    beta_cross_attn=beta_ca,
    gamma_ffn=gamma_ffn,
    beta_ffn=beta_ffn
)

print(f"Input target shape: {tgt_input.shape}")
print(f"Output sequence shape: {output_sequence.shape}")
