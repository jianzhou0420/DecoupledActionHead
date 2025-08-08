# FiLM-augmented TransformerDecoderLayer (PyTorch)
# - Adds conditional Feature-wise Linear Modulation (FiLM) to the inputs of
#   self-attention, cross-attention, and FFN sublayers.
# - Keeps the API close to nn.TransformerDecoderLayer, adding a new kwarg `cond`.
# - Works with batch_first True/False and norm_first True/False.

from typing import Optional, Union, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, MultiheadAttention, Linear, Dropout, LayerNorm


class FiLMGen(nn.Module):
    """Maps a conditioning vector to FiLM parameters (gamma, beta) per feature.

    Args:
        d_model: feature dimension of the modulated hidden states
        cond_dim: dimension of conditioning vector
    """

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, 2 * d_model),
        )
        # initialize last layer to zeros so the whole block starts near identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        # cond: (B, cond_dim) -> returns gamma, beta shaped (B, 1, D)
        gb = self.net(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma.unsqueeze(1), beta.unsqueeze(1)


def _get_activation_fn(activation: Union[str, Callable[[Tensor], Tensor]]):
    if isinstance(activation, str):
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "gelu_fast":
            return lambda x: F.gelu(x, approximate="tanh")
        raise RuntimeError(f"Unsupported activation: {activation}")
    return activation


class TransformerDecoderLayerFiLM(Module):
    """TransformerDecoderLayer with FiLM conditioning.

    Adds FiLM to the normalized inputs of: self-attn, cross-attn, and FFN.

    New arg:
        cond_dim: dimension of the conditioning vector passed at forward(cond=...)

    Forward signature difference from stock layer:
        - Add keyword-only argument `cond: Tensor` with shape (B, cond_dim)
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        cond_dim: int = 128,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs
        )
        self.multihead_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs
        )
        # FFN
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.batch_first = batch_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # FiLM generators: one per sub-layer (can be shared if desired)
        self.film1 = FiLMGen(d_model, cond_dim)  # self-attn input
        self.film2 = FiLMGen(d_model, cond_dim)  # cross-attn input
        self.film3 = FiLMGen(d_model, cond_dim)  # FFN input

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    # helper: apply FiLM on (B,T,D) or (T,B,D)
    def _apply_film(self, x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        # gamma,beta are (B,1,D). Ensure x is (B,T,D) for broadcast.
        if self.batch_first:
            y = x
        else:
            y = x.transpose(0, 1)  # (T,B,D) -> (B,T,D)
        # Use (1+gamma) to start from identity
        y = (1.0 + gamma) * y + beta
        return y if self.batch_first else y.transpose(1, 0)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        *,
        cond: Tensor,  # (B, cond_dim)
    ) -> Tensor:
        """Pass the inputs through the FiLM-augmented decoder layer."""

        x = tgt

        if self.norm_first:
            # Self-attention path
            y = self.norm1(x)
            g, b = self.film1(cond)
            y = self._apply_film(y, g, b)
            x = x + self._sa_block(y, tgt_mask, tgt_key_padding_mask, tgt_is_causal)

            # Cross-attention path
            y = self.norm2(x)
            g, b = self.film2(cond)
            y = self._apply_film(y, g, b)
            x = x + self._mha_block(y, memory, memory_mask, memory_key_padding_mask, memory_is_causal)

            # FFN path
            y = self.norm3(x)
            g, b = self.film3(cond)
            y = self._apply_film(y, g, b)
            x = x + self._ff_block(y)
        else:
            # Post-norm: FiLM the sublayer input before calling the sublayer
            g, b = self.film1(cond)
            y = self._apply_film(x, g, b)
            x = self.norm1(x + self._sa_block(y, tgt_mask, tgt_key_padding_mask, tgt_is_causal))

            g, b = self.film2(cond)
            y = self._apply_film(x, g, b)
            x = self.norm2(x + self._mha_block(y, memory, memory_mask, memory_key_padding_mask, memory_is_causal))

            g, b = self.film3(cond)
            y = self._apply_film(x, g, b)
            x = self.norm3(x + self._ff_block(y))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # cross-attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.multihead_attn(
            x, mem, mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed-forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


# --- Optional: a tiny wrapper decoder that forwards `cond` into each layer ---
class TransformerDecoderFiLM(nn.TransformerDecoder):
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        cond: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        output = tgt
        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
                cond=cond,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


# -----------------------------
# Usage Example (commented)
# -----------------------------
d_model, nhead, cond_dim = 512, 8, 128
layer = TransformerDecoderLayerFiLM(
    d_model=d_model, nhead=nhead, dim_feedforward=2048, dropout=0.1,
    batch_first=True, norm_first=True, cond_dim=cond_dim,
)
decoder = TransformerDecoderFiLM(layer, num_layers=6)

B, T_tgt, T_src = 4, 64, 64
tgt = torch.randn(B, T_tgt, d_model)
memory = torch.randn(B, T_src, d_model)
cond = torch.randn(B, cond_dim)
out = decoder(tgt, memory, cond=cond)  # (B, T_tgt, d_model)

print(cond.shape)
print(out.shape)  # Should be (B, T_tgt, d_model)
