import torch
import torch.nn as nn

from .components.attention import MultiHeadAttention
from .components.mlp import SwiGLUMLP
from .components.rms_norm import RMSNorm
from .components.rope import DefaultRope, Rope
from models.utilities.mask import create_causal_mask, create_padding_mask


class DecoderLayer(nn.Module):
    def __init__(
        self,
        rope: Rope,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_eps: float,
        dropout_prob: float,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            rope=rope,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            dropout_prob=dropout_prob,
        )
        self.mlp = SwiGLUMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.attn_norm = RMSNorm(hidden_size=hidden_size, eps=rms_eps)
        self.mlp_norm = RMSNorm(hidden_size=hidden_size, eps=rms_eps)
        self.attn_dropout = nn.Dropout(dropout_prob)
        self.mlp_dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = True,
        past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        if use_cache or past_key_values is not None:
            raise NotImplementedError("KV cache path is reserved but not implemented yet.")
        residual = hidden_states
        hidden_states = self.attention(self.attn_norm(hidden_states), position_ids, attn_mask, is_causal)
        hidden_states = residual + self.attn_dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = residual + self.mlp_dropout(hidden_states)

        return hidden_states, None


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        num_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_eps: float,
        dropout_prob: float,
        rope_base: int,
        pad_token_id: int,
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.rope = DefaultRope(
            base=rope_base, max_position_embeddings=max_position_embeddings, head_dim=hidden_size // num_attention_heads
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    rope=self.rope,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    rms_eps=rms_eps,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_layers)
            ]
        )
        self.embedd_dropout = nn.Dropout(dropout_prob)
        self.norm = RMSNorm(hidden_size=hidden_size, eps=rms_eps)
        self.pad_token_id = pad_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        use_padding_mask: bool = False,
    ) -> torch.Tensor:
        _, seq_len = input_ids.size()
        device = input_ids.device

        if use_cache or past_key_values is not None:
            raise NotImplementedError("Decoder KV cache path is not implemented yet.")

        if seq_len > self.rope.max_position_embeddings:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_position_embeddings ({self.rope.max_position_embeddings})"
            )

        if use_padding_mask:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            else:
                if attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
                    raise ValueError("attention_mask must have same shape as input_ids")

            if position_ids is None:
                position_ids = attention_mask.cumsum(dim=-1) - 1
                position_ids = position_ids.clamp_min(0)
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            else:
                if position_ids.ndim not in (1, 2):
                    raise ValueError("position_ids must be 1D or 2D")
                if position_ids.shape[-1] != seq_len:
                    raise ValueError("position_ids.shape[-1] must equal seq_len")
        else:
            position_ids = torch.arange(self.max_position_embeddings, device=device).unsqueeze(0)

        hidden_states = self.embedd_dropout(self.embeddings(input_ids))

        if use_padding_mask:
            assert attention_mask is not None
            causal_mask = create_causal_mask(seq_len, device=device)
            padding_mask = create_padding_mask(attention_mask=attention_mask, device=device)
            attn_mask = causal_mask & padding_mask
            is_causal = False
        else:
            attn_mask = None
            is_causal = True

        for idx, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[idx]
            hidden_states, _ = layer(hidden_states, position_ids, attn_mask, is_causal, layer_past, use_cache)

        return self.norm(hidden_states)
