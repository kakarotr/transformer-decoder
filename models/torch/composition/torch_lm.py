import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import TransformerConfig
from models.utilities.mask import create_causal_mask, create_padding_mask


class Rope(nn.Module):
    def __init__(self, *, base: int, max_position_embeddings: int, head_dim: int):
        super().__init__()
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.head_dim = head_dim
        cos, sin = self._precompute_cos_sin_cache()

        self.register_buffer("_cos_cached", cos, persistent=False)
        self.register_buffer("_sin_cached", sin, persistent=False)

    def _precompute_cos_sin_cache(self):
        inv_freq = 1 / self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        positons = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positons, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor):
        x_1 = x[..., : x.shape[-1] // 2]
        x_2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x_2, x_1), dim=-1)

    def _get_cos_sin(self, position_ids: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        position_ids = position_ids.to(dtype=torch.long, device=device)

        cos: torch.Tensor = self._cos_cached[position_ids].unsqueeze(1)
        sin: torch.Tensor = self._sin_cached[position_ids].unsqueeze(1)

        return cos, sin


class DefaultRope(Rope):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"x must be 4D [batch_size, num_heads, seq_len, head_dim], got {tuple(x.shape)}")

        batch_size, _, seq_len, _ = x.shape

        input_dtype = x.dtype
        x = x.float()

        cos, sin = self._get_cos_sin(position_ids=position_ids, batch_size=batch_size, seq_len=seq_len, device=x.device)
        out = (x * cos) + self._rotate_half(x) * sin
        return out.to(dtype=input_dtype)

    def apply_rotate(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        batch_size, _, seq_len, _ = q.shape

        q_dtype = q.dtype
        k_dtype = k.dtype

        q_fp32 = q.float()
        k_fp32 = k.float()

        cos, sin = self._get_cos_sin(
            position_ids=position_ids,
            batch_size=batch_size,
            seq_len=seq_len,
            device=q.device,
        )

        q_out = (q_fp32 * cos) + (self._rotate_half(q_fp32) * sin)
        k_out = (k_fp32 * cos) + (self._rotate_half(k_fp32) * sin)

        return q_out.to(dtype=q_dtype), k_out.to(dtype=k_dtype)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        *,
        rope: Rope,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.dropout_prob = dropout_prob
        self.n_rep = self.num_attention_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_attention_heads, bias=False)
        self.kv_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_key_value_heads * 2, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()

        q: torch.Tensor = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        kv: torch.Tensor = self.kv_proj(hidden_states)
        kv = kv.view(batch_size, seq_len, self.num_key_value_heads, 2, self.head_dim)
        k, v = kv.unbind(dim=3)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if position_ids.shape[-1] != seq_len:
            raise ValueError(f"position_ids.shape[-1] must equal seq_len, got {position_ids.shape[-1]} vs {seq_len}")

        q, k = self.rope.apply_rotate(q, k, position_ids)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, 1)
            v = v.repeat_interleave(self.n_rep, 1)

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                raise ValueError("attn_mask must be bool tensor")
            attn_mask = attn_mask.to(device=q.device)

        if is_causal and attn_mask is not None:
            raise ValueError("When is_causal=True, attn_mask must be None")

        context_vectors = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=is_causal,
        )
        context_vectors = context_vectors.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return self.o_proj(context_vectors)


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        hidden_states = F.silu(gate) * up
        return self.down_proj(hidden_states)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        rms = (hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        hidden_states = hidden_states * rms
        if hidden_states.dtype != self.weight.dtype:
            hidden_states = hidden_states.to(self.weight.dtype)
        hidden_states = hidden_states * self.weight
        return hidden_states.to(input_dtype)


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
                attention_mask = (input_ids != self.pad_token_id).long()
            else:
                if attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
                    raise ValueError("attention_mask must have same shape as input_ids")

            if position_ids is None:
                position_ids = attention_mask.cumsum(dim=-1) - 1
                position_ids = position_ids.clamp_min(0)
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)

            causal_mask = create_causal_mask(seq_len, device=device)
            padding_mask = create_padding_mask(attention_mask=attention_mask, device=device)

            attn_mask = causal_mask & padding_mask
            is_causal = False
        else:
            if attention_mask is not None:
                if attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
                    raise ValueError("attention_mask must have same shape as input_ids")
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            attn_mask = None
            is_causal = True

        hidden_states = self.embedd_dropout(self.embeddings(input_ids))

        for idx, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[idx]
            hidden_states, _ = layer(hidden_states, position_ids, attn_mask, is_causal, layer_past, use_cache)

        return self.norm(hidden_states)


class CausalLanguageModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.decoder = Decoder(
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            num_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            rms_eps=config.rms_eps,
            dropout_prob=config.dropout_prob,
            rope_base=config.rope_base,
            pad_token_id=config.pad_token_id,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pad_token_id = config.pad_token_id
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        base_std = 0.02
        residual_std = base_std / math.sqrt(2 * self.num_layers)

        nn.init.normal_(self.decoder.embeddings.weight, mean=0.0, std=base_std)
        for layer in self.decoder.layers:
            nn.init.normal_(layer.attention.q_proj.weight, mean=0.0, std=base_std)
            nn.init.normal_(layer.attention.kv_proj.weight, mean=0.0, std=base_std)
            nn.init.normal_(layer.attention.o_proj.weight, mean=0.0, std=residual_std)

            nn.init.normal_(layer.mlp.gate_up_proj.weight, mean=0.0, std=base_std)
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=residual_std)

        nn.init.normal_(self.lm_head.weight, mean=0.0, std=base_std)
        self.decoder.embeddings.weight[self.pad_token_id].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_padding_mask: bool = False,
    ):
        hidden_states = self.decoder(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_padding_mask=use_padding_mask,
        )
        return hidden_states
