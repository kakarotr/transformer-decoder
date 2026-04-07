import torch
import torch.nn as nn


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
