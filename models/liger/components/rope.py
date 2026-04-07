import torch
import torch.nn as nn
from liger_kernel.transformers import liger_rotary_pos_emb


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

    def apply_rotate(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        batch_size, _, seq_len, _ = q.shape

        cos, sin = self._get_cos_sin(
            position_ids=position_ids,
            batch_size=batch_size,
            seq_len=seq_len,
            device=q.device,
        )

        q_out, k_out = liger_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1)
        return q_out, k_out
