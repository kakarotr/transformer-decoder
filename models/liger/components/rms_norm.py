import torch.nn as nn
from liger_kernel.transformers import LigerRMSNorm


class RMSNorm(LigerRMSNorm):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__(
            hidden_size=hidden_size,
            eps=eps,
            offset=0.0,
            casting_mode="llama",
            init_fn="ones",
            in_place=True,
        )
