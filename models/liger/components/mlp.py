from types import SimpleNamespace

import torch
from liger_kernel.transformers import LigerSwiGLUMLP


class SwiGLUMLP(LigerSwiGLUMLP):
    def __init__(self, hidden_size: int, intermediate_size: int):
        config = SimpleNamespace(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act="silu",
        )
        super().__init__(config)
