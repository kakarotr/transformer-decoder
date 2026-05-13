"""
PackedTokenDataset

从 tokenize.py 产出的 .bin shard 文件中读取定长序列，供 DataLoader 使用。

文件格式：
  每个 shard-{n}.bin 是 shape=(N, SEQ_LEN) 的 uint16 numpy 数组（raw binary）。
  N 可以不等于 SHARD_SIZE（最后一个 shard 可能更小）。

设计决策：
  - 使用 np.memmap 惰性映射，避免一次性加载全部数据到内存。
  - memmap 对象按文件索引缓存在 dict 中，多 worker 场景下每个 worker 进程
    持有自己的缓存副本（fork/spawn 均安全）。
  - __getitem__ 返回 torch.int64 tensor（供 embedding 层直接使用）。
"""

import bisect
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from models.config import TransformerConfig

with open("artifacts/config.json", mode="r") as f:
    config = TransformerConfig.model_validate_json(f.read())

SEQ_LEN = config.max_position_embeddings


class PackedTokenDataset(Dataset):
    def __init__(self, data_dir: Path, seq_len: int):
        """
        Args:
            data_dir: 包含 shard-*.bin 文件的目录（train/ 或 eval/）。
            seq_len:  每条序列的 token 数，必须与生成时一致。
        """
        self.seq_len = seq_len
        self.files: list[Path] = sorted(data_dir.glob("shard-*.bin"))
        if not self.files:
            raise FileNotFoundError(f"No shard-*.bin files found in: {data_dir}")

        # 每个 shard 的序列数（由文件大小推断）
        self.sizes: list[int] = []
        for f in self.files:
            byte_size = f.stat().st_size
            n_seqs, remainder = divmod(byte_size, seq_len * 2)  # uint16 = 2 bytes
            if remainder != 0:
                raise ValueError(
                    f"File size of {f} ({byte_size} bytes) is not divisible by "
                    f"seq_len * 2 = {seq_len * 2}. File may be corrupted."
                )
            self.sizes.append(n_seqs)

        # 前缀和，用于 O(log n) 的全局 index → (file_idx, local_idx) 映射
        self._offsets: list[int] = [0]
        for n in self.sizes:
            self._offsets.append(self._offsets[-1] + n)

        self._total: int = self._offsets[-1]

        # memmap 缓存（惰性初始化，worker-local）
        self._memmaps: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self._total:
            raise IndexError(f"Index {idx} out of range [0, {self._total})")

        # 二分查找：找到 idx 属于哪个 shard
        file_idx = bisect.bisect_right(self._offsets, idx) - 1
        local_idx = idx - self._offsets[file_idx]

        mm = self._get_memmap(file_idx)
        # copy() 防止返回 memmap view 导致多 worker 竞争
        seq = mm[local_idx].astype(np.int64, copy=True)
        return torch.from_numpy(seq)

    def _get_memmap(self, file_idx: int) -> np.ndarray:
        if file_idx not in self._memmaps:
            self._memmaps[file_idx] = np.memmap(
                self.files[file_idx],
                dtype=np.uint16,
                mode="r",
                shape=(self.sizes[file_idx], self.seq_len),
            )
        return self._memmaps[file_idx]

    def __repr__(self) -> str:
        return (
            f"PackedTokenDataset("
            f"shards={len(self.files)}, "
            f"sequences={self._total:,}, "
            f"tokens={self._total * self.seq_len:,})"
        )
