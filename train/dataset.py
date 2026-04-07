import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        path: str | Path,
        *,
        dtype: str | None = None,
        seq_len: int | None = None,
    ):
        super().__init__()
        self.data_dir = Path(path)

        meta_files = sorted(self.data_dir.glob("*.json"))
        meta_files = [p for p in meta_files if not p.name.endswith("_summary.json")]

        self.shards: list[dict] = []
        self.cum_blocks: list[int] = []
        total_blocks = 0

        expected_dtype: np.dtype | None = np.dtype(dtype) if dtype is not None else None
        expected_seq_len = seq_len

        for meta_path in meta_files:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            bin_file = meta.get("bin_file")
            bin_path = self.data_dir / bin_file

            shard_dtype = np.dtype(meta["dtype"])
            shard_seq_len = int(meta["seq_len"])
            num_tokens = int(meta["num_tokens"])
            num_blocks = int(meta["num_blocks"])

            if num_tokens != num_blocks * shard_seq_len:
                raise ValueError(
                    f"Invalid shard meta: {meta_path}, "
                    f"num_tokens({num_tokens}) != num_blocks({num_blocks}) * seq_len({shard_seq_len})"
                )

            if expected_dtype is None:
                expected_dtype = shard_dtype
            elif shard_dtype != expected_dtype:
                raise ValueError(f"Inconsistent dtype across shards: got {shard_dtype}, expected {expected_dtype}")

            if expected_seq_len is None:
                expected_seq_len = shard_seq_len
            elif shard_seq_len != expected_seq_len:
                raise ValueError(
                    f"Inconsistent seq_len across shards: got {shard_seq_len}, expected {expected_seq_len}"
                )

            self.shards.append(
                {
                    "meta_path": meta_path,
                    "bin_path": bin_path,
                    "dtype": shard_dtype,
                    "seq_len": shard_seq_len,
                    "num_tokens": num_tokens,
                    "num_blocks": num_blocks,
                }
            )
            total_blocks += num_blocks
            self.cum_blocks.append(total_blocks)

        if expected_dtype is None or expected_seq_len is None:
            raise ValueError("Failed to infer dataset dtype / seq_len")

        self.dtype = expected_dtype
        self.seq_len = expected_seq_len
        self.total_blocks = total_blocks
        self._memmaps: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return self.total_blocks

    def __getitem__(self, index: int):
        if index < 0:
            index += self.total_blocks

        if index < 0 or index >= self.total_blocks:
            raise IndexError(f"index out of range: {index}")

        shard_idx = bisect.bisect_right(self.cum_blocks, index)
        block_start = 0 if shard_idx == 0 else self.cum_blocks[shard_idx - 1]
        local_block_idx = index - block_start

        mm = self._get_memmap(shard_idx)
        token_start = local_block_idx * self.seq_len
        token_end = token_start + self.seq_len

        block = np.asarray(mm[token_start:token_end], dtype=np.int64)
        return torch.from_numpy(block)

    def _get_memmap(self, shard_idx: int):
        mm = self._memmaps.get(shard_idx)
        if mm is None:
            shard = self.shards[shard_idx]
            mm = np.memmap(
                shard["bin_path"],
                mode="r",
                dtype=shard["dtype"],
                shape=(shard["num_tokens"],),
            )
            self._memmaps[shard_idx] = mm
        return mm
