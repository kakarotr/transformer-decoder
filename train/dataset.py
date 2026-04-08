import bisect
import json
from pathlib import Path
from typing import cast

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

    def _normalize_index(self, index: int) -> int:
        if index < 0:
            index += self.total_blocks

        if index < 0 or index >= self.total_blocks:
            raise IndexError(f"index out of range: {index}")

        return index

    def _locate_block(self, index: int) -> tuple[int, int]:
        shard_idx = bisect.bisect_right(self.cum_blocks, index)
        block_start = 0 if shard_idx == 0 else self.cum_blocks[shard_idx - 1]
        local_block_idx = index - block_start
        return shard_idx, local_block_idx

    def _read_one_block(self, shard_idx: int, local_block_idx: int) -> torch.Tensor:
        mm = self._get_memmap(shard_idx)
        token_start = local_block_idx * self.seq_len
        token_end = token_start + self.seq_len

        if self.dtype == np.int64:
            block = mm[token_start:token_end]
        else:
            block = np.asarray(mm[token_start:token_end], dtype=np.int64)

        return torch.from_numpy(block)

    def __getitem__(self, index: int) -> torch.Tensor:
        index = self._normalize_index(index)
        shard_idx, local_block_idx = self._locate_block(index)
        return self._read_one_block(shard_idx, local_block_idx)

    def __getitems__(self, indices: list[int]) -> list[torch.Tensor]:
        if not indices:
            return []

        normalized = [self._normalize_index(i) for i in indices]
        outputs: list[torch.Tensor | None] = [None] * len(normalized)

        grouped: dict[int, list[tuple[int, int]]] = {}
        for out_pos, index in enumerate(normalized):
            shard_idx, local_block_idx = self._locate_block(index)
            grouped.setdefault(shard_idx, []).append((out_pos, local_block_idx))

        for shard_idx, items in grouped.items():
            mm = self._get_memmap(shard_idx)

            items.sort(key=lambda x: x[1])

            run_start = 0
            while run_start < len(items):
                run_end = run_start + 1
                while run_end < len(items):
                    prev_local = items[run_end - 1][1]
                    curr_local = items[run_end][1]
                    if curr_local != prev_local + 1:
                        break
                    run_end += 1

                run_items = items[run_start:run_end]
                first_local = run_items[0][1]
                last_local = run_items[-1][1]

                token_start = first_local * self.seq_len
                token_end = (last_local + 1) * self.seq_len

                if self.dtype == np.int64:
                    chunk = mm[token_start:token_end]
                else:
                    chunk = np.asarray(mm[token_start:token_end], dtype=np.int64)

                chunk = chunk.reshape(-1, self.seq_len)

                for row_idx, (out_pos, _) in enumerate(run_items):
                    outputs[out_pos] = torch.from_numpy(chunk[row_idx])

                run_start = run_end

        result = cast(list[torch.Tensor], outputs)
        return result

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
