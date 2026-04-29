#!/usr/bin/env python3
"""
将预训练语料分词并打包为定长序列，保存为 .bin shard 文件。

输出格式：
  每个 .bin 文件是 shape=(SHARD_SIZE, SEQ_LEN) 的 uint16 numpy 数组（直接 tofile 写出）。
  {OUTPUT_DIR}/train/shard-{n:05d}.bin
  {OUTPUT_DIR}/eval/shard-{n:05d}.bin

打包策略：
  文档之间插入 EOS token，贪婪拼接，不跨 shard 截断。
  每个 shard 内部独立 shuffle，shard 粒度做 train/eval split（99:1）。
"""

import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from transformers import AutoTokenizer

# ── 配置 ─────────────────────────────────────────────────────────────────────

SOURCES: list[Path] = [
    Path("F:/transformer-decoder/pretraining/clean/finewiki"),
    Path("F:/transformer-decoder/pretraining/clean/Ultra-FineWeb"),
]
OUTPUT_DIR = Path("F:/transformer-decoder/pretraining/tokenize")
TOKENIZER_PATH = "artifacts/base"

SEQ_LEN = 2048
SHARD_SIZE = 50_000  # 每个 shard 的序列数，uint16 下约 200MB
TRAIN_RATIO = 0.99  # 99:1 split
TEXT_COLUMN = "text"
ENCODE_BATCH_SIZE = 2000  # 每批送给 tokenizer 的文本数
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────


def collect_parquet_files(sources: list[Path], rng: random.Random) -> list[Path]:
    """
    分别收集各 source 的文件，按比例均匀交织后返回。

    各 source 在 [0, 1) 区间内均匀分配位置（加小幅随机抖动），
    再按位置合并排序。无论两边文件数量差距多大，都能保证全程均匀混合，
    避免长时间连续喂同一个 source 的内容。
    """
    tagged: list[tuple[float, Path]] = []
    for source in sources:
        files = sorted(source.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in: {source}")
        n = len(files)
        rng.shuffle(files)
        for i, f in enumerate(files):
            # 均匀分布 + 半格以内的抖动，保证 source 间交织但不打破各自的内部顺序
            position = (i + rng.random() * 0.5) / n
            tagged.append((position, f))

    tagged.sort(key=lambda x: x[0])
    return [f for _, f in tagged]


class ShardWriter:
    """管理 shard buffer，满了就 shuffle 后落盘到 tmp 目录。"""

    def __init__(self, tmp_dir: Path, seq_len: int, shard_size: int):
        self.tmp_dir = tmp_dir
        self.seq_len = seq_len
        self.shard_size = shard_size
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self._buf = np.empty((shard_size, seq_len), dtype=np.uint16)
        self._pos = 0
        self._shard_idx = 0
        self.completed_shards: list[Path] = []
        self.total_sequences = 0

    def add(self, seq: np.ndarray) -> None:
        """添加一条长度为 seq_len 的 uint16 序列。"""
        self._buf[self._pos] = seq
        self._pos += 1
        if self._pos == self.shard_size:
            self._flush()

    def _flush(self, size: int | None = None) -> None:
        n = size if size is not None else self._pos
        if n == 0:
            return
        arr = self._buf[:n].copy()
        np.random.shuffle(arr)
        path = self.tmp_dir / f"shard-{self._shard_idx:05d}.bin"
        arr.tofile(path)
        self.completed_shards.append(path)
        self.total_sequences += n
        self._shard_idx += 1
        self._pos = 0

    def finalize(self) -> None:
        """刷出剩余不足一个 shard 的序列。"""
        if self._pos > 0:
            self._flush(size=self._pos)


def count_sequences(shards: list[Path], seq_len: int) -> int:
    """从文件大小推算序列总数（uint16，每个序列占 seq_len * 2 字节）。"""
    return sum(f.stat().st_size // (seq_len * 2) for f in shards)


def split_and_move(
    shards: list[Path],
    train_ratio: float,
    train_dir: Path,
    eval_dir: Path,
    seq_len: int,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    """随机分配 shard 到 train/eval 目录。

    Returns:
        (n_train_shards, n_eval_shards, n_train_seqs, n_eval_seqs)
    """
    shards = list(shards)
    rng.shuffle(shards)

    n_eval = max(1, round(len(shards) * (1 - train_ratio)))
    eval_shards = shards[:n_eval]
    train_shards = shards[n_eval:]

    # 在 rename 之前统计序列数（rename 后路径变了）
    n_train_seqs = count_sequences(train_shards, seq_len)
    n_eval_seqs = count_sequences(eval_shards, seq_len)

    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(train_shards):
        src.rename(train_dir / f"shard-{i:05d}.bin")
    for i, src in enumerate(eval_shards):
        src.rename(eval_dir / f"shard-{i:05d}.bin")

    return len(train_shards), len(eval_shards), n_train_seqs, n_eval_seqs


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    console = Console()
    console.print(f"[bold]Tokenizer:[/bold] {TOKENIZER_PATH}")
    console.print(f"[bold]Output:[/bold] {OUTPUT_DIR}")
    console.print(
        f"[bold]SEQ_LEN:[/bold] {SEQ_LEN}  [bold]SHARD_SIZE:[/bold] {SHARD_SIZE:,}  [bold]Split:[/bold] {TRAIN_RATIO:.0%}/{1 - TRAIN_RATIO:.0%}"
    )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    eos_id: int = tokenizer.eos_token_id
    vocab_size: int = tokenizer.vocab_size
    if vocab_size > 65535:
        raise ValueError(f"vocab_size={vocab_size} exceeds uint16 range (65535). Use uint32 instead.")

    files = collect_parquet_files(SOURCES, rng)
    console.print(f"Found [bold cyan]{len(files)}[/bold cyan] parquet files\n")

    tmp_dir = OUTPUT_DIR / "tmp"
    writer = ShardWriter(tmp_dir, SEQ_LEN, SHARD_SIZE)

    # 运行中的 token buffer（长度始终 < SEQ_LEN）
    token_buf: list[int] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.fields[fname]}", justify="left"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    )
    task = progress.add_task("tokenize", total=len(files), fname="starting...")

    with Live(progress, console=console, refresh_per_second=4):
        for file_path in files:
            progress.update(task, fname=file_path.name)

            try:
                table = pq.read_table(file_path, columns=[TEXT_COLUMN])
            except Exception as e:
                console.print(f"[yellow]Skip {file_path.name}: {e}[/yellow]")
                progress.advance(task)
                continue

            texts: list[str] = table[TEXT_COLUMN].to_pylist()

            for batch_start in range(0, len(texts), ENCODE_BATCH_SIZE):
                batch = [t for t in texts[batch_start : batch_start + ENCODE_BATCH_SIZE] if t and not t.isspace()]
                if not batch:
                    continue

                encoded: list[list[int]] = tokenizer(
                    batch,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]

                for ids in encoded:
                    if not ids:
                        continue
                    # 文档末尾追加 EOS
                    token_buf.extend(ids)
                    token_buf.append(eos_id)

                    # 贪婪打包：每凑满 SEQ_LEN 就生成一条序列
                    while len(token_buf) >= SEQ_LEN:
                        seq = np.array(token_buf[:SEQ_LEN], dtype=np.uint16)
                        writer.add(seq)
                        del token_buf[:SEQ_LEN]

            progress.advance(task)

    # 刷出尾部不足一整条的 token（丢弃，不做 padding）
    if token_buf:
        console.print(f"Discarding {len(token_buf)} trailing tokens (< SEQ_LEN)")
    writer.finalize()

    total_seqs = writer.total_sequences
    total_tokens = total_seqs * SEQ_LEN
    console.print(
        f"\n[bold green]Packing done.[/bold green] "
        f"Sequences: [cyan]{total_seqs:,}[/cyan]  "
        f"Tokens: [cyan]{total_tokens:,}[/cyan]  "
        f"Shards: [cyan]{len(writer.completed_shards)}[/cyan]"
    )

    # train/eval split
    train_dir = OUTPUT_DIR / "train"
    eval_dir = OUTPUT_DIR / "eval"
    n_train_shards, n_eval_shards, n_train_seqs, n_eval_seqs = split_and_move(
        writer.completed_shards, TRAIN_RATIO, train_dir, eval_dir, SEQ_LEN, rng
    )

    console.print(f"Train shards: [bold]{n_train_shards}[/bold]  Eval shards: [bold]{n_eval_shards}[/bold]")

    # 清理临时目录
    try:
        tmp_dir.rmdir()
    except OSError:
        pass  # 非空则跳过，不影响结果

    # 写 metadata JSON
    import json
    from datetime import datetime

    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tokenizer": TOKENIZER_PATH,
        "seq_len": SEQ_LEN,
        "shard_size": SHARD_SIZE,
        "train_ratio": TRAIN_RATIO,
        "sources": [str(s) for s in SOURCES],
        "train": {
            "shards": n_train_shards,
            "sequences": n_train_seqs,
            "tokens": n_train_seqs * SEQ_LEN,
        },
        "eval": {
            "shards": n_eval_shards,
            "sequences": n_eval_seqs,
            "tokens": n_eval_seqs * SEQ_LEN,
        },
        "total": {
            "shards": n_train_shards + n_eval_shards,
            "sequences": total_seqs,
            "tokens": total_tokens,
        },
    }
    meta_path = OUTPUT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    console.print("\n[bold green]All done.[/bold green]")
    console.print(f"  Train → {train_dir}  ({n_train_seqs:,} seqs, {n_train_seqs * SEQ_LEN:,} tokens)")
    console.print(f"  Eval  → {eval_dir}  ({n_eval_seqs:,} seqs, {n_eval_seqs * SEQ_LEN:,} tokens)")
    console.print(f"  Meta  → {meta_path}")


if __name__ == "__main__":
    main()
