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

并行策略：
  ProcessPoolExecutor（7 worker 进程）负责分词（CPU 密集）。
  主进程负责 parquet 读取、结果收集、序列打包和 shard 写盘（I/O）。
  滑动窗口限制 pending futures 数量，防止内存堆积。
"""

import json
import random
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
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
from transformers import AutoTokenizer

from models.config import TransformerConfig

# ── 配置 ─────────────────────────────────────────────────────────────────────
with open("artifacts/config.json", mode="r") as f:
    config = TransformerConfig.model_validate_json(f.read())

SOURCES: list[Path] = [
    Path("F:/transformer-decoder/pretraining/1.12B/clean/finewiki"),
    Path("F:/transformer-decoder/pretraining/1.12B/clean/Ultra-FineWeb"),
]
OUTPUT_DIR = Path("F:/transformer-decoder/pretraining/1.12B/tokenize")
TOKENIZER_PATH = "artifacts"

SEQ_LEN = config.max_position_embeddings
SHARD_SIZE = 50_000  # 每个 shard 的序列数，uint16 下约 200MB
TRAIN_RATIO = 0.99  # 99:1 split
TEXT_COLUMN = "text"
ENCODE_BATCH_SIZE = 2000  # 每批送给 tokenizer 的文本数
SEED = 42

NUM_WORKERS = 7  # 8 核留 1 给主进程
MAX_PENDING = NUM_WORKERS * 3  # 滑动窗口上限：控制 in-flight futures 数量

# ─────────────────────────────────────────────────────────────────────────────
# Worker 进程状态（每个 worker 初始化一次，后续复用）
# ─────────────────────────────────────────────────────────────────────────────

_worker_tokenizer: AutoTokenizer | None = None
_worker_eos_id: int = -1


def _worker_init(tokenizer_path: str, eos_id: int) -> None:
    """在每个 worker 进程启动时调用，加载 tokenizer 并缓存到全局变量。"""
    global _worker_tokenizer, _worker_eos_id
    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    _worker_eos_id = eos_id


def _tokenize_batch(texts: list[str]) -> list[int]:
    """
    对一批文本执行分词，返回拼接后的 flat token 列表。
    每篇文档末尾追加 EOS，文档间直接拼接。
    返回 flat list 而非 list[list[int]]，减少主进程端解包开销。
    """
    assert _worker_tokenizer is not None, "Worker not initialized"
    encoded: list[list[int]] = _worker_tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    flat: list[int] = []
    for ids in encoded:
        if ids:
            flat.extend(ids)
            flat.append(_worker_eos_id)
    return flat


# ─────────────────────────────────────────────────────────────────────────────
# 主进程工具函数（与原版保持一致）
# ─────────────────────────────────────────────────────────────────────────────


def collect_parquet_files(sources: list[Path], rng: random.Random) -> list[Path]:
    """
    分别收集各 source 的文件，按比例均匀交织后返回。
    各 source 在 [0, 1) 区间内均匀分配位置（加小幅随机抖动），
    再按位置合并排序，保证全程均匀混合。
    """
    tagged: list[tuple[float, Path]] = []
    for source in sources:
        files = sorted(source.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in: {source}")
        n = len(files)
        rng.shuffle(files)
        for i, f in enumerate(files):
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
        if self._pos > 0:
            self._flush(size=self._pos)


def count_sequences(shards: list[Path], seq_len: int) -> int:
    return sum(f.stat().st_size // (seq_len * 2) for f in shards)


def split_and_move(
    shards: list[Path],
    train_ratio: float,
    train_dir: Path,
    eval_dir: Path,
    seq_len: int,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    shards = list(shards)
    rng.shuffle(shards)

    n_eval = max(1, round(len(shards) * (1 - train_ratio)))
    eval_shards = shards[:n_eval]
    train_shards = shards[n_eval:]

    n_train_seqs = count_sequences(train_shards, seq_len)
    n_eval_seqs = count_sequences(eval_shards, seq_len)

    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(train_shards):
        src.rename(train_dir / f"shard-{i:05d}.bin")
    for i, src in enumerate(eval_shards):
        src.rename(eval_dir / f"shard-{i:05d}.bin")

    return len(train_shards), len(eval_shards), n_train_seqs, n_eval_seqs


# ─────────────────────────────────────────────────────────────────────────────
# 核心：收集 future 结果并打包进 writer
# ─────────────────────────────────────────────────────────────────────────────


def _drain_future(future: Future, token_buf: list[int], writer: ShardWriter) -> None:
    """取出一个 future 的结果，追加到 token_buf，满 SEQ_LEN 时打包写盘。"""
    tokens: list[int] = future.result()
    token_buf.extend(tokens)
    while len(token_buf) >= SEQ_LEN:
        seq = np.array(token_buf[:SEQ_LEN], dtype=np.uint16)
        writer.add(seq)
        del token_buf[:SEQ_LEN]


def _submit_batch(
    pool: ProcessPoolExecutor,
    pending: deque[Future],
    token_buf: list[int],
    writer: ShardWriter,
    batch: list[str],
) -> None:
    """提交一个分词任务；若 pending 队列满则先 drain 最老的 future。"""
    # 背压：队列满时阻塞直到最老的任务完成
    if len(pending) >= MAX_PENDING:
        _drain_future(pending.popleft(), token_buf, writer)

    future = pool.submit(_tokenize_batch, batch)
    pending.append(future)


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    console = Console()
    console.print(f"[bold]Tokenizer:[/bold] {TOKENIZER_PATH}")
    console.print(f"[bold]Output:[/bold] {OUTPUT_DIR}")
    console.print(
        f"[bold]SEQ_LEN:[/bold] {SEQ_LEN}  "
        f"[bold]SHARD_SIZE:[/bold] {SHARD_SIZE:,}  "
        f"[bold]Split:[/bold] {TRAIN_RATIO:.0%}/{1 - TRAIN_RATIO:.0%}  "
        f"[bold]Workers:[/bold] {NUM_WORKERS}"
    )

    # 提前加载一次 tokenizer 拿 eos_id / vocab_size，不在主进程中保留
    _tmp = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    eos_id: int = _tmp.eos_token_id
    vocab_size: int = _tmp.vocab_size
    del _tmp

    if vocab_size > 65535:
        raise ValueError(f"vocab_size={vocab_size} exceeds uint16 range (65535). Use uint32 instead.")

    files = collect_parquet_files(SOURCES, rng)
    console.print(f"Found [bold cyan]{len(files)}[/bold cyan] parquet files\n")

    tmp_dir = OUTPUT_DIR / "tmp"
    writer = ShardWriter(tmp_dir, SEQ_LEN, SHARD_SIZE)
    token_buf: list[int] = []
    pending: deque[Future] = deque()

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

    with (
        ProcessPoolExecutor(
            max_workers=NUM_WORKERS,
            initializer=_worker_init,
            initargs=(TOKENIZER_PATH, eos_id),
        ) as pool,
        Live(progress, console=console, refresh_per_second=4),
    ):
        for file_path in files:
            progress.update(task, fname=file_path.name)

            try:
                table = pq.read_table(file_path, columns=[TEXT_COLUMN])
            except Exception as e:
                console.print(f"[yellow]Skip {file_path.name}: {e}[/yellow]")
                progress.advance(task)
                continue

            texts: list[str] = table[TEXT_COLUMN].to_pylist()
            del table  # 尽早释放 Arrow 内存

            for batch_start in range(0, len(texts), ENCODE_BATCH_SIZE):
                batch = [t for t in texts[batch_start : batch_start + ENCODE_BATCH_SIZE] if t and not t.isspace()]
                if not batch:
                    continue
                _submit_batch(pool, pending, token_buf, writer, batch)

            progress.advance(task)

        # 等待所有 pending futures 完成
        while pending:
            _drain_future(pending.popleft(), token_buf, writer)

    # 尾部不足一整条的 token 丢弃（不 padding）
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
        pass

    # 写 metadata JSON
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
