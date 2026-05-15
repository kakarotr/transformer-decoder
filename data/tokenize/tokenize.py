#!/usr/bin/env python3
"""
将预训练语料分词并打包为定长序列，保存为 .bin shard 文件。

输出结构：
  {OUTPUT_DIR}/train/long/shard-{n:05d}.bin   # 长/中等文档主导的序列
  {OUTPUT_DIR}/train/short/shard-{n:05d}.bin  # 短文档贪婪打包序列
  {OUTPUT_DIR}/eval/long/shard-{n:05d}.bin
  {OUTPUT_DIR}/eval/short/shard-{n:05d}.bin

序列构建策略：
  长文档 (≥ SEQ_LEN)                  : 整除切块 → long；余部入 short_buf
  中等文档 [SHORT_THRESHOLD, SEQ_LEN) : 独占序列起始，从 short_buf 贪婪填充；
                                         不足则挂起（medium_pending），等待后续短文档补充
  短文档 (< SHORT_THRESHOLD)          : 优先填充 medium_pending → long；
                                         再贪婪打包 → short

并行策略：
  ProcessPoolExecutor (NUM_WORKERS 进程) 负责分词，返回 list[list[int]]（每篇文档独立）。
  主进程负责路由、打包、写盘，单线程顺序消费 future，无并发写入风险。
  滑动窗口（MAX_PENDING）限制 in-flight futures 数量，防止内存堆积。
"""

import json
import random
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
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

SEQ_LEN = config.max_position_embeddings  # 4096
SHORT_THRESHOLD = SEQ_LEN // 2  # 2048：低于此长度走短文档贪婪打包
SHARD_SIZE = 50_000  # 每个 shard 的序列数，uint16 下约 400MB
TRAIN_RATIO = 0.99
TEXT_COLUMN = "text"
ENCODE_BATCH_SIZE = 2000
SEED = 42
NUM_WORKERS = 7
MAX_PENDING = NUM_WORKERS * 3

# ─────────────────────────────────────────────────────────────────────────────
# Worker 进程
# ─────────────────────────────────────────────────────────────────────────────

_worker_tokenizer: AutoTokenizer | None = None
_worker_eos_id: int = -1


def _worker_init(tokenizer_path: str, eos_id: int) -> None:
    """在每个 worker 进程启动时加载 tokenizer。"""
    global _worker_tokenizer, _worker_eos_id
    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    _worker_eos_id = eos_id


def _tokenize_batch(texts: list[str]) -> list[list[int]]:
    """
    对一批文本分词，返回 list[list[int]]。
    每个元素是一篇文档的 token 列表（含末尾 EOS），空文档跳过。
    """
    assert _worker_tokenizer is not None, "Worker not initialized"
    encoded: list[list[int]] = _worker_tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"] # type: ignore
    result: list[list[int]] = []
    for ids in encoded:
        if ids:
            result.append(ids + [_worker_eos_id])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ShardWriter
# ─────────────────────────────────────────────────────────────────────────────


class ShardWriter:
    """管理 shard buffer，满了就 shuffle 后落盘。"""

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


# ─────────────────────────────────────────────────────────────────────────────
# 打包状态与路由
# ─────────────────────────────────────────────────────────────────────────────


class PackingState:
    """
    维护打包过程中的所有可变状态，集中管理路由逻辑。

    short_buf      : 短文档贪婪打包缓冲区，满 SEQ_LEN 时写入 short_writer
    medium_pending : 当前未填满的中等文档序列（至多同时存在一个）
    discarded_tokens: 因无法填满而丢弃的有效 token 总数
    """

    def __init__(self, long_writer: ShardWriter, short_writer: ShardWriter):
        self.long_writer = long_writer
        self.short_writer = short_writer
        self.short_buf: list[int] = []
        self.medium_pending: list[int] = []
        self.discarded_tokens = 0

    # ── 公共接口 ─────────────────────────────────────────────────────────────

    def route_doc(self, doc_tokens: list[int]) -> None:
        n = len(doc_tokens)
        if n >= SEQ_LEN:
            self._route_long(doc_tokens)
        elif n >= SHORT_THRESHOLD:
            self._route_medium(doc_tokens)
        else:
            self._route_short(doc_tokens)

    def finalize(self) -> None:
        """处理所有剩余 buffer，写盘并统计最终丢弃量。"""
        # 尝试用剩余 short_buf 填满 medium_pending
        self._fill_medium_from_buf()
        # medium_pending 仍未满则丢弃
        if self.medium_pending:
            self.discarded_tokens += len(self.medium_pending)
            self.medium_pending.clear()
        # short_buf 尾部不足一条序列，丢弃
        if self.short_buf:
            self.discarded_tokens += len(self.short_buf)
            self.short_buf.clear()
        self.long_writer.finalize()
        self.short_writer.finalize()

    # ── 路由分支 ─────────────────────────────────────────────────────────────

    def _route_long(self, doc_tokens: list[int]) -> None:
        """≥ SEQ_LEN：整除切块写入 long_writer，余部入 short_buf。"""
        n_chunks = len(doc_tokens) // SEQ_LEN
        for i in range(n_chunks):
            chunk = doc_tokens[i * SEQ_LEN : (i + 1) * SEQ_LEN]
            self.long_writer.add(np.array(chunk, dtype=np.uint16))
        remainder = doc_tokens[n_chunks * SEQ_LEN :]
        if remainder:
            self.short_buf.extend(remainder)
            # 余部入 buf 后顺带尝试填满 medium_pending
            self._fill_medium_from_buf()
            self._flush_short_buf()

    def _route_medium(self, doc_tokens: list[int]) -> None:
        """
        [SHORT_THRESHOLD, SEQ_LEN)：独占序列起始。
        新中等文档到来时，若有上一个未完成的 medium_pending，直接丢弃（两篇文档不共享序列）。
        """
        if self.medium_pending:
            self.discarded_tokens += len(self.medium_pending)
            self.medium_pending.clear()

        needed = SEQ_LEN - len(doc_tokens)
        fill = self.short_buf[:needed]
        del self.short_buf[:needed]
        combined = doc_tokens + fill

        if len(combined) == SEQ_LEN:
            self.long_writer.add(np.array(combined, dtype=np.uint16))
        else:
            # short_buf 当前不足，挂起等待后续短文档补充
            self.medium_pending.extend(combined)

    def _route_short(self, doc_tokens: list[int]) -> None:
        """
        < SHORT_THRESHOLD：优先填充 medium_pending；剩余部分贪婪打包进 short_buf。
        """
        remaining = doc_tokens

        if self.medium_pending:
            needed = SEQ_LEN - len(self.medium_pending)
            self.medium_pending.extend(remaining[:needed])
            remaining = remaining[needed:]
            if len(self.medium_pending) == SEQ_LEN:
                self.long_writer.add(np.array(self.medium_pending, dtype=np.uint16))
                self.medium_pending.clear()

        if remaining:
            self.short_buf.extend(remaining)
            self._flush_short_buf()

    # ── 内部工具 ─────────────────────────────────────────────────────────────

    def _fill_medium_from_buf(self) -> None:
        """从 short_buf 向 medium_pending 补充 token，填满则写盘。"""
        if not self.medium_pending:
            return
        needed = SEQ_LEN - len(self.medium_pending)
        take = min(needed, len(self.short_buf))
        self.medium_pending.extend(self.short_buf[:take])
        del self.short_buf[:take]
        if len(self.medium_pending) == SEQ_LEN:
            self.long_writer.add(np.array(self.medium_pending, dtype=np.uint16))
            self.medium_pending.clear()

    def _flush_short_buf(self) -> None:
        """将 short_buf 中满 SEQ_LEN 的部分写入 short_writer。"""
        while len(self.short_buf) >= SEQ_LEN:
            seq = np.array(self.short_buf[:SEQ_LEN], dtype=np.uint16)
            self.short_writer.add(seq)
            del self.short_buf[:SEQ_LEN]


# ─────────────────────────────────────────────────────────────────────────────
# Future 处理
# ─────────────────────────────────────────────────────────────────────────────


def _drain_future(future: Future, state: PackingState) -> None:
    """取出 future 结果（list[list[int]]），逐文档路由。"""
    docs: list[list[int]] = future.result()
    for doc_tokens in docs:
        state.route_doc(doc_tokens)


def _submit_batch(
    pool: ProcessPoolExecutor,
    pending: deque[Future],
    state: PackingState,
    batch: list[str],
) -> None:
    """提交分词任务；队列满时先阻塞消费最老的 future（背压控制）。"""
    if len(pending) >= MAX_PENDING:
        _drain_future(pending.popleft(), state)
    future = pool.submit(_tokenize_batch, batch)
    pending.append(future)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────


def collect_parquet_files(sources: list[Path], rng: random.Random) -> list[Path]:
    """
    分别收集各 source 的文件，按比例均匀交织后返回。
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
# 主函数
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    console = Console()
    console.print(f"[bold]Tokenizer:[/bold]       {TOKENIZER_PATH}")
    console.print(f"[bold]Output:[/bold]          {OUTPUT_DIR}")
    console.print(
        f"[bold]SEQ_LEN:[/bold] {SEQ_LEN}  "
        f"[bold]SHORT_THRESHOLD:[/bold] {SHORT_THRESHOLD}  "
        f"[bold]SHARD_SIZE:[/bold] {SHARD_SIZE:,}  "
        f"[bold]Split:[/bold] {TRAIN_RATIO:.0%}/{1 - TRAIN_RATIO:.0%}  "
        f"[bold]Workers:[/bold] {NUM_WORKERS}"
    )

    _tmp = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    eos_id: int = _tmp.eos_token_id
    vocab_size: int = _tmp.vocab_size
    del _tmp

    if vocab_size > 65535:
        raise ValueError(f"vocab_size={vocab_size} exceeds uint16 range (65535).")

    files = collect_parquet_files(SOURCES, rng)
    console.print(f"Found [bold cyan]{len(files)}[/bold cyan] parquet files\n")

    tmp_long = OUTPUT_DIR / "tmp" / "long"
    tmp_short = OUTPUT_DIR / "tmp" / "short"
    long_writer = ShardWriter(tmp_long, SEQ_LEN, SHARD_SIZE)
    short_writer = ShardWriter(tmp_short, SEQ_LEN, SHARD_SIZE)
    state = PackingState(long_writer, short_writer)
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
            del table

            for batch_start in range(0, len(texts), ENCODE_BATCH_SIZE):
                batch = [t for t in texts[batch_start : batch_start + ENCODE_BATCH_SIZE] if t and not t.isspace()]
                if not batch:
                    continue
                _submit_batch(pool, pending, state, batch)

            progress.advance(task)

        while pending:
            _drain_future(pending.popleft(), state)

    state.finalize()

    total_long_seqs = long_writer.total_sequences
    total_short_seqs = short_writer.total_sequences
    total_seqs = total_long_seqs + total_short_seqs
    total_tokens = total_seqs * SEQ_LEN

    console.print(
        f"\n[bold green]Packing done.[/bold green]\n"
        f"  Long  sequences : [cyan]{total_long_seqs:,}[/cyan]\n"
        f"  Short sequences : [cyan]{total_short_seqs:,}[/cyan]\n"
        f"  Total sequences : [cyan]{total_seqs:,}[/cyan]\n"
        f"  Total tokens    : [cyan]{total_tokens:,}[/cyan]\n"
        f"  Discarded tokens: [yellow]{state.discarded_tokens:,}[/yellow] "
        f"({state.discarded_tokens / max(total_tokens + state.discarded_tokens, 1) * 100:.2f}% of raw tokens)"
    )

    # train/eval split，long 和 short 各自独立执行
    split_stats: dict[str, dict] = {}
    for split_name, writer, tmp_dir in [
        ("long", long_writer, tmp_long),
        ("short", short_writer, tmp_short),
    ]:
        train_dir = OUTPUT_DIR / "train" / split_name
        eval_dir = OUTPUT_DIR / "eval" / split_name
        n_ts, n_es, n_tseqs, n_eseqs = split_and_move(
            writer.completed_shards, TRAIN_RATIO, train_dir, eval_dir, SEQ_LEN, rng
        )
        split_stats[split_name] = {
            "train": {"shards": n_ts, "sequences": n_tseqs, "tokens": n_tseqs * SEQ_LEN},
            "eval": {"shards": n_es, "sequences": n_eseqs, "tokens": n_eseqs * SEQ_LEN},
        }
        console.print(
            f"  [{split_name}] Train: {n_ts} shards / {n_tseqs:,} seqs  Eval: {n_es} shards / {n_eseqs:,} seqs"
        )
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    try:
        (OUTPUT_DIR / "tmp").rmdir()
    except OSError:
        pass

    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tokenizer": TOKENIZER_PATH,
        "seq_len": SEQ_LEN,
        "short_threshold": SHORT_THRESHOLD,
        "shard_size": SHARD_SIZE,
        "train_ratio": TRAIN_RATIO,
        "sources": [str(s) for s in SOURCES],
        "discarded_tokens": state.discarded_tokens,
        "long": split_stats["long"],
        "short": split_stats["short"],
        "total": {
            "sequences": total_seqs,
            "tokens": total_tokens,
        },
    }
    meta_path = OUTPUT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\n[bold green]All done.[/bold green]  Meta → {meta_path}")


if __name__ == "__main__":
    main()
