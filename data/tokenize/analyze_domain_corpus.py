"""
tokenizer_eval.py
对分词器在领域语料上做全量评估，输出标准化指标。
"""

import json
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

# ── 数据结构 ──────────────────────────────────────────────────────────────────


@dataclass
class ShardResult:
    total_chars: int
    total_tokens: int
    token_freq: Counter  # token_id → 出现次数
    token_len_dist: Counter  # token 字符长度 → 出现次数
    continued_tokens: int  # 非单字 token 的数量（多字符合并）
    single_char_tokens: int


@dataclass
class EvalResult:
    # 基础指标
    total_chars: int
    total_tokens: int
    fertility: float  # tokens / char，越低表示压缩越好
    compression: float  # chars / token，越高表示压缩越好

    # token 质量
    vocab_size: int
    vocab_coverage: float  # 语料中出现的 token 种类 / 词表大小
    active_vocab: int  # 语料中实际出现的 token 种类数

    # token 粒度分布
    single_char_ratio: float  # 单字符 token 占比（高 → 合并不够）
    mean_token_char_len: float
    token_len_distribution: dict[int, int]  # 字符长度 → token 出现次数

    # 专有名词分析
    ne_results: list[dict]  # 每个 NE 的详细分析


# ── Worker（在子进程中执行）────────────────────────────────────────────────────


def _init_worker(tokenizer_path: str):
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def _process_shard(lines: list[str]) -> ShardResult:
    global _tokenizer

    total_chars = 0
    total_tokens = 0
    token_freq: Counter = Counter()
    token_len_dist: Counter = Counter()
    continued_tokens = 0
    single_char_tokens = 0

    for line in lines:
        text = line.strip()
        if not text:
            continue

        # 字符数：只统计 Unicode 字母/汉字/标点，剔除空白
        char_count = sum(1 for c in text if not unicodedata.category(c).startswith("Z"))
        if char_count == 0:
            continue

        ids = _tokenizer.encode(text, add_special_tokens=False)

        total_chars += char_count
        total_tokens += len(ids)
        token_freq.update(ids)

        for surface in _tokenizer.convert_ids_to_tokens(ids):  # 改这里
            surface_chars = sum(1 for c in surface if not unicodedata.category(c).startswith("Z"))
            token_len_dist[surface_chars] += 1
            if surface_chars == 1:
                single_char_tokens += 1
            else:
                continued_tokens += 1

    return ShardResult(
        total_chars=total_chars,
        total_tokens=total_tokens,
        token_freq=token_freq,
        token_len_dist=token_len_dist,
        continued_tokens=continued_tokens,
        single_char_tokens=single_char_tokens,
    )


# ── 主评估流程 ─────────────────────────────────────────────────────────────────


def _analyze_named_entities(
    tokenizer,
    terms: list[str],
    corpus_lines: list[str],
) -> list[dict]:
    # 为每个 term 收集至多 5 个真实出现的行
    term_lines: dict[str, list[str]] = {t: [] for t in terms}
    for line in corpus_lines:
        for term in terms:
            if term in line and len(term_lines[term]) < 5:
                term_lines[term].append(line)

    results = []
    for term in terms:
        lines = term_lines[term]

        if not lines:
            results.append(
                {
                    "term": term,
                    "n_tokens": None,
                    "pieces": [],
                    "fertility": None,
                    "note": "未在语料中找到该词",
                }
            )
            continue

        # 统计所有出现位置的 token 数，取众数作为代表值
        token_counts: Counter = Counter()
        all_pieces: list[list[str]] = []

        for line in lines:
            start = 0
            while True:
                idx = line.find(term, start)
                if idx == -1:
                    break
                char_end = idx + len(term)

                enc = tokenizer(
                    line,
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                )
                offsets = enc["offset_mapping"]
                ids = enc["input_ids"]

                term_ids = [tid for tid, (s, e) in zip(ids, offsets) if s >= idx and e <= char_end]
                if term_ids:
                    pieces = tokenizer.convert_ids_to_tokens(term_ids)
                    token_counts[len(term_ids)] += 1
                    all_pieces.append(pieces)

                start = idx + 1

        if not token_counts:
            results.append(
                {
                    "term": term,
                    "n_tokens": None,
                    "pieces": [],
                    "fertility": None,
                    "note": "offset mapping 对齐失败",
                }
            )
            continue

        representative_n = token_counts.most_common(1)[0][0]
        representative_pieces = next(p for p in all_pieces if len(p) == representative_n)

        results.append(
            {
                "term": term,
                "n_tokens": representative_n,
                "pieces": representative_pieces,
                "fertility": representative_n / max(len(term), 1),
                "occurrences": sum(token_counts.values()),
                "token_count_dist": dict(token_counts),
            }
        )

    return sorted(
        [r for r in results if r["n_tokens"] is not None],
        key=lambda x: x["n_tokens"],
        reverse=True,
    )


def evaluate(
    tokenizer_path: str,
    corpus_dir: str | Path,  # 改为目录路径
    named_entities: list[str] | None = None,
    num_workers: int = 8,
) -> EvalResult:
    corpus_dir = Path(corpus_dir)
    md_files = sorted(corpus_dir.glob("**/*.md"))
    if not md_files:
        raise FileNotFoundError(f"在 {corpus_dir} 下未找到任何 .md 文件")

    # 每个文件作为一个 shard，天然对齐文件边界
    shards = [file.read_text(encoding="utf-8").splitlines() for file in md_files]

    print(f"共找到 {len(md_files)} 个 .md 文件")
    # 全量并行处理
    aggregated = ShardResult(
        total_chars=0,
        total_tokens=0,
        token_freq=Counter(),
        token_len_dist=Counter(),
        continued_tokens=0,
        single_char_tokens=0,
    )

    console = Console()
    total_lines = sum(len(s) for s in shards)
    with console.status(f"[cyan]处理 {len(shards)} 个 shard（{total_lines} 行）..."):
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(tokenizer_path,),
        ) as executor:
            futures = {executor.submit(_process_shard, shard): i for i, shard in enumerate(shards)}
            for future in as_completed(futures):
                result = future.result()
                aggregated.total_chars += result.total_chars
                aggregated.total_tokens += result.total_tokens
                aggregated.token_freq.update(result.token_freq)
                aggregated.token_len_dist.update(result.token_len_dist)
                aggregated.continued_tokens += result.continued_tokens
                aggregated.single_char_tokens += result.single_char_tokens

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    active_vocab = len(aggregated.token_freq)
    total = aggregated.total_tokens or 1

    # token 平均字符长度
    mean_token_char_len = sum(length * count for length, count in aggregated.token_len_dist.items()) / total

    all_lines = [line for shard in shards for line in shard]
    ne_results = _analyze_named_entities(tokenizer, named_entities or [], all_lines)

    return EvalResult(
        total_chars=aggregated.total_chars,
        total_tokens=aggregated.total_tokens,
        fertility=aggregated.total_tokens / max(aggregated.total_chars, 1),
        compression=aggregated.total_chars / max(aggregated.total_tokens, 1),
        vocab_size=vocab_size,
        vocab_coverage=active_vocab / vocab_size,
        active_vocab=active_vocab,
        single_char_ratio=aggregated.single_char_tokens / total,
        mean_token_char_len=mean_token_char_len,
        token_len_distribution=dict(sorted(aggregated.token_len_dist.items())),
        ne_results=ne_results,
    )


# ── 输出 ───────────────────────────────────────────────────────────────────────


def print_report(result: EvalResult):
    console = Console()

    # 基础指标
    table = Table(title="Tokenizer 领域评估报告", show_header=True, header_style="bold cyan")
    table.add_column("指标", style="bold")
    table.add_column("值")
    table.add_row("总字符数", f"{result.total_chars:,}")
    table.add_row("总 token 数", f"{result.total_tokens:,}")
    table.add_row("Fertility (token/char)", f"{result.fertility:.4f}")
    table.add_row("Compression (char/token)", f"{result.compression:.4f}")
    table.add_row("词表大小", f"{result.vocab_size:,}")
    table.add_row("活跃词表数", f"{result.active_vocab:,}")
    table.add_row("词表覆盖率", f"{result.vocab_coverage:.2%}")
    table.add_row("单字符 token 占比", f"{result.single_char_ratio:.2%}")
    table.add_row("平均 token 字符长度", f"{result.mean_token_char_len:.4f}")
    console.print(table)

    # token 长度分布
    dist_table = Table(title="Token 字符长度分布", show_header=True, header_style="bold cyan")
    dist_table.add_column("字符长度")
    dist_table.add_column("出现次数")
    dist_table.add_column("占比")
    total = result.total_tokens
    for length, count in sorted(result.token_len_distribution.items()):
        dist_table.add_row(str(length), f"{count:,}", f"{count / total:.2%}")
    console.print(dist_table)

    # 专有名词分析
    if result.ne_results:
        ne_table = Table(title="专有名词 Tokenization 分析", show_header=True, header_style="bold cyan")
        ne_table.add_column("专有名词")
        ne_table.add_column("token 数")
        ne_table.add_column("fertility")
        ne_table.add_column("切分结果")
        for ne in result.ne_results:
            ne_table.add_row(
                ne["term"],
                str(ne["n_tokens"]),
                f"{ne['fertility']:.2f}",
                " | ".join(ne["pieces"]),
            )
        console.print(ne_table)


def save_report(result: EvalResult, output_path: str | Path):
    output_path = Path(output_path)
    output_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── 入口 ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TOKENIZER_PATH = "artifacts"
    CORPUS_PATH = "/Users/linyongjin/Projects/Python/micro-transformer/data/knowledge"  # 每行一段文本，全量领域语料
    OUTPUT_PATH = "tokenizer_eval.json"

    NAMED_ENTITIES = [
        "本能寺之变",
        "织田信长",
        "丰臣秀吉",
        "德川家康",
        "关原之战",
        "战国大名",
        "室町幕府",
        "安土桃山时代",
        "明智光秀",
        "长篠之战",
        "上杉谦信",
        "武田信玄",
    ]

    result = evaluate(
        tokenizer_path=TOKENIZER_PATH,
        corpus_dir=CORPUS_PATH,
        named_entities=NAMED_ENTITIES,
        num_workers=8,
    )

    print_report(result)
    save_report(result, OUTPUT_PATH)
