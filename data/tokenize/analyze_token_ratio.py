"""
analyze_token_dist.py
分析语料中每个文档的 token 长度分布。

用法：
    python analyze_token_dist.py \
        --tokenizer /path/to/tokenizer \
        --dirs /data/source_a /data/source_b \
        --sample 5000 \
        --workers 8

参数：
    --tokenizer   tokenizer 目录路径
    --dirs        一个或多个语料目录（parquet 格式，含 text 字段）
    --sample      每个文件最多采样的行数（0 = 全量，默认 5000）
    --workers     并行进程数（默认 8）
    --text-col    文本列名（默认 text）
    --output      可选，将原始长度列表保存为 .npy 文件
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

console = Console()

# ---------------------------------------------------------------------------
# Worker（在子进程中运行，避免 tokenizer 序列化问题）
# ---------------------------------------------------------------------------


def _worker(args: tuple) -> tuple[str, list[int]]:
    file_path, tokenizer_path, sample_n, text_col, source_label = args

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    try:
        df = pd.read_parquet(file_path, columns=[text_col])
    except Exception as e:
        return source_label, []

    if sample_n > 0 and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)

    texts = df[text_col].dropna().tolist()
    if not texts:
        return source_label, []

    encodings = tokenizer(texts, add_special_tokens=False)
    lengths = [len(ids) for ids in encodings["input_ids"]]
    return source_label, lengths


# ---------------------------------------------------------------------------
# 渲染函数
# ---------------------------------------------------------------------------


def _percentile_table(lengths: list[int], label: str) -> Table:
    arr = np.array(lengths, dtype=np.int64)
    percentiles = [10, 25, 50, 75, 90, 95, 99, 99.9]

    table = Table(
        title=f"[bold cyan]{label}[/bold cyan]  (n={len(arr):,})",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("统计量", style="dim")
    table.add_column("值", justify="right")

    table.add_row("最小值", f"{arr.min():,}")
    table.add_row("平均值", f"{arr.mean():.1f}")
    table.add_row("中位数", f"{np.median(arr):.1f}")
    table.add_row("最大值", f"{arr.max():,}")
    table.add_row("标准差", f"{arr.std():.1f}")

    for p in percentiles:
        table.add_row(f"P{p}", f"{np.percentile(arr, p):.1f}")

    return table


def _bucket_histogram(lengths: list[int], label: str, train_seq_len: int = 2048):
    """按预设区间输出 ASCII 直方图。"""
    arr = np.array(lengths, dtype=np.int64)
    total = len(arr)

    # 区间设计：围绕训练 seq_len 分布
    edges = [0, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 8192, int(1e9)]
    labels = [
        "[0, 64)",
        "[64, 128)",
        "[128, 256)",
        "[256, 512)",
        "[512, 1024)",
        "[1024, 2048)",
        "[2048, 3072)",
        "[2048, 4096)  ← 训练长度",
        "[4096, 8192)",
        "[4096, ∞)",
    ]

    counts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        counts.append(((arr >= lo) & (arr < hi)).sum())

    bar_max = 40  # 最大 bar 宽度
    max_count = max(counts) if max(counts) > 0 else 1

    table = Table(
        title=f"[bold cyan]{label}[/bold cyan] — Token 长度分布",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold green",
    )
    table.add_column("区间", min_width=32)
    table.add_column("数量", justify="right", min_width=10)
    table.add_column("占比", justify="right", min_width=7)
    table.add_column("分布", min_width=bar_max + 2)

    for i, (cnt, lbl) in enumerate(zip(counts, labels)):
        pct = cnt / total * 100
        bar_len = int(cnt / max_count * bar_max)
        # 超出训练长度的区间用红色标注
        color = "red" if i >= 7 else ("yellow" if i == 6 else "cyan")
        bar = f"[{color}]{'█' * bar_len}[/{color}]"
        table.add_row(lbl, f"{cnt:,}", f"{pct:.2f}%", bar)

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (arr >= lo) & (arr < hi)
        token_sum = arr[mask].sum()
        print(f"[{lo}, {hi}): {token_sum / arr.sum() * 100:.2f}% of tokens")

    return table


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------


def collect_parquet_files(dirs: list[str]) -> list[tuple[str, str]]:
    """返回 [(file_path, source_label), ...]"""
    result = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            console.print(f"[red]目录不存在，跳过：{d}[/red]")
            continue
        files = sorted(p.rglob("*.parquet"))
        if not files:
            console.print(f"[yellow]目录中没有 parquet 文件：{d}[/yellow]")
        for f in files:
            result.append((str(f), p.name))
    return result


def main():
    parser = argparse.ArgumentParser(description="Token 长度分布分析")
    parser.add_argument("--tokenizer", required=True, help="tokenizer 目录路径")
    parser.add_argument("--dirs", nargs="+", required=True, help="语料目录列表")
    parser.add_argument("--sample", type=int, default=5000, help="每文件采样行数（0=全量）")
    parser.add_argument("--workers", type=int, default=8, help="并行进程数")
    parser.add_argument("--text-col", default="text", help="文本列名")
    parser.add_argument("--seq-len", type=int, default=2048, help="训练序列长度（用于直方图标注）")
    parser.add_argument("--output", default=None, help="可选：将 lengths 保存为 .npy 文件路径")
    args = parser.parse_args()

    files = collect_parquet_files(args.dirs)
    if not files:
        console.print("[red]没有找到任何 parquet 文件，退出。[/red]")
        sys.exit(1)

    console.print(f"\n[bold]共发现 {len(files)} 个文件，开始并行分析...[/bold]")
    console.print(f"  tokenizer : {args.tokenizer}")
    console.print(f"  每文件采样: {'全量' if args.sample == 0 else args.sample}")
    console.print(f"  并行进程数: {args.workers}\n")

    # 构建 worker 参数列表
    worker_args = [(fp, args.tokenizer, args.sample, args.text_col, label) for fp, label in files]

    # 按 source_label 归类
    from collections import defaultdict

    all_lengths: dict[str, list[int]] = defaultdict(list)
    global_lengths: list[int] = []

    failed = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, wa): wa[0] for wa in worker_args}

        for future in as_completed(futures):
            fp = futures[future]
            try:
                source_label, lengths = future.result()
                all_lengths[source_label].extend(lengths)
                global_lengths.extend(lengths)
                completed += 1
                console.print(
                    f"  [green]✓[/green] [{completed}/{len(files)}] {Path(fp).name}  →  {len(lengths):,} 条",
                    highlight=False,
                )
            except Exception as e:
                failed += 1
                console.print(f"  [red]✗[/red] {Path(fp).name}  →  {e}", highlight=False)

    if not global_lengths:
        console.print("[red]没有收集到任何数据，退出。[/red]")
        sys.exit(1)

    console.print(f"\n[bold]分析完成：{completed} 个文件成功，{failed} 个失败。[/bold]\n")

    # -----------------------------------------------------------------------
    # 输出各 source 分布
    # -----------------------------------------------------------------------
    sources = sorted(all_lengths.keys())

    if len(sources) > 1:
        # 各 source 单独展示
        for src in sources:
            lengths = all_lengths[src]
            console.print(_percentile_table(lengths, f"📂 {src}"))
            console.print(_bucket_histogram(lengths, f"📂 {src}", args.seq_len))
            console.print()

    # 全局汇总
    console.print(_percentile_table(global_lengths, "🌐 全局汇总"))
    console.print(_bucket_histogram(global_lengths, "🌐 全局汇总", args.seq_len))

    # -----------------------------------------------------------------------
    # 多 source 对比：关键百分位
    # -----------------------------------------------------------------------
    if len(sources) > 1:
        console.print()
        compare = Table(
            title="[bold cyan]各数据源关键百分位对比[/bold cyan]",
            box=box.ROUNDED,
            header_style="bold magenta",
        )
        compare.add_column("数据源")
        compare.add_column("n", justify="right")
        for p in ["P50", "P75", "P90", "P95", "P99"]:
            compare.add_column(p, justify="right")
        compare.add_column(f">{args.seq_len}", justify="right")

        for src in sources + ["[全局]"]:
            lengths = all_lengths[src] if src != "[全局]" else global_lengths
            arr = np.array(lengths)
            exceed = (arr > args.seq_len).sum() / len(arr) * 100
            compare.add_row(
                src,
                f"{len(arr):,}",
                *[f"{np.percentile(arr, p):.0f}" for p in [50, 75, 90, 95, 99]],
                f"{exceed:.1f}%",
            )
        console.print(compare)

    # -----------------------------------------------------------------------
    # 可选保存
    # -----------------------------------------------------------------------
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, np.array(global_lengths, dtype=np.int32))
        console.print(f"\n[green]lengths 已保存至 {out}[/green]")
    arr = np.array(global_lengths)


if __name__ == "__main__":
    main()
