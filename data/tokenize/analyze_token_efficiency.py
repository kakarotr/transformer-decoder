"""
领域语料 Token 效率分析脚本
用途：统计高频多字词的平均 token 数，判断是否需要扩充词表

使用方式：
    python analyze_token_efficiency.py \
        --tokenizer_path /path/to/tokenizer \
        --corpus_dir /path/to/markdown/files \
        [--top_n 30] \
        [--min_freq 10] \
        [--ngram_min 2] \
        [--ngram_max 6]
"""

import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

# ── 工具函数 ──────────────────────────────────────────────────────────────────


def is_chinese_char(ch: str) -> bool:
    """判断单个字符是否为中文汉字"""
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0xF900 <= cp <= 0xFAFF
        or 0x2F800 <= cp <= 0x2FA1F
    )


def is_all_chinese(s: str) -> bool:
    return len(s) > 0 and all(is_chinese_char(ch) for ch in s)


# 用于嵌入 term 的前后缀，确保 term 处于句子中间位置，避免词边界效应
_PREFIX = "这是"
_SUFFIX = "的内容"
_PREFIX_IDS_LEN: int | None = None  # 延迟初始化


def encode_in_context(tokenizer: Tokenizer, term: str) -> tuple[list[int], list[str]]:
    """
    将 term 嵌入中性句子后编码，提取 term 对应的 token 段。
    避免孤立编码时 pretokenizer 词边界处理与实际训练时不一致的问题。
    """
    global _PREFIX_IDS_LEN
    if _PREFIX_IDS_LEN is None:
        _PREFIX_IDS_LEN = len(tokenizer.encode(_PREFIX + "X").ids) - len(tokenizer.encode("X").ids)

    probe = _PREFIX + term + _SUFFIX
    enc = tokenizer.encode(probe)

    # 确定前缀占用的 token 数
    prefix_enc = tokenizer.encode(_PREFIX)
    prefix_len = len(prefix_enc.ids)

    # 确定后缀占用的 token 数
    suffix_enc = tokenizer.encode(_SUFFIX)
    suffix_len = len(suffix_enc.ids)

    # 从完整编码中裁出 term 的 token 段
    # 注意：BPE 合并可能跨越边界，所以用去掉首尾的方式近似
    total = len(enc.ids)
    term_ids = enc.ids[prefix_len : total - suffix_len]
    term_tokens = enc.tokens[prefix_len : total - suffix_len]

    # 边界裁剪后长度为 0 属于异常情况（极短 term 被合并进前后缀），退回孤立编码
    if len(term_ids) == 0:
        fallback = tokenizer.encode(term)
        return fallback.ids, fallback.tokens

    return term_ids, term_tokens


def strip_markdown(text: str) -> str:
    """去除 Markdown 标记，保留正文文字"""
    # 去除代码块
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    # 去除标题 # 符号
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # 去除链接 / 图片
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
    text = re.sub(r"\[.*?\]\(.*?\)", " ", text)
    # 去除强调符号
    text = re.sub(r"[*_]{1,3}", "", text)
    # 去除水平线
    text = re.sub(r"^[-*_]{3,}\s*$", " ", text, flags=re.MULTILINE)
    # 去除表格符号
    text = re.sub(r"\|", " ", text)
    return text


def extract_chinese_ngrams(text: str, min_n: int, max_n: int) -> list[str]:
    """从文本中提取所有纯汉字 n-gram（min_n <= n <= max_n）"""
    ngrams = []
    # 先切出连续汉字段落
    segments = re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]+", text)
    for seg in segments:
        for n in range(min_n, max_n + 1):
            for i in range(len(seg) - n + 1):
                ngrams.append(seg[i : i + n])
    return ngrams


def load_corpus(corpus_dir: Path) -> str:
    """读取目录下所有 .md 文件，拼接为一个大字符串"""
    md_files = sorted(corpus_dir.glob("**/*.md"))
    if not md_files:
        raise FileNotFoundError(f"在 {corpus_dir} 下未找到任何 .md 文件")

    print(f"找到 {len(md_files)} 个 .md 文件，正在读取...")
    parts = []
    for f in md_files:
        try:
            raw = f.read_text(encoding="utf-8", errors="ignore")
            parts.append(strip_markdown(raw))
        except Exception as e:
            print(f"  跳过 {f.name}: {e}")

    full_text = "\n".join(parts)
    char_count = len(full_text)
    print(f"语料总字符数：{char_count:,}")
    return full_text


# ── 主分析函数 ────────────────────────────────────────────────────────────────


def analyze(
    tokenizer: Tokenizer,
    corpus_dir: Path,
    top_n: int,
    min_freq: int,
    ngram_min: int,
    ngram_max: int,
):
    text = load_corpus(corpus_dir)

    print(f"\n正在提取 {ngram_min}~{ngram_max} 字纯汉字 n-gram...")
    all_ngrams = extract_chinese_ngrams(text, ngram_min, ngram_max)
    freq = Counter(all_ngrams)

    # 过滤低频
    candidates = [(term, cnt) for term, cnt in freq.items() if cnt >= min_freq]
    candidates.sort(key=lambda x: -x[1])

    print(f"频次 >= {min_freq} 的候选词数量：{len(candidates)}")

    if not candidates:
        print("没有找到符合条件的候选词，请降低 --min_freq 或增加语料。")
        return

    # 取 top_n 分析
    top_candidates = candidates[:top_n]

    print(f"\n{'词语':<12} {'频次':>8} {'token数':>8} {'字符数':>8} {'token/字':>10}  编码结果")
    print("─" * 80)

    results = []
    for term, cnt in top_candidates:
        token_ids, tokens = encode_in_context(tokenizer, term)
        n_tokens = len(token_ids)
        n_chars = len(term)
        ratio = n_tokens / n_chars
        results.append((term, cnt, n_tokens, n_chars, ratio, tokens))
        print(f"{term:<12} {cnt:>8,} {n_tokens:>8} {n_chars:>8} {ratio:>10.2f}  {tokens}")

    # 汇总统计
    ratios = [r[4] for r in results]
    avg_ratio = sum(ratios) / len(ratios)
    max_ratio_item = max(results, key=lambda x: x[4])
    inefficient = [r for r in results if r[4] > 1.5]

    print("\n" + "═" * 80)
    print(f"平均 token/字 比率：{avg_ratio:.3f}")
    print(f"比率最高词：'{max_ratio_item[0]}' → {max_ratio_item[4]:.2f} token/字  编码：{max_ratio_item[5]}")
    print(f"token/字 > 1.5 的词（低效分词）：{len(inefficient)} / {len(results)}")

    if inefficient:
        print("\n低效分词词语：")
        for term, cnt, n_tok, n_char, ratio, tokens in inefficient:
            print(f"  {term}  频次={cnt:,}  {ratio:.2f} token/字  {tokens}")

    print("\n" + "═" * 80)
    print("结论参考：")
    print("  平均比率 ≤ 1.5 → 词表对该领域覆盖良好，无需扩充")
    print("  平均比率 1.5~2.0 → 可选择性追加高频低效词")
    print("  平均比率 > 2.0  → 词表覆盖较差，扩充词表有实质收益")


# ── 入口 ─────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="领域语料 Token 效率分析")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="tokenizer.json 所在目录")
    parser.add_argument("--corpus_dir", type=str, required=True, help="领域 .md 文件目录")
    parser.add_argument("--top_n", type=int, default=30, help="取频率最高的前 N 个候选词（默认 30）")
    parser.add_argument("--min_freq", type=int, default=10, help="最低频次阈值（默认 10）")
    parser.add_argument("--ngram_min", type=int, default=2, help="最小 n-gram 长度（默认 2）")
    parser.add_argument("--ngram_max", type=int, default=6, help="最大 n-gram 长度（默认 6）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tokenizer_path = Path(args.tokenizer_path) / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"找不到 tokenizer.json：{tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()

    analyze(
        tokenizer=tokenizer,
        corpus_dir=Path(args.corpus_dir),
        top_n=args.top_n,
        min_freq=args.min_freq,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
    )
