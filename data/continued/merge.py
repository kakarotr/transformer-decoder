import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from opencc import OpenCC

OUTPUT_PATH = "/Users/linyongjin/Sengoku/Markdown"
INPUT_BASE_PATH = "/Users/linyongjin/Sengoku/Json"


@dataclass
class HeadingPattern:
    pattern: str
    level: int


opencc = OpenCC("tw2sp")


def get_heading_level(content: str, patterns: list[HeadingPattern]) -> tuple[int, str]:
    for hp in patterns:
        match = re.match(hp.pattern, content)
        if match:
            stripped = content[match.end() :].strip()
            return hp.level, stripped
    return 2, content


def normalize_quotes(text: str) -> str:
    result = []
    depth = 0
    for ch in text:
        if ch == '"':
            result.append("\u201c")
            depth += 1
        elif ch == '"':
            result.append("\u201d")
            depth = max(depth - 1, 0)
        else:
            result.append(ch)
    content = "".join(result)
    content = content.replace("「", "“").replace("」", "”")
    return content


def merge_to_markdown(
    json_dir: str | Path,
    book_name: str,
    heading_patterns: list[HeadingPattern],
    need_convert_sp: bool = False,
) -> None:
    json_dir = Path(json_dir)
    json_files = sorted(json_dir.glob("*.json"), key=lambda x: int(x.stem))

    # 每个元素：(文本内容, 类型)，类型为 "heading" 或 "body"
    parts: list[tuple[str, str]] = []
    last_body_idx: int | None = None

    # 书名作为一级标题
    parts.append((f"# {book_name}", "heading"))

    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            page = json.load(f)

        first_paragraph_has_indent: bool = page.get("first_paragraph_has_indent", True)
        paragraphs: list[dict] = page.get("paragraphs", [])

        for i, block in enumerate(paragraphs):
            block_type: str = block["type"]
            content: str = normalize_quotes(block["content"])
            if need_convert_sp:
                content = opencc.convert(content)

            if block_type == "title":
                level, title_text = get_heading_level(content, heading_patterns)
                parts.append((f"{'#' * level} {title_text}", "heading"))
                last_body_idx = None

            elif block_type in ("paragraph", "dialogue"):
                is_continuation = i == 0 and not first_paragraph_has_indent and last_body_idx is not None
                if is_continuation:
                    prev_text, prev_kind = parts[last_body_idx]  # type: ignore
                    parts[last_body_idx] = (prev_text + content, prev_kind)  # type: ignore
                else:
                    parts.append((content, "body"))
                    last_body_idx = len(parts) - 1

    # 拼接 Markdown 字符串
    output_lines: list[str] = []
    for i, (text, kind) in enumerate(parts):
        if i == 0:
            output_lines.append(text)
            continue

        prev_kind = parts[i - 1][1]

        if kind == "heading":
            output_lines.append("\n\n" + text)
        elif kind == "body":
            if prev_kind == "heading":
                output_lines.append("\n" + text)
            else:
                output_lines.append("\n\n" + text)

    output_path = f"{OUTPUT_PATH}/{book_name}.md"
    Path(output_path).write_text("".join(output_lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)

    args = parser.parse_args()

    input_path = f"{INPUT_BASE_PATH}/{args.name}"

    merge_to_markdown(
        book_name=args.name,
        json_dir=input_path,
        heading_patterns=[
            HeadingPattern(
                pattern=r"^(?=\S+ \S+$)",
                level=2,
            ),
            HeadingPattern(
                pattern=r"",
                level=3,
            ),
        ],
        need_convert_sp=True,
    )


# 一 标题
# r"(?:[一二三四五六七八九]十[一二三四五六七八九]?|十[一二三四五六七八九]?|[一二三四五六七八九])(?=\s)"

# 标题
# r""

# 第一章 标题
# r"第(?:[一二三四五六七八九]十[一二三四五六七八九]?|十[一二三四五六七八九]?|[一二三四五六七八九])章\s+"

# 前半段 后半段
# r"^(?=\S+ \S+$)"
