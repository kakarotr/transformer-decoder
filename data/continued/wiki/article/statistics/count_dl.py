#!/usr/bin/env python3
"""
统计指定路径下所有 HTML 文件中 <dl> 元素出现的文件及行号。

用法：
    python find_dl.py [路径]      # 默认扫描当前目录
"""

import json
import sys
from pathlib import Path

from bs4 import BeautifulSoup

with open(Path(__file__).parent.parent / "ignore_titles.json", mode="r", encoding="utf-8") as f:
    ignore_titles = json.load(f)


def find_dl_in_files(root_path: str) -> dict[Path, list[int]]:
    root = Path(root_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"路径不存在：{root_path}")

    # 同时匹配 .html 和 .htm，rglob 递归扫描子目录
    html_files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".html", ".htm"})

    if not html_files:
        return {}

    results: dict[Path, list[int]] = {}

    for filepath in html_files:
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"[警告] 无法读取 {filepath}：{e}", file=sys.stderr)
            continue

        # html.parser 会在每个 Tag 上注入 .sourceline（1-based 行号）
        soup = BeautifulSoup(content, "html.parser")
        for section in soup.find_all("section")[1:]:
            h2 = section.find("h2")
            if h2 and h2.get_text(strip=True) not in ignore_titles:
                dl_tags = section.find_all("dl")

                if dl_tags:
                    # sourceline 极少数情况下可能为 None，过滤掉
                    lines = [tag.sourceline for tag in dl_tags if tag.sourceline is not None]
                    if lines:
                        results[filepath] = lines

    return results


def count_dl(results: dict[Path, list[int]], root: Path) -> None:
    if not results:
        print("未找到包含 <dl> 的 HTML 文件。")
        return

    total_dl = sum(len(v) for v in results.values())
    text = []
    text = f"找到 {len(results)} 个文件，共 {total_dl} 处 <dl>：\n"

    for filepath, lines in results.items():
        # 显示相对路径，更易读
        try:
            display = filepath.relative_to(root)
        except ValueError:
            display = filepath

        tag_count = len(lines)
        text += f"  {display}  [{tag_count} 处]"
        for line in lines:
            text += f"    行 {line}\n"

    with open("tmp/dl.txt", mode="w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    root = Path(path).resolve()

    print(f"扫描路径：{root}\n")

    results = find_dl_in_files(path)
    count_dl(results, root)


if __name__ == "__main__":
    main()
