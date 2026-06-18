import json
import re
from collections import defaultdict
from pathlib import Path

TMP_DIR = Path(__file__).parent / "tmp"
LANG = "ja"

SENGOKU_RANGE = range(1467, 1616)
EXCLUDE_KEYWORDS = [
    # 以史为题材的现代作品
    "を題材とした",
    "の映画",
    "のテレビ",
    "のアニメ",
    "の漫画",
    "のゲーム",
    "の小説",
    "の文学",
    "の演劇",
    # 物质遗存
    "の遺跡",
    "の建築",
    "の城郭",  # 注意：城は残す可能性あり
    # 自然事件
    "の災害",
    "の地震",
    "の飢饉",
    "の疫病",
    # 现代学术/机构
    "の研究",
    "博物館",
    "に関する書籍",
    # 过于宽泛的跨地区分类
    "アジアの軍人",
    "世界の",
]

INCLUDE_KEYWORDS = [
    "武将",
    "武士",
    "大名",
    "将軍",
    "守護",
    "地頭",
    "合戦",
    "の戦い",
    "の乱",
    "の変",
    "の役",
    "幕府",
    "の人物",
    "氏",  # 氏 = 家族/氏族
]


def classify_year_category(name: str) -> str | None:
    """
    匹配 '1573年の日本' 或 '1470年代の日本' 格式。
    返回 'include'/'exclude'，无法匹配返回 None。
    """
    m = re.match(r"^(\d{4})年の日本$", name)
    if m:
        return "include" if int(m.group(1)) in SENGOKU_RANGE else "exclude"

    m = re.match(r"^(\d{3,4})年代の日本$", name)
    if m:
        decade = int(m.group(1))
        return "include" if any(y in SENGOKU_RANGE for y in range(decade, decade + 10)) else "exclude"

    return None


def classify_category(name: str) -> str:
    year_result = classify_year_category(name)
    if year_result is not None:
        return year_result

    if any(kw in name for kw in INCLUDE_KEYWORDS):
        return "include"

    if any(kw in name for kw in EXCLUDE_KEYWORDS):
        return "exclude"

    return "review"


if __name__ == "__main__":
    in_file = TMP_DIR / f"{LANG}_categories.json"
    out_file = TMP_DIR / f"{LANG}_classify.json"

    names: list[str] = json.loads(in_file.read_text(encoding="utf-8"))

    classify_result: dict[str, list[str]] = defaultdict(list)
    for name in names:
        classify_result[classify_category(name)].append(name)

    out_file.write_text(
        json.dumps({k: sorted(v) for k, v in classify_result.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    counts = {k: len(v) for k, v in classify_result.items()}
    print(f"✓ 分类完成: {counts}")
    print(f"  include={counts.get('include', 0)}  review={counts.get('review', 0)}  exclude={counts.get('exclude', 0)}")
    print(f"  → {out_file}")
    if counts.get("review", 0):
        print(f"\n请人工审查 review 部分，确认后可执行 insert.py")
