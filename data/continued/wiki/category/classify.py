import json
import re
from collections import defaultdict

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
    # 具体年份：1573年の日本
    m = re.match(r"^(\d{4})年の日本$", name)
    if m:
        return "include" if int(m.group(1)) in SENGOKU_RANGE else "exclude"

    # 年代：1470年代の日本
    m = re.match(r"^(\d{3,4})年代の日本$", name)
    if m:
        decade = int(m.group(1))
        # 年代与目标范围有重叠则保留
        return "include" if any(y in SENGOKU_RANGE for y in range(decade, decade + 10)) else "exclude"

    return None  # 不是年份分类，交给其他规则处理


def classify_category(name: str) -> str:
    # 年份分类优先处理
    year_result = classify_year_category(name)
    if year_result is not None:
        return year_result

    # 保留关键词
    if any(kw in name for kw in INCLUDE_KEYWORDS):
        return "include"

    # 排除关键词
    if any(kw in name for kw in EXCLUDE_KEYWORDS):
        return "exclude"

    return "review"


if __name__ == "__main__":
    classify_result: dict[str, set[str]] = defaultdict(set)
    with open("data/continued/wiki/category/ja_categories.txt", mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            match = re.findall(r"name='([^']+)'", line)
            if not match:
                continue
            name = match[0]
            result = classify_category(name)
            classify_result[result].add(name)
    with open("data/continued/wiki/ja_classify_category.json", mode="w+", encoding="utf-8") as f:
        f.write(json.dumps({k: list(v) for k, v in classify_result.items()}, ensure_ascii=False, indent=2))
