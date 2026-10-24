"""批次的请求/响应协议。

输出 schema 按批动态生成，properties 恰好等于该批的 slot id 集合，
配合 additionalProperties=false + required 全量，
"key 集合不一致"在 structured output 层面就不可能发生。
"""

from __future__ import annotations

import json
import re
from typing import Any

from data.continued.paths import WIKI_TRANSLATE

from .slots import Batch

SYSTEM_PROMPT = """你是一个日译中翻译引擎，负责将日本战国相关的日文维基百科内容翻译为简体中文，产出用于语言模型训练的语料。你只输出 JSON，不输出任何其他内容。
 
【输入输出说明】
输入是一个 JSON 对象，key 是片段 ID，value 含 role（片段类型）和 text（日文原文）。
输出必须是一个 JSON 对象，key 与输入完全一致，value 是对应的中文译文。
 
【翻译总则】
- 简体中文，百科书面语，忠实原文，不增删信息，不加译者注。
- 条目名在正文中出现时，一律使用 article 给定的译名。
- 术语表中出现的词条为强制对照，命中即使用指定译名，不得改写。
- 遇到日文人名、地名中的日制汉字（国字）或中文罕用异体字（如篠、惣、辻、峠、榊、塙、曽、樋等），一律保留原字，不要替换为形近简体字或意译。
- 以假名书写的人名、作品名等，glossary 未指定时：有通行中文译名的用通行译名，没有的按惯例音译或意译，不得保留假名。
- 拉丁字母内容（罗马字、英文等）原样保留。
- 「〜一揆」中的"一揆"为固定术语，保留不译，不得译作起义、暴动等。
 
【数字与纪年（硬性要求）】
- 一切数字照抄原文字形：阿拉伯数字保持阿拉伯数字，汉字数字保持汉字数字，不转换、不换算。
- 年号纪年不追加公历年份；原文已括注的公历年份照抄。
- 以上只约束数字字符本身，标点仍按标点规则正常转换。
 
【标点】
- 输出一律使用简体中文全角标点；日文读点「、」按语义译为逗号或顿号。
- 「」：一般引用与对话 → ""；若标记的是短篇作品名（歌曲、单篇文章、章节等）→ 《》。
- 『』：作品名（书籍、报刊、电影、游戏等）→ 《》，出现在引号内部时同样保持《》，不降级；若是引号内的再引用而非作品名 → ''。
- 引号嵌套：外层用 ""，内层用 ''。
- 人名间隔点 ・ → ·。
 
【事件名后缀（术语表未命中时的默认规则）】
〜の戦い → 〜之战；〜の乱 → 〜之乱；〜の変 → 〜之变；〜の役 → 〜之役；〜の陣 → 〜之阵。术语表命中时以术语表为准。"""

titles: list[dict] = []
glossaries: list[dict] = []

with open(f"{WIKI_TRANSLATE}/titles.jsonl", encoding="utf-8") as f:
    for line in f:
        titles.append(json.loads(line))

with open(f"{WIKI_TRANSLATE}/glossaries.jsonl", encoding="utf-8") as f:
    for line in f:
        glossaries.append(json.loads(line))


def _safe_filename(value: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", value).strip("_")
    return name[:120] or "untitled"


def build_input(batch: Batch) -> str:
    payload = {s.sid: {"role": s.role, "text": s.text} for s in batch.slots}
    return json.dumps(payload, ensure_ascii=False, indent=None)


def lookup_title(title_ja: str) -> str:
    """返回该条目已译好的中文标题；查不到返回 ""。

    契约：
      - 两个入参任选其一作为 key——你的译名表按文章 id 存就用 article_id，
        按日文标题存就用 title_ja。
      - 返回 "" 表示无译名。此时正文仍照常翻译，article.title 保持日文原样，
        且不会向 prompt 注入"本条目主题"那一段。
      - 每篇文章只调一次，不是每批。同步函数，若内部有 DB/磁盘 IO
        请预先全量载入内存，否则会阻塞事件循环。
    """
    for item in titles:
        if title_ja == item.get("ja_title"):
            return item.get("zh_title", "")
    return ""


def lookup_glossary(batch: Batch, title_ja: str) -> dict[str, str]:
    """返回本批需要注入的术语：{日文表层形式: 中文译名}。

    契约：
      - 只返回**本批文本里真正出现的**词条。全量表塞进每个请求既贵又稀释注意力。
      - key 是日文表层形式，不是词典原形。正文里出现 `信長` 就要有 `信長` 这条，
        只放 `織田信長` 命中不了。
      - 若同时存在长短两种写法（織田信長 / 信長），两条都返回，
        或按 key 长度降序去重——顺序会原样进 prompt。
      - 返回 {} 或 None 表示本批无需注入。

    可用 batch.slots 里每个 slot 的 .text 拿到日文原文，title_ja 是条目原标题。
    """
    # first_sid = batch.slots[0].sid if batch.slots else "empty"
    # last_sid = batch.slots[-1].sid if batch.slots else "empty"
    # debug_dir = WIKI_TRANSLATE / "debug_lookup_glossary"
    # debug_dir.mkdir(parents=True, exist_ok=True)
    # debug_path = debug_dir / f"{_safe_filename(title_ja)}__{first_sid}-{last_sid}.json"
    # debug_path.write_text(batch.to_json(), encoding="utf-8")

    result = {}
    for slot in batch.slots:
        text = getattr(slot, "text")
        for item in glossaries:
            ja_title = item.get("ja_title", "")
            zh_title = item.get("zh_title", "")
            if ja_title and zh_title and ja_title in text:
                result[ja_title] = zh_title
    return result


def build_user_message(batch: Batch, *, title_ja: str = "", title_zh: str = "") -> str:
    parts: list[str] = []

    if title_ja and title_zh:
        parts.append(f"条目名：{title_ja} → {title_zh}")

    if batch.heading_path:
        parts.append("当前位置：" + " › ".join(batch.heading_path))

    hits = lookup_glossary(batch, title_ja) or {}
    if hits:
        lines = "\n".join(f"{ja} → {zh}" for ja, zh in hits.items())
        parts.append(f"术语表：\n{lines}")

    parts.append(f"待翻译片段：\n{build_input(batch)}")
    return "\n\n".join(parts)


def build_schema(batch: Batch) -> dict:
    sids = [s.sid for s in batch.slots]
    return {
        "type": "object",
        "properties": {sid: {"type": "string"} for sid in sids},
        "required": sids,
        "additionalProperties": False,
    }


class ProtocolError(RuntimeError):
    pass


def parse_response(raw: str, batch: Batch) -> dict[str, str]:
    """解析并校验。任何不合格都抛异常，交由上层重试。"""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"响应不是合法 JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ProtocolError(f"响应顶层不是对象: {type(data).__name__}")

    expected = {s.sid for s in batch.slots}
    got = set(data)
    if got != expected:
        raise ProtocolError(f"key 集合不匹配，缺失={sorted(expected - got)} 多余={sorted(got - expected)}")

    for sid, value in data.items():
        if not isinstance(value, str) or not value.strip():
            raise ProtocolError(f"{sid} 译文为空或非字符串")

    return data
