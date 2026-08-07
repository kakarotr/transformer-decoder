"""把 WikiArticle 的叶子文本抽成可翻译 slot，并切成批次。

核心不变量：结构由 Pydantic 对象本身承载，翻译只做 setattr 原地改写。
模型的输出永远不参与结构重建。
"""

from __future__ import annotations

import itertools
import json
from typing import Any, NamedTuple

# 一个 slot 的字符数超过它，说明单个段落本身就撑爆了批次上限。
# 这种 slot 只能单独成批，无法再切——记录下来但不阻断。
_ROLE_BY_KIND = {"heading": "heading", "paragraph": "paragraph"}


class TextSlot(NamedTuple):
    """指向某个 Pydantic 实例上的一个字符串字段。"""

    sid: str
    owner: Any
    field: str
    role: str

    @property
    def text(self) -> str:
        return getattr(self.owner, self.field)

    def write(self, value: str) -> None:
        setattr(self.owner, self.field, value)


class Unit(NamedTuple):
    """一个顶层 Block 产出的 slot 集合。

    groups 是该 Unit 内部**合法的**切分点划分：
      - dl   → 每个 DescriptionItem 一组（term 与其 body 必然同组）
      - list → 每个顶层 ListItem 一组（含其 children）
      - 其他 → 只有一组，不可再分
    只有当整个 Unit 超过批次上限时才会降级到 groups 切分。
    """

    groups: list[list[TextSlot]]
    section_start: bool
    kind: str
    level: int = 0  # heading 的层级，非 heading 为 0

    @property
    def slots(self) -> list[TextSlot]:
        return [s for g in self.groups for s in g]


class Batch(NamedTuple):
    slots: list[TextSlot]
    heading_path: list[str]  # 该批起始处的标题祖先链，仅作上下文，不参与翻译

    def to_json(self) -> str:
        return json.dumps(
            {
                "heading_path": self.heading_path,
                "slot_count": len(self.slots),
                "slots": [
                    {
                        "sid": slot.sid,
                        "owner_type": type(slot.owner).__name__,
                        "field": slot.field,
                        "role": slot.role,
                        "text": slot.text,
                    }
                    for slot in self.slots
                ],
            },
            ensure_ascii=False,
            indent=2,
        )


def collect_units(article: Any) -> list[Unit]:
    """遍历文章，产出按顶层 Block 组织的 slot。

    article.title 不产出 slot——标题由上游单独翻译，此处只作为上下文注入。
    category == "enumerative" 的 ListBlock 整块跳过（不入 CPT 语料，不翻）。
    category is None 视为待翻译——判别器漏跑造成的漏译是不可逆的，
    宁可多翻也不要静默丢内容。
    """
    counter = itertools.count()

    def new_sid() -> str:
        return f"t{next(counter):04d}"

    def emit(owner: Any, field: str, role: str, out: list[TextSlot]) -> None:
        value = getattr(owner, field, None)
        if isinstance(value, str) and value.strip():
            out.append(TextSlot(new_sid(), owner, field, role))

    def walk_list_items(items: list[Any], out: list[TextSlot]) -> None:
        for item in items:
            emit(item, "text", "list_item", out)
            walk_list_items(item.children, out)

    def walk_node(node: Any, out: list[TextSlot]) -> None:
        """把一个 Block 的所有 slot 追加到 out。用于 dl.body 内的嵌套块。"""
        kind = node.kind
        if kind in _ROLE_BY_KIND:
            emit(node, "text", _ROLE_BY_KIND[kind], out)
        elif kind == "list":
            if node.category != "enumerative":
                walk_list_items(node.items, out)
        elif kind == "quote":
            emit(node, "title", "quote_title", out)
            emit(node, "text", "quote", out)
            emit(node, "citation", "citation", out)
        elif kind == "dl":
            for desc_item in node.items:
                emit(desc_item, "term", "term", out)
                for sub in desc_item.body:
                    walk_node(sub, out)
        else:
            raise ValueError(f"unknown block kind: {kind!r}")

    def build_groups(node: Any) -> list[list[TextSlot]]:
        """产出该 Block 的内部合法切分组。"""
        if node.kind == "dl":
            groups = []
            for desc_item in node.items:
                g: list[TextSlot] = []
                emit(desc_item, "term", "term", g)
                for sub in desc_item.body:
                    walk_node(sub, g)
                if g:
                    groups.append(g)
            return groups
        if node.kind == "list":
            if node.category == "enumerative":
                return []
            groups = []
            for item in node.items:
                g = []
                emit(item, "text", "list_item", g)
                walk_list_items(item.children, g)
                if g:
                    groups.append(g)
            return groups
        g = []
        walk_node(node, g)
        return [g] if g else []

    units: list[Unit] = []

    # article.title 不在此翻译：标题已由上游统一译好，
    # 在这里重译会产生第二个真值来源，正文与 canonical 标题可能不一致。

    for para in article.lead:
        g = []
        emit(para, "text", "paragraph", g)
        if g:
            units.append(Unit([g], False, "paragraph"))

    for block in article.blocks:
        groups = build_groups(block)
        if not groups:
            continue
        level = block.level if block.kind == "heading" else 0
        units.append(Unit(groups, level != 0 and level <= 2, block.kind, level))

    return units


def _chars(slots: list[TextSlot]) -> int:
    return sum(len(s.text) for s in slots)


def make_batches(
    units: list[Unit],
    *,
    max_slots: int = 40,
    max_chars: int = 6000,
) -> list[Batch]:
    """切批。规则：
      1. H2（level<=2）标题强制开新批
      2. 切点只落在顶层 Block 之间
      3. 单个 Block 超上限时，降级到它的 groups 切分
      4. 切点不落在标题与其正文之间——批尾的连续 heading 顺延到下一批
    每批携带起始处的标题祖先链（heading_path），供 prompt 定位。
    """
    batches: list[Batch] = []
    current: list[TextSlot] = []
    current_chars = 0
    stack: list[tuple[int, str]] = []  # (level, text) 标题栈
    start_path: list[str] = []  # 当前批起始时的祖先链快照

    def snapshot() -> list[str]:
        return [text for _, text in stack]

    def flush() -> None:
        nonlocal current, current_chars
        if current:
            batches.append(Batch(current, list(start_path)))
            current, current_chars = [], 0

    def trailing_heading_cut(slots: list[TextSlot]) -> int:
        """返回尾部连续 heading 的起始下标。"""
        i = len(slots)
        while i > 0 and slots[i - 1].role == "heading":
            i -= 1
        return i

    def add(chunk: list[TextSlot]) -> None:
        nonlocal current, current_chars, start_path
        n = _chars(chunk)
        over = current and (len(current) + len(chunk) > max_slots or current_chars + n > max_chars)
        if over:
            cut = trailing_heading_cut(current)
            if 0 < cut < len(current):
                # 批尾是孤立标题：留在本批会让它脱离正文，顺延到下一批
                tail = current[cut:]
                current = current[:cut]
                current_chars = _chars(current)
                flush()
                start_path = snapshot()
                current = list(tail)
                current_chars = _chars(tail)
            else:
                flush()
                start_path = snapshot()
        elif not current:
            start_path = snapshot()
        current.extend(chunk)
        current_chars += n

    for unit in units:
        if unit.section_start:
            flush()

        # 栈更新必须早于 add：批次以标题开头时，
        # 快照要包含该标题本身，否则路径指向上一个小节
        if unit.level:
            while stack and stack[-1][0] >= unit.level:
                stack.pop()
            stack.append((unit.level, unit.slots[0].text))

        whole = unit.slots
        if len(whole) <= max_slots and _chars(whole) <= max_chars:
            add(whole)
        else:
            # 超上限，降级到组切分；单组仍超限就让它独占一批
            for group in unit.groups:
                if len(group) > max_slots or _chars(group) > max_chars:
                    flush()
                    start_path = snapshot()
                    batches.append(Batch(group, list(start_path)))
                else:
                    add(group)

    flush()
    return batches
