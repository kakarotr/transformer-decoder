from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from data.continued.paths import WIKI_ERA, WIKI_FUSED, WIKI_PARSED, WIKI_PREVIEW
from data.continued.wiki.db import WikiAliases, WikiArticles


class Heading(BaseModel):
    kind: Literal["heading"] = "heading"
    level: int
    text: str


class Paragraph(BaseModel):
    kind: Literal["paragraph"] = "paragraph"
    text: str


class ListItem(BaseModel):
    text: str
    children: list["ListItem"] = Field(default_factory=list)  # 块内嵌套


class ListBlock(BaseModel):
    kind: Literal["list"] = "list"
    ordered: bool
    items: list[ListItem]
    category: Literal["descriptive", "enumerative"] | None = None  # Stage 2 判别填


class Quote(BaseModel):
    title: str | None = None
    kind: Literal["quote"] = "quote"
    text: str
    citation: str | None = None


class DescriptionItem(BaseModel):
    term: str
    body: list[Paragraph | ListBlock | Quote]


class DescriptionBlock(BaseModel):
    kind: Literal["dl"] = "dl"
    items: list[DescriptionItem]


type Block = Annotated[Heading | Paragraph | ListBlock | DescriptionBlock | Quote, Field(discriminator="kind")]


class Infobox(BaseModel):
    fields: list[tuple[str, str]]


def normalize_text(text: str):
    text.replace("\n", "")


def _is_empty_block(block: Block) -> bool:
    match block.kind:
        case "paragraph":
            return not block.text.strip()
        case "heading":
            return not block.text.strip()
        case _:
            return False


def _render_list_item(item: ListItem, *, indent: int = 0, ordered: bool = False, index: int = 1) -> str:
    prefix = "  " * indent
    marker = f"{index}. " if ordered else "- "
    lines = [f"{prefix}{marker}{item.text}"]
    for i, child in enumerate(item.children, 1):
        lines.append(_render_list_item(child, indent=indent + 1, ordered=ordered, index=i))
    return "\n".join(lines)


def _render_list_block(block: ListBlock) -> str:
    return "\n".join(_render_list_item(item, ordered=block.ordered, index=i) for i, item in enumerate(block.items, 1))


def _render_quote(block: Quote) -> str:
    if block.citation:
        return f"{block.text}——{block.citation}"
    return block.text


def _render_description_block(block: DescriptionBlock) -> str:
    parts: list[str] = []
    for di in block.items:
        rendered_body: list[str] = []
        for body_item in di.body:
            if isinstance(body_item, Paragraph):
                rendered_body.append(body_item.text)
            elif isinstance(body_item, ListBlock):
                rendered_body.append(_render_list_block(body_item))
            elif isinstance(body_item, Quote):
                rendered_body.append(_render_quote(body_item))
        parts.append(f"**{di.term}**\n{'\n\n'.join(rendered_body)}")
    return "\n\n".join(parts)


def _render_block(block: Block) -> str:
    match block.kind:
        case "heading":
            return f"{'#' * block.level} {block.text}"
        case "paragraph":
            return block.text
        case "list":
            return _render_list_block(block)
        case "dl":
            return _render_description_block(block)
        case "quote":
            return _render_quote(block)
        case _:
            raise ValueError(f"Unknown block kind: {block.kind}")


def _join_blocks(blocks: list[Block]) -> str:
    if not blocks:
        return ""
    parts = [_render_block(blocks[0])]
    for i in range(1, len(blocks)):
        prev, curr = blocks[i - 1], blocks[i]
        sep = "\n" if prev.kind == "heading" and curr.kind != "heading" else "\n\n"
        parts.append(sep + _render_block(curr))
    return "".join(parts)


def _filter_description_block(block: DescriptionBlock) -> DescriptionBlock | None:
    filtered_items = []
    for item in block.items:
        filtered_body = [b for b in item.body if not (isinstance(b, ListBlock) and b.category == "enumerative")]
        if filtered_body:
            filtered_items.append(DescriptionItem(term=item.term, body=filtered_body))

    if not filtered_items:
        return None
    return DescriptionBlock(items=filtered_items)


def _filter_blocks(blocks: list[Block]) -> list[Block]:
    # Step 1: 过滤 enumerative list
    filtered: list[Block] = []
    for b in blocks:
        if b.kind == "paragraph" and b.text.startswith("※"):
            continue
        elif b.kind == "list" and b.category == "enumerative":
            continue
        elif b.kind == "dl":
            result = _filter_description_block(b)
            if result is not None:
                filtered.append(result)
        else:
            filtered.append(b)

    # Step 2: 移除过滤后变空的 heading
    stack: list[Block] = []
    for block in filtered:
        if block.kind == "heading":
            while stack and stack[-1].kind == "heading" and stack[-1].level >= block.level:
                stack.pop()
        stack.append(block)

    while stack and stack[-1].kind == "heading":
        stack.pop()

    return stack


class WikiArticle(BaseModel):
    title: str
    infobox: dict[str, str] | None = None
    lead: list[Paragraph]
    blocks: list[Block]

    def merge_to_md(self):
        sections: list[str] = [f"# {self.title}"]

        if self.lead:
            sections.append("\n\n".join(p.text for p in self.lead))

        filtered_blocks = _filter_blocks(self.blocks)
        if filtered_blocks:
            sections.append(_join_blocks(filtered_blocks))

        return "\n\n".join(sections)

    @model_validator(mode="after")
    def filter_empty_blocks(self) -> "WikiArticle":
        self.lead = [p for p in self.lead if p.text.strip()]
        self.blocks = [b for b in self.blocks if not _is_empty_block(b)]
        return self


def preivew(count: int):
    files = sorted(Path(WIKI_ERA).glob("*.json"), key=lambda x: x.stat().st_size, reverse=True)
    titles = [p.name for p in files[count - 100 : count]]
    output_dir = WIKI_PREVIEW / str(count)
    output_dir.mkdir(parents=True, exist_ok=True)
    for title in titles:
        article = WikiArticle.model_validate_json(Path(WIKI_ERA / title).read_text())
        content = article.merge_to_md()
        file = Path(output_dir / f"{title.split('.')[0]}.md")
        if not file.exists():
            file.touch()
        file.write_text(content)


def preview_single(title: str, preview_dir: str):
    article = WikiArticle.model_validate_json(Path(WIKI_ERA / f"{title}.json").read_text())
    content = article.merge_to_md()
    output_dir = WIKI_PREVIEW / preview_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    file = output_dir / f"{title.split('.')[0]}.md"
    file.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    preivew(200)
    preivew(300)
    preivew(400)
