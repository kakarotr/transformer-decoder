from typing import Annotated, Literal

from pydantic import BaseModel, Field


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


type Block = Annotated[Heading | Paragraph | ListBlock | Quote, Field(discriminator="kind")]


class Infobox(BaseModel):
    fields: list[tuple[str, str]]


class WikiArticle(BaseModel):
    title: str
    infobox: dict[str, str] | None = None
    lead: list[Paragraph]
    blocks: list[Block]
