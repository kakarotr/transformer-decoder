from typing import Literal

from pydantic import BaseModel, Field

class BookParagraph(BaseModel):
    type: Literal["title", "paragraph"] = Field(description="段落类型")
    content: str = Field(description="段落内容")

class BookPage(BaseModel):
    first_paragraph_has_indent: bool = Field(description="首个段落是否有首行缩进")
    paragraphs: list[BookParagraph] = Field(description="段落列表")