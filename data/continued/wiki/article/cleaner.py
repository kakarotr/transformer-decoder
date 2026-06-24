import re
from typing import ClassVar

from bs4 import BeautifulSoup, Comment, NavigableString, Tag  # type: ignore


class WikiCleaner:
    DECOMPOSE_CLASSES: ClassVar[set[str]] = {
        "gallery",
        "NavFrame",
        "thumb",
        "sistersitebox",
        "rellink",
        "reflist",
        "noprint",
        "navbox",
        "refbegin",
    }

    DECOMPOSE_TAGS: ClassVar[set[str]] = {"style", "meta", "link", "figure", "small"}

    _RE_WHITESPACE: ClassVar[re.Pattern[str]] = re.compile(r"\s+")
    _RE_CJK_SPACE: ClassVar[re.Pattern[str]] = re.compile(r"(?<=[^\x00-\x7F]) (?=[^\x00-\x7F])")
    _RE_BOLD_PRE: ClassVar[re.Pattern[str]] = re.compile(r"(?<=[^\x00-\x7F]) (?=\*\*)")
    _RE_BOLD_POST: ClassVar[re.Pattern[str]] = re.compile(r"(?<=\*\*) (?=[^\x00-\x7F])")

    def __init__(self, content: str) -> None:
        self.document = BeautifulSoup(content, "html.parser")

    def clean(self) -> str:
        head = self.document.find("head")
        if head:
            head.decompose()

        # 先删编辑入口（连同内部 a 标签一起删除）
        for tag in self.document.find_all("span", class_="mw-editsection"):
            tag.decompose()

        self._remove_comments()
        self._remove_by_tag()
        self._remove_by_class()
        self._clean_ruby_tags()
        self._clean_sup_tag()
        self._clean_table()
        self._strip_parsoid_attrs()
        self._unwrap_a()
        self._process_blockquote_tags()
        # _process_span_tags 必须先于 _process_inline_tags，确保 <b> 内的嵌套 span 在生成 ** 标记前已展开
        self._process_span_tags()
        self._process_inline_tags()
        self._normalize_whitespace()

        return self.document.prettify()

    def _remove_comments(self) -> None:
        for comment in self.document.find_all(string=lambda node: isinstance(node, Comment)):
            comment.extract()

    def _remove_by_class(self) -> None:
        selector = ", ".join(f".{cls}" for cls in self.DECOMPOSE_CLASSES)
        for tag in self.document.select(selector):
            tag.decompose()

    def _remove_by_tag(self) -> None:
        for tag in self.DECOMPOSE_TAGS:
            for ele in self.document.find_all(tag):
                ele.decompose()

    def _clean_sup_tag(self) -> None:
        def _should_delete_sup(sup_tag) -> bool:
            classes = set(sup_tag.get("class", []))
            if "reference" in classes:
                return True
            # 所有 Template-* 系列都是编辑标记
            if any(c.startswith("Template-") or c.endswith("-Template") for c in classes):
                return True
            return False

        for sup in self.document.find_all("sup"):
            if _should_delete_sup(sup):
                sup.decompose()
            else:
                sup.unwrap()

    def _clean_ruby_tags(self) -> None:
        for ruby in self.document.find_all("ruby"):
            # 删掉读音标注和括号，保留基底汉字
            for tag in ruby.find_all(["rt", "rp"]):
                tag.decompose()

            # 剥掉 rb 容器，保留文字
            for rb in ruby.find_all("rb"):
                rb.unwrap()
            ruby.unwrap()

    def _strip_parsoid_attrs(self) -> None:
        for tag in self.document.find_all(True):
            for attr in {"data-mw", "about", "typeof"}:
                tag.attrs.pop(attr, None)

    def _unwrap_a(self) -> None:
        for a in self.document.find_all("a"):
            a.unwrap()

    def _process_inline_tags(self) -> None:
        for tag in self.document.find_all("i"):
            tag.unwrap()

        for tag in self.document.find_all("b"):
            text = tag.get_text().strip()
            tag.replace_with(f"**{text}**")

    def _process_span_tags(self) -> None:
        for span in self.document.find_all("span"):
            if span.parent is None:  # 已被外层 span 的 decompose 带走
                continue

            if span.find("img"):
                span.decompose()
                continue

            if not span.get_text(strip=True):  # 空或纯空白
                span.decompose()
                continue

            span.unwrap()

    def _clean_table(self) -> None:
        for table in self.document.select("table:not(.infobox):not(.wikitable)"):
            table.decompose()

    def _append_blockquote_text(self, value: str, pieces: list) -> None:
        text = " ".join(value.split())
        if not text:
            return
        if pieces and isinstance(pieces[-1], str) and not pieces[-1].endswith(" "):
            pieces[-1] += " "
        pieces.append(text)

    def _walk_blockquote(self, node, pieces: list) -> None:
        if isinstance(node, NavigableString):
            self._append_blockquote_text(str(node), pieces)
            return
        if not isinstance(node, Tag):
            return
        if node.name == "br":
            pieces.append(self.document.new_tag("br"))
            return
        if node.name == "cite":
            text = " ".join(node.get_text(" ", strip=True).split())
            if text:
                self._append_blockquote_text(text, pieces)
            return
        for child in node.children:
            self._walk_blockquote(child, pieces)

    def _process_blockquote_tags(self) -> None:
        for blockquote in self.document.find_all("blockquote"):
            pieces: list[str | Tag] = []
            self._walk_blockquote(blockquote, pieces)
            blockquote.clear()
            for piece in pieces:
                if isinstance(piece, str):
                    blockquote.append(NavigableString(piece))
                else:
                    blockquote.append(piece)

    def _normalize_whitespace(self) -> None:
        self.document.smooth()

        for text_node in self.document.find_all(string=lambda s: type(s) is NavigableString):
            s = self._RE_WHITESPACE.sub(" ", str(text_node))
            s = self._RE_CJK_SPACE.sub("", s)
            s = self._RE_BOLD_PRE.sub("", s)
            s = self._RE_BOLD_POST.sub("", s)
            s = s.strip()
            text_node.replace_with(s)
