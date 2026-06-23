import re

from bs4 import BeautifulSoup, NavigableString, Tag  # type: ignore


class WikiCleaner:
    DECOMPOSE_CLASSES = {
        "gallery",
        "NavFrame",
        "thumb",
        "sistersitebox",
        "rellink",
        "reflist",
        "noprint",
        "navbox",
    }

    DECOMPOSE_TAGS = {"style", "meta", "link", "figure"}

    def __init__(self, content: str) -> None:
        self.document = BeautifulSoup(content, "html.parser")

    def clean(self):
        head = self.document.find("head")
        if head:
            head.decompose()

        # 先删编辑入口（连同内部 a 标签一起删除）
        for tag in self.document.find_all("span", class_="mw-editsection"):
            tag.decompose()

        self._remove_by_tag()
        self._remove_by_class()
        self._clean_ruby_tags()
        self._clean_sup_tag()
        self._clean_table()
        self._strip_parsoid_attrs()
        self._unwrap_a()
        self._process_blockquote_tags()
        self._process_span_tags()
        self._process_inline_tags()
        self._normalize_whitespace()

        return self.document.prettify()

    def _remove_by_class(self) -> None:
        for cls in self.DECOMPOSE_CLASSES:
            for tag in self.document.find_all(class_=cls):
                tag.decompose()

    def _remove_by_tag(self) -> None:
        for tag in self.DECOMPOSE_TAGS:
            for ele in self.document.find_all(tag):
                ele.decompose()

    def _clean_sup_tag(self):
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

    def _unwrap_a(self):
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

    def _clean_table(self):
        for table in self.document.select("table:not(.infobox):not(.wikitable)"):
            table.decompose()

    def _merge_paragraph(self, root: Tag) -> None:
        for section in root.find_all("section", recursive=False):
            self._merge_paragraph(section)

        current_group: list[Tag] = []

        def flush_group() -> None:
            if len(current_group) <= 1:
                current_group.clear()
                return
            first = current_group[0]
            for p in current_group[1:]:
                first.append("\n")
                for child in list(p.children):
                    first.append(child)
                p.decompose()
            current_group.clear()

        for child in list(root.children):
            if isinstance(child, Tag) and child.name == "p":
                current_group.append(child)
            elif isinstance(child, NavigableString) and not child.strip():
                continue  # 标签间的空白文本节点不中断分组
            else:
                flush_group()

        flush_group()

    def _process_blockquote_tags(self):
        for blockquote in self.document.find_all("blockquote"):
            pieces: list[str | Tag] = []

            def append_text(value: str) -> None:
                text = " ".join(value.split())
                if not text:
                    return
                if pieces and isinstance(pieces[-1], str) and not pieces[-1].endswith(" "):
                    pieces[-1] += " "
                pieces.append(text)

            def walk(node) -> None:
                if isinstance(node, NavigableString):
                    append_text(str(node))
                    return
                if not isinstance(node, Tag):
                    return
                if node.name == "br":
                    pieces.append(self.document.new_tag("br"))
                    return
                if node.name == "cite":
                    text = " ".join(node.get_text(" ", strip=True).split())
                    if text:
                        append_text(text)
                    return
                for child in node.children:
                    walk(child)

            walk(blockquote)
            blockquote.clear()
            for piece in pieces:
                if isinstance(piece, str):
                    blockquote.append(NavigableString(piece))
                else:
                    blockquote.append(piece)

    def _normalize_whitespace(self) -> None:
        self.document.smooth()

        for text_node in self.document.find_all(string=True):
            s = re.sub(r"\s+", " ", str(text_node))
            # 非 ASCII 字符之间去空格
            s = re.sub(r"(?<=[^\x00-\x7F]) (?=[^\x00-\x7F])", "", s)

            # 非 ASCII 字符与 ** 标记之间去空格
            s = re.sub(r"(?<=[^\x00-\x7F]) (?=\*\*)", "", s)
            s = re.sub(r"(?<=\*\*) (?=[^\x00-\x7F])", "", s)

            s = s.strip()
            text_node.replace_with(s)
