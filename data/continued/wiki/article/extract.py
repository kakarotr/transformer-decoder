import json

import bs4
from bs4 import BeautifulSoup

from data.continued.wiki.article.structure import (
    Block,
    Heading,
    Paragraph,
    Quote,
    WikiArticle,
)

with open("data/continued/wiki/article/ignore_titles.json", mode="r", encoding="utf-8") as f:
    ignore_titles = json.load(f)


def process_infobox(document: BeautifulSoup):
    rs = document.select("table.infobox")
    infobox: dict[str, str] | None = None
    if rs:
        for tr in rs[0].find_all("tr"):
            key, value = "", ""
            th = tr.find("th")
            if th:
                key = th.get_text(strip=True)
            td = tr.find("td")
            if td:
                for br in td.find_all("br"):
                    br.replace_with("，")
                value = td.get_text(strip=True)
            if key and value:
                if infobox is None:
                    infobox = {}
                infobox[key] = value
    return infobox


def process_lead(document: BeautifulSoup):
    rs = document.select("section[data-mw-section-id='0']")
    leads = [Paragraph(text=p.get_text(strip=True)) for p in rs[0].find_all("p")]
    return leads


def process_section(section: bs4.Tag):
    blocks: list[Block] = []

    heading = section.find(["h2", "h3", "h4"], recursive=False)
    assert heading is not None

    leading = Heading(level=int(heading.name[-1]), text=heading.get_text(strip=True))
    if leading.level == 2 and leading.text in ignore_titles:
        return blocks
    blocks.append(leading)
    for child in section.find_all(recursive=False):
        if child.name == "p":
            blocks.append(Paragraph(text=child.get_text(strip=True)))

        if child.name == "ul":
            pass

        if child.name == "dl":
            pass

        if child.name == "blockquote":
            for br in child.find_all("br"):
                print("bb")
                br.replace_with("<|new_line|>")

            text = child.get_text(strip=True).replace("<|new_line|>", "\n")
            segments = text.split("—")

            content = text
            citation: str | None = None

            if len(segments) > 1:
                content = segments[0]
                citation = segments[1]

            attr_title = child.get("title")
            title: str | None = None
            if attr_title:
                title = str(attr_title)

            blocks.append(Quote(title=title, text=content, citation=citation))

        if child.name == "section":
            blocks.extend(process_section(section=child))

    return blocks


if __name__ == "__main__":
    title = "豊臣秀吉"
    with open(f"/Users/kakarot/Data/CPT/Sengoku/Wiki/cleaned_html/{title}.html", mode="r", encoding="utf-8") as f:
        document = BeautifulSoup(f.read(), "html.parser")

    infobox = process_infobox(document)
    lead = process_lead(document)

    sections = [
        tag
        for tag in document.find("body").find_all("section", attrs={"data-mw-section-id": True}, recursive=False)  # type: ignore
        if int(tag["data-mw-section-id"]) > 0  # type: ignore
    ]

    blocks: list[Block] = []
    for section in sections:
        blocks.extend(process_section(section=section))

    article = WikiArticle(title=title, infobox=infobox, lead=lead, blocks=blocks)
    with open(f"{title}.json", mode="w", encoding="utf-8") as f:
        f.write(article.model_dump_json(ensure_ascii=False, indent=2))
