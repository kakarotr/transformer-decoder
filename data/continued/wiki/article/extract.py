import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString, Tag  # type: ignore
from tqdm import tqdm

from data.continued.wiki.article.structure import (
    Block,
    DescriptionBlock,
    DescriptionItem,
    Heading,
    ListBlock,
    ListItem,
    Paragraph,
    Quote,
    WikiArticle,
)

# ---------------------------------------------------------------------------
# Logging — 同时写文件和终端，worker 进程的异常在主进程这里统一记录
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("parse_errors.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 模块级常量（每个 worker 进程 import 时各自加载一次，开销可忽略）
# ---------------------------------------------------------------------------
_SENTENCE_END_RE = re.compile(r"[。！？」]")

with open("data/continued/wiki/article/ignore_titles.json", mode="r", encoding="utf-8") as f:
    ignore_titles = json.load(f)

# ---------------------------------------------------------------------------
# 解析辅助函数（与原来完全相同）
# ---------------------------------------------------------------------------


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


def process_list(lst: Tag, is_order: bool = False):
    texts = [t for li in lst.find_all("li") if li.find(["ul", "ol"]) is None and (t := li.get_text(strip=True))]

    def classify_list(texts: list[str], *, sentence_ratio_thr: float = 0.15):
        n = len(texts)
        if n == 0:
            return "descriptive"
        sentence_ratio = sum(bool(_SENTENCE_END_RE.search(t)) for t in texts) / n
        return "enumerative" if sentence_ratio < sentence_ratio_thr else "descriptive"

    def parse_item(li_tag: Tag) -> ListItem:
        text_parts: list[str] = []
        children: list[ListItem] = []

        for child in li_tag.children:
            if isinstance(child, Comment):
                continue
            elif isinstance(child, NavigableString):
                text_parts.append(str(child))
            elif child.name in ("ul", "ol"):  # type: ignore
                for nested_li in child.find_all("li", recursive=False):  # type: ignore
                    children.append(parse_item(nested_li))
            else:
                text_parts.append(child.get_text())

        return ListItem(
            text="".join(text_parts).strip(),
            children=children,
        )

    return ListBlock(
        ordered=is_order,
        items=[parse_item(li) for li in lst.find_all("li", recursive=False)],
        category=classify_list(texts),
    )


def process_blockquote(blockquote: Tag):
    for br in blockquote.find_all("br"):
        br.replace_with("<|new_line|>")

    text = blockquote.get_text(strip=True).replace("<|new_line|>", "\n")
    segments = text.split("—")

    content = text
    citation: str | None = None

    if len(segments) > 1:
        content = segments[0]
        citation = segments[1]

    attr_title = blockquote.get("title")
    title: str | None = None
    if attr_title:
        title = str(attr_title)
    if content:
        return Quote(title=title, text=content, citation=citation)


def process_description_list(dl: Tag):
    groups: list[tuple[str, list[Tag]]] = []
    dds: list[Tag] = []
    result: list[Block] = []
    has_dt = len(dl.find_all("dt"))
    for child in dl.children:
        if not isinstance(child, Tag):
            continue

        if child.name == "dt":
            groups.append((child.get_text(strip=True), []))

        if child.name == "dd":
            nest_dt = child.find_all("dt")
            if nest_dt:
                # 针对 dd 内部嵌套 dl/dt 的特殊情况
                groups.append((child.get_text(strip=True), []))
            else:
                if has_dt:
                    try:
                        groups[-1][1].append(child)
                    except:
                        pass
                else:
                    dds.append(child)

    if has_dt:
        result.append(DescriptionBlock(items=[]))
        for group in groups:
            title, content = group
            di = DescriptionItem(term=title, body=[])

            for item in content:
                has_child = item.find(True) is not None
                if not has_child:
                    di.body.append(Paragraph(text=item.get_text(strip=True)))
                    continue

                for child in item.find_all(recursive=False):
                    if child.name == "p":
                        di.body.append(Paragraph(text=child.get_text(strip=True)))

                    if child.name in ("ul", "ol"):
                        di.body.append(process_list(lst=child, is_order=child.name == "ol"))

                    if child.name == "blockquote":
                        quote = process_blockquote(blockquote=child)
                        if quote:
                            di.body.append(quote)
            assert isinstance(result[0], DescriptionBlock)
            result[0].items.append(di)
    else:
        for item in dds:
            has_child = item.find(True) is not None
            if not has_child:
                result.append(Paragraph(text=item.get_text(strip=True)))
                continue

            for child in item.find_all(recursive=False):
                if child.name == "p":
                    result.append(Paragraph(text=child.get_text(strip=True)))

                if child.name in ("ul", "ol"):
                    result.append(process_list(lst=child, is_order=child.name == "ol"))

                if child.name == "blockquote":
                    quote = process_blockquote(blockquote=child)
                    if quote:
                        result.append(quote)
    return result


def process_section(section: Tag):
    blocks: list[Block] = []

    heading = section.find(["h2", "h3", "h4", "h5", "h6"], recursive=False)

    if heading is None:
        section_id = section.get("data-mw-section-id", "?")
        raise ValueError(f"Section (id={section_id}) has no h2/h3/h4/h5/h6 heading")

    if not any(tag.get_text(strip=True) for tag in section.find_all(["p", "ul", "ol", "dl", "blockquote"])):
        return blocks

    level = int(heading.name[-1])
    level = level if level < 5 else 4
    heading = Heading(level=level, text=heading.get_text(strip=True))
    if heading.level == 2 and heading.text in ignore_titles:
        return blocks
    blocks.append(heading)

    for child in section.find_all(recursive=False):
        if child.name == "p":
            blocks.append(Paragraph(text=child.get_text(strip=True)))

        if child.name == "ul" or child.name == "ol":
            blocks.append(process_list(lst=child, is_order=child.name == "ol"))

        if child.name == "div" and "columns-list__wrapper" in child.get("class", []):  # type: ignore
            lst = child.find(["ol", "ul"])
            if lst:
                blocks.append(process_list(lst=lst, is_order=lst.name == "ol"))

        if child.name == "dl":
            blocks.extend(process_description_list(dl=child))

        if child.name == "blockquote":
            quote = process_blockquote(blockquote=child)
            if quote:
                blocks.append(quote)

        if child.name == "section":
            blocks.extend(process_section(section=child))

    return blocks


INPUT_DIR = Path("/Users/kakarot/Data/CPT/Sengoku/Wiki/cleaned_html")
OUTPUT_DIR = Path("/Users/kakarot/Data/CPT/Sengoku/Wiki/structure")


def process_file(html_path: Path) -> None:
    title = html_path.stem
    document = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")

    infobox = process_infobox(document)
    lead = process_lead(document)

    sections = [
        tag
        for tag in document.find("body").find_all(  # type: ignore
            "section", attrs={"data-mw-section-id": True}, recursive=False
        )
        if int(tag["data-mw-section-id"]) > 0  # type: ignore
    ]

    blocks: list[Block] = []
    for section in sections:
        blocks.extend(process_section(section=section))

    article = WikiArticle(title=title, infobox=infobox, lead=lead, blocks=blocks)
    out_path = OUTPUT_DIR / f"{title}.json"
    out_path.write_text(article.model_dump_json(ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    htmls = list(INPUT_DIR.glob("*.html"))

    failed: list[str] = []

    with ProcessPoolExecutor() as executor:
        future_to_path = {executor.submit(process_file, html): html for html in htmls}

        with tqdm(total=len(htmls), desc="Parsing", unit="file") as pbar:
            for future in as_completed(future_to_path):
                html_path = future_to_path[future]
                try:
                    future.result()
                except Exception as e:
                    title = html_path.stem
                    failed.append(title)
                    # exc_info=True 会把完整 traceback 写进日志文件
                    logger.warning("Failed: %s — %s: %s", title, type(e).__name__, e, exc_info=True)

                pbar.set_postfix_str(html_path.stem[:24])
                pbar.update(1)

    total = len(htmls)
    logger.info("Done. %d / %d succeeded, %d failed.", total - len(failed), total, len(failed))
    if failed:
        logger.warning("Failed titles: %s", ", ".join(failed))
