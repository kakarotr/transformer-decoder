import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, Tag  # type: ignore
from tqdm import tqdm

from data.continued.paths import WIKI_CLEANED_HTML, WIKI_FUSED, WIKI_PARSED
from data.continued.wiki.article.parsed import process_section
from data.continued.wiki.article.structure import Block


logger = logging.getLogger(__name__)


@dataclass
class ArticleResult:
    title: str
    parsed_changed: bool = False
    fused_changed: bool = False
    fused_exists: bool = False
    old_parsed_blocks: int = 0
    new_blocks: int = 0


def parse_blocks(html_path: Path) -> list[Block]:
    document = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    body = document.find("body")
    if body is None:
        raise ValueError(f"HTML has no body: {html_path}")

    sections = [
        tag
        for tag in body.find_all("section", attrs={"data-mw-section-id": True}, recursive=False)
        if isinstance(tag, Tag) and int(tag["data-mw-section-id"]) > 0
    ]

    blocks: list[Block] = []
    for section in sections:
        blocks.extend(process_section(section=section))
    return blocks


def read_article(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    article = json.loads(path.read_text(encoding="utf-8"))
    blocks = article.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError(f"JSON has no list blocks: {path}")
    return article


def replace_blocks(path: Path, article: dict, new_blocks: list[dict], *, dry_run: bool) -> bool:
    old_blocks = article["blocks"]
    is_changed = old_blocks != new_blocks
    if is_changed and not dry_run:
        article["blocks"] = new_blocks
        path.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
    return is_changed


def refresh_article(title: str, *, dry_run: bool = False) -> ArticleResult:
    html_path = WIKI_CLEANED_HTML / f"{title}.html"
    parsed_path = WIKI_PARSED / f"{title}.json"
    fused_path = WIKI_FUSED / f"{title}.json"

    blocks = parse_blocks(html_path)
    new_blocks = [block.model_dump(mode="json") for block in blocks]

    parsed_article = read_article(parsed_path)
    result = ArticleResult(
        title=title,
        old_parsed_blocks=len(parsed_article["blocks"]),
        new_blocks=len(new_blocks),
    )
    result.parsed_changed = replace_blocks(parsed_path, parsed_article, new_blocks, dry_run=dry_run)

    if fused_path.exists():
        fused_article = read_article(fused_path)
        result.fused_exists = True
        result.fused_changed = replace_blocks(fused_path, fused_article, new_blocks, dry_run=dry_run)

    return result


def iter_titles(title: str | None = None) -> list[str]:
    if title:
        return [title]
    return [path.stem for path in sorted(WIKI_CLEANED_HTML.glob("*.html"))]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reparse wiki HTML once, then replace only blocks in parsed JSON and existing fused JSON."
        )
    )
    parser.add_argument("--title", help="Only refresh one article title, without .html/.json suffix.")
    parser.add_argument("--limit", type=int, help="Only refresh the first N HTML files after sorting.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and compare without writing JSON files.")
    parser.add_argument("--verbose", action="store_true", help="Print per-article block counts.")
    args = parser.parse_args()

    titles = iter_titles(args.title)
    if args.limit is not None:
        titles = titles[: args.limit]

    parsed_changed = 0
    fused_changed = 0
    fused_existing = 0
    failed: list[str] = []

    for title in tqdm(titles, desc="Refreshing blocks", unit="article"):
        try:
            result = refresh_article(title, dry_run=args.dry_run)
        except Exception as exc:
            failed.append(title)
            logger.warning("Failed: %s - %s: %s", title, type(exc).__name__, exc)
            continue

        parsed_changed += int(result.parsed_changed)
        fused_changed += int(result.fused_changed)
        fused_existing += int(result.fused_exists)

        if args.verbose:
            tqdm.write(
                f"{title}: parsed {result.old_parsed_blocks} -> {result.new_blocks}"
                f"{' changed' if result.parsed_changed else ''}; "
                f"fused {'changed' if result.fused_changed else 'unchanged' if result.fused_exists else 'missing'}"
            )

    print(
        json.dumps(
            {
                "total": len(titles),
                "parsed_changed": parsed_changed,
                "fused_existing": fused_existing,
                "fused_changed": fused_changed,
                "failed": len(failed),
                "failed_titles": failed,
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
