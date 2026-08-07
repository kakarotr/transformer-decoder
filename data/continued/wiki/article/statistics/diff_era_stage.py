import argparse
import csv
import unicodedata
from collections import Counter
from pathlib import Path

from data.continued.paths import WIKI_ERA
from data.continued.wiki.db import WikiArticles, db


DEFAULT_STAGES = ("skip_fused", "fused")
CSV_FIELDS = [
    "reason",
    "db_title",
    "expected_file_title",
    "actual_file_title",
    "stage",
    "is_redirect",
    "redirect_to",
]


def collect_era_titles(era_dir: Path) -> set[str]:
    return {item.stem for item in era_dir.glob("*.json") if item.is_file()}


def collect_db_rows(lang: str, stages: tuple[str, ...]) -> list[dict[str, str]]:
    query = (
        WikiArticles.select(
            WikiArticles.title,
            WikiArticles.stage,
            WikiArticles.is_redirect,
            WikiArticles.redirect_to,
        )
        .where((WikiArticles.lang == lang) & (WikiArticles.stage.in_(stages)))
    )
    return [
        {
            "title": row.title,
            "stage": row.stage,
            "is_redirect": str(row.is_redirect),
            "redirect_to": row.redirect_to or "",
            "expected_file_title": expected_file_title(row.title, row.is_redirect, row.redirect_to or ""),
        }
        for row in query
    ]


def collect_all_db_stages(lang: str) -> dict[str, str]:
    query = (
        WikiArticles.select(WikiArticles.title, WikiArticles.stage)
        .where(WikiArticles.lang == lang)
        .tuples()
    )
    return {title: stage for title, stage in query}


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def title_key(title: str) -> str:
    return unicodedata.normalize("NFKC", title)


def expected_file_title(title: str, is_redirect: int, redirect_to: str) -> str:
    if is_redirect == 1 and redirect_to:
        return redirect_to.split("#", 1)[0]
    return title


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare WIKI_ERA json files with wiki_articles rows in fused-like stages."
    )
    parser.add_argument("--lang", default="ja")
    parser.add_argument("--era-dir", type=Path, default=WIKI_ERA)
    parser.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/wiki_era_stage_diff.csv"),
        help="CSV path for all mismatched titles.",
    )
    parser.add_argument(
        "--print-titles",
        action="store_true",
        help="Print every mismatched title to stdout as well as writing CSV.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    stages = tuple(args.stages)

    era_titles = collect_era_titles(args.era_dir)
    era_key_to_titles: dict[str, list[str]] = {}
    for title in era_titles:
        era_key_to_titles.setdefault(title_key(title), []).append(title)

    with db.connection_context():
        db_rows = collect_db_rows(args.lang, stages)
        all_db_title_to_stage = collect_all_db_stages(args.lang)
    all_db_key_to_stage = {title_key(title): stage for title, stage in all_db_title_to_stage.items()}

    expected_keys = {title_key(row["expected_file_title"]) for row in db_rows}
    missing_db_rows = sorted(
        (row for row in db_rows if title_key(row["expected_file_title"]) not in era_key_to_titles),
        key=lambda row: (row["expected_file_title"], row["title"]),
    )
    extra_in_era = sorted(title for title in era_titles if title_key(title) not in expected_keys)
    normalized_filename_matches = sum(
        1
        for row in db_rows
        if row["expected_file_title"] not in era_titles and title_key(row["expected_file_title"]) in era_key_to_titles
    )

    rows = []
    for row in missing_db_rows:
        rows.append(
            {
                "reason": "db_stage_but_expected_era_file_missing",
                "db_title": row["title"],
                "expected_file_title": row["expected_file_title"],
                "actual_file_title": "",
                "stage": row["stage"],
                "is_redirect": row["is_redirect"],
                "redirect_to": row["redirect_to"],
            }
        )
    rows.extend(
        {
            "reason": "era_file_but_not_expected_by_db_stage",
            "db_title": "",
            "expected_file_title": title,
            "actual_file_title": title,
            "stage": all_db_key_to_stage.get(title_key(title), "<missing_db_row>"),
            "is_redirect": "",
            "redirect_to": "",
        }
        for title in extra_in_era
    )

    write_rows(args.output, rows)

    print(f"ERA dir: {args.era_dir}")
    print(f"lang: {args.lang}")
    print(f"stages: {', '.join(stages)}")
    print(f"WIKI_ERA files: {len(era_titles)}")
    print(f"DB rows in stages: {len(db_rows)}")
    print(f"DB redirect rows in stages: {sum(row['is_redirect'] == '1' for row in db_rows)}")
    print(f"Expected unique ERA files after redirects and title normalization: {len(expected_keys)}")
    print(f"Duplicate DB rows after redirect resolution and title normalization: {len(db_rows) - len(expected_keys)}")
    print(f"DB rows matched only after title normalization: {normalized_filename_matches}")
    print(f"DB rows whose expected ERA file is missing: {len(missing_db_rows)}")
    print(f"Unique expected ERA files missing: {len({title_key(row['expected_file_title']) for row in missing_db_rows})}")
    print(f"ERA files not expected by DB stages: {len(extra_in_era)}")
    print(f"wrote: {args.output}")

    for reason, count in Counter(row["reason"] for row in rows).items():
        print(f"{reason}: {count}")
    for (reason, stage), count in Counter((row["reason"], row["stage"]) for row in rows).most_common():
        print(f"{reason} | stage={stage}: {count}")

    if args.print_titles:
        for row in rows:
            stage = f"\t{row['stage']}" if row["stage"] else ""
            print(f"{row['reason']}\t{row['db_title']}\t{row['expected_file_title']}{stage}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
