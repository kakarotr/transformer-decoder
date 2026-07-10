import argparse

from data.continued.wiki.article.structure import preview_single
from data.continued.wiki.article.tmp.refresh_blocks import refresh_article


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh one wiki article's blocks, then regenerate its Markdown preview."
    )
    parser.add_argument("--title", required=True, help="Article title, without .html/.json suffix.")
    parser.add_argument(
        "--preview-dir",
        required=True,
        help="Preview subdirectory under WIKI_PREVIEW, for example 200.",
    )
    args = parser.parse_args()

    result = refresh_article(args.title)
    if not result.fused_exists:
        parser.error(f"Fused article does not exist: {args.title}.json")

    preview_single(args.title, args.preview_dir)
    print(
        f"{args.title}: parsed {result.old_parsed_blocks} -> {result.new_blocks}; "
        f"fused {'changed' if result.fused_changed else 'unchanged'}; "
        f"preview_dir={args.preview_dir}"
    )


if __name__ == "__main__":
    main()
