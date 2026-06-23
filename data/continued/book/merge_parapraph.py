import json
import os
import sys
from pathlib import Path


def get_sorted_json_files(directory: Path) -> list[tuple[int, Path]]:
    """Collect all numerically-named JSON files, sorted by number."""
    files = []
    for f in directory.glob("*.json"):
        try:
            num = int(f.stem)
            files.append((num, f))
        except ValueError:
            pass
    return sorted(files, key=lambda x: x[0])


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process(directory: Path) -> None:
    files = get_sorted_json_files(directory)

    if not files:
        print("[INFO] No numerically-named JSON files found.")
        return

    for i, (num, filepath) in enumerate(files):
        data = load_json(filepath)

        # Only act when first_paragraph_has_indent is explicitly false
        if data.get("first_paragraph_has_indent") is not False:
            continue

        # ── Guard: no previous file ──────────────────────────────────────────
        if i == 0:
            print(f"[SKIP] {filepath.name}: first_paragraph_has_indent=false but no previous file exists.")
            continue

        prev_num, prev_path = files[i - 1]
        prev_data = load_json(prev_path)

        prev_paragraphs = prev_data.get("paragraphs", [])
        curr_paragraphs = data.get("paragraphs", [])

        # ── Guard: previous file has no paragraphs ───────────────────────────
        if not prev_paragraphs:
            print(f"[SKIP] {filepath.name}: {prev_path.name} has no paragraphs.")
            continue

        # ── Guard: current file has no paragraphs ────────────────────────────
        if not curr_paragraphs:
            print(f"[SKIP] {filepath.name}: paragraphs list is empty.")
            continue

        last_para = prev_paragraphs[-1]

        # ── Guard: last paragraph in previous file is not type "paragraph" ───
        if last_para.get("type") != "paragraph":
            print(
                f"[SKIP] {filepath.name}: last element in {prev_path.name} "
                f"has type='{last_para.get('type')}', expected 'paragraph'."
            )
            continue

        # ── Merge ────────────────────────────────────────────────────────────
        first_content = curr_paragraphs[0].get("content", "")
        last_para["content"] += first_content

        # Remove the merged paragraph from the current file
        data["paragraphs"] = curr_paragraphs[1:]

        # Mark as merged so this file won't be re-processed
        data["first_paragraph_has_indent"] = True

        save_json(prev_path, prev_data)
        save_json(filepath, data)

        print(
            f"[MERGED] {filepath.name} first paragraph → appended to "
            f"last paragraph of {prev_path.name}, removed from {filepath.name}."
        )


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python merge_paragraphs.py <directory>")
        sys.exit(1)

    directory = Path(sys.argv[1])

    if not directory.is_dir():
        print(f"[ERROR] '{directory}' is not a valid directory.")
        sys.exit(1)

    process(directory)


if __name__ == "__main__":
    main()
