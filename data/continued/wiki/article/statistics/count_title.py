from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from data.continued.paths import WIKI_CLEANED_HTML


def process_html(html_path: Path) -> list[tuple[str, str]]:
    document = BeautifulSoup(html_path.read_text(), "html.parser")
    stem = html_path.stem
    return [(h2.get_text(strip=True), stem) for h2 in document.find_all("h2")]


def count_title():
    path = WIKI_CLEANED_HTML
    html_files = list(path.glob("*.html"))

    title_counts: dict[str, int] = defaultdict(int)
    title_files: dict[str, set[str]] = defaultdict(set)

    with ProcessPoolExecutor() as executor:
        for results in executor.map(process_html, html_files):
            for title, stem in results:
                title_counts[title] += 1
                title_files[title].add(stem)

    rows = []
    for title, count in sorted(title_counts.items(), key=lambda x: x[1], reverse=True):
        files = sorted(title_files[title])
        if count <= 10:
            files_str = " | ".join(files)
        else:
            files_str = " | ".join(files[:5])
        rows.append({"h2标题": title, "文件名": files_str, "数量": count})

    df = pd.DataFrame(rows, columns=["h2标题", "文件名", "数量"])
    df.to_excel("count_title.xlsx", index=False)


if __name__ == "__main__":
    path = WIKI_CLEANED_HTML
    count = 0
    for html in list(path.glob("*.html")):
        if html.stem in {}:
            pass
