import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

from data.continued.paths import WIKI_CLEANED_HTML
from data.continued.wiki.article.cleaner import WikiCleaner
from data.continued.wiki.db import WikiArticles


def fetch_single(name: str, lang: str):
    try:
        response = requests.get(
            f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{name}",
            headers={
                "User-Agent": "WarringStatesBot/1.0 (https://github.com/kakarot/transformer-decoder; kakarotter7@gmail@gmail.com) python-requests/2.32.0"
            },
        )
        response.raise_for_status()

        cleaner = WikiCleaner(content=response.text)
        cleaned_content = cleaner.clean()
        safe_name = name.replace("/", "_")
        with open(WIKI_CLEANED_HTML / f"{safe_name}.html", mode="w", encoding="utf-8") as f:
            f.write(cleaned_content)
        WikiArticles.update(stage="fetched").where((WikiArticles.title == name) & (WikiArticles.lang == lang)).execute()
        print(f"✅ {name}")
    except Exception as e:
        print(f"❌ {name} 获取失败 {str(e)}")
        WikiArticles.update(stage="fetch_failed").where(
            (WikiArticles.title == name) & (WikiArticles.lang == lang)
        ).execute()


def fetch_all(lang: str, workers: int = 8):
    queued_titles: list[str] = [
        row[0]
        for row in WikiArticles.select(WikiArticles.title)
        .where((WikiArticles.lang == lang) & (WikiArticles.stage.in_(["queued", "fetch_failed"])))
        .tuples()
    ]

    success = 0
    failed = 0
    lock = threading.Lock()

    def _fetch_one(name: str) -> None:
        response = requests.get(
            f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{name}",
            headers={
                "User-Agent": "WarringStatesBot/1.0 (https://github.com/kakarot/transformer-decoder; kakarotter7@gmail@gmail.com) python-requests/2.32.0"
            },
        )
        response.raise_for_status()
        cleaner = WikiCleaner(content=response.text)
        cleaned_content = cleaner.clean()
        safe_name = name.replace("/", "_")
        with open(WIKI_CLEANED_HTML / f"{safe_name}.html", mode="w", encoding="utf-8") as f:
            f.write(cleaned_content)
        WikiArticles.update(stage="fetched").where(
            (WikiArticles.title == name) & (WikiArticles.lang == lang)
        ).execute()

    with tqdm(total=len(queued_titles), unit="篇", dynamic_ncols=True) as pbar:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fetch_one, name): name for name in queued_titles}
            for future in as_completed(futures):
                name = futures[future]
                with lock:
                    try:
                        future.result()
                        success += 1
                    except Exception as e:
                        tqdm.write(f"❌ {name}: {e}")
                        WikiArticles.update(stage="fetch_failed").where(
                            (WikiArticles.title == name) & (WikiArticles.lang == lang)
                        ).execute()
                        failed += 1
                    pbar.update(1)
                    pbar.set_postfix({"✅": success, "❌": failed}, refresh=False)


if __name__ == "__main__":
    load_dotenv()

    fetch_all(lang="ja")
