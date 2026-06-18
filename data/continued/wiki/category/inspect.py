import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

TMP_DIR = Path(__file__).parent / "tmp"


@dataclass
class Category:
    lang: str
    name: str
    depth: int


HEADERS = {
    "User-Agent": "WarringStatesBot/1.0 (https://github.com/kakarot/transformer-decoder; kakarotter7@gmail@gmail.com) python-requests/2.32.0"
}

ROOT_CATEGORIES = {
    "ja": [
        # Category(lang="ja", name="治承・寿永の乱", depth=1),
        # Category(lang="ja", name="鎌倉時代", depth=1),
        # Category(lang="ja", name="南北朝時代の人物 (日本)", depth=2),
        # Category(lang="ja", name="室町時代", depth=2),
        # Category(lang="ja", name="安土桃山時代", depth=3),
        # Category(lang="ja", name="戦国時代_(日本)", depth=3),
        Category(lang="ja", name="戦国大名", depth=3),
        # Category(lang="ja", name="戦国武将", depth=3),
    ]
}


def build_session(total_retries: int = 5, backoff_factor: float = 1.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(HEADERS)
    return session


def inspect_category_tree(category: Category, delay: float = 1.5) -> list[str]:
    """遍历分类树，返回所有发现的分类名称列表。"""
    session = build_session()
    visited: set[str] = set()
    queue: deque[tuple[Category, int]] = deque([(category, 0)])

    while queue:
        cat, depth = queue.popleft()
        if cat.name in visited:
            continue
        visited.add(cat.name)
        print(f"{'  ' * depth}[{depth}] {cat.name}")

        if depth >= cat.depth:
            continue

        cmcontinue = None
        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{cat.name}",
                "cmlimit": 500,
                "cmtype": "subcat",
                "format": "json",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue

            resp = session.get(
                f"https://{cat.lang}.wikipedia.org/w/api.php",
                params=params,
                timeout=100,
            )
            data = resp.json()

            for member in data["query"]["categorymembers"]:
                subcat = member["title"].removeprefix("Category:")
                if subcat not in visited:
                    queue.append((Category(lang=cat.lang, name=subcat, depth=cat.depth), depth + 1))

            cmcontinue = data.get("continue", {}).get("cmcontinue")
            time.sleep(delay)
            if not cmcontinue:
                break

    return list(visited)


if __name__ == "__main__":
    TMP_DIR.mkdir(exist_ok=True)

    for lang, categories in ROOT_CATEGORIES.items():
        collected: set[str] = set()
        for category in categories:
            names = inspect_category_tree(category)
            collected.update(names)

        out_file = TMP_DIR / f"{lang}_categories.json"
        out_file.write_text(
            json.dumps(sorted(collected), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n✓ {lang}: {len(collected)} 个分类 → {out_file}")
