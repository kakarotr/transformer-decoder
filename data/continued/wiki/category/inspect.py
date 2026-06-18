import time
from collections import deque
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry


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
    """带重试和指数退避的 Session，专门应对 SSL/读取错误。"""
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,  # 覆盖 SSLEOFError / UNEXPECTED_EOF
        backoff_factor=backoff_factor,  # 等待时间：1.5s、3s、6s、12s、24s
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(HEADERS)
    return session


def inspect_category_tree(category: Category, delay: float = 1.5):
    session = build_session()
    visited: set[str] = set()
    queue: deque[tuple[Category, int]] = deque([(category, 0)])

    while queue:
        cat, depth = queue.popleft()
        if cat.name in visited:
            continue
        visited.add(cat.name)
        print(f"{'  ' * depth}[{depth}] {cat}")

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
            time.sleep(delay)  # ← 移到 break 判断之前，每次请求后必然执行
            if not cmcontinue:
                break


for item in ROOT_CATEGORIES.items():
    for category in item[1]:
        inspect_category_tree(category)
