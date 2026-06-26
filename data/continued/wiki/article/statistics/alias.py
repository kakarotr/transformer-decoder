import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from data.continued.wiki.db import WikiAliases, WikiArticles

console = Console()
write_lock = threading.Lock()
_thread_local = threading.local()

MAX_WORKERS = 8
USER_AGENT = "WarringStatesBot/1.0 (https://github.com/kakarot/transformer-decoder; kakarotter7@gmail.com) python-requests/2.32.0"


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
        _thread_local.session = session
    return _thread_local.session


def process_article(title: str, lang: str) -> tuple[str, str, str | None]:
    try:
        response = get_session().get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "titles": title,
                "prop": "info",
                "redirects": 1,
                "format": "json",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        redirects = data["query"].get("redirects", [])

        with write_lock:
            if redirects:
                canonical = list(data["query"]["pages"].values())[0]["title"]
                tofragment = redirects[0].get("tofragment", None)

                WikiArticles.update(
                    is_redirect=1,
                    redirect_to=canonical,
                    redirect_has_anchor=int(tofragment is not None),
                ).where((WikiArticles.title == title) & (WikiArticles.lang == lang)).execute()

                if tofragment is None:
                    WikiAliases.insert({"alias_title": title, "canonical_title": canonical, "lang": lang}).execute()
            else:
                WikiArticles.update(is_redirect=0).where(
                    (WikiArticles.title == title) & (WikiArticles.lang == lang)
                ).execute()

        return title, lang, None

    except Exception as e:
        return title, lang, str(e)


if __name__ == "__main__":
    articles = [
        (a.title, a.lang)
        for a in WikiArticles.select(WikiArticles.title, WikiArticles.lang).where(
            (WikiArticles.stage == "fetched") & (WikiArticles.is_redirect.is_null())
        )
    ]

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("checking redirects", total=len(articles))

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_article, title, lang): (title, lang) for title, lang in articles}

            for future in as_completed(futures):
                title, lang, error = future.result()
                if error:
                    console.print(f"[red][error] {title} ({lang}): {error}[/red]")
                progress.advance(task)
