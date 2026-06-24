import time
from typing import Any

import requests
from psycopg2 import DatabaseError

from data.continued.wiki.db import (
    WikiArticleCategories,
    WikiArticles,
    WikiCategories,
    db,
)

_HEADERS = {
    "User-Agent": "WarringStatesBot/1.0 (https://github.com/kakarot/transformer-decoder; kakarotter7@gmail@gmail.com) python-requests/2.32.0"
}


def insert_by_category(lang: str, category_name: str) -> None:
    WikiCategories.insert(name=category_name, lang=lang, status="pending").on_conflict_ignore().execute()

    params: dict[str, Any] = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category_name}",
        "cmlimit": 500,
        "cmtype": "page",
        "format": "json",
    }

    existing_titles: set[str] = {
        row[0] for row in WikiArticles.select(WikiArticles.title).where(WikiArticles.lang == lang).tuples()
    }
    existing_article_categories: set[str] = {
        row[0]
        for row in WikiArticleCategories.select(WikiArticleCategories.title)
        .where((WikiArticleCategories.lang == lang) & (WikiArticleCategories.category == category_name))
        .tuples()
    }

    articles: list[dict[str, Any]] = []
    article_categories: list[dict[str, Any]] = []

    try:
        while True:
            response = requests.get(f"https://{lang}.wikipedia.org/w/api.php", headers=_HEADERS, params=params)
            response.raise_for_status()
            data = response.json()

            for member in data["query"]["categorymembers"]:
                if member["ns"] != 0:
                    continue
                title: str = member["title"]
                if title not in existing_article_categories:
                    existing_article_categories.add(title)
                    article_categories.append({"title": title, "category": category_name, "lang": lang})
                if title not in existing_titles:
                    existing_titles.add(title)
                    articles.append({"title": title, "lang": lang, "stage": "pending"})

            if "continue" not in data:
                break

            params["cmcontinue"] = data["continue"]["cmcontinue"]
            time.sleep(0.5)
    except (requests.RequestException, KeyError) as e:
        print(f"[SKIP] {lang}:{category_name} - {e}")
        return

    try:
        with db.atomic():
            if article_categories:
                WikiArticleCategories.insert_many(article_categories).on_conflict_ignore().execute()
            if articles:
                WikiArticles.insert_many(articles).on_conflict_ignore().execute()
            WikiCategories.update(status="complete").where(
                (WikiCategories.name == category_name) & (WikiCategories.lang == lang)
            ).execute()
    except DatabaseError as e:
        print(f"[DB ERROR] {lang}:{category_name} - {e}")
    else:
        print(f"✅ {lang}:{category_name} - {len(articles)} 篇文章")


if __name__ == "__main__":
    # for category in WikiCategories.select().where(WikiCategories.status == "pending"):
    #     insert_by_category(category.lang, category.name)
    insert_by_category("ja", "日本の旧国名")
