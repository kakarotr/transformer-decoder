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

if __name__ == "__main__":
    fetched_articles: set[str] = {row[0] for row in WikiArticles.select(WikiArticles.title).tuples()}
    categories = list(WikiCategories.select().where(WikiCategories.status == "pending"))

    for category in categories:
        category_name = category.name
        lang = category.lang

        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category_name}",
            "cmlimit": 500,
            "cmtype": "page",
            "format": "json",
        }
        headers = {
            "User-Agent": "WarringStatesBot/1.0 (https://github.com/kakarot/transformer-decoder; kakarotter7@gmail@gmail.com) python-requests/2.32.0"
        }

        articles: list[dict[str, Any]] = []
        article_categories: list[dict[str, Any]] = []

        try:
            while True:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                response_data = response.json()

                members = response_data["query"]["categorymembers"]
                for member in members:
                    title: str = member["title"]
                    if member["ns"] == 0:
                        article_categories.append({"title": title, "category": category_name, "lang": lang})

                        if title not in fetched_articles:
                            fetched_articles.add(title)
                            articles.append({"title": title, "lang": lang, "stage": "pending"})

                if "continue" not in response_data:
                    break

                params["cmcontinue"] = response_data["continue"]["cmcontinue"]
                time.sleep(0.5)
        except (requests.RequestException, KeyError) as e:
            print(f"[SKIP] {lang}:{category_name} - {e}")
            continue

        print(f"✅ 分类: {category_name}")

        try:
            with db.atomic():
                if article_categories:
                    WikiArticleCategories.insert_many(article_categories).execute()
                if articles:
                    WikiArticles.insert_many(articles).execute()
                WikiCategories.update(status="complete").where(
                    (WikiCategories.name == category_name) & (WikiCategories.lang == lang)
                ).execute()
        except DatabaseError as e:
            print(f"[DB ERROR] {lang}:{category_name} - {e}")
