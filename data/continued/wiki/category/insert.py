import json

from data.continued.wiki.db import WikiCategories

if __name__ == "__main__":
    with open("data/continued/wiki/category/tmp/ja_classify_category.json", mode="r", encoding="utf-8") as f:
        data = json.load(f)

    WikiCategories.insert_many([{"name": category, "lang": "ja"} for category in data["include"]]).execute()
