import json
from pathlib import Path

from data.continued.wiki.db import WikiCategories

TMP_DIR = Path(__file__).parent / "tmp"
LANG = "ja"

if __name__ == "__main__":
    in_file = TMP_DIR / f"{LANG}_classify.json"
    data = json.loads(in_file.read_text(encoding="utf-8"))

    categories = data.get("include", [])
    if not categories:
        print("没有 include 分类，跳过。")
    else:
        WikiCategories.insert_many([{"name": name, "lang": LANG} for name in categories]).execute()
        print(f"✓ 已插入 {len(categories)} 个分类")
