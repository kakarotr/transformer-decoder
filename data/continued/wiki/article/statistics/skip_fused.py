import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from data.continued.paths import WIKI_PARSED
from data.continued.wiki.db import WikiArticles


def _get_stem_if_no_infobox(item: Path) -> str | None:
    # 跳过 Pydantic，只检查 infobox 键，read_bytes 省去编解码
    return item.stem if json.loads(item.read_bytes()).get("infobox") is None else None


if __name__ == "__main__":
    path = WIKI_PARSED
    items = list(path.glob("*.json"))

    # IO 密集型任务，ThreadPoolExecutor 直接用默认 worker 数
    with ThreadPoolExecutor() as executor:
        titles_to_skip = [t for t in executor.map(_get_stem_if_no_infobox, items) if t is not None]

    if titles_to_skip:
        WikiArticles.update(stage="skip_fused").where(
            WikiArticles.title.in_(titles_to_skip) & (WikiArticles.lang == "ja")
        ).execute()
