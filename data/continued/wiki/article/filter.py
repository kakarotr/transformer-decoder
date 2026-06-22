import asyncio
import json
import os

from openai import AsyncOpenAI
from peewee import JOIN, SQL
from tqdm.asyncio import tqdm

from data.continued.utils import get_model
from data.continued.wiki.db import WikiArticleCategories, WikiArticles

base_url, api_key, model = get_model(provider="DEEPSEEK")
client = AsyncOpenAI(base_url=base_url, api_key=api_key)

prompt = """你是日本中世史专家，判断维基百科条目是否与日本中世史（1180–1615年）直接相关。

保留：武将・大名・政治家、合戦・政治事件、幕府・大名家、城郭・军事据点、
      经济・农业・商业、建造物、对目标时期有重大作用的宗教势力

排除：纯文化艺术人物（无政治关联）、纯宗教教义、地方民俗・祭事、
      与重要事件无关的地方小寺社、现代媒体作品、江户中期以后内容

只输出 JSON，格式：{"results": [{"title": "...", "keep": true}]}"""


async def classify_batch(
    articles: list[dict],
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """articles: [{"title": str, "categories": list[str]}]"""
    lines = [f"{a['title']} | {', '.join(a['categories'][:5])}" for a in articles]
    user_content = "\n".join(lines)

    async with semaphore:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            reasoning_effort="medium",
            extra_body={"thinking": {"type": "enabled"}},
            response_format={"type": "json_object"},
        )

    return json.loads(resp.choices[0].message.content)["results"]  # type: ignore


async def classify_all(batch_size: int = 50, concurrency: int = 10):
    query = (
        WikiArticles.select(
            WikiArticles.title,
            WikiArticles.lang,
            SQL('STRING_AGG("category", \',\' ORDER BY "category")').alias("categories"),
        )
        .join(
            WikiArticleCategories,
            JOIN.LEFT_OUTER,
            on=(
                (WikiArticles.title == WikiArticleCategories.title) & (WikiArticles.lang == WikiArticleCategories.lang)
            ),
        )
        .where(WikiArticles.stage == "pending")
        .group_by(WikiArticles.title, WikiArticles.lang)
    )

    articles = [{"title": row.title, "categories": "|".join(row.categories.split(","))} for row in query.namedtuples()]

    batches = [articles[i : i + batch_size] for i in range(0, len(articles), batch_size)]

    async def classify_and_save(batch: list[dict], pbar: tqdm) -> None:
        try:
            batch_result = await classify_batch(batch, semaphore)
        except Exception as e:
            print(f"Batch error: {e}")
            return
        finally:
            pbar.update(1)

        skip_titles = [item["title"] for item in batch_result if not item["keep"]]  # type: ignore
        keep_titles = [item["title"] for item in batch_result if item["keep"]]  # type: ignore
        if skip_titles:
            WikiArticles.update(stage="skip").where(WikiArticles.title.in_(skip_titles)).execute()
        if keep_titles:
            WikiArticles.update(stage="queued").where(WikiArticles.title.in_(keep_titles)).execute()

    semaphore = asyncio.Semaphore(concurrency)
    with tqdm(total=len(batches), desc="Classifying", unit="batch") as pbar:
        await asyncio.gather(*[classify_and_save(b, pbar) for b in batches])


if __name__ == "__main__":
    asyncio.run(classify_all())
