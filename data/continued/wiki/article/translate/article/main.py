"""日文维基翻译入口（OpenAI 协议）。

用法：
    export OPENAI_API_KEY=...
    python main.py --db wiki.db --model gpt-4.1 --limit 20            # 官方端点，先标定
    python main.py --db wiki.db --base-url https://api.deepseek.com \
                   --model deepseek-chat --json-mode object           # 兼容端点

中断安全：Ctrl-C 后未 save 的文章 translated 仍为 0，重跑直接续上。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
import time
from pathlib import Path

from data.continued.paths import WIKI_ERA, WIKI_FUSED, WIKI_TRANSLATE
from data.continued.utils import get_model
from data.continued.wiki.article.structure import WikiArticle
from data.continued.wiki.db import WikiArticles

from .protocol import SYSTEM_PROMPT
from .runner import run

# ↓↓↓ 改成你的实际表名列名 ↓↓↓
TABLE = "articles"
ID_COL = "id"
PAYLOAD_COL = "payload"
FLAG_COL = "translated"


# ---------------------------------------------------------------- 存储层


def build_storage(lang: str):
    def pending(limit: int | None) -> list[str]:
        query = (
            WikiArticles.select(WikiArticles.title)
            .where(
                (WikiArticles.lang == lang)
                & (WikiArticles.stage.in_(["fused", "skip_fused"]))
                & (WikiArticles.is_redirect == 0)
                & (WikiArticles.title == "織田信長")
            )
            .limit(limit)
        ).execute()
        return [row.title for row in query]

    def load(title: str) -> WikiArticle | None:
        file = Path(f"{WIKI_ERA}/{title}.json")
        if file.exists():
            return WikiArticle.model_validate_json(file.read_text())

    def save(title: str, article: WikiArticle) -> None:
        file = Path(f"{WIKI_TRANSLATE}/{title}.json")
        file.write_text(article.model_dump_json(indent=2))
        WikiArticles.update(stage="translated").where(
            (WikiArticles.title == title) & (WikiArticles.lang == lang)
        ).execute()

    return pending, load, save


# ---------------------------------------------------------------- provider


def build_call(args: argparse.Namespace):
    from openai import AsyncOpenAI

    base_url, api_key, _ = get_model(args.provider)

    client = AsyncOpenAI(base_url=base_url, max_retries=5, timeout=args.timeout, api_key=api_key)

    async def call(system: str, user: str, schema: dict) -> str:
        # if args.json_mode == "schema":
        #     # strict 模式下 schema 本身强制 key 集合，模型无法漏 key 或多 key
        #     response_format = {
        #         "type": "json_schema",
        #         "json_schema": {"name": "translations", "strict": True, "schema": schema},
        #     }
        # else:
        #     # 兼容端点常见：只保证输出是合法 JSON，key 集合由 parse_response 校验
        #     response_format = {"type": "json_object"}

        print(system)
        print(user)
        # resp = await client.chat.completions.create(
        #     model=args.model,
        #     messages=[
        #         {"role": "system", "content": system},
        #         {"role": "user", "content": user},
        #     ],
        #     response_format={"type": "json_object"},
        #     temperature=args.temperature,
        #     max_tokens=args.max_tokens,
        # )
        # choice = resp.choices[0]
        # if choice.finish_reason == "length":
        #     raise OSError("输出被截断，调小 --max-chars 或调大 --max-tokens")
        # return choice.message.content
        return ""

    return call


# ---------------------------------------------------------------- 主流程


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--provider", default="doubao", help="模型供应商")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 篇，用于标定")
    p.add_argument("--request-concurrency", type=int, default=16, help="按 API rate limit 设")
    p.add_argument("--article-concurrency", type=int, default=8)
    p.add_argument("--max-slots", type=int, default=40)
    p.add_argument("--max-chars", type=int, default=6000)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args()

    pending, load, save = build_storage(lang="ja")
    titles = pending(args.limit)
    if not titles:
        print("没有待翻译文章")
        return 0

    call = build_call(args)
    total, done, failed, start = len(titles), 0, 0, time.perf_counter()
    print(f"待翻译 {total} 篇 | {args.provider} @ {args.model or 'openai'} | 请求并发 {args.request_concurrency}")

    def save_and_report(title: str, article: WikiArticle) -> None:
        nonlocal done
        save(title, article)
        done += 1
        elapsed = time.perf_counter() - start
        print(f"[{done + failed}/{total}] ok title={title} | {elapsed / (done + failed):.1f}s/篇", flush=True)

    def on_error(aid: int, exc: Exception) -> None:
        nonlocal failed
        failed += 1
        print(f"[{done + failed}/{total}] FAIL id={aid}: {exc}", file=sys.stderr, flush=True)

    try:
        asyncio.run(
            run(
                titles,
                load=load,
                save=save_and_report,
                call=call,
                system=SYSTEM_PROMPT,
                request_concurrency=args.request_concurrency,
                article_concurrency=args.article_concurrency,
                max_slots=args.max_slots,
                max_chars=args.max_chars,
                on_error=on_error,
            )
        )
    except KeyboardInterrupt:
        print("\n已中断，未完成的文章 translated 仍为 0", file=sys.stderr)

    print(f"完成 {done} 失败 {failed} 耗时 {time.perf_counter() - start:.0f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
