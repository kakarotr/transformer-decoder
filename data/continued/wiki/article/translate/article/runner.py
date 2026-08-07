"""并发执行与按篇幂等。

失败语义：一篇文章内任何一批最终失败 → 整篇不回填、不入库、不标记，
下次直接重跑整篇。因为回填发生在全部批次成功之后，
失败时 WikiArticle 对象保持原样，不存在半翻译状态。
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Awaitable, Callable

from . import protocol  # 模块级引用：钩子可在运行时替换，不被 import 绑死
from .protocol import ProtocolError, build_schema, build_user_message, parse_response
from .slots import Batch, collect_units, make_batches

# 调用方提供：接收 (system, user, json_schema)，返回模型输出的原始字符串。
CallFn = Callable[[str, str, dict], Awaitable[str]]


async def _run_batch(
    batch: Batch,
    call: CallFn,
    system: str,
    sem: asyncio.Semaphore,
    max_retries: int,
    title_ja: str,
    title_zh: str,
) -> dict[str, str]:
    user = build_user_message(batch, title_ja=title_ja, title_zh=title_zh)
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            async with sem:
                raw = await call(system, user, build_schema(batch))
            return parse_response(raw, batch)
        except (ProtocolError, asyncio.TimeoutError, OSError) as exc:
            last = exc
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt + random.random())
    raise RuntimeError(f"批次失败（{len(batch.slots)} slots）: {last}") from last


async def translate_article(
    article: Any,
    *,
    call: CallFn,
    system: str,
    sem: asyncio.Semaphore,
    title: str,
    max_retries: int = 3,
    max_slots: int = 40,
    max_chars: int = 6000,
) -> int:
    """原地翻译一篇文章，返回改写的 slot 数。失败则抛异常且不改动 article。

    中文标题经 lookup_title 取得，只作为 prompt 上下文注入；
    全批成功后写回 article.title。查不到译名则保持日文原标题。
    """
    title_zh = protocol.lookup_title(title) or ""

    units = collect_units(article)
    batches = make_batches(units, max_slots=max_slots, max_chars=max_chars)
    if not batches:
        return 0

    results = await asyncio.gather(
        *(_run_batch(b, call, system, sem, max_retries, title, title_zh) for b in batches)
    )

    written = 0
    for batch, mapping in zip(batches, results):
        for slot in batch.slots:
            slot.write(mapping[slot.sid])
            written += 1
    if title_zh:
        article.title = title_zh
    return written


async def run(
    titles: list[str],
    *,
    load: Callable[[Any], Any],
    save: Callable[[Any, Any], None],
    call: CallFn,
    system: str,
    request_concurrency: int,
    article_concurrency: int = 8,
    max_slots: int = 40,
    max_chars: int = 6000,
    on_error: Callable[[Any, Exception], None] | None = None,
) -> None:
    """load/save 由调用方接。save 只在整篇成功后被调用，内部负责置 translated=1。"""
    request_sem = asyncio.Semaphore(request_concurrency)
    article_sem = asyncio.Semaphore(article_concurrency)

    async def one(title: str) -> None:
        async with article_sem:
            article = await asyncio.to_thread(load, title)
            try:
                await translate_article(
                    article,
                    call=call,
                    system=system,
                    sem=request_sem,
                    title=title,
                    max_slots=max_slots,
                    max_chars=max_chars,
                )
            except Exception as exc:  # noqa: BLE001 — 单篇失败不影响其他篇
                if on_error:
                    on_error(title, exc)
                return
            await asyncio.to_thread(save, title, article)

    await asyncio.gather(*(one(title) for title in titles))
