import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from data.continued.paths import WIKI_FUSED, WIKI_TRANSLATE
from data.continued.utils import get_model
from data.continued.wiki.article.translate.prompt import translate_title_prompt

base_url, api_key, _ = get_model(provider="deepseek")

BATCH_SIZE = 100
MAX_WORKERS = 8
MAX_ATTEMPTS = 3
PROGRESS_EVERY_BATCHES = 1
OUTPUT_PATH = WIKI_TRANSLATE / "titles.jsonl"
TRANSLATE_TITLE_BATCH_PROMPT = f"""{translate_title_prompt}

注意：本次接口要求最外层为 JSON 对象，因此请输出以下格式：
{{"items": [...]}}

items 数组中的每个对象仍须严格遵守上文对输出对象的字段要求。
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

write_lock = threading.Lock()
thread_local = threading.local()


def get_client() -> OpenAI:
    client = getattr(thread_local, "client", None)
    if client is None:
        client = OpenAI(base_url=base_url, api_key=api_key)
        thread_local.client = client
    return client


def chunks(items: list[dict], size: int) -> list[list[dict]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def load_done_titles(path: Path) -> set[str]:
    if not path.exists():
        return set()

    done: set[str] = set()
    with open(path, mode="r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("%s:%d 不是合法 JSON，已忽略", path, line_no)
                continue
            ja_title = item.get("ja_title")
            if isinstance(ja_title, str):
                done.add(ja_title)
    return done


def parse_response(content: str) -> list[dict]:
    data = json.loads(content)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("items", "titles", "translations", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("模型返回 JSON 不是数组，也不是包含数组字段的对象")


def validate_results(batch: list[dict], results: list[dict]) -> list[dict]:
    if len(results) != len(batch):
        raise ValueError(f"返回数量不一致：输入 {len(batch)}，输出 {len(results)}")

    expected_ids = [item["id"] for item in batch]
    result_ids = [item.get("id") for item in results]
    if result_ids != expected_ids:
        raise ValueError(f"id 顺序不一致：expected={expected_ids} actual={result_ids}")

    required_keys = {"id", "ja_title", "zh_title", "confidence", "note"}
    normalized: list[dict] = []
    for input_item, result in zip(batch, results):
        if not isinstance(result, dict):
            raise ValueError(f"{input_item['ja_title']}: 返回条目不是对象")
        missing = required_keys - result.keys()
        if missing:
            raise ValueError(f"{input_item['ja_title']}: 返回缺少字段 {sorted(missing)}")
        if result["ja_title"] != input_item["ja_title"]:
            raise ValueError(
                f"id={input_item['id']}: ja_title 不一致："
                f"expected={input_item['ja_title']} actual={result['ja_title']}"
            )
        normalized.append({key: result[key] for key in ("id", "ja_title", "zh_title", "confidence", "note")})
    return normalized


def append_results(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with write_lock:
        with open(path, mode="a", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def translate_batch(batch_index: int, batch: list[dict]) -> tuple[int, int]:
    if len(batch) > BATCH_SIZE:
        raise ValueError(f"batch={batch_index} 包含 {len(batch)} 条，超过单次翻译上限 {BATCH_SIZE}")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = get_client().chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": TRANSLATE_TITLE_BATCH_PROMPT},
                    {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("API 返回空内容")

            results = validate_results(batch, parse_response(content))
            append_results(OUTPUT_PATH, results)
            return len(results), len(batch)
        except Exception:
            if attempt == MAX_ATTEMPTS:
                logger.exception("batch=%d 处理失败，已放弃", batch_index)
                return 0, len(batch)
            logger.exception("batch=%d 处理失败，准备重试（%d/%d）", batch_index, attempt, MAX_ATTEMPTS)
            time.sleep(2**attempt)

    return 0, len(batch)

if __name__ == "__main__":
    ja_titles = sorted(file.stem for file in Path(WIKI_FUSED).glob("*.json"))
    entities = [{"id": index, "ja_title": title} for index, title in enumerate(ja_titles)]

    done_titles = load_done_titles(OUTPUT_PATH)
    pending_entities = [item for item in entities if item["ja_title"] not in done_titles]
    batches = chunks(pending_entities, BATCH_SIZE)

    logger.info(
        "标题总数：%d，已存在：%d，待翻译：%d，批大小：%d，并发数：%d",
        len(entities),
        len(done_titles),
        len(pending_entities),
        BATCH_SIZE,
        MAX_WORKERS,
    )

    if not batches:
        logger.info("没有待翻译标题")
        raise SystemExit(0)

    total_success = 0
    total_processed = 0
    total_pending = len(pending_entities)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(translate_batch, batch_index, batch): batch_index
            for batch_index, batch in enumerate(batches, start=1)
        }
        for finished, future in enumerate(as_completed(futures), start=1):
            success_count, processed_count = future.result()
            total_success += success_count
            total_processed += processed_count
            if finished % PROGRESS_EVERY_BATCHES == 0 or finished == len(futures):
                failed_count = total_processed - total_success
                logger.info(
                    "进度：批次 %d/%d (%.1f%%)，标题 %d/%d (%.1f%%)，成功 %d，失败 %d",
                    finished,
                    len(futures),
                    finished / len(futures) * 100,
                    total_processed,
                    total_pending,
                    total_processed / total_pending * 100,
                    total_success,
                    failed_count,
                )
