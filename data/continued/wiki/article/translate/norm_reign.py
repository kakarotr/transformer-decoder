import json
import re
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from data.continued.paths import WIKI_ERA, WIKI_FUSED
from data.continued.wiki.article.structure import WikiArticle

_CN_DIGITS = "零一二三四五六七八九"


def _int_to_cn(n: int) -> str:
    if n < 10:
        return _CN_DIGITS[n]
    if n < 20:
        return "十" + ("" if n == 10 else _CN_DIGITS[n % 10])
    tens, ones = divmod(n, 10)
    return _CN_DIGITS[tens] + "十" + ("" if ones == 0 else _CN_DIGITS[ones])


_FW = str.maketrans("０１２３４５６７８９", "0123456789")


def _to_int(s: str) -> int:
    return int(s.translate(_FW))


_D = r"[0-9０-９]"  # 半角 + 全角数字


def build_date_normalizer(era_names, *, year_one_as_gen: bool = True):
    eras = sorted({e for e in era_names if e}, key=len, reverse=True)
    if not eras:
        raise ValueError("era_names 不能为空")
    era_alt = "|".join(re.escape(e) for e in eras)

    pattern = re.compile(
        # 1) 公历日期整段保护：4位年(+可选月)(+可选日) 全部保留阿拉伯，优先匹配
        rf"(?P<greg>(?<!{_D}){_D}{{4}}年(?:{_D}{{1,2}}月)?(?:{_D}{{1,2}}日)?)"
        # 2) 年号年（闸门）
        rf"|(?P<era>{era_alt})(?P<ey>{_D}{{1,2}})年"
        # 3) 月
        rf"|(?<!{_D})(?P<mon>{_D}{{1,2}})月"
        # 4) 日
        rf"|(?<!{_D})(?P<day>{_D}{{1,2}})日"
    )

    def _repl(m):
        if m.group("greg") is not None:
            return m.group(0)  # 公历日期：原样保留
        if m.group("era") is not None:
            n = _to_int(m.group("ey"))
            if not (1 <= n <= 99):
                return m.group(0)
            cn = "元" if (n == 1 and year_one_as_gen) else _int_to_cn(n)
            return f"{m.group('era')}{cn}年"
        if m.group("mon") is not None:
            n = _to_int(m.group("mon"))
            return f"{_int_to_cn(n)}月" if 1 <= n <= 12 else m.group(0)
        if m.group("day") is not None:
            n = _to_int(m.group("day"))
            return f"{_int_to_cn(n)}日" if 1 <= n <= 31 else m.group(0)
        return m.group(0)

    return lambda text: pattern.sub(_repl, text)


def transform_all_text_fields(obj: Any, normalizer: Callable) -> None:
    if isinstance(obj, BaseModel):
        for field_name in obj.__class__.model_fields:
            value = getattr(obj, field_name)

            if field_name == "text" and isinstance(value, str):
                setattr(obj, field_name, normalizer(value))
            else:
                transform_all_text_fields(value, normalizer)

    elif isinstance(obj, list):
        for item in obj:
            transform_all_text_fields(item, normalizer)

    elif isinstance(obj, tuple):
        for item in obj:
            transform_all_text_fields(item, normalizer)

    elif isinstance(obj, dict):
        for value in obj.values():
            transform_all_text_fields(value, normalizer)


if __name__ == "__main__":
    eras: list[str] = json.loads(Path("data/continued/wiki/article/translate/eras.json").read_text())
    normalizer = build_date_normalizer(eras)
    for file in Path(WIKI_FUSED).glob("*.json"):
        article = WikiArticle.model_validate_json(file.read_text())
        transform_all_text_fields(article, normalizer)
        output = Path(WIKI_ERA / f"{article.title}.json")
        output.write_text(article.model_dump_json(indent=2))
