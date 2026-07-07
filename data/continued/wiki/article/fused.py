import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from peewee import InterfaceError, OperationalError
from pydantic import BaseModel, Field

from data.continued.paths import WIKI_FUSED, WIKI_PARSED
from data.continued.utils import get_model
from data.continued.wiki.article.structure import Paragraph, WikiArticle
from data.continued.wiki.db import WikiArticles, db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("infobox_fusion.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

db_lock = threading.Lock()

MAX_WORKERS = 16

MAX_TOKENS = 8000

DB_MAX_ATTEMPTS = 3


class Paragraphs(BaseModel):
    texts: list[str] = Field(description="段落列表")


prompt = """あなたは編集アシスタントです。Wikipedia記事のタイトル、infobox、
リード文が与えられます。infoboxの中から、主題の理解に実質的な
価値のある情報をリード文に自然に融合してください。

共通ルール：
- リード文の既存の文言を書き換えない。語順の変更、表現の言い換え、
  句読点の変更もしない。情報の挿入のみ行う
- リード文に既に含まれている情報は繰り返さない
- リード文の元の段落構成を維持すること。段落の統合や削除はしない。
  情報の追加により段落を増やすことは許容する
- 画像、テンプレートメタデータ、墓所、戒名、神号などの
  補助的フィールドは除外する
- infoboxで「→」を使って変遷を示すフィールドは、
  自然な叙述に変換し、矢印や読点の羅列をそのまま残さない
- infoboxとリード文に含まれない情報を補足しない
- 出力は日本語

人物記事：
以下のフィールドはinfoboxに存在する場合、必ず融合すること。
省略は許容しない：
- 生没年
- 氏族
- 主君：重要な数名を選び、自然な叙述でつなぐ。つなぎはinfoboxとリード文に
  明記されている情報のみを用いること（記事に明記されていない歴史的経緯を
  外部知識で補わない）。全ての主君を機械的に列挙しない
- 改名：幼名と広く知られた名を残し、改名の全経緯を逐一列挙しない
- 官位：生前に任じられた官位のうち、最高位または最も代表的なもののみ残す。
  贈位（贈正一位など死後に追贈されたもの）は除外する
- 父母・配偶者・子：著名な人物を必ず含める。
  完全なリストをそのまま転記しない
- 兄弟：著名な人物のみ残す

合戦記事：
- 時期・場所・結果は必ず残す
- 交戦双方は指揮官のみ残し、参戦武将を全て列挙しない
- 兵力・損害の数値は明確な場合のみ残す

その他：
- 上記の方針に準じて、実質的な情報価値のあるフィールドを判断する

出力形式：
以下のJSON形式で出力してください。他のテキストは一切含めないでください。
{{
  "texts": ["段落1", "段落2", ...]
}}
textsには融合後のリード文を段落単位で格納してください。

記事タイトル：{title}

infobox：
{infobox_fields}

リード文：
{lead_text}

以下に融合の例を示します。

例：
記事タイトル：徳川家康

infobox：
時代: 戦国時代 - 江戸時代前期
生誕: 天文11年12月26日（1543年1月31日）
死没: 元和2年4月17日（1616年6月1日）
改名: 竹千代（幼名）→ 松平元信 → 松平元康 → 松平家康 → 徳川家康
主君: 今川義元 → 今川氏真 → 足利義昭 → 織田信長 → 豊臣秀吉 → 豊臣秀頼
氏族: 三河松平氏 → 徳川氏
官位: 贈正一位、太政大臣、征夷大将軍
父母: 父：松平広忠、母：於大の方
妻: 正室：築山殿、継室：朝日姫
子: 信康、秀康、秀忠、忠吉、信吉、忠輝、松千代、義直、頼宣、頼房

リード文：
徳川家康は、日本の戦国時代から江戸時代初期にかけての武将・大名。戦国の三英傑の一人。

三河国の土豪・松平氏に生まれ、幼少期は今川氏の人質として過ごした。桶狭間の戦い後に独立し、織田信長と同盟を結んで勢力を拡大した。信長没後は豊臣秀吉と対立・臣従を経て、関ヶ原の戦いに勝利した。

出力：
{{
  "texts": [
    "徳川家康は、日本の戦国時代から江戸時代初期にかけての武将・大名。戦国の三英傑の一人。天文11年（1543年）に三河松平氏の当主・松平広忠の子として生まれ、幼名は竹千代。",
    "三河国の土豪・松平氏に生まれ、幼少期は今川義元の人質として過ごした。桶狭間の戦い後に独立し、織田信長と同盟を結んで勢力を拡大した。信長没後は豊臣秀吉と対立・臣従を経て、関ヶ原の戦いに勝利した。元和2年（1616年）に没する。正室は築山殿、嫡男は松平信康。征夷大将軍に任じられ、太政大臣にも任官した。"
  ]
}}"""

base_url, api_key, model = get_model(provider="doubao")
client = OpenAI(base_url=base_url, api_key=api_key)


def mark_fused(title: str, lang: str) -> None:
    """使用短连接更新状态，并在连接失效时重连重试。"""
    for attempt in range(1, DB_MAX_ATTEMPTS + 1):
        try:
            with db.connection_context():
                WikiArticles.update(stage="fused").where(
                    (WikiArticles.title == title) & (WikiArticles.lang == lang)
                ).execute()
            return
        except (InterfaceError, OperationalError):
            # Peewee 只记录逻辑连接状态；底层连接被服务端关闭后，需要显式
            # 重置当前 worker 的线程局部状态，下一次尝试才会创建新连接。
            db.close()
            if attempt == DB_MAX_ATTEMPTS:
                raise
            logger.warning(
                "%s: 数据库连接失效，正在重连（%d/%d）",
                title,
                attempt,
                DB_MAX_ATTEMPTS,
            )


def process_title(title: str, lang: str) -> None:
    """处理单个标题的 infobox 融合。任何异常都在内部捕获并记录，不向上抛出，
    保证一个标题失败不会影响其他标题的处理。"""
    try:
        article = WikiArticle.model_validate_json((WIKI_PARSED / f"{title}.json").read_text())
        infobox = article.infobox
        assert infobox is not None, f"{title}: stage=parsed 但 infobox 为空，与既定假设不符，需要排查上游过滤逻辑"

        infobox_text = "\n".join(f"{key}: {value}" for key, value in infobox.items())
        lead_text = "\n\n".join(item.text for item in article.lead)

        response = client.chat.completions.create(
            model="doubao-seed-1-8-251228",
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(title=title, infobox_fields=infobox_text, lead_text=lead_text),
                },
            ],
            temperature=0.3,
            max_tokens=MAX_TOKENS,
            reasoning_effort="medium",
            extra_body={"thinking": {"type": "enabled"}},
            response_format={"type": "json_object"},
        )

        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning(f"{title}: 响应被 max_tokens 截断（finish_reason=length），需要复核或调大 MAX_TOKENS")

        result = choice.message.content
        assert result is not None, f"{title}: API 返回空内容"

        fused_lead = Paragraphs.model_validate_json(result)
        article.lead = [Paragraph(text=item) for item in fused_lead.texts]

        with open(WIKI_FUSED / f"{title}.json", mode="w", encoding="utf-8") as f:
            f.write(article.model_dump_json(indent=2))

        with db_lock:
            mark_fused(title, lang)

    except Exception:
        logger.exception(f"{title}: 处理失败，已跳过")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    args = parser.parse_args()

    with db.connection_context():
        titles: list[str] = [
            row[0]
            for row in WikiArticles.select(WikiArticles.title)
            .where((WikiArticles.stage == "parsed") & (WikiArticles.lang == args.lang))
            .tuples()
        ]

    logger.info(f"待处理标题数：{len(titles)}，并发数：{MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_title, title, args.lang) for title in titles]
        total = len(futures)
        for i, future in enumerate(as_completed(futures), start=1):
            future.result()
            if i % 100 == 0 or i == total:
                logger.info(f"进度：{i}/{total}")
