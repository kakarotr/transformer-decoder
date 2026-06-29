import argparse
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from data.continued.paths import WIKI_PARSED
from data.continued.utils import get_model
from data.continued.wiki.article.structure import WikiArticle
from data.continued.wiki.db import WikiArticles


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
- 主君：重要な数名を選び、自然な叙述でつなぐ。
  つなぎは史実に基づくこと。全ての主君を機械的に列挙しない
- 改名：幼名と広く知られた名を残し、
  改名の全経緯を逐一列挙しない
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
官位: 正一位太政大臣、征夷大将軍
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
    "三河国の土豪・松平氏に生まれ、幼少期は今川義元の人質として過ごした。桶狭間の戦い後に独立し、織田信長と同盟を結んで勢力を拡大した。信長没後は豊臣秀吉と対立・臣従を経て、関ヶ原の戦いに勝利した。元和2年（1616年）に没する。正室は築山殿、嫡男は松平信康。征夷大将軍に任じられ、正一位太政大臣を贈られた。"
  ]
}}"""

base_url, api_key, model = get_model(provider="doubao")
client = OpenAI(base_url=base_url, api_key=api_key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    args = parser.parse_args()

    titles: list[str] = [
        row[0]
        for row in WikiArticles.select(WikiArticles.title)
        .where((WikiArticles.stage == "parsed") & (WikiArticles.lang == args.lang))
        .tuples()
    ]

    for title in titles:
        article = WikiArticle.model_validate_json((WIKI_PARSED / f"{title}.json").read_text())
        infobox = article.infobox
        lead = article.lead

        assert infobox is not None
        content = []
        for key, value in infobox.items():
            content.append(f"{key}: {value}")
        infobox = "\n".join(content)

        content = []
        for item in lead:
            content.append(item.text)
        lead = "\n\n".join(content)

        client.responses.create(
            model="doubao-seed-1-8-251228",
            input=[
                {"role": "user", "content": prompt.format(title=title, infobox_fields=infobox, lead_text=lead)},
            ],
            temperature=0.3,
            reasoning={"effort": "medium"},
            extra_body={"thinking": {"type": "enabled"}},
        )
