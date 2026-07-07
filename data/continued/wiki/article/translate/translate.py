"""
ListBlock 展平 / 回填 最小示例
================================
演示回路：结构体 --(DFS 展平)--> 扁平 ID 条目 --(模型)--> {id: 译文} --(寻址表)--> 写回结构体

核心结论：
1. 结构信息（ordered / children 嵌套 / level / category）从不序列化、从不恢复，
   源结构体全程保留，id 只是"写回地址"。
2. kind 是纯模型侧元信息（触发语域规则），回填不使用。
3. 同一次遍历同时产出 payload 和 id -> (节点, 字段名) 寻址表，回填 = 查表 setattr。
   跨进程场景不需要持久化寻址表：对源 JSON 重跑 flatten 即可重建（遍历确定 -> id 稳定）。

沙盒无 pydantic，此处用 dataclass 定义同名同字段 schema。
flatten_article / apply_translations 只依赖 getattr / setattr / isinstance，
换回项目里的 Pydantic 模型时这两个函数无需修改
（deepcopy -> model_copy(deep=True)，asdict -> model_dump）。
"""

from __future__ import annotations

import copy
import json
import re

from openai import OpenAI

from data.continued.utils import get_model
from data.continued.wiki.article.structure import (
    Block,
    DescriptionBlock,
    Heading,
    ListBlock,
    ListItem,
    Paragraph,
    Quote,
    WikiArticle,
)

# 寻址表类型：id -> (持有文本的节点对象, 字段名)
Addr = dict[str, tuple[object, str]]


def _emit(node: object, attr: str, item_id: str, kind: str, out: list[dict], addr: Addr) -> None:
    """登记一条待译文本。payload 里的 kind 只给模型看（语域/风格），回填不用。"""
    text = getattr(node, attr)
    if text is None or not text.strip():
        return  # 纯容器 / 空节点：不送翻译，原样保留
    out.append({"id": item_id, "kind": kind, "text": text})
    addr[item_id] = (node, attr)


def _flatten_list_items(items: list[ListItem], prefix: str, out: list[dict], addr: Addr) -> None:
    for i, it in enumerate(items):
        pid = f"{prefix}.{i}"
        _emit(it, "text", pid, "list_item", out, addr)
        _flatten_list_items(it.children, pid, out, addr)  # 嵌套只体现为 id 前缀，结构不进 payload


def _flatten_block(block: Block, bid: str, out: list[dict], addr: Addr) -> None:
    if isinstance(block, Heading):
        _emit(block, "text", bid, "heading", out, addr)
    elif isinstance(block, Paragraph):
        _emit(block, "text", bid, "paragraph", out, addr)
    elif isinstance(block, ListBlock):
        _flatten_list_items(block.items, bid, out, addr)
    elif isinstance(block, Quote):
        _emit(block, "title", f"{bid}.t", "quote_title", out, addr)
        _emit(block, "text", f"{bid}.q", "quote", out, addr)  # 触发半文言语域规则
        _emit(block, "citation", f"{bid}.c", "citation", out, addr)
    elif isinstance(block, DescriptionBlock):
        for j, item in enumerate(block.items):
            _emit(item, "term", f"{bid}.{j}.dt", "dt", out, addr)
            for k, sub in enumerate(item.body):
                _flatten_block(sub, f"{bid}.{j}.{k}", out, addr)  # body 递归复用同一套逻辑
    else:
        raise TypeError(f"unknown block type: {type(block)}")


def flatten_article(article: WikiArticle) -> tuple[list[dict], Addr]:
    """title 不在此处理：按既定方案单独先译一次，作为元信息注入各 chunk。"""
    out: list[dict] = []
    addr: Addr = {}
    for i, p in enumerate(article.lead):
        _emit(p, "text", f"l{i}", "paragraph", out, addr)
    for i, b in enumerate(article.blocks):
        _flatten_block(b, f"b{i}", out, addr)
    return out, addr


_KANA_RE = re.compile(r"[ぁ-ゖァ-ヶー]")  # 假名残留检测


def apply_translations(addr: Addr, translations: dict[str, str]) -> None:
    if addr.keys() != translations.keys():
        missing = sorted(addr.keys() - translations.keys())
        extra = sorted(translations.keys() - addr.keys())
        raise ValueError(f"id 集合不一致 missing={missing} extra={extra}")

    need_retry = [i for i, t in translations.items() if not t.strip() or _KANA_RE.search(t)]
    if need_retry:
        raise ValueError(f"空译文或假名残留，需单条重译: {need_retry}")

    for item_id, text in translations.items():
        node, attr = addr[item_id]
        setattr(node, attr, text)  # 全部回填都是这一行


_TEXT_KEYS = {"text", "term", "citation", "title"}

data = {
    "title": "織田信長",
    "lead": [
        {
            "kind": "paragraph",
            "text": "**織田信長**（おだのぶなが）は、日本の戦国時代から安土桃山時代にかけての武将・大名。戦国の三英傑の一人。天文3年（1534年）に生まれ、天正10年（1582年）に没し享年49。幼名は吉法師、後に信長と改名した。織田弾正忠家（勝幡織田氏）出身で、生前の最高官位は正二位右大臣。斯波義銀、足利義昭に仕えた。",
        },
        {
            "kind": "paragraph",
            "text": "尾張国（現在の愛知県）出身。織田信秀の嫡男。母は土田御前で、弟に信勝、長益、妹にお市の方がいる。正室には斎藤道三の娘である鷺山殿（濃姫）がおり、嫡男は信忠のほか、信雄、信孝らの子をもった。家督争いの混乱を収めた後、桶狭間の戦いで今川義元を討ち取り、勢力を拡大した。足利義昭を奉じて上洛し、後には足利義昭を追放することで、畿内を中心に独自の中央政権（「織田政権」）を確立して天下人となった。しかし、天正10年6月2日（1582年6月21日）、家臣・明智光秀に謀反を起こされ、本能寺で自害した。",
        },
        {
            "kind": "paragraph",
            "text": "これまで信長の政権は、豊臣秀吉による豊臣政権、徳川家康が開いた江戸幕府への流れをつくった画期的なもので、その政治手法も革新的なものであるとみなされてきた。しかし、近年の歴史学界ではその政策の前時代性が指摘されるようになり、しばしば「中世社会の最終段階」とも評され、その革新性を否定する研究が主流となっている。",
        },
    ],
    "blocks": [],
    # "blocks": [
    #     {"kind": "heading", "level": 2, "text": "概要"},
    #     {
    #         "kind": "paragraph",
    #         "text": "織田信長は、織田弾正忠家の当主・織田信秀の子に生まれ、尾張（愛知県西部）の一地方領主としてその生涯を歩み始めた。信長は織田弾正忠家の家督を継いだ後、尾張守護代の織田大和守家、織田伊勢守家を滅ぼすとともに、弟の織田信行を排除して、尾張一国の支配を徐々に固めていった。",
    #     },
    #     {
    #         "kind": "paragraph",
    #         "text": "永禄3年（1560年）、信長は桶狭間の戦いにおいて駿河の戦国大名・今川義元を撃破した。そして、三河の領主・徳川家康（松平元康）と同盟を結ぶ。永禄8年（1565年）、犬山城の織田信清を破ることで尾張の統一を達成した。",
    #     },
    #     {
    #         "kind": "paragraph",
    #         "text": "一方で、室町幕府の将軍・足利義輝が殺害された（永禄の変）後に、弟の足利義昭から室町幕府及び足利将軍家再興の呼びかけを受けており、信長も永禄9年（1566年）には上洛を図ろうとした。美濃の戦国大名・斉藤氏（一色氏）との対立のためこれは実現しなかったが、永禄10年（1567年）には斎藤氏の駆逐に成功し（稲葉山城の戦い）、尾張・美濃の二カ国を領する戦国大名となった。そして、改めて幕府再興を志す意を込めて、「天下布武」の印を使用した。",
    #     },
    #     {
    #         "kind": "paragraph",
    #         "text": "翌年10月、足利義昭とともに信長は上洛し、三好三人衆などを撃破して、室町幕府の再興を果たす。信長は、室町幕府との二重政権（連合政権）を築いて、「天下」（五畿内）の静謐を実現することを目指した。しかし、敵対勢力も多く、元亀元年（1570年）6月、越前の朝倉義景・北近江の浅井長政を姉川の戦いで破ることには成功したものの、三好三人衆や比叡山延暦寺、石山本願寺などに追い詰められる。同年末に、信長と義昭は一部の敵対勢力と講和を結び、ようやく窮地を脱した。",
    #     },
    #     {
    #         "kind": "paragraph",
    #         "text": "元亀2年（1571年）9月、比叡山を焼き討ちする。しかし、その後も苦しい情勢は続き、三方ヶ原の戦いで織田・徳川連合軍が武田信玄に敗れた後、元亀4年（1573年）、将軍・足利義昭は信長を見限る。信長は義昭と敵対することとなり、同年中には義昭を京都から追放した（槇島城の戦い）。",
    #     },
    #     {
    #         "kind": "paragraph",
    #         "text": "将軍不在のまま中央政権を維持しなければならなくなった信長は、天下人への道を進み始める。元亀から天正への改元を実現すると、天正元年（1573年）中には浅井長政・朝倉義景・三好義継を攻め、これらの諸勢力を滅ぼすことに成功した。天正3年（1575年）には、長篠の戦いでの武田氏に対して勝利するとともに、右近衛大将に就任し、室町幕府に代わる新政権の構築に乗り出した。翌年には安土城の築城も開始している。しかし、天正5年（1577年）以降、松永久秀、別所長治、荒木村重らが次々と信長に叛いた。",
    #     },
    #     {
    #         "kind": "paragraph",
    #         "text": "天正8年（1580年）、長きにわたった石山合戦（大坂本願寺戦争）に決着をつけ、翌年には京都で大規模な馬揃え（京都御馬揃え）を行い、その勢威を誇示している。",
    #     },
    #     {
    #         "kind": "paragraph",
    #         "text": "天正10年（1582年）、甲州征伐を行い、武田勝頼を自害に追いやって武田氏を滅亡させ、東国の大名の多くを自身に従属させた。同年には信長を太政大臣・関白・征夷大将軍のいずれかに任ずるという構想が持ち上がっている（三職推任）。その後、信長は長宗我部元親討伐のために四国攻めを決定し、三男の信孝に出兵の準備をさせている。そして、信長自身も毛利輝元ら毛利氏討伐のため、中国地方攻略に赴く準備を進めていた。しかし、6月2日、重臣の明智光秀の謀反によって、京の本能寺で自害に追い込まれた（本能寺の変）。",
    #     },
    #     {
    #         "kind": "paragraph",
    #         "text": "一般に、信長の性格は、極めて残虐で、また、常人とは異なる感性を持ち、家臣に対して酷薄であったと言われている。一方、信長は世間の評判を非常に重視し、家臣たちの意見にも耳を傾けていたという異論も存在する。なお、信長は武芸の鍛錬に励み、趣味として鷹狩り・茶の湯・相撲などを愛好した。南蛮などの異国に興味を持っていたとも言われる。",
    #     },
    #     {
    #         "kind": "paragraph",
    #         "text": "政策面では、信長は室町幕府将軍から「天下」を委任されるという形で自らの政権を築いた。天皇や朝廷に対しては協調的な姿勢を取っていたという見方が有力となっている。",
    #     },
    #     {
    #         "kind": "paragraph",
    #         "text": "江戸時代には、新井白石らが信長の残虐性を強く非難したように、信長の評価は低かった。",
    #     },
    #     {
    #         "kind": "paragraph",
    #         "text": "とはいえ、やがて信長は勤王家として称賛されるようになり、明治時代には神として祀られている。第二次世界大戦後には、信長はその政策の新しさから、革新者として評価されるようになった。しかし、このような革新者としての信長像には疑義が呈されつつあり、近年の歴史学界では信長の評価の見直しが進んでいる。",
    #     },
    # ],
}


def structure_fingerprint(article: WikiArticle) -> dict:
    def scrub(x):
        if isinstance(x, dict):
            return {k: ("" if k in _TEXT_KEYS and isinstance(v, str) else scrub(v)) for k, v in x.items()}
        if isinstance(x, list):
            return [scrub(v) for v in x]
        return x

    return scrub(article.model_dump())  # type: ignore


prompt = """你是一个日译中翻译引擎，负责将日本战国相关的日文维基百科内容翻译为简体中文，产出用于语言模型训练的语料。你只输出 JSON，不输出任何其他内容。
 
【输入格式】
用户消息是一个 JSON 对象，字段如下：
- article：条目原名与既定中文译名，格式为「原名 → 译名」
- path：当前内容所在的章节标题路径，仅用于理解语境，不翻译、不输出
- glossary：术语强制对照表（该字段可能缺省，缺省即无强制术语）
- items：待译条目数组，每条含 id、kind、text 三个字段
 
【输出格式（硬性要求）】
1. 只输出一个 JSON 对象：{"<id>": "<译文>", ...}，不输出解释、注释、Markdown 代码围栏或任何 JSON 以外的字符。
2. 输出的 id 集合必须与输入 items 的 id 集合完全一致，不增不减。
3. 每条译文只对应该 id 自身的 text 内容，禁止将相邻条目的内容合并、拆分或调换。
4. 译文不得为空。text 无需改动时（如纯数字、纯拉丁字母），将原文照抄作为该 id 的译文。
5. 译文中不得残留任何平假名或片假名。
 
【翻译总则】
- 简体中文，百科书面语，忠实原文，不增删信息，不加译者注。
- 条目名在正文中出现时，一律使用 article 给定的译名。
- glossary 中出现的词条为强制对照，命中即使用指定译名。
- 日文汉字人名、地名按汉字直接对译。日语国字（辻、畑、峠、榊、笹等）没有对应汉字，在专名中原样保留，不得替换或音译。
- 以假名书写的人名、作品名等，glossary 未指定时：有通行中文译名的用通行译名，没有的按惯例音译或意译，不得保留假名。
- 拉丁字母内容（罗马字、英文等）原样保留。
- 「〜一揆」中的"一揆"为固定术语，保留不译，不得译作起义、暴动等。
 
【数字与纪年（硬性要求）】
- 一切数字照抄原文字形：阿拉伯数字保持阿拉伯数字，汉字数字保持汉字数字，不转换、不换算。
- 年号纪年不追加公历年份；原文已括注的公历年份照抄。
- 以上只约束数字字符本身，标点仍按标点规则正常转换。
 
【标点】
- 输出一律使用简体中文全角标点；日文读点「、」按语义译为逗号或顿号。
- 「」：一般引用与对话 → ""；若标记的是短篇作品名（歌曲、单篇文章、章节等）→ 《》。
- 『』：作品名（书籍、报刊、电影、游戏等）→ 《》，出现在引号内部时同样保持《》，不降级；若是引号内的再引用而非作品名 → ''。
- 引号嵌套：外层用 ""，内层用 ''。
- 人名间隔点 ・ → ·。
 
【kind 规则】
- heading：简洁的名词性标题，不加句末标点。
- paragraph：百科正文语体。
- list_item：简洁；同一批相邻的 list_item 属于同一列表，保持句式一致。
- dt：术语或主题词，名词性，不加句末标点。
- quote：引文。古文、和歌、书信、军记物语等历史引文用半文半白语域译出；现代人的引语用白话。
- quote_title、citation：出处信息，书名用《》。
 
【事件名后缀（glossary 未命中时的默认规则）】
〜の戦い → 〜合战；〜の乱 → 〜之乱；〜の変 → 〜之变；〜の役 → 〜之役；〜の陣 → 〜之阵。glossary 命中时以 glossary 为准。
 
【示例，使用 JSON 格式输出】
输入：
{"article": "織田信長 → 织田信长", "path": "生涯 > 桶狭間の戦い", "glossary": {"ルイス・フロイス": "路易斯·弗洛伊斯"}, "items": [{"id": "b12", "kind": "heading", "text": "桶狭間の戦い"}, {"id": "b13", "kind": "paragraph", "text": "永禄3年（1560年）5月19日、織田信長は桶狭間において今川義元を討ち取った。宣教師ルイス・フロイスは、信長を「稀代の英雄」と記している。"}, {"id": "b14.q", "kind": "quote", "text": "人間五十年、下天のうちを比ぶれば、夢幻の如くなり"}, {"id": "b14.c", "kind": "citation", "text": "幸若舞『敦盛』"}]}
输出：
{"b12": "桶狭间之战", "b13": "永禄3年（1560年）5月19日，织田信长于桶狭间讨取今川义元。传教士路易斯·弗洛伊斯记载，信长是"稀代英雄"。", "b14.q": "人间五十年，与天相比，不过梦幻", "b14.c": "幸若舞《敦盛》"}"""


def main() -> None:
    # ---- 源文（日文，模拟解析产物）----
    article = WikiArticle.model_validate(data)

    # ---- 1. 深拷贝出译文载体（源文对象保持不动）----
    translated = copy.deepcopy(article)
    translated.title = "织田信长"  # title 单独先译，作为元信息进各 chunk

    # ---- 2. 展平：payload 给模型，addr 留在本地 ----
    payload, addr = flatten_article(translated)
    # print("=" * 30, "payload（进 prompt 的 items，贪心分块就是切这个列表）", "=" * 30)
    # print(json.dumps(payload, ensure_ascii=False, indent=2))

    # ---- 3. 模型返回 {id: 译文}（此处硬编码模拟，保证演示可复现）----
    base_url, api_key, model = get_model(provider="doubao")
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model="doubao-seed-1-8-251228",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"title": "織田信長 → 织田信长", "path": "引言", "items": payload},
                    ensure_ascii=False,
                ),
            },
        ],
        temperature=0.3,
        reasoning_effort="low",
        extra_body={"thinking": {"type": "enabled"}},
        response_format={"type": "json_object"},
        max_tokens=16 * 1024,
    )

    mock_model_output = json.loads(response.choices[0].message.content)  # type: ignore

    # ---- 4. 回填 ----
    apply_translations(addr, mock_model_output)

    # ---- 5. 断言一：结构指纹逐位相同（ordered / children / level / category 全部原样）----
    assert structure_fingerprint(article) == structure_fingerprint(translated)
    print("\n[断言通过] 译前/译后结构指纹逐位相同，仅文本字段变化")

    # ---- 6. 断言二：确定性 —— 对源结构重跑 flatten，id 与 payload 完全一致 ----
    #      这意味着跨进程批翻时无需持久化寻址表，重跑即可重建
    replay_payload, _ = flatten_article(copy.deepcopy(article))
    assert replay_payload == payload
    print("[断言通过] 重跑 flatten 产出的 id 序列与首次一致（寻址表可随时重建）")

    # ---- 7. 译后结构体（交给你的 CPT 渲染器的就是它）----
    print("\n" + "=" * 30, "译后 WikiArticle（结构原封不动）", "=" * 30)
    print(translated.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
