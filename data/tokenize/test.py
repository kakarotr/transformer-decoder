from tokenizers import Regex
from tokenizers.pre_tokenizers import Digits, Sequence, Split, UnicodeScripts
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def test_pre_tokenize():
    pre_tok = Sequence(
        [
            UnicodeScripts(),
            Digits(individual_digits=True),
            Split(pattern=Regex(r"\n+"), behavior="isolated"),
            Split(pattern=Regex(r" +"), behavior="isolated"),
        ]
    )
    print(pre_tok.pre_tokenize_str("## Hello"))
    print(pre_tok.pre_tokenize_str("### World"))
    print(pre_tok.pre_tokenize_str("**bold**"))


def test_md_token_in_vocab():
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("artifacts")
    vocab = tokenizer.get_vocab()
    candidates = ["#", "##", "###", "####", "```", "**", "`", "|", ">", "-", "."]
    missing = [t for t in candidates if t not in vocab]
    print("不在词表里的 markdown token：", missing)


def test_tokenize():
    texts = [
        "人工智能技术的发展正在深刻改变人类社会的生产方式和生活方式。",
        "2024年中国GDP增速为5.2%，总量约为126万亿元人民币。",
        "深度学习中的Transformer架构由Google在2017年提出，论文标题为Attention Is All You Need。",
        '他问道："你真的确定吗？"她回答说："当然，我已经考虑了很久！"',
        "织田信长",
        "德川家康",
        "室町幕府",
        "上杉谦信",
        "德川家",
        "大名",
        "丰臣秀吉",
        """## 战国时代概述

织田信长于1568年进入京都，开启了统一天下的进程：

- 桶狭间之战（1560年）
- 长篠之战（1575年）
- 本能寺之变（1582年）
### 三级标题
#### 四级标题
""",
        "德川家康是江户幕府的创立者，他在关原之战中击败石田三成领导的西军，确立了德川氏对日本的统治地位。此后长达260余年的江户时代，日本实现了相对稳定的社会秩序。",
    ]

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("artifacts/base")
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        ratio = len(text) / len(token_ids)
        print(f"字符数: {len(text)} | token数: {len(token_ids)} | 压缩率: {ratio:.2f}")
    print("---")
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        print(repr("|".join([tokenizer.decode([token_id]) for token_id in token_ids])))


test_md_token_in_vocab()
test_tokenize()
