from tokenizers import Regex
from tokenizers.pre_tokenizers import Digits, Sequence, Split, UnicodeScripts
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def test_pre_tokenize():
    pre_tokenizer = Sequence(
        [
            UnicodeScripts(),
            Digits(individual_digits=True),
            Split(pattern=Regex(r" +"), behavior="isolated"),
            Split(pattern=Regex(r"\n+"), behavior="isolated"),
        ]
    )

    print(pre_tokenizer.pre_tokenize_str("## 一级标题\n这是一段内容\n\n### 下一级标题"))


def test_md_token_in_vocab():
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("artifacts/base")
    vocab = tokenizer.get_vocab()
    candidates = ["#", "##", "###", "####", "```", "**", "`", "|", ">", "-", "."]
    missing = [t for t in candidates if t not in vocab]
    print("不在词表里的 markdown token：", missing)


def test_tokenize():
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("artifacts/base")
    print(tokenizer.encode("你好，   今天天气怎么样？"))


test_md_token_in_vocab()
