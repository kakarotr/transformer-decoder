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
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("artifacts/base")
    vocab = tokenizer.get_vocab()
    candidates = ["#", "##", "###", "####", "```", "**", "`", "|", ">", "-", "."]
    missing = [t for t in candidates if t not in vocab]
    print("不在词表里的 markdown token：", missing)


def test_tokenize():
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("artifacts/base")
    print(tokenizer.encode("你好，   今天天气怎么样？"))
    print(tokenizer.encode("###", add_special_tokens=False))
    print(tokenizer.encode("####", add_special_tokens=False))
    print(len(tokenizer.get_vocab()))  # 应该还是 39 * 1024


test_md_token_in_vocab()
