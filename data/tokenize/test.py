from tokenizers import Regex
from tokenizers.pre_tokenizers import Digits, Sequence, Split, UnicodeScripts

pre_tokenizer = Sequence(
    [
        UnicodeScripts(),
        Digits(individual_digits=True),
        Split(pattern=Regex(r" +"), behavior="isolated"),
        Split(pattern=Regex(r"\n+"), behavior="isolated"),
    ]
)

print(pre_tokenizer.pre_tokenize_str("## 一级标题\n这是一段内容\n\n### 下一级标题"))
