import random
from datetime import datetime
from pathlib import Path

import pandas as pd
from tokenizers import AddedToken, Regex, Tokenizer, models, trainers
from tokenizers.pre_tokenizers import Digits, Sequence, Split, UnicodeScripts
from transformers.tokenization_utils_tokenizers import PreTrainedTokenizerFast

vocab_size = 39 * 1024
special_tokens_dict = {
    "unk": "<|unk|>",
    "pad": "<|pad|>",
    "eos": "<|im_end|>",
    "system": "<|im_system|>",
    "user": "<|im_user|>",
    "assistant": "<|im_assistant|>",
}
markdown_tokens = ["```"]


def get_training_data():
    texts = []
    finewiki_path = Path("/workspace/transformer-decoder-2/pretraining/finewiki")
    fineweb_path = Path("/workspace/transformer-decoder-2/pretraining/Ultra-FineWeb")

    print(f"{datetime.now().strftime('%H:%M:%S')} 开始采样训练数据...")

    for path in sorted(finewiki_path.glob("*.parquet")):
        df = pd.read_parquet(path, columns=["text"])
        texts.extend(df["text"].to_list())

    all_files = sorted(fineweb_path.glob("*.parquet"))
    random.seed(42)
    sampled_files = random.sample(all_files, 13)
    for path in sampled_files:
        df = pd.read_parquet(path, columns=["text"])
        texts.extend(df["text"].tolist())

    print(f"{datetime.now().strftime('%H:%M:%S')} 训练数据采样完成")

    return texts


tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = Sequence(
    [
        UnicodeScripts(),
        Digits(individual_digits=True),
        Split(pattern=Regex(r"\n+"), behavior="isolated"),
        Split(pattern=Regex(r" {5,}"), behavior="isolated"),
    ]
)

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size - len(markdown_tokens),
    special_tokens=list(special_tokens_dict.values()),
    min_frequency=5,
    show_progress=True,
)

tokenizer.train_from_iterator(get_training_data(), trainer)

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    pad_token=AddedToken(special_tokens_dict["pad"], lstrip=False, rstrip=False, normalized=False, special=True),
    eos_token=AddedToken(special_tokens_dict["eos"], lstrip=False, rstrip=False, normalized=False, special=True),
    bos_token=None,
    unk_token=AddedToken(special_tokens_dict["unk"], lstrip=False, rstrip=False, normalized=False, special=True),
    additional_special_tokens=[
        AddedToken(special_tokens_dict["user"], lstrip=False, rstrip=False, normalized=False, special=True),
        AddedToken(special_tokens_dict["assistant"], lstrip=False, rstrip=False, normalized=False, special=True),
        AddedToken(special_tokens_dict["system"], lstrip=False, rstrip=False, normalized=False, special=True),
    ],
    clean_up_tokenization_spaces=False,
    model_max_length=102400,
)
fast_tokenizer.add_bos_token = False
fast_tokenizer.add_eos_token = False
fast_tokenizer.add_tokens(markdown_tokens)

# current_size = len(fast_tokenizer)
# if current_size < vocab_size:
#     pad_count = vocab_size - current_size
#     dummy_tokens = [f"<|dummy_{i}|>" for i in range(pad_count)]
#     fast_tokenizer.add_tokens(dummy_tokens)

output_dir = Path("artifacts/base")
if not output_dir.exists():
    output_dir.mkdir()
fast_tokenizer.save_pretrained(output_dir)
