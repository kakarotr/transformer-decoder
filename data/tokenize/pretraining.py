from pathlib import Path
import random

import pandas as pd
from tokenizers import AddedToken, Tokenizer, models, trainers
from tokenizers.pre_tokenizers import Digits, Sequence, UnicodeScripts
from transformers.tokenization_utils_tokenizers import PreTrainedTokenizerFast

vocab_size = 40960
special_tokens_dict = {
    "pad": "<|pad|>",
    "eos": "<|im_end|>",
    "system": "<|im_system|>",
    "user": "<|im_user|>",
    "assistant": "<|im_assistant|>",
}
markdown_tokens = ["#", "##", "###", "####", "-", "`", "**", "|", ">"]


def get_training_data():
    texts = []
    finewiki_path = Path("~/transformer-decoder/pretraing/finewiki")
    fineweb_path = Path("~/transformer-decoder/pretraing/Ultra-FineWeb")

    for path in sorted(finewiki_path.glob("*.parquet")):
        df = pd.read_parquet(path, columns=["text"])
        texts.append(df["text"].to_list())
    
    all_files = sorted(fineweb_path.glob("*.parquet"))
    sampled_files = random.sample(all_files, 50)
    for path in sampled_files:
        df = pd.read_parquet(path, columns=["text"])
        texts.extend(df["text"].tolist())

    
    


tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = Sequence([UnicodeScripts(), Digits(individual_digits=True)])

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

current_size = len(fast_tokenizer)
if current_size < vocab_size:
    pad_count = vocab_size - current_size
    dummy_tokens = [f"<|dummy_{i}|>" for i in range(pad_count)]
    fast_tokenizer.add_tokens(dummy_tokens)

output_dir = Path("artifacts/base")
if not output_dir.exists():
    output_dir.mkdir()
fast_tokenizer.save_pretrained(output_dir)
