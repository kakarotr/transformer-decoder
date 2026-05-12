import random
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from tokenizers import AddedToken, Regex, Tokenizer, models, trainers
from tokenizers.pre_tokenizers import Digits, Sequence, Split, UnicodeScripts
from transformers.tokenization_utils_tokenizers import PreTrainedTokenizerFast

vocab_size = 40 * 1024
special_tokens_dict = {
    "unk": "<|unk|>",
    "pad": "<|pad|>",
    "eos": "<|im_end|>",
    "system": "<|im_system|>",
    "user": "<|im_user|>",
    "assistant": "<|im_assistant|>",
}


def prepare_corpus(output_dir: Path) -> None:
    """将训练所需的 parquet 文件复制到 output_dir，供后续上传云主机。"""
    finewiki_path = Path("/workspace/transformer-decoder-2/pretraining/finewiki")
    fineweb_path = Path("/workspace/transformer-decoder-2/pretraining/Ultra-FineWeb")

    output_dir = Path(output_dir)
    finewiki_out = output_dir / "finewiki"
    fineweb_out = output_dir / "fineweb"
    finewiki_out.mkdir(parents=True, exist_ok=True)
    fineweb_out.mkdir(parents=True, exist_ok=True)

    print(f"{datetime.now().strftime('%H:%M:%S')} 开始复制语料文件到: {output_dir}")

    for src in sorted(finewiki_path.glob("*.parquet")):
        shutil.copy2(src, finewiki_out / src.name)
        print(f"  copied {src.name}")

    all_files = sorted(fineweb_path.glob("*.parquet"))
    random.seed(42)
    sampled_files = random.sample(all_files, 13)
    for src in sampled_files:
        shutil.copy2(src, fineweb_out / src.name)
        print(f"  copied {src.name}")

    print(f"{datetime.now().strftime('%H:%M:%S')} 复制完成，共 {len(list(finewiki_out.glob('*.parquet'))) + 13} 个文件")


def iter_corpus(corpus_dir: Path):
    """从 corpus_dir 下的所有 parquet 文件中迭代文本，供 tokenizer 训练。"""
    corpus_dir = Path(corpus_dir)
    for path in sorted(corpus_dir.rglob("*.parquet")):
        df = pd.read_parquet(path, columns=["text"])
        yield from df["text"]


def train_tokenizer(corpus_dir: Path, output_dir: Path) -> None:
    """直接从 corpus_dir 里的 parquet 文件训练 BPE tokenizer 并保存。"""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"语料目录不存在: {corpus_dir}")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = Sequence(
        [
            UnicodeScripts(),
            Digits(individual_digits=True),
            Split(pattern=Regex(r"\n+"), behavior="isolated"),
            Split(pattern=Regex(r" +"), behavior="isolated"),
        ]
    )

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(special_tokens_dict.values()),
        min_frequency=5,
        show_progress=True,
    )

    print(f"{datetime.now().strftime('%H:%M:%S')} 开始训练 tokenizer...")
    tokenizer.train_from_iterator(iter_corpus(corpus_dir), trainer)
    print(f"{datetime.now().strftime('%H:%M:%S')} 训练完成")

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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fast_tokenizer.save_pretrained(output_dir)
    print(f"{datetime.now().strftime('%H:%M:%S')} Tokenizer 已保存至: {output_dir}")


if __name__ == "__main__":
    # 本地运行：选文件并复制
    prepare_corpus(Path("artifacts/corpus"))

    # 云主机上运行：直接从上传的目录训练
    train_tokenizer(Path("artifacts/corpus"), Path("artifacts"))
