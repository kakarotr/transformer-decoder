import os

from datasets import load_dataset

from data.clean.base_cleaner import BaseCleaner

cleaner = BaseCleaner()


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

ds = load_dataset("HuggingFaceFW/finewiki", name="zh", split="train", streaming=True)
for i, sample in enumerate(ds):
    text = sample["text"]
    result = cleaner.clean(text)
    print(result)
