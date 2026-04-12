import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from integrations.transformers.config import GllamaConfig
from integrations.transformers.model import GllamaForCausalLM
from models.config import TransformerConfig
from models.torch.causal_lm import CausalLanguageModel


@torch.no_grad()
def greedy_decode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(text)


@torch.no_grad()
def sample():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(text)


def test():
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[:, -1, :]
        values, indices = torch.topk(logits, k=10, dim=-1)

    for idx, val in zip(indices[0].tolist(), values[0].tolist()):
        print(idx, repr(tokenizer.decode([idx])), float(val))


device = torch.device("cuda")

config = GllamaConfig.from_pretrained("artifacts")
model = GllamaForCausalLM.from_pretrained("artifacts", config=config).to(device)
tokenizer = AutoTokenizer.from_pretrained("artifacts")

inputs = tokenizer(
    "迄今人类所有的食物确实都来自野生生物",
    return_tensors="pt",
    return_token_type_ids=False,
)
inputs.pop("attention_mask", None)
inputs = {k: v.to(device) for k, v in inputs.items()}
