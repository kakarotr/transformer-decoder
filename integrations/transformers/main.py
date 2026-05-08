from tokenizers import decoders
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
    print(type(text))
    print(repr(text))


def test():
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[:, -1, :]
        values, indices = torch.topk(logits, k=10, dim=-1)

    for idx, val in zip(indices[0].tolist(), values[0].tolist()):
        print(idx, repr(tokenizer.decode([idx])), float(val))


def lm_nrom():
    # 取 lm_head 权重 shape: [vocab_size, hidden_dim]
    W = model.lm_head.weight.detach().float()

    # 计算行向量的归一化内积（行间余弦相似度）
    W_norm = F.normalize(W, dim=-1)
    # 随机采样 1000 对行，计算余弦
    idx = torch.randint(0, W.size(0), (1000, 2))
    cos = (W_norm[idx[:, 0]] * W_norm[idx[:, 1]]).sum(dim=-1)
    print(cos.mean(), cos.std())


device = torch.device("cuda")

config = GllamaConfig.from_pretrained("artifacts")
model = GllamaForCausalLM.from_pretrained("artifacts", config=config).to(device)
tokenizer = AutoTokenizer.from_pretrained("artifacts/base")

inputs = tokenizer(
    "春天来了，万物复苏",
    return_tensors="pt",
    return_token_type_ids=False,
)
inputs.pop("attention_mask", None)
inputs = {k: v.to(device) for k, v in inputs.items()}

sample()
