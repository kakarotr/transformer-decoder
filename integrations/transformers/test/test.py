import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from integrations.transformers.config import GllamaConfig
from integrations.transformers.model import GllamaForCausalLM
from models.config import TransformerConfig
from models.liger.causal_lm import CausalLanguageModel


@torch.no_grad()
def compare_prompts(model, tokenizer, prompts, device="cuda"):
    vecs = []
    for text in prompts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        inputs.pop("attention_mask", None)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logits = model(**inputs, return_dict=True).logits[0, -1].float()
        vecs.append(F.normalize(logits, dim=0))

    sims = torch.zeros(len(vecs), len(vecs), device=device)
    for i in range(len(vecs)):
        for j in range(len(vecs)):
            sims[i, j] = torch.dot(vecs[i], vecs[j])

    print(sims.cpu())


@torch.no_grad()
def next_token_rank(model, tokenizer, text, device="cuda"):
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) >= 2

    prefix = torch.tensor([ids[:-1]], device=device)
    target = ids[-1]

    logits = model(input_ids=prefix, return_dict=True).logits[0, -1].float()
    rank = (logits > logits[target]).sum().item() + 1

    print("target token id:", target)
    print("target token   :", repr(tokenizer.decode([target])))
    print("target logit   :", logits[target].item())
    print("rank           :", rank)


@torch.no_grad()
def inspect_lm_head_norm(model, tokenizer):
    weight = model.lm_head.weight.float()  # [vocab, hidden]
    norms = weight.norm(dim=1)

    topk = torch.topk(norms, k=20)
    for tid, n in zip(topk.indices.tolist(), topk.values.tolist()):
        print(tid, repr(tokenizer.decode([tid])), n)


prompts = [
    "今天天气很好，所以我决定",
    "秦始皇统一六国之后，",
    "这段 Python 代码报错的原因是",
    "猫坐在窗台上，突然",
    "迄今人类所有的食物确实都来自野生生物",
]

device = torch.device("cuda")
config = GllamaConfig.from_pretrained("artifacts")
model = GllamaForCausalLM.from_pretrained("artifacts", config=config).to(device)
tokenizer = AutoTokenizer.from_pretrained("artifacts")

compare_prompts(model, tokenizer, prompts)
print("---")
next_token_rank(model, tokenizer, "今天天气很好，适合出门散")
print("---")
inspect_lm_head_norm(model, tokenizer)
