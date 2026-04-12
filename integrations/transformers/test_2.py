import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# ===== 你的 HF 模型导入 =====
# 按你自己的项目路径改
from integrations.transformers.model import GllamaForCausalLM
from integrations.transformers.config import GllamaConfig


DEVICE = "cuda"
MODEL_PATH = "artifacts"


@torch.no_grad()
def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = GllamaConfig.from_pretrained(model_path)
    model = GllamaForCausalLM.from_pretrained(model_path, config=config).to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def get_last_hidden_and_logits(
    model: GllamaForCausalLM,
    tokenizer,
    text: str,
    device: str = "cuda",
):
    # 按你之前的测试方式，移除 attention_mask，强制走无 padding 的 causal 路径
    inputs = tokenizer(
        text,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    inputs.pop("attention_mask", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]

    # 直接从 decoder 取 hidden_states
    hidden_states = model.decoder(
        input_ids=input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        use_padding_mask=False,
    )  # [1, T, H]

    logits = model.lm_head(hidden_states)  # [1, T, V]

    last_hidden = hidden_states[0, -1].float()  # [H]
    last_logits = logits[0, -1].float()  # [V]

    return {
        "input_ids": input_ids,
        "last_hidden": last_hidden,
        "last_logits": last_logits,
    }


@torch.no_grad()
def cosine_matrix(vectors: list[torch.Tensor]) -> torch.Tensor:
    xs = [F.normalize(x, dim=0) for x in vectors]
    n = len(xs)
    sims = torch.empty(n, n, dtype=torch.float32, device=xs[0].device)
    for i in range(n):
        for j in range(n):
            sims[i, j] = torch.dot(xs[i], xs[j])
    return sims


@torch.no_grad()
def compare_prompts(
    model: GllamaForCausalLM,
    tokenizer,
    prompts: list[str],
    device: str = "cuda",
):
    hidden_list = []
    logits_list = []

    for text in prompts:
        out = get_last_hidden_and_logits(model, tokenizer, text, device=device)
        hidden_list.append(out["last_hidden"])
        logits_list.append(out["last_logits"])

    hidden_sims = cosine_matrix(hidden_list).cpu()
    logits_sims = cosine_matrix(logits_list).cpu()

    print("=== prompts ===")
    for i, p in enumerate(prompts):
        print(f"[{i}] {p}")

    print("\n=== last hidden cosine ===")
    print(hidden_sims)

    print("\n=== last logits cosine ===")
    print(logits_sims)

    return hidden_sims, logits_sims


@torch.no_grad()
def topk_from_prefix(
    model: GllamaForCausalLM,
    tokenizer,
    prefix: str,
    k: int = 10,
    device: str = "cuda",
):
    out = get_last_hidden_and_logits(model, tokenizer, prefix, device=device)
    last_logits = out["last_logits"]

    topk = torch.topk(last_logits, k=k)
    print(f"=== top-{k} for prefix ===")
    print(prefix)
    for token_id, score in zip(topk.indices.tolist(), topk.values.tolist()):
        piece = tokenizer.decode([token_id])
        print(token_id, repr(piece), float(score))


@torch.no_grad()
def target_rank_and_margin(
    model: GllamaForCausalLM,
    tokenizer,
    full_text: str,
    device: str = "cuda",
):
    """
    用 full_text 的最后一个 token 当 target。
    注意：这是“最后一个 token”测试，不是“最后一个字/词”测试。
    如果你的目标词会被切成多个 token，这个函数只测最后那个 token。
    """
    ids = tokenizer.encode(full_text, add_special_tokens=False)
    if len(ids) < 2:
        raise ValueError("文本至少要有 2 个 token")

    prefix_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    target_id = ids[-1]

    hidden_states = model.decoder(
        input_ids=prefix_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        use_padding_mask=False,
    )
    logits = model.lm_head(hidden_states)[0, -1].float()

    target_logit = logits[target_id]
    rank = (logits > target_logit).sum().item() + 1

    top1_val, top1_idx = torch.topk(logits, k=1)
    top1_id = top1_idx.item()
    top1_logit = top1_val.item()

    # margin:
    # - 如果 target 不是 top1，则 margin = target - top1_wrong，通常为负
    # - 如果 target 恰好是 top1，则 margin = target - second_best，为正
    if top1_id != target_id:
        margin = (target_logit - logits[top1_id]).item()
    else:
        top2_val, top2_idx = torch.topk(logits, k=2)
        second_id = top2_idx[1].item()
        margin = (target_logit - logits[second_id]).item()

    print("=== target rank / margin ===")
    print("full_text       :", full_text)
    print("target token id :", target_id)
    print("target token    :", repr(tokenizer.decode([target_id])))
    print("target logit    :", float(target_logit))
    print("rank            :", rank)
    print("margin          :", float(margin))
    print("top1 token      :", repr(tokenizer.decode([top1_id])))
    print("top1 logit      :", float(top1_logit))

    return {
        "target_id": target_id,
        "target_token": tokenizer.decode([target_id]),
        "target_logit": float(target_logit),
        "rank": int(rank),
        "margin": float(margin),
        "top1_id": int(top1_id),
        "top1_token": tokenizer.decode([top1_id]),
        "top1_logit": float(top1_logit),
    }


if __name__ == "__main__":
    tokenizer, model = load_model_and_tokenizer(MODEL_PATH, device=DEVICE)

    prompts = [
        "今天天气很好，所以我决定",
        "秦始皇统一六国之后，",
        "这段 Python 代码报错的原因是",
        "猫坐在窗台上，突然",
        "迄今人类所有的食物确实都来自野生生物",
    ]

    compare_prompts(model, tokenizer, prompts, device=DEVICE)

    print()
    topk_from_prefix(model, tokenizer, "今天天气很好，适合出门散", k=10, device=DEVICE)

    print()
    target_rank_and_margin(model, tokenizer, "今天天气很好，适合出门散步", device=DEVICE)
