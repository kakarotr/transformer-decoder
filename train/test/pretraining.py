import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from models.config import TransformerConfig
from models.torch.causal_lm import CausalLanguageModel


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens=200,
    device="cuda",
    temperature=0.9,
    top_p=0.9,
    repetition_penalty=1.15,
):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    # 处理 DDP/compile 包装
    raw_model = model
    if hasattr(model, "module"):
        raw_model = model.module

    for _ in range(max_new_tokens):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden_states = raw_model(generated)
            logits = raw_model.lm_head(hidden_states[:, -1, :]).float()

        # 正确的 repetition penalty
        for token_id in set(generated[0].tolist()):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

        logits = logits / temperature

        # Top-p sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[remove_mask] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


@torch.no_grad()
def inspect_next_token(model, tokenizer, prompt, device="cuda"):
    raw_model = model.module if hasattr(model, "module") else model
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden_states = raw_model(input_ids)
        logits = raw_model.lm_head(hidden_states[:, -1, :]).float()

    probs = torch.softmax(logits, dim=-1)[0]
    top_probs, top_ids = probs.topk(10)

    for prob, tid in zip(top_probs, top_ids):
        print(f"{tokenizer.decode([tid.item()])!r:15s} {prob.item():.4f}")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("artifacts")
    with open("artifacts/config.json", mode="r", encoding="utf-8") as f:
        config = TransformerConfig.model_validate_json(f.read())
    model = CausalLanguageModel(config).to("cuda")
    state_dict = load_file("artifacts/model.safetensors")
    model.load_state_dict(state_dict)

    test_cases = {
        "续写_新闻": "据新华社报道，本次会议的主要议题是",
        "续写_叙述": "他走进图书馆，发现里面几乎没有人。他找了个靠窗的位置坐下，",
        "续写_说明": "量子计算机与传统计算机的本质区别在于",
        "知识_历史": "中国古代四大发明分别是",
        "知识_地理": "长江是中国最长的河流，全长约",
        "知识_科学": "光合作用是植物利用",
        "领域_战国": "日本战国时代，织田信长最重要的军事革新是",
        "领域_二战": "第二次世界大战爆发的直接导火索是",
    }
    for prompt in test_cases.values():
        print(f"输入: {prompt}")
        print(f"输出: {generate(model, tokenizer, prompt)}")
        print("---\n")

    # while True:
    #     prompt = input("请输入：")
    #     print(generate(model, tokenizer, prompt))
    #     print("\n")
