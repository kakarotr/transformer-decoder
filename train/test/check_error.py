import math

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from integrations.transformers.config import GllamaConfig
from integrations.transformers.model import GllamaForCausalLM

device = torch.device("cuda")
config = GllamaConfig.from_pretrained("artifacts")
model = GllamaForCausalLM.from_pretrained("artifacts", config=config).to(device)
tokenizer = AutoTokenizer.from_pretrained("artifacts/base")

prompt = "产品设计中被低估的建模插件，"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    logits = outputs.logits  # [1, seq_len, vocab_size]
    hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]

print("检查 logit 分布")
last_logits = logits[0, -1]  # 最后一个位置
print(f"logit max: {last_logits.max():.2f}")
print(f"logit min: {last_logits.min():.2f}")
print(f"logit std: {last_logits.std():.2f}")
print(f"top-5 token ids: {last_logits.topk(5).indices.tolist()}")
print(f"top-5 tokens: {[tokenizer.decode([i]) for i in last_logits.topk(5).indices.tolist()]}")
print("\n\n")

print("检查 hidden state 的 DC 分量")
print(f"\nhidden mean (DC): {hidden.mean().item():.4f}")
print(f"hidden std: {hidden.std().item():.4f}")
print("\n\n")

print("Loss 和 PPL")
with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    print(f"loss: {loss.item():.4f}")
    print(f"ppl:  {math.exp(loss.item()):.2f}")
    print("\n\n")

print("检查 权重文件的 lm_head 是否被正确映射")
ckpt = load_file("artifacts/model.safetensors")
lm_head_keys = [k for k in ckpt.keys() if "lm_head" in k]
print("checkpoint 里的 lm_head keys:", lm_head_keys)
model_keys = [k for k in model.state_dict().keys() if "lm_head" in k]
print("transformers model 里的 lm_head keys:", model_keys)
print("\n\n")

print("检查 transformers 集成的 lm_head 权重的范数")
lm_head_weight = model.lm_head.weight
print(f"lm_head weight norm: {lm_head_weight.norm().item():.4f}")
print(f"lm_head weight std:  {lm_head_weight.std().item():.6f}")
print(f"lm_head weight mean: {lm_head_weight.mean().item():.6f}")
print("\n\n")

print("检查权重文件的 lm_head 权重的范数")
lm_head_weight = ckpt["lm_head.weight"]
print(f"lm_head weight norm: {lm_head_weight.norm().item():.4f}")
print(f"lm_head weight std:  {lm_head_weight.std().item():.6f}")
print(f"lm_head weight mean: {lm_head_weight.mean().item():.6f}")
print("\n\n")

print(f"{'pos':>3} {'label token':>15} {'correct logit':>14} {'max logit':>10} {'max token':>15}")
for i in range(shift_labels.size(1)):
    correct_id = shift_labels[0, i].item()
    correct_logit = shift_logits[0, i, correct_id].item()
    max_logit = shift_logits[0, i].max().item()
    max_id = shift_logits[0, i].argmax().item()
    print(
        f"{i:>3} {repr(tokenizer.decode([correct_id])):>15} {correct_logit:>14.3f} {max_logit:>10.3f} {repr(tokenizer.decode([max_id])):>15}"
    )

# 找到 ' ' 的 token id
space_id = [8]
print(f"space token id: {space_id}")

# 检查 lm_head 中这个 token 的权重范数
lm_weight = model.lm_head.weight.float()
for tid in space_id:
    print(f"token {tid} ({repr(tokenizer.decode([tid]))}) lm_head row norm: {lm_weight[tid].norm():.4f}")

# 对比几个普通 token
for tid in [19540, 21234, 21483]:  # 首、都是、北京
    print(f"token {tid} ({repr(tokenizer.decode([tid]))}) lm_head row norm: {lm_weight[tid].norm():.4f}")

# 全局分布
row_norms = lm_weight.norm(dim=1)
print(f"\nlm_head row norm: mean={row_norms.mean():.4f}, std={row_norms.std():.4f}, max={row_norms.max():.4f}")
print(f"max norm token: {row_norms.argmax().item()} = {repr(tokenizer.decode([row_norms.argmax().item()]))}")
print("\n\n")

embed_weight = model.decoder.embeddings.weight.float()
print(f"embedding std:  {embed_weight.std():.6f}")
print(f"embedding mean: {embed_weight.mean():.6f}")

row_norms = embed_weight.norm(dim=1)
print(f"embedding row norm: mean={row_norms.mean():.4f}, max={row_norms.max():.4f}, std={row_norms.std():.4f}")