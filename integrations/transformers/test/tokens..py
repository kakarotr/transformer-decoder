import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from integrations.transformers.config import GllamaConfig
from integrations.transformers.model import GllamaForCausalLM

device = torch.device("cuda")

config = GllamaConfig.from_pretrained("artifacts")
model = GllamaForCausalLM.from_pretrained("artifacts", config=config).to(device)
tokenizer = AutoTokenizer.from_pretrained("artifacts")

model.eval()
high_freq_tokens = ["的", "了", "是", "在", "，", "。", "我", "也", "有", "不"]
normal_tokens = ["学习", "方法", "科学", "经济", "文化", "发展", "社会", "政治", "教育", "技术"]
# 收集多个 prompt 的 hidden state，提取均值向量（DC 分量）
prompts = [
    "秦始皇统一六国之后，",
    "今天天气很好，",
    "这段代码报错的原因是",
    "猫坐在窗台上，",
    "学习的方法有很多，",
]

all_hidden = []
handle = model.decoder.norm.register_forward_hook(lambda m, i, o: all_hidden.append(o.detach()[0, -1].float()))

model.eval()
with torch.no_grad():
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        _ = model(**inputs)

handle.remove()

stacked = torch.stack(all_hidden)  # [N, hidden_dim]
dc = stacked.mean(dim=0)  # DC 分量，[hidden_dim]
dc_norm = F.normalize(dc.unsqueeze(0), dim=-1)

W = model.lm_head.weight.detach().float()
W_norm = F.normalize(W, dim=-1)  # [vocab_size, hidden_dim]

# DC 分量和所有 token 行向量的余弦
cos_all = (W_norm @ dc_norm.T).squeeze()  # [vocab_size]

# 取 top20 最对齐的 token
topk = torch.topk(cos_all, 20)
print("和 DC 分量余弦最高的 20 个 token：")
for score, idx in zip(topk.values, topk.indices):
    print(f"  {tokenizer.decode([idx.item()])!r} (id={idx.item()}): {score.item():.4f}")

print(f"\nDC 分量的模长: {dc.norm().item():.4f}")
print(f"单个样本模长均值: {stacked.norm(dim=-1).mean().item():.4f}")
