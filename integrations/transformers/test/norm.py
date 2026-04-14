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
prompts = [
    "今天天气很好，所以我决定",
    "秦始皇统一六国之后，",
    "这段 Python 代码报错的原因是",
    "猫坐在窗台上，突然",
    "迄今人类所有的食物确实都来自野生生物",
]

# 根据你的模型结构调整这里的 hook 位置
hidden_before_norm = {}
hidden_after_norm = {}


def hook_before(module, input):
    hidden_before_norm["val"] = input[0].detach()


def hook_after(module, input, output):
    hidden_after_norm["val"] = output.detach()


# 注册 hook，具体层名根据你的模型结构调整
# 常见命名：model.norm、model.final_norm、transformer.ln_f 等
handle_before = model.decoder.norm.register_forward_pre_hook(hook_before)
handle_after = model.decoder.norm.register_forward_hook(hook_after)

all_before = []
all_after = []

with torch.no_grad():
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        _ = model(**inputs)
        # 取最后一个 token 的向量
        all_before.append(hidden_before_norm["val"][0, -1])
        all_after.append(hidden_after_norm["val"][0, -1])

handle_before.remove()
handle_after.remove()


# 计算两两余弦
def pairwise_cosine(vecs):
    vecs = torch.stack(vecs)
    normed = F.normalize(vecs, dim=-1)
    sim = normed @ normed.T
    n = len(vecs)
    vals = [sim[i, j].item() for i in range(n) for j in range(i + 1, n)]
    return sum(vals) / len(vals)


print("final norm 之前的余弦均值:", pairwise_cosine(all_before))
print("final norm 之后的余弦均值:", pairwise_cosine(all_after))

norm = model.decoder.norm
print("weight mean:", norm.weight.mean().item())
print("weight std:", norm.weight.std().item())
print("weight max:", norm.weight.max().item())
print("weight min:", norm.weight.min().item())
