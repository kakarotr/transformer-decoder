import torch
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from transformers import AutoTokenizer

from models.config import TransformerConfig
from models.liger.causal_lm import CausalLanguageModel

tokenizer = AutoTokenizer.from_pretrained("artifacts/base")
with open("artifacts/config.json", mode="r", encoding="utf-8") as f:
    config = TransformerConfig.model_validate_json(f.read())
model = CausalLanguageModel(config=config)

model.train()
sample_ids = tokenizer("中国的首都是北京。", return_tensors="pt")["input_ids"].to("cuda")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer.zero_grad()

with torch.autocast("cuda", dtype=torch.bfloat16):
    hidden_states = model.decoder(sample_ids)
    print(f"hidden states type: {hidden_states.dtype}")
    print(f"lm weight type: {model.lm_head.weight.dtype}")
    weight = model.lm_head.weight.to(hidden_states.dtype)

    loss_fn = LigerFusedLinearCrossEntropyLoss()

    shift_h = hidden_states[:, :-1].contiguous().view(-1, hidden_states.size(-1))
    shift_l = sample_ids[:, 1:].contiguous().view(-1)
    loss = loss_fn(weight, shift_h, shift_l)

loss.backward()

print(f"lm_head.weight.grad: {model.lm_head.weight.grad}")
