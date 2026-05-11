import os
import subprocess
from contextlib import nullcontext

import torch
import torch.distributed
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from transformers import AutoTokenizer

from models.config import TransformerConfig
from models.liger.causal_lm import CausalLanguageModel

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

backend = "nccl" if torch.distributed.is_nccl_available() else "gloo"
torch.distributed.init_process_group(backend=backend)
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

tokenizer = AutoTokenizer.from_pretrained("artifacts/base")
with open("artifacts/config.json", mode="r", encoding="utf-8") as f:
    config = TransformerConfig.model_validate_json(f.read())
model = CausalLanguageModel(config=config).to(device)
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False,
    gradient_as_bucket_view=True,
    static_graph=False,
)

model.train()
sample_ids = tokenizer("中国的首都是北京。", return_tensors="pt")["input_ids"].to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer.zero_grad()

gradient_accumulation_steps = 4

optimizer.zero_grad(set_to_none=True)

for step in range(gradient_accumulation_steps):
    is_update_step = (step + 1) == gradient_accumulation_steps
    sync_ctx = model.no_sync() if not is_update_step else nullcontext()

    with sync_ctx:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden_states = model(sample_ids)
            weight = model.module.lm_head.weight.to(hidden_states.dtype)

            loss_fn = LigerFusedLinearCrossEntropyLoss()

            shift_h = hidden_states[:, :-1].contiguous().view(-1, hidden_states.size(-1))
            shift_l = sample_ids[:, 1:].contiguous().view(-1)
            loss = loss_fn(weight, shift_h, shift_l) / gradient_accumulation_steps
        loss.backward()

grad = model.module.lm_head.weight.grad
print(f"grad is None: {grad is None}")
if grad is not None:
    print(f"grad norm:     {grad.norm():.6f}")
    print(f"grad all zero: {grad.abs().max().item() == 0.0}")
