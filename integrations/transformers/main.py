import torch
from transformers import AutoTokenizer

from integrations.transformers.model import GllamaForCausalLM

model = GllamaForCausalLM.from_pretrained("artifacts")
tokenizer = AutoTokenizer.from_pretrained("artifacts")

inputs = tokenizer(
    "你好",
    return_tensors="pt",
    return_token_type_ids=False,
)
with torch.no_grad():
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
