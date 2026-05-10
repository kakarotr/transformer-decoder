import math

import torch
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from integrations.transformers.config import GllamaConfig
from models.liger.decoder import Decoder


class GllamaPreTrainedModel(PreTrainedModel):
    config_class = GllamaConfig
    base_model_prefix = "decoder"
    main_input_name = "input_ids"

    def _init_weights(self, module):
        base_std = 0.02
        residual_std = base_std / math.sqrt(2 * self.config.num_layers)

        # Embedding
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=base_std)

            pad_token_id = getattr(self.config, "pad_token_id", None)
            if pad_token_id is not None and 0 <= pad_token_id < module.num_embeddings:
                with torch.no_grad():
                    module.weight[pad_token_id].zero_()
            return

        # Linear
        if isinstance(module, nn.Linear):
            name = getattr(module, "_hf_init_name", "")

            if name in {"q_proj", "kv_proj", "gate_up_proj", "gate_proj", "up_proj", "lm_head"}:
                nn.init.normal_(module.weight, mean=0.0, std=base_std)
            elif name in {"o_proj", "down_proj"}:
                nn.init.normal_(module.weight, mean=0.0, std=residual_std)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=base_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            return

        # RMSNorm
        if hasattr(module, "weight") and isinstance(getattr(module, "weight"), torch.nn.Parameter):
            weight = getattr(module, "weight")
            if weight is not None and weight.ndim == 1:
                nn.init.ones_(weight)

        if hasattr(module, "bias") and isinstance(getattr(module, "bias"), torch.nn.Parameter):
            bias = getattr(module, "bias")
            if bias is not None:
                nn.init.zeros_(bias)


class GllamaForCausalLM(GllamaPreTrainedModel, GenerationMixin):
    def __init__(self, config: GllamaConfig):
        super().__init__(config)
        self.decoder = Decoder(
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            num_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            rms_eps=config.rms_eps,
            dropout_prob=config.dropout_prob,
            rope_base=config.rope_base,
            pad_token_id=config.pad_token_id,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embeddings

    def set_input_embeddings(self, value):
        self.decoder.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache:
            raise NotImplementedError("当前还没实现 KV cache")

        hidden_states = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            use_padding_mask=attention_mask is not None,
        )
        logits = self.lm_head(hidden_states)

        loss = None

        if not return_dict:
            return (loss, logits, None)

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None, hidden_states=(hidden_states,))

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "use_cache": False,
        }
