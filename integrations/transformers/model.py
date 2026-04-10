import math

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from integrations.transformers.config import GllamaConfig
from models.torch.decoder import Decoder


class GllamaPreTrainedModel(PreTrainedModel):
    config_class = GllamaConfig
    base_model_prefix = "model"
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


class GllamaModel(GllamaPreTrainedModel):
    def __init__(self, config: GllamaConfig):
        super().__init__(config)
        self.model = Decoder(
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
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value  # type: ignore

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache:
            raise NotImplementedError("当前还没实现 KV cache")

        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            use_padding_mask=attention_mask is not None,
        )

        if not return_dict:
            return (hidden_states, None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )


class GllamaForCausalLM(GllamaPreTrainedModel, GenerationMixin):
    def __init__(self, config: GllamaConfig):
        super().__init__(config)
        self.model = GllamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

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

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if not return_dict:
            return (loss, logits, outputs.past_key_values)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

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


if __name__ == "__main__":
    AutoConfig.register("gllama_decoder", GllamaConfig)
    AutoModel.register(GllamaConfig, GllamaModel)
    AutoModelForCausalLM.register(GllamaConfig, GllamaForCausalLM)

    config = GllamaConfig(
        vocab_size=32768,
        max_position_embeddings=2048,
        hidden_size=1536,
        num_layers=18,
        num_attention_heads=12,
        num_key_value_heads=12,
        intermediate_size=4608,
        rms_eps=1e-5,
        rope_base=10000,
        dropout_prob=0.0,
        pad_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        use_cache=False,
    )
    model = GllamaForCausalLM(config)
    model.save_pretrained("artifacts", safe_serialization=True)
