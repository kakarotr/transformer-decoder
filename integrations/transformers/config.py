from typing import no_type_check

from transformers import PretrainedConfig


class GllamaConfig(PretrainedConfig):
    model_type = "gllama_decoder"

    @no_type_check
    def __init__(
        self,
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
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.rms_eps = rms_eps
        self.rope_base = rope_base
        self.dropout_prob = dropout_prob
        self.use_cache = use_cache
        self.is_decoder = True
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_act = "silu"
