from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class TransformerConfig(BaseModel):
    vocab_size: Annotated[int, Field(description="词表大小")]
    max_position_embeddings: Annotated[int, Field(description="输入序列最大Token长度")]
    hidden_size: Annotated[int, Field(description="Token维度")]
    num_layers: Annotated[int, Field(description="隐藏层数量")]
    num_attention_heads: Annotated[int, Field(description="多头注意力头数")]
    num_key_value_heads: Annotated[int, Field(description="KV组数")]
    dropout_prob: Annotated[float, Field(description="Dropout强度")]
    intermediate_size: Annotated[int, Field(description="多层感知机升维维度")]
    rms_eps: Annotated[float, Field(description="RMS指数")]
    rope_base: Annotated[int, Field(description="ROPE旋转基数")]
    pad_token_id: Annotated[int, Field(description="Pad Token Id")]

    @model_validator(mode="after")
    def validate_model_params(self) -> "TransformerConfig":
        if self.num_attention_heads == 0:
            raise ValueError("num_attention_heads cannot be zero")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if (self.hidden_size // self.num_attention_heads) % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.hidden_size // self.num_attention_heads}")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                "num_key_value_heads ({self.num_key_value_heads}) to support GQA/MQA."
            )
        return self

    def compute_size(self):
        attention_size = (
            self.hidden_size * self.hidden_size
            + self.hidden_size * (self.hidden_size // self.num_attention_heads) * self.num_attention_heads * 2
            + self.hidden_size * self.hidden_size
        )
        mlp_size = self.hidden_size * self.intermediate_size * 3
        norm_size = self.hidden_size
        one_layer_size = attention_size + mlp_size + norm_size * 2

        embedding_size = self.hidden_size * self.vocab_size
        lm_head_size = self.hidden_size * self.vocab_size

        total = one_layer_size * self.num_layers + embedding_size + norm_size + lm_head_size
        return {
            "total": one_layer_size * self.num_layers + embedding_size + norm_size + lm_head_size,
            "compture_size": total - embedding_size - lm_head_size,
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    with open("artifacts/config.json", mode="r") as f:
        print(TransformerConfig.model_validate_json(f.read()).compute_size())
    tokenizer = AutoTokenizer.from_pretrained("artifacts/base")
    print(len(tokenizer))
