from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from integrations.transformers.config import GllamaConfig
from integrations.transformers.model import GllamaModel

AutoConfig.register("gllama_decoder", GllamaConfig)
AutoModel.register(GllamaConfig, GllamaModel)
AutoModelForCausalLM.register(GllamaConfig, GllamaModel)
