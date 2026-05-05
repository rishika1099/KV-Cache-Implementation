"""
LlamaMLALatentForCausalLM — Llama MLA model that uses latent KV caching.

This module provides LlamaMLALatentForCausalLM, a drop-in replacement for
LlamaMLAForCausalLM where every attention layer is swapped to MLALatentAttention.
When past_key_values is an MLALatentCache, the model caches (c_kv_norm, k_rot_roped)
instead of full (K, V), achieving ~14-18x KV-cache memory reduction.  When
past_key_values is a standard DynamicCache, behaviour is identical to the base class.

Architecture
------------
  LlamaMLALatentForCausalLM
    └─ LlamaMLALatentModel
         └─ [LlamaMLALatentDecoderLayer × num_layers]
              └─ MLALatentAttention  (from mla_latent.py)

Loading pre-converted TransMLA model weights
--------------------------------------------
The TransMLA converter saves model weights + config to a directory.  Load them
directly with this class to bypass trust_remote_code and get latent caching:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path("<repo>/TransMLA_NeurIPS_2025")))

    from transmla.transformers.llama.configuration_llamamla import LlamaMLAConfig
    from transmla.transformers.llama.modeling_llamamla_latent import LlamaMLALatentForCausalLM
    from transmla.transformers.mla_latent import MLALatentCache

    mla_path = "xxx/llama2-7B-deepseek"
    config   = LlamaMLAConfig.from_pretrained(mla_path)
    model    = LlamaMLALatentForCausalLM.from_pretrained(mla_path, config=config)
    model.eval().to("cuda")

    cache = MLALatentCache()
    out   = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    # Decode loop:
    for _ in range(max_new_tokens):
        next_tok = out.logits[:, -1].argmax(-1, keepdim=True)
        out = model(input_ids=next_tok, past_key_values=out.past_key_values, use_cache=True)
    print(cache.get_cache_bytes() / 1e6, "MB")   # actual latent cache size
"""

from torch import nn

from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaDecoderLayer,
    LlamaForCausalLM,
)

from .configuration_llamamla import LlamaMLAConfig
from .modeling_llamamla import LlamaMLAPreTrainedModel
from ..mla_latent import MLALatentAttention


class LlamaMLALatentDecoderLayer(LlamaDecoderLayer):
    """LlamaDecoderLayer with MLALatentAttention instead of standard attention."""

    def __init__(self, config: LlamaMLAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MLALatentAttention(config, layer_idx)


class LlamaMLALatentModel(LlamaMLAPreTrainedModel, LlamaModel):

    def __init__(self, config: LlamaMLAConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LlamaMLALatentDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )


class LlamaMLALatentForCausalLM(LlamaMLAPreTrainedModel, LlamaForCausalLM):
    """
    Llama MLA causal LM with latent KV caching.

    Weights are fully compatible with LlamaMLAForCausalLM checkpoints — all
    parameter names and shapes are identical.  Only the attention forward() is
    different (latent cache path when past_key_values is MLALatentCache).
    """

    def __init__(self, config: LlamaMLAConfig):
        super().__init__(config)
        self.model = LlamaMLALatentModel(config)


__all__ = [
    "LlamaMLALatentForCausalLM",
    "LlamaMLALatentModel",
]
