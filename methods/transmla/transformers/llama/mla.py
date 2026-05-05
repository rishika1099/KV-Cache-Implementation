# Shim for source-tree imports.
# modeling_llamamla.py uses `from .mla import ...` which resolves to this file
# when importing from the TransMLA source tree (mla.py lives one level up at
# transmla/transformers/mla.py, not inside this llama/ subfolder).
# When trust_remote_code loads from a flat save directory, the real mla.py is
# present and this shim is not needed — but it does not conflict either.
from ..mla import MLAAttention, eager_attention_forward
