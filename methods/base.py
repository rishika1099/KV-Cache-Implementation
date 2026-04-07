from abc import ABC, abstractmethod


class MethodWrapper(ABC):
    """
    All methods implement this interface.
    The runner calls these hooks — methods never touch the generate loop directly.
    """

    @abstractmethod
    def process_prefill(self, past_key_values, attention_weights=None):
        """
        Called once after prefill completes.
        past_key_values: tuple of (K, V) tensors per layer, each shape
                         (batch, num_heads, seq_len, head_dim)
        attention_weights: tuple of attn weight tensors per layer (may be None)
        Returns: modified past_key_values
        """
        pass

    @abstractmethod
    def process_step(self, past_key_values, step, attention_weights=None):
        """
        Called after every decode step.
        step: int, current decode step index (0-indexed)
        Returns: modified past_key_values
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset all internal state between runs."""
        pass

    @property
    def needs_attention_weights(self) -> bool:
        """Return True if this method requires attention weights from prefill.
        Only SnapKV needs this. All others return False to avoid the massive
        memory allocation of output_attentions=True (which OOMs at seq_len>=8192).
        """
        return False

    def get_kv_size_bytes(self, past_key_values):
        """
        Compute actual bytes used by the KV cache representation.
        Override in each method to count method-specific storage correctly.
        Default: count all tensors in past_key_values.
        """
        total = 0
        for layer in past_key_values:
            for tensor in layer:
                if hasattr(tensor, 'nbytes'):
                    total += tensor.nbytes
                elif hasattr(tensor, 'element_size'):
                    total += tensor.numel() * tensor.element_size()
        return total
