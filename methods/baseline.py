import torch
from .base import MethodWrapper


class BaselineMethod(MethodWrapper):
    """
    FP16 full KV cache with no modification.
    This is the reference point. All compression ratios are computed relative to this.
    """

    def process_prefill(self, past_key_values, attention_weights=None):
        return past_key_values

    def process_step(self, past_key_values, step, attention_weights=None):
        return past_key_values

    def reset(self):
        pass

    def get_kv_size_bytes(self, past_key_values):
        """Sum of all K and V tensor bytes in fp16."""
        total = 0
        for layer in past_key_values:
            k, v = layer[0], layer[1]
            total += k.numel() * k.element_size()
            total += v.numel() * v.element_size()
        return total


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from benchmark.runner import generate_with_method

    model_name = "facebook/opt-125m"  # small model for smoke test
    print(f"Loading {model_name} for baseline smoke test...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    method = BaselineMethod()
    text, metrics = generate_with_method(
        model, tokenizer, method,
        prompt="Hello, my name is",
        max_new_tokens=10,
        device="cuda",
    )
    print(f"Generated: {text}")
    print(f"Metrics: {metrics}")
    print("Baseline smoke test PASSED")
