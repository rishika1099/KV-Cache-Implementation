"""
Method registry — single source of truth for all KV-cache methods.

Every experiment (kernel bench, ablation, K-sweep, profile, CLI runner)
loops over this registry. New methods (e.g. KIVI+TopK hybrid in Phase B)
are added by appending one entry — no experiment code is rewritten.

Each entry is a *factory* `(**kwargs) -> MethodWrapper`. Factories accept
any/all knobs and silently ignore the ones they don't use, so CLI tools
can pass a single flat kwargs dict to every method without branching.
"""

from typing import Callable, Dict, List, Any
from .base import MethodWrapper
from .baseline import BaselineMethod
from .kivi_quant import KIVIMethod
from .topk_selection import TopKMethod
from .kivi_topk_hybrid import KIVI_TopK_Method


# ── Factory functions ─────────────────────────────────────────────────────────
# Each factory takes **kwargs and pulls out only the params it needs. This lets
# the CLI / experiments pass one flat config dict to every method uniformly.

def _make_baseline(**_) -> MethodWrapper:
    return BaselineMethod()


def _make_kivi(*, bits: int = 4, residual_length: int = 128,
               group_size: int = 32, **_) -> MethodWrapper:
    return KIVIMethod(
        bits=bits,
        residual_length=residual_length,
        group_size=group_size,
    )


def _make_topk(*, K: int = 512, n_sink: int = 128, n_local: int = 512,
               refresh_interval: int = 50, page_size: int = 64,
               cache_similarity_threshold: float = 0.95,
               chunk_size: int = 2048, use_kernels: bool = True,
               # Position-encoding (BUG-2 fix)
               head_dim: int = 128,
               rope_theta: float = 10000.0,
               apply_rope_correction: bool = False,
               # ── Phase A ablation flags ──
               use_head_softmax: bool = True,
               use_selection_cache: bool = True,
               use_sink_tokens: bool = True,
               use_local_tokens: bool = True,
               use_criticality_weights: bool = True,
               **_) -> MethodWrapper:
    return TopKMethod(
        K=K,
        n_sink=n_sink,
        n_local=n_local,
        refresh_interval=refresh_interval,
        page_size=page_size,
        cache_similarity_threshold=cache_similarity_threshold,
        chunk_size=chunk_size,
        use_kernels=use_kernels,
        head_dim=head_dim,
        rope_theta=rope_theta,
        apply_rope_correction=apply_rope_correction,
        use_head_softmax=use_head_softmax,
        use_selection_cache=use_selection_cache,
        use_sink_tokens=use_sink_tokens,
        use_local_tokens=use_local_tokens,
        use_criticality_weights=use_criticality_weights,
    )


def _make_kivi_topk(*,
                    # KIVI side
                    bits: int = 4, residual_length: int = 128,
                    group_size: int = 32,
                    # TopK side
                    K: int = 1024, n_sink: int = 128, n_local: int = 512,
                    refresh_interval: int = 50,
                    cache_similarity_threshold: float = 0.95,
                    # Hybrid scoring path: "centroid" (a) | "quantized" (c)
                    score_mode: str = "centroid",
                    # Position-encoding (BUG-2 fix)
                    head_dim: int = 128,
                    rope_theta: float = 10000.0,
                    apply_rope_correction: bool = False,
                    # Ablation flags (consistent with topk)
                    use_head_softmax: bool = True,
                    use_criticality_weights: bool = True,
                    use_selection_cache: bool = True,
                    use_sink_tokens: bool = True,
                    use_local_tokens: bool = True,
                    **_) -> MethodWrapper:
    return KIVI_TopK_Method(
        bits=bits, residual_length=residual_length, group_size=group_size,
        K=K, n_sink=n_sink, n_local=n_local,
        refresh_interval=refresh_interval,
        cache_similarity_threshold=cache_similarity_threshold,
        score_mode=score_mode,
        head_dim=head_dim,
        rope_theta=rope_theta,
        apply_rope_correction=apply_rope_correction,
        use_head_softmax=use_head_softmax,
        use_criticality_weights=use_criticality_weights,
        use_selection_cache=use_selection_cache,
        use_sink_tokens=use_sink_tokens,
        use_local_tokens=use_local_tokens,
    )


def _make_kivi_topk_c(**kwargs) -> MethodWrapper:
    """Design (c): same hybrid, score_mode='quantized'. Convenience alias."""
    kwargs["score_mode"] = "quantized"
    return _make_kivi_topk(**kwargs)


# ── Public registry ──────────────────────────────────────────────────────────

METHOD_REGISTRY: Dict[str, Callable[..., MethodWrapper]] = {
    "baseline":     _make_baseline,
    "kivi":         _make_kivi,
    "topk":         _make_topk,
    "kivi_topk":    _make_kivi_topk,    # Phase B hybrid (design a)
    "kivi_topk_c":  _make_kivi_topk_c,  # Phase B hybrid (design c)
}


def make_method(name: str, **kwargs) -> MethodWrapper:
    """Construct a method by name. Extra kwargs are forwarded to the factory."""
    if name not in METHOD_REGISTRY:
        raise KeyError(
            f"Unknown method '{name}'. "
            f"Registered: {sorted(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[name](**kwargs)


def list_methods() -> List[str]:
    return sorted(METHOD_REGISTRY.keys())


def describe(name: str) -> Dict[str, Any]:
    """Return a short summary of a method (used by `--help` in CLI)."""
    descriptions = {
        "baseline":  "FP16 full KV cache (reference point)",
        "kivi":      "KIVI asymmetric block-wise quantisation (4-bit / 2-bit)",
        "topk":      "TokenSelect dynamic top-K selection (Triton fused kernels)",
        "kivi_topk":   "Phase B hybrid: KIVI storage + centroid-scored top-K (design a)",
        "kivi_topk_c": "Phase B hybrid: KIVI storage + Triton quant-aware top-K (design c)",
    }
    return {"name": name, "description": descriptions.get(name, "")}
