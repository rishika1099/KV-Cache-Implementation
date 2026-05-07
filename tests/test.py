"""
test_unit.py — fast mechanical tests for SnapKV, no model required.

Runs in seconds on CPU with fake tensors.
Tests the algorithm logic directly, not model quality.

Usage:
    python tests/test_unit.py
"""

import sys
import os
import torch
import torch.nn.functional as F
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from methods.snapkv_eviction import SnapKVMethod

PASS = "✓"
FAIL = "✗"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name))
    print(f"  {status}  {name}")
    if not condition and detail:
        print(f"       → {detail}")

def section(title):
    print(f"\n── {title}")

# ---------------------------------------------------------------------------
# Fake KV and attention tensors
# ---------------------------------------------------------------------------

def make_fake_inputs(batch=1, heads=4, seq_len=40, head_dim=8, obs=8):
    """
    Returns (past_key_values, attention_weights) in the shape
    process_prefill expects, with one layer.
    Attention is crafted so tokens 2 and 5 in the prefix are clearly important.
    """
    prefix_len = seq_len - obs

    # Attention: obs window strongly attends to positions 2 and 5
    attn = torch.zeros(batch, heads, seq_len, seq_len)
    # causal: each token attends to itself and earlier tokens
    for i in range(seq_len):
        attn[:, :, i, :i+1] = 0.01
    # obs queries heavily attend to prefix positions 2 and 5
    attn[:, :, -obs:, 2] = 1.0
    attn[:, :, -obs:, 5] = 0.8
    # softmax-normalise along key dim (approximate)
    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)

    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)

    past_key_values = ((k, v),)          # one layer
    attention_weights = (attn,)
    return past_key_values, attention_weights, k, v

# ---------------------------------------------------------------------------

section("1. Output shape is correct")

method = SnapKVMethod(budget_ratio=0.4, observation_window=8, kernel_size=3)
seq_len, obs = 40, 8
pkv, attn, k, v = make_fake_inputs(seq_len=seq_len, obs=obs)

out = method.process_prefill(pkv, attention_weights=attn)
k_out, v_out = out[0]

prefix_len = seq_len - obs
budget_k = max(1, int(0.4 * seq_len))
expected_len = min(budget_k, prefix_len) + obs

check("output has same number of layers as input", len(out) == 1)
check("k and v have same seq dim", k_out.shape[2] == v_out.shape[2])
check(
    f"compressed seq len = budget_k + obs ({expected_len})",
    k_out.shape[2] == expected_len,
    f"got {k_out.shape[2]}, expected {expected_len}"
)
check("head_dim preserved", k_out.shape[3] == k.shape[3])
check("heads preserved", k_out.shape[1] == k.shape[1])

# ---------------------------------------------------------------------------

section("2. Observation window is always fully retained")

# The last `obs` tokens of k should appear verbatim in k_out
k_obs_original = k[:, :, -obs:, :]
k_obs_in_out   = k_out[:, :, -obs:, :]

check(
    "last obs tokens in output match original obs tokens",
    torch.allclose(k_obs_original, k_obs_in_out),
    "obs window tokens were modified or dropped"
)

# ---------------------------------------------------------------------------

section("3. Clearly important tokens are selected")

# Tokens 2 and 5 have very high attention from the obs window.
# After voting and pooling they should be in the top-k for every head.
# We check by looking at which prefix positions ended up in k_out.

# Recover selected indices by matching columns of k_out[:,:,:prefix_k,:]
# against k[:,:,:prefix_len,:]
prefix_k = k_out.shape[2] - obs
k_prefix_out = k_out[:, :, :prefix_k, :]   # selected prefix KVs
k_prefix_all = k[:, :, :prefix_len, :]     # all original prefix KVs

# For head 0, batch 0: find which original indices appear in the output
selected_indices = []
for pos in range(prefix_len):
    col = k_prefix_all[0, 0, pos, :]  # (head_dim,)
    for j in range(prefix_k):
        if torch.allclose(col, k_prefix_out[0, 0, j, :], atol=1e-5):
            selected_indices.append(pos)
            break

check(
    "token 2 (high attention) is selected",
    2 in selected_indices,
    f"selected indices: {sorted(selected_indices)}"
)
check(
    "token 5 (high attention) is selected",
    5 in selected_indices,
    f"selected indices: {sorted(selected_indices)}"
)

# ---------------------------------------------------------------------------

section("4. Low-attention tokens are evicted")

# Token 15 gets no special attention — should be evicted when budget is tight.
# (budget_ratio=0.4 of 40 = 16 slots; prefix has 32 tokens; so ~half evicted)
check(
    "not all prefix tokens are kept (compression happened)",
    prefix_k < prefix_len,
    f"prefix_k={prefix_k} should be < prefix_len={prefix_len}"
)

# ---------------------------------------------------------------------------

section("5. Pooling is applied before topk (order matters)")

# With kernel=1 (no pool), selection is purely by raw vote.
# With kernel=7, neighbouring tokens get boosted.
# We test: a token with a mediocre raw score but a high-scoring neighbour
# should be selected with pooling but potentially not without.

method_no_pool = SnapKVMethod(budget_ratio=0.15, observation_window=8, kernel_size=1)
method_pooled  = SnapKVMethod(budget_ratio=0.15, observation_window=8, kernel_size=7)

# Craft attention: token 10 is very important, token 11 has zero raw score
# but is right next to 10 — pooling should pull 11 in.
seq2 = 40
attn2 = torch.zeros(1, 4, seq2, seq2)
attn2[:, :, -8:, 10] = 1.0   # token 10: very important
attn2[:, :, -8:, 11] = 0.0   # token 11: zero raw score, neighbour of 10
attn2 = attn2 / (attn2.sum(dim=-1, keepdim=True) + 1e-9)
k2 = torch.randn(1, 4, seq2, 8)
v2 = torch.randn(1, 4, seq2, 8)
pkv2 = ((k2, v2),)
attn2_t = (attn2,)

out_np = method_no_pool.process_prefill(pkv2, attention_weights=attn2_t)
out_p  = method_pooled.process_prefill(pkv2, attention_weights=attn2_t)

prefix_len2 = seq2 - 8

def get_selected(k_original, k_compressed, prefix_len):
    """Return set of original prefix indices that appear in k_compressed."""
    n_selected = k_compressed.shape[2] - 8  # subtract obs
    k_pre = k_original[0, 0, :prefix_len, :]
    k_sel = k_compressed[0, 0, :n_selected, :]
    sel = set()
    for pos in range(prefix_len):
        for j in range(n_selected):
            if torch.allclose(k_pre[pos], k_sel[j], atol=1e-5):
                sel.add(pos)
                break
    return sel

sel_np = get_selected(k2, out_np[0][0], prefix_len2)
sel_p  = get_selected(k2, out_p[0][0],  prefix_len2)

check("token 10 selected by both (raw score is high)", 10 in sel_np and 10 in sel_p)
check(
    "token 11 selected by pooled but not no-pool (neighbour boost)",
    11 in sel_p and 11 not in sel_np,
    f"no-pool selected: {sorted(sel_np)}, pooled selected: {sorted(sel_p)}"
)

# ---------------------------------------------------------------------------

section("6. process_step is identity (KV unchanged during decode)")

method_s = SnapKVMethod()
fake_pkv = ((torch.randn(1, 4, 10, 8), torch.randn(1, 4, 10, 8)),)
out_step = method_s.process_step(fake_pkv, step=0)

check("process_step returns same object", out_step is fake_pkv)

# ---------------------------------------------------------------------------

section("7. No-op when prompt shorter than obs window")

method_short = SnapKVMethod(budget_ratio=0.4, observation_window=50)
short_seq = 20  # shorter than obs window
k_s = torch.randn(1, 4, short_seq, 8)
v_s = torch.randn(1, 4, short_seq, 8)
attn_s = torch.zeros(1, 4, short_seq, short_seq)
attn_s = attn_s / (attn_s.sum(dim=-1, keepdim=True) + 1e-9)

out_s = method_short.process_prefill(((k_s, v_s),), attention_weights=(attn_s,))
k_s_out, v_s_out = out_s[0]

check(
    "short prompt passes through unchanged",
    k_s_out.shape == k_s.shape,
    f"got {k_s_out.shape}, expected {k_s.shape}"
)

# ---------------------------------------------------------------------------

section("8. Handles None attention_weights gracefully")

out_none = method.process_prefill(pkv, attention_weights=None)
check(
    "returns original pkv when attention_weights=None",
    out_none is pkv
)

# ---------------------------------------------------------------------------

section("9. KV cache is strictly smaller after compression")

method_c = SnapKVMethod(budget_ratio=0.3, observation_window=8, kernel_size=3)
pkv_c, attn_c, k_c, v_c = make_fake_inputs(seq_len=60, obs=8)
out_c = method_c.process_prefill(pkv_c, attention_weights=attn_c)

bytes_before = k_c.numel() * k_c.element_size() * 2  # k + v
bytes_after  = method_c.get_kv_size_bytes(out_c)

check(
    "compressed KV is smaller than original",
    bytes_after < bytes_before,
    f"before={bytes_before}B after={bytes_after}B"
)

ratio = bytes_after / bytes_before
check(
    f"compression ratio is meaningful (<80% of original, got {ratio:.0%})",
    ratio < 0.80,
    f"ratio={ratio:.0%} — budget_ratio=0.3 should compress more aggressively"
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

passed = sum(1 for s, _ in results if s == PASS)
total  = len(results)
print(f"\n{'='*45}")
print(f"  {passed}/{total} tests passed")
print(f"{'='*45}")
if passed == total:
    print("  All mechanical tests pass. Safe to run the benchmark.")
else:
    failed = [(s, n) for s, n in results if s == FAIL]
    print(f"  {len(failed)} failure(s):")
    for _, name in failed:
        print(f"    ✗ {name}")
    sys.exit(1)