"""Triton kernel: fused two-shape decode attention, GLOBAL form
(exp-0013). The shared @triton.jit body lives in attn_swa.py
(_attn_decode_kernel; contract + numerics documented there); this
wrapper instantiates its IS_SWA=0 specialization: K/V read in place
from the layer's kv.GlobalPool flat page pool through the torch-built
`flat` [B, Lg] masked table-gather indices, key length Lg a runtime
argument (the eager path passes max(pos)+1, the CUDA-graph path the
static 512-bucket L_pad — the SAME padded columns the torch chain
masked, dead by l < n in-kernel)."""
import torch

from engine.pyengine.kernels.attn_swa import _attn_decode_kernel


def attn_global(q, pool_kp, pool_vp, flat, pos, rel, Lg):
    """Global decode attention over the layer's kv.GlobalPool, in place.
    q [B, HQ, D] bf16 (post q-norm + tau), pool_kp/pool_vp
    [(P+1), ps, HK, D] bf16 (indexed flat [(P+1)*ps, HK, D]), flat
    [B, Lg] i64 (masked page-table gather indices, model call site #3),
    pos [B] i64, rel [B, HQ, E] bf16 (tau-scaled rel-logit mix,
    E = rel_extent 1024). Returns [B, HQ, D]."""
    B, HQ, D = q.shape
    HK = pool_kp.shape[2]
    E = rel.shape[2]
    #1. contract checks — fail loud (contiguity is load-bearing)
    assert q.is_contiguous() and rel.is_contiguous()
    assert pool_kp.is_contiguous() and pool_vp.is_contiguous()
    assert flat.is_contiguous() and flat.shape == (B, Lg)
    assert flat.dtype == torch.int64 and pos.dtype == torch.int64
    out = torch.empty_like(q)
    _attn_decode_kernel[(B, HK)](
        q, out, pool_kp, pool_vp, rel, pos, pos, flat, Lg, 1.0 / D,
        E=E, HK=HK, REP=HQ // HK, HQ=HQ, D=D, W=0, IS_SWA=False,
        BLOCK_L=128, REP_PAD=16, num_warps=4)
    return out
