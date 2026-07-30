"""Triton kernel: fused two-shape decode attention, SWA form (exp-0013;
D4 "two-shape attention w/ fused rel-bias" — ranked attack (2) of the
exp-0012 graphed-step kernel table: ~128 ms/step of matmul-contiguous
direct_copy on the pooled K/V gathers, 260 copies/step, 28% of the
GPU-bound step).

One @triton.jit body serves BOTH shapes (attn_global.py imports it):
program (b, kv_head) processes that row's REP GQA query heads (SWA 4,
global 8; padded to REP_PAD=16 for tl.dot) against the layer's K/V read
IN PLACE from the batch pool — no [B, L, H, D] gather copy, no
matmul-contiguity copy, no [B, H, 1, L] bias/score intermediates:

  - SWA (IS_SWA=1): K/V rows come from kv.SwaPool [S, W, HK, D] at row
    `slot`, ring geometry recomputed in-kernel from the row's position
    exactly as model.decode_batch_ctx #1: base = max(0, n-W), slot s
    holds p = base + ((s-base) mod W), live iff dist = pos-p in [0, W).
  - global (IS_SWA=0): K/V rows come from the flat page pool
    [(P+1)*ps, HK, D] through `flat` [B, Lg] — the SAME masked
    table-gather indices the torch path built (dead lanes point at row
    0 and are masked by l < n), so page indirection stays one tiny
    int64 tensor per layer instead of a 134 MB K/V copy.

Per-key math is attn_decode_batch_pooled #4-#6 exactly: score =
q.k * (1/head_dim) + rel_bias(dist), bias gathered from the precomputed
rel-logit mix rel[b, h, clamp(dist, 0, E-1)] and zeroed out of band
(dist < 0 or >= E), dead key slots overwritten with a large-negative
constant (selection, never addition — stale pool rows can hold any
finite value), softmax in fp32, probs downcast to bf16 for the PV dot
(the torch path's post-softmax downcast), fp32 accumulate, one bf16
round on store. Differences vs the torch chain are reduction ORDER
only (fp32 flash accumulation vs cuBLAS tiling — the same B3.1/B3.3
drift class as every batched rewrite, inside the D13 envelope);
everything is fixed-order, fixed-shape, atomics-free, so same-schedule
bitwise determinism holds (t_b3 arm). No autotune — fixed BLOCK_L and
num_warps, so the kernel sequence is a pure function of the schedule
and capture-safe (graphrun warms each segment twice before capture,
which JIT-compiles both specializations outside graph capture).

All-masked tiles never NaN: masked lanes carry -1e38 (finite, far below
any real |score| <= ~1e4), so a tile with no live key contributes
exp(0)=1 garbage that the first live tile's rescale multiplies by
exp(-1e38 - m) = 0 exactly; >= 1 live key always exists (position 0's
own key — graphrun dead rows run pos 0 on the scratch slot)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_decode_kernel(q_ptr, o_ptr, k_ptr, v_ptr, rel_ptr, pos_ptr,
                        slots_ptr, flat_ptr, Lg, scale,
                        E: tl.constexpr, HK: tl.constexpr,
                        REP: tl.constexpr, HQ: tl.constexpr,
                        D: tl.constexpr, W: tl.constexpr,
                        IS_SWA: tl.constexpr, BLOCK_L: tl.constexpr,
                        REP_PAD: tl.constexpr):
    #1. one program = (batch row, kv head): its REP query heads vs the
    #   row's whole key set
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    pos = tl.load(pos_ptr + b)                       # i64 scalar
    n = pos + 1
    r_off = tl.arange(0, REP_PAD)
    d_off = tl.arange(0, D)
    hq = kvh * REP + r_off                           # absolute q head
    r_mask = r_off < REP
    q = tl.load(q_ptr + b * HQ * D + hq[:, None] * D + d_off[None, :],
                mask=r_mask[:, None], other=0.0)     # [REP_PAD, D] bf16
    if IS_SWA:
        slot = tl.load(slots_ptr + b)                # pool row (i64)
        kv_row = slot * W * HK * D + kvh * D
        base = tl.maximum(n - W, 0)
    #2. online-softmax state (fp32)
    m_i = tl.full([REP_PAD], float("-inf"), tl.float32)
    l_i = tl.zeros([REP_PAD], tl.float32)
    acc = tl.zeros([REP_PAD, D], tl.float32)
    L_end = W if IS_SWA else Lg
    for l0 in range(0, L_end, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        #3. per-shape key addressing + geometry (decode_batch_ctx math)
        if IS_SWA:
            in_l = ls < W
            p = base + (((ls - base) % W) + W) % W   # ring slot -> pos
            dist = pos - p
            valid = in_l & (dist >= 0) & (dist < W)
            kv_off = kv_row + ls[:, None] * HK * D + d_off[None, :]
        else:
            in_l = ls < Lg
            fl = tl.load(flat_ptr + b * Lg + ls, mask=in_l, other=0)
            dist = pos - ls
            valid = in_l & (ls < n)
            kv_off = fl[:, None] * HK * D + kvh * D + d_off[None, :]
        kk = tl.load(k_ptr + kv_off, mask=in_l[:, None], other=0.0)
        #4. scores + fused rel bias (band-clamped gather, zero outside),
        #   dead slots overwritten by selection
        s = tl.dot(q, tl.trans(kk), out_dtype=tl.float32) * scale
        band = (dist >= 0) & (dist < E)
        dcl = tl.minimum(tl.maximum(dist, 0), E - 1)
        bias = tl.load(rel_ptr + b * HQ * E + hq[:, None] * E
                       + dcl[None, :],
                       mask=r_mask[:, None] & band[None, :], other=0.0)
        s = s + bias.to(tl.float32)
        s = tl.where(valid[None, :], s, -1e38)
        #5. online softmax update; probs downcast to bf16 for the PV dot
        #   (attn_decode_batch_pooled's post-softmax downcast)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        prob = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(prob, 1)
        vv = tl.load(v_ptr + kv_off, mask=in_l[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(
            prob.to(v_ptr.dtype.element_ty), vv, out_dtype=tl.float32)
        m_i = m_new
    #6. normalize, one bf16 round on store
    out = acc / l_i[:, None]
    tl.store(o_ptr + b * HQ * D + hq[:, None] * D + d_off[None, :],
             out.to(o_ptr.dtype.element_ty), mask=r_mask[:, None])


def attn_swa(q, pool_k, pool_v, slots, pos, rel):
    """SWA decode attention over the layer's kv.SwaPool, in place.
    q [B, HQ, D] bf16 (post q-norm), pool_k/pool_v [S, W, HK, D] bf16,
    slots/pos [B] i64 (ctx["slots"]/ctx["pos"]), rel [B, HQ, W] bf16
    (the rel-logit mix; extent == W on SWA layers). Returns [B, HQ, D]."""
    B, HQ, D = q.shape
    S, W, HK, _ = pool_k.shape
    #1. contract checks — fail loud (contiguity is load-bearing: the
    #   kernel indexes raw strides)
    assert q.is_contiguous() and rel.is_contiguous()
    assert pool_k.is_contiguous() and pool_v.is_contiguous()
    assert rel.shape == (B, HQ, W), (rel.shape, (B, HQ, W))
    assert slots.dtype == torch.int64 and pos.dtype == torch.int64
    out = torch.empty_like(q)
    #2. launch on the layer's device (layers are split across devices;
    #   Triton launches on the CURRENT device — moe_gemm.py precedent)
    with torch.cuda.device(q.device):
        _attn_decode_kernel[(B, HK)](
            q, out, pool_k, pool_v, rel, pos, slots, pos, W, 1.0 / D,
            E=W, HK=HK, REP=HQ // HK, HQ=HQ, D=D, W=W, IS_SWA=True,
            BLOCK_L=128, REP_PAD=16, num_warps=4)
    return out
