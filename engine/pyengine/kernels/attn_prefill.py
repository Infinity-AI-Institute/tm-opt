"""Triton kernel: fused two-shape PREFILL attention (exp-0023) — the
prefill twin of exp-0013's fused decode attention, and the same D4 item
("two-shape attention with fused rel-bias") applied to the other phase.

WHY. model.attn_prefill #3-#8 is the last bring-up-era eager chain in the
engine and it runs entirely through a materialized [B, heads, T, T] bf16
score tensor. At the canonical cohort shape (B=6 rows, T=1536 padded, 64
heads) that tensor is 1.81 GB and the chain touches score-shaped memory
about a dozen times per layer: rel_bias builds the [B,H,T,E] logit mix
(604 MB), gathers it to [B,H,T,T] and masked_fills it; the QK matmul
writes scores; `* scale`, `+ bias` and `+ mask` each read and write them;
softmax(dtype=fp32) writes a 3.62 GB fp32 copy that `.to(bf16)` reads back;
the PV matmul reads the result; and repeat_kv materializes K and V at 64
heads first (302 MB). ~26 GB of HBM traffic per layer, ~1.7 TB per
traversal, to produce a 19 MB output — on devices that exp-0023's DEADENDS
finding 1 shows are already saturated during prefill.

WHAT. One @triton.jit body serves both layer shapes (IS_SWA constexpr),
flash-attention style: a program owns a BLOCK_M block of query rows for one
(row, query head), streams the causal key range in BLOCK_N tiles, and keeps
only the online-softmax state in registers. Nothing score-shaped is ever
written to memory. Per tile the math is attn_prefill's, in its order:

  score = q.k * (1/head_dim)                       (#6, fp32 accumulate)
        + rel[b, h, q_pos, clamp(q_pos - k_pos, 0, E-1)]   (#3 gathered
          in-kernel from the SAME [B,H,T,E] mix rel_bias computes, zeroed
          outside 0 <= dist < E — rel_bias #4)
  masked by causality (dist >= 0) and, on SWA layers, the window
  (dist < W) — the additive_causal_mask predicate exactly (#155-175),
  applied as SELECTION to -1e38 rather than by adding finfo.min
  fp32 softmax, probs downcast to bf16 for the PV dot (#7's `.to(q.dtype)`)
  fp32 accumulate, one bf16 round on store (#8)

GQA is internal (kv head = q head // rep), so repeat_kv's expand+reshape
copies disappear; the output is written straight in [B, T, H, D] layout,
so #8's transpose(1,2).contiguous() disappears too.

NUMERICS. Same class as exp-0013 and every batched rewrite before it: the
per-element ops are the reference's, the DIFFERENCE is reduction order —
fp32 online softmax + fp32 flash accumulation instead of a bf16-rounded
score tensor summed by cuBLAS tiling (B3.1/B3.3 drift, D13 envelope is the
binding gate). Bitwise identity vs the eager chain is unattainable in
principle and is not claimed. Everything is fixed-shape, fixed-order and
atomics-free, with no autotune, so the kernel sequence is a pure function
of the schedule and same-schedule bitwise determinism holds (t_b3 arm).

MASK CONTRACT. The kernel derives the mask from POSITIONS, so it reproduces
additive_causal_mask(q_positions, kv_positions, dtype, window) for any
positions — which is what layer_prefill's docstring already requires of the
caller's mask ("the caller's mask must match the layer type") and what both
in-tree builders (Engine._mask_pos, generate_greedy._masks) construct. The
loop BOUNDS additionally assume positions are non-decreasing in index (both
builders pass torch.arange); a caller with unsorted positions must run with
PYENGINE_FUSED_PREFILL_ATTN=0.

ALL-MASKED BLOCKS never NaN, by exp-0013's argument: masked lanes carry
-1e38 (finite), so a row with no live key in a tile accumulates exp(0)=1
garbage that the first live tile's rescale multiplies by exp(-1e38-m) = 0
exactly. Right-padded rows are covered by the same reasoning from the other
side: a pad query sits at a real position and always sees at least its own
key, and a real query never sees a pad because pads occupy FUTURE key
positions (attn_prefill's causality argument), so pad rows produce finite
garbage that the caller's [:lens[i]] slices drop — exactly as in the eager
chain.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_prefill_kernel(q_ptr, k_ptr, v_ptr, o_ptr, rel_ptr,
                         qpos_ptr, kpos_ptr, T, scale,
                         sqb, sqh, sqt, sqd, skb, skh, skt, skd,
                         svb, svh, svt, svd, sob, sot, soh, srb, srh, srt,
                         H: tl.constexpr, REP: tl.constexpr,
                         D: tl.constexpr, E: tl.constexpr,
                         W: tl.constexpr, IS_SWA: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    #1. one program = (query block, (row, query head)); its KV head is
    #   fixed by GQA (repeat_kv :145-154 repeats each KV head REP times)
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    kvh = h // REP
    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_off = tl.arange(0, D)
    m_mask = m_off < T
    q = tl.load(q_ptr + b * sqb + h * sqh + m_off[:, None] * sqt
                + d_off[None, :] * sqd, mask=m_mask[:, None], other=0.0)
    qp = tl.load(qpos_ptr + m_off, mask=m_mask, other=0).to(tl.int32)
    #2. online-softmax state (fp32), flash accumulation
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    #3. key range: causal end, plus the window's start on SWA layers —
    #   the out-of-band tiles the eager chain computes and then masks are
    #   simply never visited (block bounds assume non-decreasing positions)
    hi = tl.minimum((pid_m + 1) * BLOCK_M, T)
    if IS_SWA:
        lo = (tl.maximum(pid_m * BLOCK_M - W + 1, 0) // BLOCK_N) * BLOCK_N
    else:
        lo = 0
    for n0 in range(lo, hi, BLOCK_N):
        n_off = n0 + tl.arange(0, BLOCK_N)
        n_mask = n_off < T
        kk = tl.load(k_ptr + b * skb + kvh * skh + n_off[:, None] * skt
                     + d_off[None, :] * skd, mask=n_mask[:, None],
                     other=0.0)
        kp = tl.load(kpos_ptr + n_off, mask=n_mask, other=0).to(tl.int32)
        #4. scores + fused rel bias (band-clamped gather, zero outside —
        #   rel_bias #3/#4), then the mask predicate as selection
        s = tl.dot(q, tl.trans(kk), out_dtype=tl.float32) * scale
        dist = qp[:, None] - kp[None, :]
        band = (dist >= 0) & (dist < E)
        dcl = tl.minimum(tl.maximum(dist, 0), E - 1)
        bias = tl.load(rel_ptr + b * srb + h * srh + m_off[:, None] * srt
                       + dcl, mask=m_mask[:, None] & band, other=0.0)
        s = s + bias.to(tl.float32)
        valid = (dist >= 0) & m_mask[:, None] & n_mask[None, :]
        if IS_SWA:
            valid = valid & (dist < W)
        s = tl.where(valid, s, -1e38)
        #5. online softmax update; probs downcast to bf16 for the PV dot
        #   (attn_prefill #7's post-softmax `.to(q.dtype)`)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        prob = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(prob, 1)
        vv = tl.load(v_ptr + b * svb + kvh * svh + n_off[:, None] * svt
                     + d_off[None, :] * svd, mask=n_mask[:, None],
                     other=0.0)
        acc = acc * alpha[:, None] + tl.dot(
            prob.to(v_ptr.dtype.element_ty), vv, out_dtype=tl.float32)
        m_i = m_new
    #6. normalize, one bf16 round on store, straight into [B, T, H, D]
    out = acc / l_i[:, None]
    tl.store(o_ptr + b * sob + m_off[:, None] * sot + h * soh
             + d_off[None, :], out.to(o_ptr.dtype.element_ty),
             mask=m_mask[:, None])


def attn_prefill_fused(q, k, v, rel, q_positions, kv_positions, window,
                       block_m=64, block_n=64):
    """Fused prefill attention over the layer's in-batch K/V.

    q   [B, H, T, D]  bf16, post q-rmsnorm (and post log-scaling on global
                      layers) — the transposed view attn_prefill #2 builds,
                      strides are read, contiguity is NOT required
    k,v [B, HK, T, D] bf16, post k-rmsnorm / post v-sconv, BEFORE repeat_kv
    rel [B, H, T, E]  bf16, the rel_bias #1 logit mix (r @ proj).transpose
                      (1,2), tau-scaled on global layers; E = rel extent
    q_positions/kv_positions [T] i64, the arange the engine builds
    window            None on global layers, mc.window (512) on SWA layers

    Returns [B, T, H, D] bf16 — the layout attn_prefill #8 feeds to wo, so
    the caller reshapes and never copies."""
    B, H, T, D = q.shape
    HK = k.shape[1]
    E = rel.shape[3]
    #1. contract checks — fail loud. Q/K/V strides are read as given and
    #   NOT assumed contiguous: sconv_prefill returns a channel-major
    #   tensor (its own-input residual `conv.transpose(1,2) + xf` keeps the
    #   transposed layout, so K and V arrive here with stride 1 along T and
    #   stride T along head_dim), which is also why the eager chain's
    #   matmul pays a contiguity copy on them. rel comes from a matmul and
    #   is genuinely last-dim contiguous, which the bias gather needs.
    assert k.shape == (B, HK, T, D) and v.shape == (B, HK, T, D)
    assert rel.shape == (B, H, T, E) and rel.stride(3) == 1
    assert q_positions.shape == (T,) and kv_positions.shape == (T,)
    assert q_positions.dtype == torch.int64
    assert kv_positions.dtype == torch.int64
    assert H % HK == 0
    is_swa = window is not None
    if is_swa:
        #2. SWA layers carry extent == window (rel_bias docstring: extent
        #   comes from proj — 1024 global, 512 = the window on SWA), which
        #   is what lets one constexpr serve the band and the bias gather
        assert E == window, (E, window)
    out = torch.empty((B, T, H, D), dtype=q.dtype, device=q.device)
    #3. launch on the layer's device (layers are split across devices;
    #   Triton launches on the CURRENT device — moe_gemm.py precedent)
    with torch.cuda.device(q.device):
        _attn_prefill_kernel[(triton.cdiv(T, block_m), B * H)](
            q, k, v, out, rel, q_positions, kv_positions, T, 1.0 / D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            rel.stride(0), rel.stride(1), rel.stride(2),
            H=H, REP=H // HK, D=D, E=E, W=(window or 0), IS_SWA=is_swa,
            BLOCK_M=block_m, BLOCK_N=block_n, num_warps=8, num_stages=2)
    return out
