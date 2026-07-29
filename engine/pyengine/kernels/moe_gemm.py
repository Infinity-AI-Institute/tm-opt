"""Triton kernels: grouped routed-expert GEMM directly on PACKED NVFP4
weights (exp-0003; attack surface #3 "skinny-expert MoE grouped-GEMM").

Why (exp-0003 step profile, 2026-07-29, post-exp-0002): moe_experts is
645.9 ms of the 918 ms batched decode step at B=8 (70%) — 10.1 ms/layer for
~44 hit experts. Even with the bit-exact Triton dequant (exp-0002), every
hit expert still MATERIALIZES 112 MB of bf16 weight (w13 75 MB + w2 37 MB)
and re-reads it in cuBLAS to serve 1-2 token rows: ~225 MB of HBM round-trip
per expert against 21+10.5 MB of checkpoint-form packed bytes — plus ~11
eager-op launches per expert and one unique().tolist() CPU sync per layer.
The pair-bmm dead-end (DEADENDS.md 2026-07-29) showed grouping WITHOUT
removing the dequant traffic loses; this kernel removes both at once: the
GEMM reads packed u8 nibbles + f8 group scales, dequantizes in-register
(bit-equal construction to kernels/dequant.py), and runs all (token, slot)
pairs of a layer in ONE launch per matrix. The roofline headline
(docs/ROOFLINE.md) is exactly this traffic cut on the path that sets
decode tok/s.

Numerics (D13-envelope semantics, NOT bitwise vs the eager loop — B3.1/B3.3
established bitwise across execution shapes is unattainable in principle):
- dequantized weight VALUES are bit-equal to loader.dequant_nvfp4: same
  e2m1 fp32 constants, low nibble first, s = fp32(scale_f8) * scale2 (f8 ->
  fp32 widening is exact, so the in-kernel product is the same IEEE fp32 mul
  the eager reference computes), one fp32 value*s multiply, one RTNE round
  to bf16 — kernels/dequant.py's proven sequence;
- every post-GEMM rounding mirrors the eager op chain: GEMM acc fp32 ->
  bf16 (cuBLAS store), silu at fp32 opmath -> bf16, silu*up at fp32 opmath
  -> bf16, down-GEMM acc fp32 -> bf16, *routing-weight at fp32 opmath ->
  bf16. The remaining drift sources are fp32 GEMM accumulation ORDER (tile
  loop vs cuBLAS split-k) and the final per-token slot-sum (fp32-accumulated
  in slot order vs the loop's sequential bf16 adds in expert-ascending
  order) — the same drift class the D13 envelope already covers.
Determinism (the t_b3-adapted worker arm): stable sort, exact-integer
bincount, fixed grids, no atomics, each output row written exactly once —
two runs of the same schedule are bitwise identical."""
import torch
import triton
import triton.language as tl

#launch config, fixed (no autotune: deterministic, one JIT specialization);
#chosen by the exp-0003 sweep on synthetic real-shape packs
BLOCKS = (16, 128, 128)  # BLOCK_M, BLOCK_N, BLOCK_K
NUM_WARPS = 4
NUM_STAGES = 3


@triton.jit
def _e2m1(n):
    """e2m1 nibble -> fp32 by direct bit construction (ALU-lean form of
    kernels/dequant.py's where-chain; ~3x fewer ops on the hot path).
    Magnitudes m>=2 are (1 + 0.5*(m&1)) * 2^((m>>1)-1): exponent field
    126+(m>>1), mantissa msb (m&1)<<22; m in {0,1} maps to {+0.0, 0.5} =
    m * 0x3F000000. Sign = nibble bit 3 -> fp32 bit 31 (m=0 gives -0.0,
    exactly like the reference's -v of +0.0). Bit patterns ARE the exact
    constants {0,.5,1,1.5,2,3,4,6} — bitwise-checked against
    loader.dequant_nvfp4 over all 256 byte patterns (t_moe_grouped arm 0)."""
    m = n & 7
    bits = tl.where(m < 2, m * 0x3F000000,
                    ((126 + (m >> 1)) << 23) | ((m & 1) << 22))
    return (bits | ((n & 8) << 28)).to(tl.float32, bitcast=True)


@triton.jit
def _gate_up_silu_kernel(x_ptr, pk_ptr, sc_ptr, s2_ptr, tok_ptr, offs_ptr,
                         act_ptr, K, FFN,
                         stride_pe, stride_pr, stride_se, stride_sr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr):
    """Grouped gate/up GEMM + silu*up on packed w13. Grid (E, FFN/BLOCK_N);
    programs of experts with no rows exit immediately. w13 rows are stored
    INTERLEAVED [g0,u0,g1,u1,..]: deinterleaved gate row n = packed row 2n,
    up row n = packed row 2n+1 (loader.PackedExperts). This expert's pair
    rows are sorted-pair rows [offs[e], offs[e+1]); tok_ptr maps a pair row
    to its token row in x. Byte j of a packed row holds elements 2j (low
    nibble) and 2j+1 (high); scale groups are 16 elements = 8 bytes. act out
    [P, FFN] bf16 in sorted-pair order, epilogue mirroring the eager
    rounding chain: acc->bf16, fp32 silu ->bf16, fp32 mul ->bf16."""
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    start = tl.load(offs_ptr + pid_e)
    end = tl.load(offs_ptr + pid_e + 1)
    rows = end - start
    if rows == 0:
        return
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    s2 = tl.load(s2_ptr + pid_e)
    #1. expert base offsets in int64 — pid_e * stride_pe reaches 255 *
    #   18.9M ~ 4.8e9 for w13, past int32
    pk_e = pk_ptr + pid_e.to(tl.int64) * stride_pe
    sc_e = sc_ptr + pid_e.to(tl.int64) * stride_se
    HALF: tl.constexpr = BLOCK_K // 2
    for m0 in range(0, rows, BLOCK_M):
        m_idx = m0 + tl.arange(0, BLOCK_M)
        m_mask = m_idx < rows
        tok = tl.load(tok_ptr + start + m_idx, mask=m_mask, other=0)
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            #1. one contiguous x tile for the whole k block
            xt = tl.load(x_ptr + tok[:, None] * K
                         + (k0 + tl.arange(0, BLOCK_K))[None, :],
                         mask=m_mask[:, None], other=0.0)
            #2. gate rows (2n) then up rows (2n+1): packed bytes, per-group
            #   scales, in-register dequant; byte j holds elements 2j (low
            #   nibble) / 2j+1 (high) -> tl.interleave restores k order
            pb = k0 // 2 + tl.arange(0, HALF)
            g_rows = 2 * n_offs
            u_rows = 2 * n_offs + 1
            bg = tl.load(pk_e + g_rows[:, None] * stride_pr
                         + pb[None, :]).to(tl.int32)
            sg = tl.load(sc_e + g_rows[:, None] * stride_sr
                         + (pb // 8)[None, :]).to(tl.float32) * s2
            wg = tl.interleave((_e2m1(bg & 0x0F) * sg).to(tl.bfloat16),
                               (_e2m1((bg >> 4) & 0x0F) * sg)
                               .to(tl.bfloat16))
            acc_g += tl.dot(xt, tl.trans(wg))
            bu = tl.load(pk_e + u_rows[:, None] * stride_pr
                         + pb[None, :]).to(tl.int32)
            su = tl.load(sc_e + u_rows[:, None] * stride_sr
                         + (pb // 8)[None, :]).to(tl.float32) * s2
            wu = tl.interleave((_e2m1(bu & 0x0F) * su).to(tl.bfloat16),
                               (_e2m1((bu >> 4) & 0x0F) * su)
                               .to(tl.bfloat16))
            acc_u += tl.dot(xt, tl.trans(wu))
        #3. epilogue mirrors the eager rounding chain exactly
        g = acc_g.to(tl.bfloat16).to(tl.float32)
        u = acc_u.to(tl.bfloat16).to(tl.float32)
        act = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32) * u
        tl.store(act_ptr + (start + m_idx)[:, None] * FFN + n_offs[None, :],
                 act.to(tl.bfloat16), mask=m_mask[:, None])


@triton.jit
def _down_scale_kernel(act_ptr, pk_ptr, sc_ptr, s2_ptr, wt_ptr, perm_ptr,
                       offs_ptr, out_ptr, K, N,
                       stride_pe, stride_pr, stride_se, stride_sr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr):
    """Grouped down GEMM on packed w2 + routing-weight epilogue. Grid
    (E, N/BLOCK_N). Reads act rows in sorted-pair order, stores each result
    row SCATTERED to its original pair row (token*top_k + slot) via
    perm_ptr — each row written exactly once (deterministic, no atomics).
    Epilogue mirrors eager: acc->bf16, * pair weight at fp32 opmath ->bf16."""
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    start = tl.load(offs_ptr + pid_e)
    end = tl.load(offs_ptr + pid_e + 1)
    rows = end - start
    if rows == 0:
        return
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    s2 = tl.load(s2_ptr + pid_e)
    #1. expert base offsets in int64 — pid_e * stride_pe reaches 255 *
    #   9.4M ~ 2.4e9 for w2, past int32
    pk_e = pk_ptr + pid_e.to(tl.int64) * stride_pe
    sc_e = sc_ptr + pid_e.to(tl.int64) * stride_se
    HALF: tl.constexpr = BLOCK_K // 2
    for m0 in range(0, rows, BLOCK_M):
        m_idx = m0 + tl.arange(0, BLOCK_M)
        m_mask = m_idx < rows
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            at = tl.load(act_ptr + (start + m_idx)[:, None] * K
                         + (k0 + tl.arange(0, BLOCK_K))[None, :],
                         mask=m_mask[:, None], other=0.0)
            pb = k0 // 2 + tl.arange(0, HALF)
            b = tl.load(pk_e + n_offs[:, None] * stride_pr
                        + pb[None, :]).to(tl.int32)
            s = tl.load(sc_e + n_offs[:, None] * stride_sr
                        + (pb // 8)[None, :]).to(tl.float32) * s2
            w = tl.interleave((_e2m1(b & 0x0F) * s).to(tl.bfloat16),
                              (_e2m1((b >> 4) & 0x0F) * s).to(tl.bfloat16))
            acc += tl.dot(at, tl.trans(w))
        #1. epilogue: bf16 GEMM store, then * routing weight (fp32 opmath,
        #   one round) — eager's (h * w).to(bf16); scatter to original row
        h = acc.to(tl.bfloat16).to(tl.float32)
        wt = tl.load(wt_ptr + start + m_idx, mask=m_mask,
                     other=0.0).to(tl.float32)
        dest = tl.load(perm_ptr + start + m_idx, mask=m_mask, other=0)
        tl.store(out_ptr + dest[:, None].to(tl.int64) * N + n_offs[None, :],
                 (h * wt[:, None]).to(tl.bfloat16), mask=m_mask[:, None])


def moe_experts_packed(x, pack13, pack2, topk_indices, topk_weights):
    """Grouped drop-in for model.moe_experts when the routed experts are
    loader.PackedExperts (the NVFP4 MoE layers 3-65; layer 2's plain bf16
    tensors keep the reference loop — caller dispatches). x [T, hidden]
    bf16 flattened tokens; returns [T, hidden] bf16 = each token's top_k
    routed-expert outputs summed over the slot axis (fixed order,
    fp32-accumulated torch.sum, one bf16 round — deterministic)."""
    T, K = x.shape
    top_k = topk_indices.shape[1]
    E, R13, PK13 = pack13.packed.shape
    ffn = R13 // 2
    #1. contract: shapes/layout this kernel is specialized to
    assert pack13.deinterleave and not pack2.deinterleave
    assert pack13.group_size == 16 and pack2.group_size == 16
    assert PK13 * 2 == K, (PK13, K)
    assert pack2.packed.shape[1] == K and pack2.packed.shape[2] * 2 == ffn
    assert K % 128 == 0 and ffn % 128 == 0

    #2. pair list sorted by expert id (stable -> deterministic), pair->token
    #   map, per-expert segment offsets — all on the layer's device, no CPU
    #   sync (replaces the loop's per-layer unique().tolist() sync)
    flat = topk_indices.reshape(-1)
    sorted_e, perm = torch.sort(flat, stable=True)
    perm32 = perm.to(torch.int32)
    tok32 = (perm // top_k).to(torch.int32)
    offs = torch.zeros(E + 1, device=x.device, dtype=torch.int32)
    offs[1:] = torch.cumsum(torch.bincount(sorted_e, minlength=E),
                            0).to(torch.int32)
    wt = topk_weights.reshape(-1)[perm].contiguous()

    P = flat.shape[0]
    act = torch.empty(P, ffn, device=x.device, dtype=torch.bfloat16)
    out_pairs = torch.empty(P, K, device=x.device, dtype=torch.bfloat16)
    BM, BN, BK = BLOCKS
    #3. launch on the pack's own device — layers live spread across 4 GPUs
    #   and Triton launches on the CURRENT device (exp-0002 lesson)
    with torch.cuda.device(x.device):
        _gate_up_silu_kernel[(E, ffn // BN)](
            x, pack13.packed, pack13.scale, pack13.scale2, tok32, offs, act,
            K, ffn, pack13.packed.stride(0), pack13.packed.stride(1),
            pack13.scale.stride(0), pack13.scale.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES)
        _down_scale_kernel[(E, K // BN)](
            act, pack2.packed, pack2.scale, pack2.scale2, wt, perm32, offs,
            out_pairs, ffn, K,
            pack2.packed.stride(0), pack2.packed.stride(1),
            pack2.scale.stride(0), pack2.scale.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    #4. per-token reduction over the top_k slots
    return out_pairs.view(T, top_k, K).sum(dim=1)
