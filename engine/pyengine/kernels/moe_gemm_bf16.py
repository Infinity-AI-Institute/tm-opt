"""Triton kernels: grouped routed-expert GEMM on PLAIN BF16 expert stacks
— the bf16 twin of kernels/moe_gemm.py, for the ONE MoE layer the NVFP4
export skipped (layer `dense_mlp_idx` = 2, whose w13/w2 ship unquantized:
CLAUDE.md model facts, hf_quant_config exclude list, census B1.2/B1.3).

Why (exp-0012 graphed-step structure + exp-0018's residual): every other
MoE layer runs moe_experts_packed — ONE launch per matrix for all
(token, slot) pairs, sync-free and fixed-shape, hence legal inside
CUDA-graph capture. Layer 2 alone kept model.moe_experts' reference loop
(`torch.unique(topk_indices).tolist()` + one pair of GEMMs per hit
expert), which is BOTH a device->host sync per traversal AND ~2 x 200
launches serving 1-2 rows each. exp-0012 could not capture it, so it
carved layer 2 out of the graph as the eager island `graphrun._run_island`
and split device 0's segment into two graphs around it. This kernel
removes the reason the island exists: same grouped structure, same
rounding chain, no dequant pipe (the weights are already bf16).

Numerics (D13-envelope semantics, NOT bitwise vs the eager loop — B3.1/
B3.3 established that bitwise across execution shapes is unattainable in
principle). Every post-GEMM rounding mirrors the eager op chain exactly as
moe_gemm.py's packed kernels do: GEMM acc fp32 -> bf16 (cuBLAS store),
silu at fp32 opmath -> bf16, silu*up at fp32 opmath -> bf16, down-GEMM acc
fp32 -> bf16, *routing-weight at fp32 opmath -> bf16. Unlike the packed
kernels there is no dequant step at all, so the ONLY drift sources are the
two the packed path already contributes on all 63 other MoE layers: fp32
GEMM accumulation ORDER (tile loop vs cuBLAS split-k) and the final
per-token slot-sum (fp32-accumulated in slot order vs the loop's
sequential bf16 adds in expert-ascending order). Same drift class the D13
envelope already covers.

Determinism (the t_b3-adapted worker arm): stable sort, exact-integer
searchsorted segment offsets (never bincount — its CUDA path sizes the
histogram from self.max().cpu(), a hidden sync that is illegal under
capture, exp-0015b), fixed grids, no atomics, each output row written
exactly once. Two runs of the same schedule are bitwise identical.
Sync-free and fixed-shape at fixed T, so the whole call is legal inside
CUDA-graph capture (exp-0012)."""
import torch
import triton
import triton.language as tl

#launch config, fixed (no autotune: deterministic, one JIT specialization).
#Deliberately the SAME constants moe_gemm.py uses, so this experiment moves
#one thing (which path layer 2 takes) and not two (path + tile shape);
#bf16-specific tile tuning is a separate Stage-3 hypothesis. The exp-0009
#BLOCK_M rule carries over for a DIFFERENT but parallel reason: the packed
#kernels re-DEQUANTIZE the full weight tile per M-chunk (ALU), these
#re-READ it (HBM traffic), and both scale with ceil(rows/BLOCK_M) — at
#prefill (~17-24 rows/expert) BLOCK_M 16 pays the weight tile twice.
BLOCKS = (16, 128, 128)  # BLOCK_M, BLOCK_N, BLOCK_K
BLOCKS_PREFILL = (32, 128, 128)
PREFILL_MIN_PAIRS = 2048
NUM_WARPS = 4
NUM_STAGES = 3


@triton.jit
def _gate_up_silu_bf16_kernel(x_ptr, w_ptr, tok_ptr, offs_ptr, act_ptr,
                              K, FFN, stride_we, stride_wr,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                              BLOCK_K: tl.constexpr):
    """Grouped gate/up GEMM + silu*up on a bf16 w13 stack [E, 2*FFN, K].
    Grid (E, FFN/BLOCK_N); programs of experts with no rows exit at once.
    Rows are DE-INTERLEAVED here (loader applies Interleave(dim=1), same as
    vLLM inkling nvidia/moe.py:583-595 and HF's conversion_mapping), i.e.
    the stack is [gates; ups] HALVES — gate row n = row n, up row n =
    row FFN + n. That is the one layout difference from the packed twin,
    whose checkpoint-form rows stay interleaved [g0,u0,g1,u1,..].
    This expert's pair rows are sorted-pair rows [offs[e], offs[e+1]);
    tok_ptr maps a pair row to its token row in x. act out [P, FFN] bf16
    in sorted-pair order."""
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    start = tl.load(offs_ptr + pid_e)
    end = tl.load(offs_ptr + pid_e + 1)
    rows = end - start
    if rows == 0:
        return
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    #1. expert base offset in int64 — pid_e * stride_we reaches 255 *
    #   2*3072*6144 ~ 9.6e9 for w13, far past int32
    w_e = w_ptr + pid_e.to(tl.int64) * stride_we
    for m0 in range(0, rows, BLOCK_M):
        m_idx = m0 + tl.arange(0, BLOCK_M)
        m_mask = m_idx < rows
        tok = tl.load(tok_ptr + start + m_idx, mask=m_mask, other=0)
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            #1. one contiguous x tile for the whole k block
            xt = tl.load(x_ptr + tok[:, None] * K + k_idx[None, :],
                         mask=m_mask[:, None], other=0.0)
            #2. gate half (row n) then up half (row FFN + n) — plain bf16
            #   tiles, no dequant pipe
            wg = tl.load(w_e + n_offs[:, None] * stride_wr + k_idx[None, :])
            acc_g += tl.dot(xt, tl.trans(wg))
            wu = tl.load(w_e + (FFN + n_offs)[:, None] * stride_wr
                         + k_idx[None, :])
            acc_u += tl.dot(xt, tl.trans(wu))
        #3. epilogue mirrors the eager rounding chain exactly
        g = acc_g.to(tl.bfloat16).to(tl.float32)
        u = acc_u.to(tl.bfloat16).to(tl.float32)
        act = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32) * u
        tl.store(act_ptr + (start + m_idx)[:, None] * FFN + n_offs[None, :],
                 act.to(tl.bfloat16), mask=m_mask[:, None])


@triton.jit
def _down_scale_bf16_kernel(act_ptr, w_ptr, wt_ptr, perm_ptr, offs_ptr,
                            out_ptr, K, N, stride_we, stride_wr,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                            BLOCK_K: tl.constexpr):
    """Grouped down GEMM on a bf16 w2 stack [E, N, K] (N = hidden,
    K = FFN) + routing-weight epilogue. Grid (E, N/BLOCK_N). Reads act rows
    in sorted-pair order, stores each result row SCATTERED to its original
    pair row (token*top_k + slot) via perm_ptr — each row written exactly
    once (deterministic, no atomics)."""
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    start = tl.load(offs_ptr + pid_e)
    end = tl.load(offs_ptr + pid_e + 1)
    rows = end - start
    if rows == 0:
        return
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    w_e = w_ptr + pid_e.to(tl.int64) * stride_we
    for m0 in range(0, rows, BLOCK_M):
        m_idx = m0 + tl.arange(0, BLOCK_M)
        m_mask = m_idx < rows
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            at = tl.load(act_ptr + (start + m_idx)[:, None] * K
                         + k_idx[None, :], mask=m_mask[:, None], other=0.0)
            w = tl.load(w_e + n_offs[:, None] * stride_wr + k_idx[None, :])
            acc += tl.dot(at, tl.trans(w))
        #1. epilogue: bf16 GEMM store, then * routing weight (fp32 opmath,
        #   one round) — eager's (h * w).to(bf16); scatter to original row
        h = acc.to(tl.bfloat16).to(tl.float32)
        wt = tl.load(wt_ptr + start + m_idx, mask=m_mask,
                     other=0.0).to(tl.float32)
        dest = tl.load(perm_ptr + start + m_idx, mask=m_mask, other=0)
        tl.store(out_ptr + dest[:, None].to(tl.int64) * N + n_offs[None, :],
                 (h * wt[:, None]).to(tl.bfloat16), mask=m_mask[:, None])


def moe_experts_bf16(x, w13, w2, topk_indices, topk_weights):
    """Grouped drop-in for model.moe_experts when the routed experts are
    PLAIN bf16 stacks (the un-exported MoE layer 2). x [T, hidden] bf16
    flattened tokens; returns [T, hidden] bf16 = each token's top_k routed-
    expert outputs summed over the slot axis (fixed order, fp32-accumulated
    torch.sum, one bf16 round — deterministic). Structurally identical to
    moe_gemm.moe_experts_packed; see this module's docstring for the two
    layout differences (de-interleaved w13 halves, no scales)."""
    T, K = x.shape
    top_k = topk_indices.shape[1]
    E, R13, K13 = w13.shape
    ffn = R13 // 2
    #1. contract: shapes/layout this kernel is specialized to
    assert w13.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16
    assert K13 == K, (K13, K)
    assert w2.shape[0] == E and w2.shape[1] == K and w2.shape[2] == ffn
    assert K % 128 == 0 and ffn % 128 == 0
    assert w13.is_contiguous() and w2.is_contiguous() and x.is_contiguous()

    #2. pair list sorted by expert id (stable -> deterministic), pair->token
    #   map, per-expert segment offsets — all on the layer's device, no CPU
    #   sync (this is what replaces the loop's unique().tolist(), the sync
    #   that kept layer 2 out of the captured step)
    flat = topk_indices.reshape(-1)
    sorted_e, perm = torch.sort(flat, stable=True)
    perm32 = perm.to(torch.int32)
    tok32 = (perm // top_k).to(torch.int32)
    offs = torch.searchsorted(
        sorted_e, torch.arange(E + 1, device=x.device,
                               dtype=sorted_e.dtype)).to(torch.int32)
    wt = topk_weights.reshape(-1)[perm].contiguous()

    P = flat.shape[0]
    act = torch.empty(P, ffn, device=x.device, dtype=torch.bfloat16)
    out_pairs = torch.empty(P, K, device=x.device, dtype=torch.bfloat16)
    BM, BN, BK = BLOCKS if P < PREFILL_MIN_PAIRS else BLOCKS_PREFILL
    #3. launch on the weight's own device — layers live spread across 4 GPUs
    #   and Triton launches on the CURRENT device (exp-0002 lesson)
    with torch.cuda.device(x.device):
        _gate_up_silu_bf16_kernel[(E, ffn // BN)](
            x, w13, tok32, offs, act, K, ffn,
            w13.stride(0), w13.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES)
        _down_scale_bf16_kernel[(E, K // BN)](
            act, w2, wt, perm32, offs, out_pairs, ffn, K,
            w2.stride(0), w2.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    #4. per-token reduction over the top_k slots
    return out_pairs.view(T, top_k, K).sum(dim=1)
