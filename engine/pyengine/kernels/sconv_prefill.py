"""Triton kernel: fused PREFILL sconv (exp-0025) — D4 item #3 ("sconv
fusion", ARCHITECTURE "Where the wins must come from") applied to the
prefill phase, the twin of what exp-0008 did for decode.

WHY. model.sconv_prefill is the engine's last un-fused per-layer eager
chain and it is run FOUR times per layer (attention K, attention V,
attention output, MoE output — modeling_inkling.py:500-542), i.e. 264
times per traversal. Every call runs the whole window-4 depthwise
convolution in fp32 through torch:

  xf = x.float()                       read bf16, write fp32
  conv1d(xf.transpose(1, 2), w, ...)   the transposed input is not
                                       contiguous, so ATen copies it,
                                       then the depthwise conv reads and
                                       writes another fp32 [B, C, T]
  conv.transpose(1, 2) + xf            reads two fp32 tensors, writes one
  .to(x.dtype)                         reads fp32, writes bf16

That is ~36 bytes of HBM traffic per element to compute a 4-tap dot that
needs 4. Per layer the four sites carry 6144 + 6144 + 2048 + 2048 = 16,384
channels per token (1024-wide K/V on the 11 global layers), so a canonical
6-row cohort padded to T≈1400 moves ~5 GB per layer, ~340 GB per traversal,
on devices exp-0023's DEADENDS finding 1 shows are SATURATED during
prefill (utilization.gpu 92-100% on the active device of the layer split
through the whole prefill phase of the worker's own canonical bench).

LAYOUT, the second half of the same mechanism. `conv.transpose(1, 2) + xf`
resolves to the TRANSPOSED operand's layout: the eager result is
CHANNEL-MAJOR ([B, T, C] with strides (T*C, 1, T)), verified on CPU and
GPU by t_fused_sconv arm e. That layout is then what the rest of the layer
consumes — the per-head K rmsnorm reduces over a stride-T axis, the fused
attention kernel reads K/V with a stride-T head_dim, and `x = x + h` in
layer_prefill propagates it into the residual stream, so the NEXT layer's
input_layernorm reduces a strided hidden axis too. The kernel writes its
output in the natural contiguous layout, which is the same mechanism (a
kernel stores where its consumers read), not a second variable.

WHAT. One @triton.jit body: a program owns a [BLOCK_T, BLOCK_C] tile of
one batch row, loads the 4 shifted input tiles it needs (out-of-range
history reads 0.0 — causal_conv1d_fn's left pad), accumulates in fp32 in
ASCENDING TAP ORDER, adds the module's own-input residual in fp32 and
stores one bf16 round. Tap j taps position t-(k-1)+j, so weight[..., -1]
taps the CURRENT token (modeling_inkling.py:461-481, no kernel flip —
torch conv1d is cross-correlation).

NUMERICS. Unlike every other fused kernel in this engine this one does not
change reduction order: a 4-term dot has exactly one sensible order and it
is the one the reference uses. t_fused_sconv arm a gates BITWISE equality
against the eager chain at both prefill shapes and both dtypes; if a future
toolchain contracts the multiply-add differently the arm will say so
(fp32 FMA vs mul+add is the only degree of freedom left). Fixed grid, no
atomics, no autotune → same-schedule bitwise determinism holds (arm b).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sconv_prefill_kernel(x_ptr, w_ptr, o_ptr, T, C,
                          sxb, sxt, sxc, sob, sot, soc, swc, swk,
                          K: tl.constexpr, BLOCK_T: tl.constexpr,
                          BLOCK_C: tl.constexpr):
    #1. one program = one [BLOCK_T, BLOCK_C] tile of one batch row
    b = tl.program_id(0)
    t_off = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    c_off = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    t_mask = t_off < T
    c_mask = c_off < C
    acc = tl.zeros([BLOCK_T, BLOCK_C], tl.float32)
    #2. taps in ASCENDING order: tap j reads position t - (K-1) + j, so
    #   j = K-1 is the current token (causal_conv1d_fn :461-481). Positions
    #   before 0 are the conv's left zero-pad and contribute nothing
    for j in tl.static_range(K):
        p = t_off - (K - 1) + j
        w = tl.load(w_ptr + c_off * swc + j * swk, mask=c_mask, other=0.0)
        xj = tl.load(x_ptr + b * sxb + p[:, None] * sxt + c_off[None, :] * sxc,
                     mask=(p >= 0)[:, None] & t_mask[:, None] & c_mask[None, :],
                     other=0.0)
        acc += w.to(tl.float32)[None, :] * xj.to(tl.float32)
    #3. the module's OWN-INPUT residual, in fp32, then one bf16 round
    #   (:515 + :542 — the residual lives INSIDE the module)
    x0 = tl.load(x_ptr + b * sxb + t_off[:, None] * sxt
                 + c_off[None, :] * sxc,
                 mask=t_mask[:, None] & c_mask[None, :], other=0.0)
    out = acc + x0.to(tl.float32)
    tl.store(o_ptr + b * sob + t_off[:, None] * sot + c_off[None, :] * soc,
             out.to(o_ptr.dtype.element_ty),
             mask=t_mask[:, None] & c_mask[None, :])


def sconv_prefill_fused(x, weight):
    """Drop-in for model.sconv_prefill. x [B, T, C] (any strides), weight
    (C, 1, k) as the module holds it; returns a CONTIGUOUS [B, T, C] tensor
    of x's dtype. The caller keeps the eager path for shapes this kernel
    does not serve (non-CUDA tensors, k != the kernel's static range)."""
    B, T, C = x.shape
    out = torch.empty((B, T, C), dtype=x.dtype, device=x.device)
    k = weight.shape[-1]
    #1. tiles: 64 x 256 keeps the 4 shifted loads inside L2 and gives the
    #   6144/2048/1024-wide sites whole-warp rows (swept in t_fused_sconv)
    block_t, block_c = 64, 256
    grid = (B, triton.cdiv(T, block_t), triton.cdiv(C, block_c))
    #2. the layer walk hops devices (loader splits the 66 layers over 4
    #   GPUs) while the ambient CUDA context stays on device 0, so the
    #   launch is pinned to the tensors' device like every other kernel
    #   in engine/pyengine/kernels
    with torch.cuda.device(x.device):
        _sconv_prefill_kernel[grid](
            x, weight, out, T, C,
            x.stride(0), x.stride(1), x.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            weight.stride(0), weight.stride(2),
            K=k, BLOCK_T=block_t, BLOCK_C=block_c, num_warps=4)
    return out
