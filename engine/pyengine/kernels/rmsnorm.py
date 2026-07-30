"""Triton kernel: rmsnorm (B2.3). Replays the torch reference model.rmsnorm
— exact InklingRMSNorm semantics (transformers modeling_inkling.py:99-113):
mean-of-squares in fp32, normalized value downcast to the INPUT dtype BEFORE
the weight multiply; that multiply runs at fp32 opmath with a single rounding
on store, which is exactly torch's bf16 elementwise-mul behavior (two 8-bit
mantissas multiply exactly in fp32, one round to bf16). The only divergence
source vs torch is fp32 reduction order in the variance (few-ulp fp32, at
most 1-2 bf16 ulp after downcast) — t_b2 rmsnorm gates on that.

One program per row; the whole hidden dim fits one block (6144 -> BLOCK 8192;
q/k-norm head_dim 128 -> BLOCK 128)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, o_ptr, n_cols, row_stride, eps,
                    BLOCK_N: tl.constexpr):
    #1. one program handles one row of the flattened (rows, n_cols) input
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0)
    #2. fp32 mean of squares over the real n_cols (masked lanes contribute 0)
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / n_cols
    #3. normalize in fp32, downcast to the input dtype BEFORE the weight
    #   multiply (InklingRMSNorm order: norm -> round -> scale)
    xhat = (xf * tl.math.rsqrt(var + eps)).to(x_ptr.dtype.element_ty)
    #4. weight multiply at fp32 opmath, single rounding on the store —
    #   matches torch's promoted bf16 mul bit-for-bit given equal inputs
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    of = xhat.to(tl.float32) * w.to(tl.float32)
    tl.store(o_ptr + row * row_stride + cols,
             of.to(x_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x, weight, eps=1e-6):
    """Drop-in for model.rmsnorm: normalize over the last dim, scale by a
    1-D weight of the same dtype (all Inkling norms are bf16 per the B1.3
    dtype map). Returns a new tensor shaped like x."""
    #1. contract checks — fail loud, no silent promotion
    n = x.shape[-1]
    assert x.is_cuda and weight.is_cuda, (x.device, weight.device)
    assert weight.shape == (n,), (tuple(x.shape), tuple(weight.shape))
    assert weight.dtype == x.dtype, (x.dtype, weight.dtype)
    #2. flatten leading dims to rows; one block must span a whole row
    x2 = x.contiguous().view(-1, n)
    out = torch.empty_like(x2)
    BLOCK_N = triton.next_power_of_2(n)
    assert BLOCK_N <= 16384, f"row width {n} exceeds single-block rmsnorm"
    #3. launch in the TENSOR's device context, like every other kernel
    #   wrapper here (attn_*, sconv_prefill, moe_gemm*, fp4_quant): the
    #   engine splits the 66 layers over 4 GPUs, so x rarely lives on the
    #   current device and Triton validates its pointers against the
    #   current context ("cannot be accessed from Triton"). Launch
    #   configuration only — no numerics, so B2.3's gate is untouched.
    if x2.shape[0]:
        with torch.cuda.device(x.device):
            _rmsnorm_kernel[(x2.shape[0],)](
                x2, weight.contiguous(), out, n, x2.stride(0), eps,
                BLOCK_N=BLOCK_N, num_warps=8 if BLOCK_N >= 2048 else 4)
    return out.view(x.shape)
