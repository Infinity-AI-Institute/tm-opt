"""Triton kernel: one-pass NVFP4 expert dequant, BIT-EQUAL to
loader.dequant_nvfp4 (the B1.4 reference) by construction.

Why it exists (exp-0002 profile, 2026-07-29): the eager reference
materializes per-ELEMENT intermediates — an int64 LUT index tensor
(8 B/elem), an fp32 value tensor, an fp32 where() result and an fp32
scale-broadcast product — ~750 MB of HBM traffic per expert w13 for a
75 MB bf16 result, ~1.1 ms per expert on B300. On-demand dequant runs
under EVERY routed-expert GEMM (loader.PackedExperts.__getitem__), so at
decode batch 8 that is ~44 hit experts x 64 MoE layers ~ 3.1 s/step —
measured to be essentially the whole batched-decode step wall. This
kernel reads the packed bytes + the per-group fp32 scales and writes
bf16 once: ~100 MB traffic per expert w13, no per-element temporaries.

Bit-equality argument (each step mirrors the reference op-for-op):
- the per-group scale product s = scale.float() * scale2 stays in EAGER
  torch (tiny: k/16 per row) — bit-identical because it IS the same op;
- e2m1 magnitudes are exact fp32 constants {0,.5,1,1.5,2,3,4,6} (same
  values as loader.E2M1_VALUES), sign from bit 3, LOW nibble first —
  dequant_nvfp4 #2/#3 exactly;
- one fp32 multiply value*s (IEEE mul.f32, same operand pairing as the
  reference's val * s broadcast), then ONE round-to-nearest-even cast
  to bf16 — dequant_nvfp4 #5's rounding sequence exactly.
Verified bitwise vs dequant_nvfp4 over all 256 byte patterns and random
expert-shaped packs (t_dequant arm in the exp-0002 evidence log)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1(n):
    #1. 3-bit magnitude via exact fp32 constants; sign is bit 3
    mag = n & 7
    v = tl.where(mag == 1, 0.5,
        tl.where(mag == 2, 1.0,
        tl.where(mag == 3, 1.5,
        tl.where(mag == 4, 2.0,
        tl.where(mag == 5, 3.0,
        tl.where(mag == 6, 4.0,
        tl.where(mag == 7, 6.0, 0.0)))))))
    return tl.where((n & 8) != 0, -v, v)


@triton.jit
def _nvfp4_dequant_kernel(packed_ptr, s_ptr, out_ptr, n_bytes, k2, grp2,
                          BLOCK: tl.constexpr):
    #1. one program handles BLOCK packed bytes = 2*BLOCK output elements
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_bytes
    b = tl.load(packed_ptr + offs, mask=mask, other=0).to(tl.int32)
    #2. LOW nibble = element 2j, HIGH nibble = element 2j+1 (ref #2)
    lo = b & 0x0F
    hi = (b >> 4) & 0x0F
    #3. this byte's fp32 group scale: k2 bytes per row, grp2 bytes per
    #   scale group (= group_size/2), k2//grp2 scales per row
    row = offs // k2
    grp = (offs % k2) // grp2
    s = tl.load(s_ptr + row * (k2 // grp2) + grp, mask=mask, other=0.0)
    #4. fp32 value*scale, single RTNE downcast to bf16 (ref #3-#5 order)
    o = offs.to(tl.int64) * 2
    tl.store(out_ptr + o, (_e2m1(lo) * s).to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + o + 1, (_e2m1(hi) * s).to(tl.bfloat16), mask=mask)


def dequant_nvfp4_triton(packed, scale, scale2, group_size,
                         out_dtype=torch.bfloat16):
    """Drop-in for loader.dequant_nvfp4 (same contract, same asserts,
    bf16 out only — the only dtype the serving path uses)."""
    #1. contract checks — verbatim from the reference
    assert packed.dtype == torch.uint8, packed.dtype
    assert scale.dtype == torch.float8_e4m3fn, scale.dtype
    assert scale2.dtype == torch.float32, scale2.dtype
    assert out_dtype == torch.bfloat16, out_dtype
    *lead, rows, pk = packed.shape
    k = pk * 2
    assert k % group_size == 0
    assert tuple(scale.shape) == (*lead, rows, k // group_size), scale.shape
    assert tuple(scale2.shape) == tuple(lead), scale2.shape

    #2. per-group fp32 scale product in eager torch — the reference's #4
    #   verbatim (k/16 elements per row; the cheap part stays bit-identical
    #   because it is the same op)
    s = scale.float() * scale2.reshape(*scale2.shape, 1, 1)

    #3. one kernel pass over the flattened bytes
    p = packed.reshape(-1, pk).contiguous()
    s = s.reshape(p.shape[0], k // group_size).contiguous()
    out = torch.empty(p.shape[0], k, device=packed.device, dtype=out_dtype)
    n_bytes = p.numel()
    BLOCK = 1024
    grid = ((n_bytes + BLOCK - 1) // BLOCK,)
    _nvfp4_dequant_kernel[grid](p, s, out, n_bytes, pk, group_size // 2,
                                BLOCK=BLOCK)
    return out.reshape(*lead, rows, k)
