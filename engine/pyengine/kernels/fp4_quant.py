"""NVFP4 activation-quant Triton kernel (exp-0015a; groundwork arm L,
/workspace/logs/fp4_quant_groundwork_exp0015.py — validated bitwise vs the
torch reference on packed bytes AND swizzled scales, end-to-end identical
through torch._scaled_mm, deterministic across reruns).

Recipe (vLLM ModelOpt NVFP4, models/inkling/nvidia/moe.py — the REFERENCE'S
OWN numeric path for these layers, so this is parity work, not a shortcut):
  input_scale = input_amax / (448 * 6)                     [fixed at load]
  sf_e4m3     = cast_rn_satfinite_e4m3(group_amax / (6 * input_scale))
  q_e2m1      = cvt.rn.satfinite.e2m1x2.f32(x / (fp32(sf_e4m3) * input_scale))
The GEMM computes sum(q_a q_b sfa sfb); the per-expert global product
(input_scale * weight_scale2) is applied in the GEMM epilogue.

Scales are written DIRECTLY in the Blackwell 32_4_4 blocked swizzle (the
cuBLAS/CuTe scale-factor atom layout) — verified bitwise against the CUTLASS
example's to_blocked() by construction and again in t_moe_w4a4 arm a.
Row-map variant (HAS_MAP) reads source row src[i] of x and writes packed
bytes + scales at destination row dst[i] — used by the grouped MoE path
where real (token, slot) pair rows scatter into a 128-padded per-expert
row space (pad rows stay all-zero: zero packed bytes = +0.0 e2m1, zero
scale bytes = 0.0 e4m3, contributing exactly 0 to any MMA that reads them;
e4m3 NaN encodings can never appear).

Numerics: RTNE everywhere via the HARDWARE converts (fp32->e4m3 through
Triton's native float8e4nv cast; e2m1 via inline asm cvt.rn.satfinite,
pack=1, operand order $1=high/$2=low probe-verified). Deterministic: pure
elementwise, fixed grid, each output written once.
Constraint: the k-loop is unmasked — BLOCK_K must divide K (engine shapes
6144/3072 satisfy every candidate; the wrapper auto-picks)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _nvfp4_quant_kernel(x_ptr, pk_ptr, sf_ptr, src_ptr, dst_ptr,
                        inv_gs, gs, K, NC,
                        HAS_MAP: tl.constexpr, BLOCK_K: tl.constexpr):
    """One program per quantized row. K = row length (mult of 16), processed
    in BLOCK_K chunks (BLOCK_K mult of 16). Packed out [Mp, K/2]
    low-nibble-first; scales written straight into the 32_4_4 blocked
    swizzle (flat), NC = padded scale-column count / 4 (= ceil(K/16/4)).
    HAS_MAP: read x row src[i], write packed/scale row dst[i] (dst rows of
    distinct programs are distinct -> each output written exactly once)."""
    i = tl.program_id(0)
    if HAS_MAP:
        m_src = tl.load(src_ptr + i)
        m_dst = tl.load(dst_ptr + i)
    else:
        m_src = i
        m_dst = i
    g32 = (m_dst % 128) % 32
    g4 = (m_dst % 128) // 32
    base_g = (m_dst // 128) * NC
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + m_src.to(tl.int64) * K + offs).to(tl.float32)
        xg = tl.reshape(x, (BLOCK_K // 16, 16))
        amax = tl.max(tl.abs(xg), axis=1)
        # e4m3 scale through the global scale; hardware RTNE+satfinite cast
        sf8 = (amax * (inv_gs / 6.0)).to(tl.float8e4nv)
        scf = sf8.to(tl.float32) * gs
        # guard all-zero groups (0/0 -> nan): x is 0 there, any divisor works
        div = tl.where(scf == 0.0, 1.0, scf)
        y = xg / div[:, None]
        ylo, yhi = tl.split(tl.reshape(y, (BLOCK_K // 2, 2)))
        # cvt.rn.satfinite.e2m1x2.f32 d, a, b packs a->HIGH nibble, b->LOW
        # (probe-verified); we want elem0 (ylo) in the LOW nibble.
        byte = tl.inline_asm_elementwise(
            "{\n.reg .b8 t;\ncvt.rn.satfinite.e2m1x2.f32 t, $1, $2;\n"
            "cvt.u32.u8 $0, t;\n}",
            "=r,f,f", [yhi, ylo], dtype=tl.uint32, is_pure=True, pack=1)
        tl.store(pk_ptr + m_dst.to(tl.int64) * (K // 2) + k0 // 2
                 + tl.arange(0, BLOCK_K // 2), byte.to(tl.uint8))
        # swizzled scale offsets: group index kb; blocked 32_4_4 layout
        # off = ((G*32 + g32)*4 + g4)*4 + kb%4, G = (m_dst//128)*NC + kb//4
        kb = k0 // 16 + tl.arange(0, BLOCK_K // 16)
        G = base_g + kb // 4
        off = ((G * 32 + g32) * 4 + g4) * 4 + kb % 4
        tl.store(sf_ptr + off, sf8)


def _pick_block(K):
    return next(b for b in (2048, 1024, 512, 256, 128, 64, 32, 16)
                if K % b == 0)


def nvfp4_quant(x, input_scale):
    """x [M, K] bf16 contiguous -> (packed [M, K/2] uint8, sf flat e4m3 of
    size ceil(M/128)*128 * ceil(K/16/4)*4 in the 32_4_4 blocked swizzle)."""
    M, K = x.shape
    assert K % 16 == 0 and x.stride(1) == 1 and x.stride(0) == K
    kg = K // 16
    nr, nc = -(-M // 128), -(-kg // 4)
    packed = torch.empty(M, K // 2, device=x.device, dtype=torch.uint8)
    # zero-init: swizzle rows/cols beyond (M, kg) must be 0; the kernel
    # only writes live positions
    sf = torch.zeros(nr * 128 * nc * 4, device=x.device,
                     dtype=torch.float8_e4m3fn)
    with torch.cuda.device(x.device):
        _nvfp4_quant_kernel[(M,)](
            x, packed, sf, x, x, 1.0 / input_scale, input_scale, K, nc,
            HAS_MAP=False, BLOCK_K=_pick_block(K), num_warps=4)
    return packed, sf


def nvfp4_quant_scatter(x, src, dst, m_pad, input_scale):
    """Quantize the P rows x[src[i]] into row dst[i] of a zeroed
    128-row-padded destination space (m_pad rows, multiple of 128; every
    dst < m_pad). Returns (packed [m_pad, K/2] uint8 zeros outside dst,
    sf flat e4m3 [m_pad * ceil(K/16/4)*4] zeros outside dst) — with all
    expert groups 128-row-aligned the per-group 32_4_4 swizzle equals this
    global one, so the buffer is directly the grouped GEMM's SFA."""
    P = src.shape[0]
    M, K = x.shape
    assert K % 16 == 0 and m_pad % 128 == 0
    assert src.dtype == torch.int32 and dst.dtype == torch.int32
    kg = K // 16
    nc = -(-kg // 4)
    packed = torch.zeros(m_pad, K // 2, device=x.device, dtype=torch.uint8)
    sf = torch.zeros(m_pad * nc * 4, device=x.device,
                     dtype=torch.float8_e4m3fn)
    if P:
        with torch.cuda.device(x.device):
            _nvfp4_quant_kernel[(P,)](
                x, packed, sf, src, dst, 1.0 / input_scale, input_scale,
                K, nc, HAS_MAP=True, BLOCK_K=_pick_block(K), num_warps=4)
    return packed, sf
