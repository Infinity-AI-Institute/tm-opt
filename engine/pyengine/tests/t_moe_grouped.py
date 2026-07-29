"""exp-0003 evidence: grouped packed-NVFP4 routed-expert GEMM vs the
reference per-hit-expert loop.

Arms (GPUs 4-7 only; no model load — synthetic packs at exact checkpoint
shapes):
  1. small-shape value check: grouped path vs model's reference loop run on
     the densely dequantized weights (dequant is bit-equal by exp-0002, so
     the reference loop on dense tensors IS the pre-exp-0003 numeric path).
     Expect bf16 accumulation-order drift only (B3.1/B3.3 class), not
     structural error: assert allclose atol catches wrong rows/scales/
     interleave (those give O(1) errors everywhere).
  2. real-shape value check (E=256, hidden 6144, ffn 3072) at decode T=8
     and prefill-chunk T=512 (exercises the in-program multi-BLOCK_M row
     loop: 512*6/256 ~ 12 rows/expert avg, tail experts > 16).
  3. determinism: two identical calls bitwise-equal (stable sort, fixed
     grids, no atomics).
  4. timing at real shapes: grouped vs the CURRENT production path (the
     loop dequanting on demand through PackedExperts.__getitem__).

    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m engine.pyengine.tests.t_moe_grouped
"""
import time

import torch
import triton
import triton.language as tl

from engine.pyengine import loader as pload
from engine.pyengine import model as pmodel
from engine.pyengine.kernels.moe_gemm import _e2m1, moe_experts_packed


@triton.jit
def _lut_probe(b_ptr, o_ptr, s, BLOCK: tl.constexpr):
    """Test-only: dequant BLOCK bytes with a constant fp32 group scale s
    through the production _e2m1 + (val*s)->bf16 sequence."""
    offs = tl.arange(0, BLOCK)
    b = tl.load(b_ptr + offs).to(tl.int32)
    tl.store(o_ptr + 2 * offs, (_e2m1(b & 0x0F) * s).to(tl.bfloat16))
    tl.store(o_ptr + 2 * offs + 1,
             (_e2m1((b >> 4) & 0x0F) * s).to(tl.bfloat16))


def check_lut_bitwise(device):
    """Arm 0: the kernel's bit-constructed e2m1 LUT + fp32 mul + RTNE bf16
    round, bitwise vs loader.dequant_nvfp4 (the proven B1.4 eager
    reference) over ALL 256 byte patterns x edge/typical scales."""
    bytes_all = torch.arange(256, dtype=torch.uint8, device=device)
    for f8 in [0.0, 1.0, 448.0, -1.5, 2.0 ** -9, 0.017578125]:
        for s2 in [1.0, -0.25, 3.7e-3]:
            scale = torch.full((2, 16), f8).to(torch.float8_e4m3fn)
            s2_t = torch.tensor(s2, dtype=torch.float32)
            expect = pload.dequant_nvfp4(
                bytes_all.reshape(2, 128).cpu(), scale, s2_t, 16).to(device)
            #1. fp32 product exactly as the reference computes it (a python
            #   double product would double-round)
            s = float(scale.float()[0, 0] * s2_t)
            out = torch.empty(512, dtype=torch.bfloat16, device=device)
            _lut_probe[(1,)](bytes_all, out, s, BLOCK=256)
            assert torch.equal(out.view(2, 256), expect), \
                f"LUT mismatch at f8={f8} s2={s2}"
    print("[lut] bitwise vs loader.dequant_nvfp4, all 256 bytes x 18 "
          "scales: PASS", flush=True)


def make_pack(E, rows, k, device, deinterleave, seed):
    """Synthetic checkpoint-form pack: u8 nibbles, f8e4m3 group scales,
    per-expert f32 scale2 — value distributions irrelevant to the check
    (the reference consumes the same bytes)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    packed = torch.randint(0, 256, (E, rows, k // 2), generator=g,
                           dtype=torch.uint8).to(device)
    scale = (torch.rand(E, rows, k // 16, generator=g) * 2 + 0.01) \
        .to(torch.float8_e4m3fn).to(device)
    scale2 = (torch.rand(E, generator=g) * 0.01 + 1e-4).to(device)
    return pload.PackedExperts(packed, scale, scale2, 16, deinterleave)


def dense(pack):
    """Dense bf16 stack through the proven bit-equal per-expert dequant —
    the pre-exp-0003 numeric path's exact weight values."""
    return torch.stack([pack[e] for e in range(pack.packed.shape[0])])


def routing(T, E, top_k, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.stack([torch.randperm(E, generator=g)[:top_k]
                       for _ in range(T)]).to(device)
    wts = (torch.rand(T, top_k, generator=g) * 2 - 0.5) \
        .to(torch.bfloat16).to(device)
    return idx, wts


def check(tag, E, K, ffn, T, top_k, device, atol):
    pack13 = make_pack(E, 2 * ffn, K, device, True, seed=hash(tag) % 9973)
    pack2 = make_pack(E, K, ffn, device, False, seed=hash(tag) % 9973 + 1)
    g = torch.Generator(device="cpu").manual_seed(42)
    x = (torch.randn(T, K, generator=g) * 0.5).to(torch.bfloat16).to(device)
    idx, wts = routing(T, E, top_k, device, seed=7)

    out = moe_experts_packed(x, pack13, pack2, idx, wts)
    ref = pmodel.moe_experts(x, dense(pack13), dense(pack2), idx, wts)
    d = (out.float() - ref.float()).abs()
    scale = ref.float().abs().mean()
    print(f"[{tag}] max|d| {d.max():.6f} mean|d| {d.mean():.6f} "
          f"(ref mean|.| {scale:.4f})", flush=True)
    assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=0.05), \
        f"{tag}: structural mismatch (max|d| {d.max():.6f})"

    out2 = moe_experts_packed(x, pack13, pack2, idx, wts)
    assert torch.equal(out, out2), f"{tag}: NOT deterministic"
    print(f"[{tag}] determinism bitwise PASS", flush=True)
    return pack13, pack2, x, idx, wts


def main():
    device = "cuda:0"
    #0. LUT bit-equality vs the proven eager dequant
    check_lut_bitwise(device)

    #1. small shapes: fast, catches indexing/interleave/scale bugs
    check("small", E=8, K=256, ffn=128, T=5, top_k=3, device=device,
          atol=0.05)

    #2+3. real checkpoint shapes, decode and prefill-chunk row counts
    p13, p2, x8, idx8, wts8 = check(
        "real-T8", E=256, K=6144, ffn=3072, T=8, top_k=6, device=device,
        atol=0.35)
    check("real-T512", E=256, K=6144, ffn=3072, T=512, top_k=6,
          device=device, atol=0.35)

    #4. timing vs the current production path (loop + on-demand dequant)
    for T in (8, 512):
        g = torch.Generator(device="cpu").manual_seed(3)
        x = (torch.randn(T, 6144, generator=g) * 0.5) \
            .to(torch.bfloat16).to(device)
        idx, wts = routing(T, 256, 6, device, seed=T)
        for _ in range(3):
            moe_experts_packed(x, p13, p2, idx, wts)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            moe_experts_packed(x, p13, p2, idx, wts)
        torch.cuda.synchronize()
        t_new = (time.time() - t0) / 10
        #1. production loop path: reach it by passing the packs through a
        #   shim WITHOUT the .packed attribute (dispatch stays untouched)
        class Shim:
            def __init__(self, p):
                self._p = p
            def __getitem__(self, e):
                return self._p[e]
        for _ in range(2):
            pmodel.moe_experts(x, Shim(p13), Shim(p2), idx, wts)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            pmodel.moe_experts(x, Shim(p13), Shim(p2), idx, wts)
        torch.cuda.synchronize()
        t_old = (time.time() - t0) / 5
        print(f"[time T={T}] grouped {t_new * 1e3:.3f} ms vs loop "
              f"{t_old * 1e3:.3f} ms -> {t_old / t_new:.2f}x", flush=True)


if __name__ == "__main__":
    main()
