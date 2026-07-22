#1. B0.2: prove triton compiles + runs on the visible GPUs (expects CUDA_VISIBLE_DEVICES=4,5,6,7)
import torch, triton, triton.language as tl

@triton.jit
def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

def main():
    n = 4096
    x, y = (torch.randn(n, device="cuda") for _ in range(2))
    o = torch.empty(n, device="cuda")
    _add[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
    assert torch.allclose(o, x + y), "triton smoke mismatch"
    print(f"triton ok on {torch.cuda.get_device_name(0)}; triton {triton.__version__}")

if __name__ == "__main__":
    main()
