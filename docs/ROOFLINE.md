# Roofline notes — B300, back-of-hand

Per-kernel roofline analyses are appended here by each accepted experiment.

## Machine balance (fill exact numbers from device query on the pod)
- HBM3e bandwidth: ~8 TB/s class
- NVFP4 tensor core peak: ~10+ PFLOPS class (dense)
- Balance point: peak_flops / bandwidth => arithmetic intensity threshold;
  kernels below it are bandwidth-bound (all decode-path kernels here are)

## Template per kernel
1. Bytes moved (weights + activations + KV, per token per layer)
2. FLOPs (2 * m * n * k terms summed)
3. AI = FLOPs / bytes -> which side of the balance point
4. Ceiling: min(peak_flops, AI * bandwidth) -> tok/s upper bound
5. Measured vs ceiling -> % of roofline, and the named limiter

## Decode-path headline (why NVFP4 DP wins)
Per decode token, dominant traffic is the 12B active parameters. At NVFP4
(~0.5 byte/param + scales) that is ~6-7 GB/token/GPU vs ~24 GB in BF16 —
a ~4x roofline lift on the exact path that sets tok/s.