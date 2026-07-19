/*
 * @brief  Launchers for all custom CUDA kernels. Implementations in
 *         engine/kernels/*.cu; every kernel change goes through the harness
 *         (parity gate + benchmark) before merge.
 *         Kernel set mirrors docs/ARCHITECTURE.md rev 2 (Inkling has NO RoPE;
 *         attention carries a relative-position bias instead).
 */
#pragma once
#include <cstdint>            
#include <cuda_runtime.h>     

namespace tmopt {
namespace kernels {

//1. fused RMSNorm (eps from ModelConfig); embed_norm variant shares the kernel
void fused_rmsnorm(void *x, const void *gamma, uint32_t n_tokens,
                   uint32_t hidden, float eps, cudaStream_t stream);

//2. sconv: window-4 depthwise conv on attn K/V/output and MoE output;
//   per-layer ring state (kernel size from ModelConfig.sconv_kernel)
void sconv(void *x, const void *w, void *ring_state, uint32_t n_tokens,
           uint32_t channels, cudaStream_t stream);

//3. MoE: fused sigmoid gate(+bias) -> top-6 -> norm_after_topk -> permute ->
//   grouped GEMM over selected experts (skinny N=3072, NVFP4 B-operand,
//   device-side work queue) + shared-expert(sink) GEMM in the same launch
void moe_grouped_gemm(const void *x, const void *w_experts, const void *gate_w,
                      void *y, uint32_t n_tokens, cudaStream_t stream);

//4. decode attention: 1 query token vs paged (global, 8 KV heads) or
//   ring-512 (SWA, 16 KV heads) cache; relative bias + log scaling fused
//   pre-softmax; deterministic reduction order (parity gate is batch-sensitive)
void decode_attention(const void *q, const void *kv, const void *rel_bias,
                      void *out, uint32_t batch, uint32_t max_seq,
                      bool is_swa_layer, cudaStream_t stream);

}  //namespace kernels
}  //namespace tmopt
