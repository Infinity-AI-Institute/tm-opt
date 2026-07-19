/*
 * @brief  Fused MoE block for Inkling (6-of-256 + 2 shared experts).
 *         gate GEMV -> sigmoid(+bias) -> top-6 -> norm_after_topk -> token permute -> grouped GEMM
 *         (NVFP4 weights via B300 tensor cores) -> weighted unpermute-add.
 */
#include <cuda_runtime.h>   
#include <cstdint>          
#include "tmopt/kernels.h"

namespace tmopt {
namespace kernels {

void moe_grouped_gemm(const void *x, const void *w_experts, const void *gate_w,
                      void *y, uint32_t n_tokens, cudaStream_t stream)
{
    //1. gate: [n_tokens, hidden] x [hidden, 256] -> logits (small GEMM)
    //2. fused sigmoid gate (+bias) + top-6 select + norm_after_topk (route_scale 8),
    //   write (expert_id, weight, src_row)
    //3. exclusive scan over expert counts -> permuted row offsets
    //4. permute activations into expert-contiguous staging (coalesced)
    //5. grouped GEMM: per-expert tiles, NVFP4 B-operand, BF16 accumulate out
    //6. shared experts: dense GEMM over all tokens in the same launch
    //7. unpermute + weighted sum back into y
    //TODO(exp-0001): first correct version via CUTLASS grouped GEMM,
    //                then iterate tiles/stages through the harness
    (void)x; (void)w_experts; (void)gate_w; (void)y; (void)n_tokens; (void)stream;
}

}  //namespace kernels
}  //namespace tmopt
