/*
 * @brief  Fused RMSNorm: one global read + one global write per token.
 *         (Replaces rev-1 fused_rmsnorm_rope.cu — Inkling has no RoPE;
 *         position handling lives in the attention kernels as a relative bias.)
 */
#include <cuda_runtime.h>   
#include <cstdint>          
#include "tmopt/kernels.h"

namespace tmopt {
namespace kernels {

void fused_rmsnorm(void *x, const void *gamma, uint32_t n_tokens,
                   uint32_t hidden, float eps, cudaStream_t stream)
{
    //1. block per token; warp-shuffle reduction for sum of squares
    //2. normalize, scale by gamma
    //TODO(exp): implement; hidden=6144 => 48 float4 loads/thread-block pass
    (void)x; (void)gamma; (void)n_tokens; (void)hidden; (void)eps; (void)stream;
}

}  //namespace kernels
}  //namespace tmopt
