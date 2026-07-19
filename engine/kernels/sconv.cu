/*
 * @brief  Short convolution (window 4) applied to attention K, V, output,
 *         and MoE output each layer. Decode-path state is a 4-deep per-channel
 *         ring; prefill is a causal depthwise conv.
 *         Fusion targets: fold into the producing GEMM epilogue or the
 *         consuming kernel's prologue — standalone launch is the fallback.
 */
#include <cuda_runtime.h>   //for launch config
#include <cstdint>          
#include "tmopt/kernels.h"

namespace tmopt {
namespace kernels {

void sconv(void *x, const void *w, void *ring_state, uint32_t n_tokens,
           uint32_t channels, cudaStream_t stream)
{
    //1. decode: y[c] = sum_{k=0..3} w[c,k] * ring[c, (head-k) mod 4]
    //2. prefill: causal window-4 depthwise conv along sequence dim
    //3. update ring state after compute (deterministic order)
    //TODO(exp): implement fused variants after correct standalone version
    (void)x; (void)w; (void)ring_state; (void)n_tokens; (void)channels; (void)stream;
}

}  //namespace kernels
}  //namespace tmopt
