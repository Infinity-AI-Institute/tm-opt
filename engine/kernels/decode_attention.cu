/*
 * @brief  Decode attention: 1 query token per sequence.
 *         Two cache kinds per ModelConfig: paged global (8 KV heads) and
 *         ring-512 SWA (16 KV heads); head_dim 128 for both. Relative-position
 *         bias (d_rel=16, extent 1024) + log length scaling fused pre-softmax
 *         (Inkling has no RoPE). Warp per KV-head; float4 loads; split-KV
 *         reduction ordered deterministically for batch-invariant parity.
 */
#include <cuda_runtime.h>   
#include <cstdint>          
#include "tmopt/kernels.h"

namespace tmopt {
namespace kernels {

void decode_attention(const void *q, const void *kv, const void *rel_bias,
                      void *out, uint32_t batch, uint32_t max_seq,
                      bool is_swa_layer, cudaStream_t stream)
{
    //1. block = (seq, kv_head); warp iterates pages, float4 loads into regs
    //2. online softmax (m, l running stats) — fixed reduction tree order so
    //   results are batch-invariant (parity gate depends on this)
    //3. local layers read the ring buffer window; global read page table
    //TODO(exp-0006)
    (void)q; (void)kv; (void)rel_bias; (void)out; (void)batch;
    (void)max_seq; (void)is_swa_layer; (void)stream;
}

}  //namespace kernels
}  //namespace tmopt
