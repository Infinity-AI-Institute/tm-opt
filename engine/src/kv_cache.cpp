/*
 * @brief  KvCache implementation. See kv_cache.h for layout rationale.
 */
#include "tmopt/kv_cache.h"

namespace tmopt {

KvCache::KvCache(const ModelConfig &mc, const RuntimeConfig &rc)
    : mc_(mc), rc_(rc)
{
    //1. size global pool from gpu_mem_fraction minus weights + activations
    //2. cudaMalloc one slab per pool; build free list
    //TODO(exp-0004): local ring buffers sized to sliding_window
}

KvCache::~KvCache()
{
    //1. single cudaFree per slab; pointers nulled
    global_pool_ = nullptr;
    local_ring_  = nullptr;
}

bool KvCache::alloc_sequence(uint32_t, uint32_t) { return false; }  //TODO
bool KvCache::extend(uint32_t)                   { return false; }  //TODO
void KvCache::release(uint32_t)                  {}                 //TODO
uint32_t KvCache::free_pages() const             { return 0; }      //TODO

}  //namespace tmopt
