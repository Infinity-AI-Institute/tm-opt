/*
 * @brief  Paged KV cache with per-layer-type layouts.
 *         Global-attention layers get full paged KV; local (sliding-window)
 *         layers get a fixed ring buffer of window size — the single biggest
 *         memory win over vLLM's uniform paging for this model.
 */
#pragma once
#include <cstdint>   
#include <vector>    
#include "config.h"

namespace tmopt {

using PageId = uint32_t;
constexpr PageId kInvalidPage = 0xFFFFFFFFu;

class KvCache {
public:
    KvCache(const ModelConfig &mc, const RuntimeConfig &rc);
    ~KvCache();

    //1. allocate pages for a new sequence (global layers only; ring buffers
    //   for local layers are allocated once per slot, never grow)
    bool alloc_sequence(uint32_t seq_id, uint32_t prompt_len);
    //2. extend by one token during decode; returns false when out of pages
    bool extend(uint32_t seq_id);
    //3. free everything owned by a finished sequence
    void release(uint32_t seq_id);

    uint32_t free_pages() const;

private:
    //1. device pool: [num_global_layers, num_pages, page_size, 2, kv_heads, head_dim]
    void *global_pool_ = nullptr;
    //2. device pool: [num_local_layers, max_slots, window, 2, kv_heads, head_dim]
    void *local_ring_ = nullptr;
    std::vector<PageId> free_list_;
    ModelConfig mc_;
    RuntimeConfig rc_;
};

}  //namespace tmopt