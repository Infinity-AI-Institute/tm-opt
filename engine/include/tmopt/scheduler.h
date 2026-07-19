/*
 * @brief  Continuous-batching scheduler: one prefill/decode iteration per
 *         step under a token budget, chunked prefill, FCFS with decode
 *         priority so TTFT stays bounded under load.
 */
#pragma once
#include <cstdint>   
#include <deque>     
#include <vector>    
#include "kv_cache.h"

namespace tmopt {

struct Sequence {
    uint32_t seq_id;
    std::vector<uint32_t> token_ids;
    uint32_t generated = 0;
    uint32_t max_new_tokens;
    bool     finished = false;
};

struct StepPlan {
    std::vector<Sequence *> prefill;   //chunked to fit token budget
    std::vector<Sequence *> decode;    //one token each
};

class Scheduler {
public:
    Scheduler(KvCache &kv, uint32_t max_batch_tokens);
    void     enqueue(Sequence *seq);
    StepPlan next_step();              //called once per engine iteration

private:
    KvCache &kv_;
    uint32_t budget_;
    std::deque<Sequence *> waiting_;
    std::vector<Sequence *> running_;
};

}  //namespace tmopt
