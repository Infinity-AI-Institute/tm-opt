/*
 * @brief  Scheduler implementation. Decode-first admission keeps ITL flat;
 *         chunked prefill keeps TTFT bounded.
 */
#include "tmopt/scheduler.h"

namespace tmopt {

Scheduler::Scheduler(KvCache &kv, uint32_t max_batch_tokens)
    : kv_(kv), budget_(max_batch_tokens) {}

void Scheduler::enqueue(Sequence *seq) { waiting_.push_back(seq); }

StepPlan Scheduler::next_step()
{
    StepPlan plan;
    //1. admit all running decodes first (1 token each against budget)
    //2. fill remaining budget with chunked prefill from waiting_, FCFS
    //3. preempt-on-oom policy: youngest sequence recomputes later
    //TODO(exp-0005)
    return plan;
}

}  //namespace tmopt