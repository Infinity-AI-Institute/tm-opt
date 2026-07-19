/*
 * @brief  Engine entry point: loads Inkling, starts the OpenAI-compatible
 *         HTTP server (/v1/completions, /health), runs the step loop.
 *         SKELETON .
 */
#include <cstdio>            
#include "tmopt/config.h"
#include "tmopt/kv_cache.h"
#include "tmopt/scheduler.h"

int main(int argc, char **argv)
{
    //1. parse flags: --model, --port, --max-batch-tokens, --mtp
    //2. load config.json -> ModelConfig; mmap safetensors shards (NVFP4)
    //3. construct KvCache (paged global pool + local ring buffers)
    //4. construct Scheduler with token budget
    //5. start HTTP server thread: enqueue Sequence per request, stream SSE out
    //6. step loop: plan = scheduler.next_step(); run prefill batch, decode
    //   batch, (MTP draft+verify when enabled); sample; stream tokens
    //TODO(exp-0002): step loop with CUDA graphs for the decode path
    std::printf("tmopt engine skeleton — not yet serving\n");
    (void)argc; (void)argv;
    return 0;
}
