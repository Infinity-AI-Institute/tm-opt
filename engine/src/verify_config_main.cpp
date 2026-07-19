/*
 * @brief  Standalone entry: verify config.h against a checkpoint directory.
 *         Usage: ./verify_config /workspace/models/inkling-nvfp4
 *         Exit 0 = all fields match; abort with diff otherwise.
 *         Run this in CI / after any checkpoint update / in worker preflight.
 */
#include <cstdio>            
#include "tmopt/config.h"

int main(int argc, char **argv)
{
    if (argc != 2) { std::printf("usage: %s <model_dir>\n", argv[0]); return 2; }
    tmopt::load_and_verify_model_config(argv[1]);
    return 0;
}
