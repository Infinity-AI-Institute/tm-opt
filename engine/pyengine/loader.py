"""B1: checkpoint -> GPU tensors. NVFP4 block-scale layout per vLLM modelopt
(cite file:line when implemented). TP=4 plan: attention head-parallel,
MoE expert-parallel (64 experts/GPU), embeddings replicated."""
#TODO(B1.1..B1.6): implemented item-by-item by the build loop.
