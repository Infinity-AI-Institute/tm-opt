"""B2: the 66-layer graph. Order per layer: rmsnorm -> attention (global or
SWA by layer id; rel-bias + log scaling pre-softmax; sconv on K,V,attn-out)
-> residual -> rmsnorm -> MoE (sigmoid gate+bias, top-6, norm_after_topk,
route_scale 8, +2 shared sink experts; layer 2 = dense MLP) -> sconv on
moe-out -> residual. No RoPE anywhere."""
#TODO(B2.2..B2.11)
