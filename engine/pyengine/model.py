"""B2: the 66-layer graph. Order per layer: rmsnorm -> attention (global or
SWA by layer id; rel-bias + log scaling pre-softmax; sconv on K,V,attn-out)
-> residual -> rmsnorm -> MoE (sigmoid gate+bias, top-6, norm_after_topk,
route_scale 8, +2 shared sink experts; layers 0-1 = dense MLP) -> sconv on
moe-out -> residual. No RoPE anywhere."""
import torch


def rmsnorm(x, weight, eps=1e-6):
    """Torch-reference RMSNorm with exact InklingRMSNorm semantics
    (transformers models/inkling/modeling_inkling.py:99-113): variance in
    fp32, normalized value downcast to the INPUT dtype BEFORE the weight
    multiply, so the final scale runs in bf16. B2.3's Triton kernel is
    checked against this function."""
    #1. fp32 upcast + mean-of-squares over the hidden dim
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    #2. normalize in fp32, downcast, THEN scale by weight (bf16 x bf16)
    return weight * (xf * torch.rsqrt(var + eps)).to(x.dtype)


def embed(input_ids, w_embed, w_norm, eps=1e-6):
    """Token embedding + embed_norm. transformers does
    inputs_embeds = embed_norm(embed_tokens(input_ids)) with a plain
    nn.Embedding lookup, no multiplier/scale (modeling_inkling.py:654,
    659, 682). Returns (lookup, normed) so tests can gate both boundaries."""
    #1. plain row lookup from the (vocab, hidden) table
    h = w_embed[input_ids]
    #2. RMSNorm with the embed_norm weight
    return h, rmsnorm(h, w_norm, eps)
