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


def rel_bias(relative_states, proj, q_positions, kv_positions):
    """Relative-attention position bias, exact InklingRelativeLogits
    semantics (modeling_inkling.py:131-142). relative_states [B, Q, H, d_rel]
    (the r_proj output viewed per head) mixes proj — a trained bank of
    bias-vs-backward-distance profiles, shape (d_rel, extent) — into one bias
    value per (q, k) pair, gathered at distance d = q_pos - k_pos and zeroed
    outside 0 <= d < extent (causality/padding stay in the attention mask).
    extent comes from proj: rel_extent 1024 on global layers, window 512 on
    SWA layers (modeling_inkling.py:196). Returns [B, H, Q, K], proj dtype.
    B2.4 is the pure-torch form; a fused Triton kernel is later work (D4)."""
    #1. mix the profiles: [B,Q,H,d_rel] @ (d_rel,extent) -> [B,H,Q,extent]
    rel_logits = (relative_states @ proj).transpose(1, 2)
    #2. backward distance per (q, k) pair: [1,1,Q,K]
    extent = proj.shape[1]
    distance = (q_positions[:, None] - kv_positions[None, :])[None, None, :, :]
    #3. gather each pair's bias at its clamped distance
    gather_index = distance.clamp(0, extent - 1).expand(
        *rel_logits.shape[:2], -1, -1)
    position_bias = rel_logits.gather(-1, gather_index)
    #4. zero everything out of band (future keys, beyond-extent past)
    return position_bias.masked_fill(
        (distance < 0) | (distance >= extent), 0.0)


def log_scale_tau(q_positions, alpha, n_floor):
    """Per-query log length-scaling factor tau in fp32, exact ref math
    (modeling_inkling.py:254-258): tau = 1 + alpha*ln(clamp((pos+1)/floor,
    min=1)). The clamp makes tau == 1.0 EXACTLY for pos+1 <= floor, so with
    floor 128000 log scaling is inert across our whole 16K serving window;
    it is applied on GLOBAL layers only (caller's job, :254 `if not
    self.is_sliding`)."""
    #1. 1-based effective length, fp32 (exact for positions < 2^24)
    effective_n = (q_positions + 1).float()
    #2. clamp-to-1 kills the log below the floor
    return 1.0 + alpha * torch.log((effective_n / n_floor).clamp(min=1.0))


def apply_log_scaling(query_states, position_bias, q_positions, alpha,
                      n_floor):
    """Scale Q [B,H,Q,D] and position_bias [B,H,Q,K] by tau along the query
    axis: fp32 multiply, downcast back to the input dtype — exact op order
    of modeling_inkling.py:259-261. Global layers only."""
    #1. tau as a column vector over the query axis
    tau = log_scale_tau(q_positions, alpha, n_floor).view(1, 1, -1, 1)
    #2. fp32 multiply then downcast (matches the ref's rounding)
    q = (query_states.float() * tau).to(query_states.dtype)
    b = (position_bias.float() * tau).to(position_bias.dtype)
    return q, b
