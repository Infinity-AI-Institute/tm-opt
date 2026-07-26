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


def sconv_prefill(x, weight):
    """Window-4 short convolution (sconv), PREFILL form — exact
    InklingShortConvolution semantics (transformers models/inkling/
    modeling_inkling.py:500-542). The whole module runs in fp32: conv
    weights are held fp32 (_keep_in_fp32_modules_strict :610; checkpoint
    stores bf16 — the upcast is exact) and the input is upcast on entry.
    Depthwise causal conv1d, no bias (conv1d bias=False :498) and no
    activation (causal_conv1d_fn :461-481: pad kernel-1 both ends, keep the
    first T outputs — so out[t] = sum_j w[c,0,j] * x[t-(k-1)+j, c], i.e.
    weight[..., -1] taps the CURRENT token; torch conv1d is
    cross-correlation, no kernel flip). The module then adds its OWN input
    back (residual INSIDE the module, in fp32, :515+:542) and downcasts to
    the input dtype. x [B, T, C]; weight (C, 1, k). The decode form (ring
    state, causal_conv1d_update :441-457) is B3.1's job; the fused Triton
    kernel is Stage-3 work (D4)."""
    #1. fp32 upcast — module computes everything in fp32
    xf = x.float()
    w = weight.float()
    #2. depthwise causal conv over time, channels-first, same F.conv1d call
    #   as the ref (pad k-1, truncate to the first T outputs = causal)
    conv = torch.nn.functional.conv1d(
        xf.transpose(1, 2), w, bias=None, padding=w.shape[-1] - 1,
        groups=w.shape[0])[:, :, : x.shape[1]]
    #3. own-input residual in fp32, then downcast to the input dtype
    return (conv.transpose(1, 2) + xf).to(x.dtype)


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


def additive_causal_mask(q_positions, kv_positions, dtype, window=None):
    """Additive attention mask in transformers' eager form
    (masking_utils.py eager_mask: 0.0 where the (q, k) pair takes part,
    torch.finfo(dtype).min where it does not; shape [1, 1, Q, K] — the
    batch axis broadcasts; no padding in text-only v1 scope). Allowed =
    causal (causal_mask_function: kv_pos <= q_pos) AND, when `window` is
    given, inside the sliding window (sliding_window_overlay:
    kv_pos > q_pos - window, i.e. distance < window). Pure selection
    between the two eager constants — no arithmetic — so the tensor is
    bitwise identical to the native builder's."""
    #1. backward distance per (q, k) pair
    d = q_positions[:, None] - kv_positions[None, :]
    #2. in-window causal predicate
    allowed = d >= 0
    if window is not None:
        allowed &= d < window
    #3. select the two eager constants, add leading [1, 1] broadcast axes
    zero = torch.zeros((), dtype=dtype, device=allowed.device)
    return torch.where(allowed, zero, torch.finfo(dtype).min)[None, None]


def attn_prefill(x, wq, wk, wv, wr, wo, w_k_sconv, w_v_sconv, w_q_norm,
                 w_k_norm, rel_proj, attn_mask, q_positions, kv_positions,
                 head_dim, is_global, alpha=None, n_floor=None, eps=1e-6):
    """One attention layer, PREFILL (no-cache) form — exact InklingAttention
    semantics (transformers models/inkling/modeling_inkling.py:217-282 with
    eager_attention_forward :157-182). Both layer shapes run through here:
    global (64Q/8KV heads, rel extent 1024, log scaling) and SWA (64Q/16KV,
    extent = window 512, no scaling) differ only in weights, mask and
    `is_global`. x [B, T, C] is the POST-input_layernorm hidden state;
    attn_mask is the additive eager mask (additive_causal_mask). Scaling is
    1/head_dim, not 1/sqrt, because q/k are per-head RMS-normalized
    (:197-198). No biases anywhere. The decode form (cached KV + sconv ring
    state) is B3's job; fused kernels are Stage-3 work (D4)."""
    F = torch.nn.functional
    B, T = x.shape[0], x.shape[1]
    n_heads, n_kv = wq.shape[0] // head_dim, wk.shape[0] // head_dim
    #1. projections (:228-231); K and V pass their window-4 sconv first
    q = F.linear(x, wq)
    k = sconv_prefill(F.linear(x, wk), w_k_sconv)
    v = sconv_prefill(F.linear(x, wv), w_v_sconv)
    r = F.linear(x, wr)
    #2. per-head q/k rmsnorm over head_dim, then [B,T,H,D] -> [B,H,T,D]
    #   (:233-235; v is not normalized)
    q = rmsnorm(q.view(B, T, n_heads, head_dim), w_q_norm, eps).transpose(1, 2)
    k = rmsnorm(k.view(B, T, n_kv, head_dim), w_k_norm, eps).transpose(1, 2)
    v = v.view(B, T, n_kv, head_dim).transpose(1, 2)
    #3. rel-pos bias from the relative states (:248-251)
    bias = rel_bias(r.view(B, T, n_heads, -1), rel_proj, q_positions,
                    kv_positions)
    #4. log length scaling on global layers only (:254-261). tau == 1.0
    #   below the 128000 floor, but the ref runs the fp32 round-trip
    #   unconditionally on global layers — so do we, for bit parity
    if is_global and n_floor is not None:
        q, bias = apply_log_scaling(q, bias, q_positions, alpha, n_floor)
    #5. GQA: repeat each KV head n_heads/n_kv times (repeat_kv :145-154 —
    #   expand + reshape materializes contiguous copies, like the ref)
    rep = n_heads // n_kv
    k = k[:, :, None].expand(B, n_kv, rep, T, head_dim).reshape(
        B, n_heads, T, head_dim)
    v = v[:, :, None].expand(B, n_kv, rep, T, head_dim).reshape(
        B, n_heads, T, head_dim)
    #6. bf16 scores * 1/head_dim, + bias, + additive mask (:171-175 — same
    #   association order)
    attn = torch.matmul(q, k.transpose(2, 3)) * (1.0 / head_dim)
    attn = attn + bias
    if attn_mask is not None:
        attn = attn + attn_mask
    #7. fp32 softmax, downcast to the input dtype (:177; dropout is p=0 at
    #   eval — identity, skipped)
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    #8. weighted values -> [B,T,H*D] -> output projection (:179-181 + :280f)
    out = torch.matmul(attn, v).transpose(1, 2).contiguous()
    return F.linear(out.reshape(B, T, n_heads * head_dim), wo)
