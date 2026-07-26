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


def moe_gate(x, w_gate, bias, global_scale, top_k, n_shared, route_scale):
    """MoE router — exact InklingTopkRouter semantics (transformers
    models/inkling/modeling_inkling.py:342-377). w_gate is
    (n_routed + n_shared, hidden): the router scores the 2 shared experts
    alongside the 256 routed ones in ONE linear (:357-358). Selection and
    weighting are DIFFERENT functions of those logits: the top_k CHOICE
    ranks sigmoid(routed logits) + e_score_correction_bias (:361-364 —
    the bias steers which experts win, DeepSeek-V3 style), while the
    WEIGHTS ignore the bias entirely — softmax over logsigmoid (= sigmoids
    normalized to sum 1) of the chosen routed logits AND the shared logits
    jointly (:366-370, the norm_after_topk form), then * route_scale *
    global_scale (:372). Slot order [top_k routed | n_shared shared];
    shared slots are returned separately as shared_gammas, consumed by
    InklingSharedExperts (:374-375). Tokens are flattened (:357): returns
    (routed_logits [T, n_routed], topk_weights [T, top_k], topk_indices
    [T, top_k] int64, shared_gammas [T, n_shared]) in the input dtype —
    the ref run holds the whole router in bf16 (checkpoint f32
    bias/global_scale downcast at load; router NOT in
    _keep_in_fp32_modules_strict :610). Fused Triton form is Stage-3 (D4)."""
    F = torch.nn.functional
    #1. flatten tokens, one linear over all routed+shared slots (:357-358)
    flat = x.reshape(-1, x.shape[-1])
    router_logits = F.linear(flat, w_gate)
    #2. choice scores: sigmoid of the ROUTED logits + correction bias;
    #   top-k indices only, unsorted (:361-364)
    scores = router_logits.sigmoid()
    scores_for_choice = scores[..., :-n_shared] + bias
    topk_indices = torch.topk(scores_for_choice, top_k, dim=-1,
                              sorted=False)[1]
    #3. weights from the LOGITS (no bias): gather the chosen routed logits,
    #   append the shared logits, normalize their sigmoids (:366-370)
    routed_logits = router_logits[..., :-n_shared]
    shared_logits = router_logits[..., -n_shared:]
    topk_logits = torch.cat(
        [routed_logits.gather(-1, topk_indices), shared_logits], dim=-1)
    topk_log_probs = F.logsigmoid(topk_logits)
    topk_weights = torch.exp(
        topk_log_probs - torch.logsumexp(topk_log_probs, dim=-1,
                                         keepdim=True))
    #4. scale, then split the [top_k | n_shared] slots (:372-375)
    topk_weights = topk_weights * route_scale * global_scale
    shared_gammas = topk_weights[..., -n_shared:].contiguous()
    topk_weights = topk_weights[..., :top_k].contiguous()
    return routed_logits, topk_weights, topk_indices, shared_gammas


def moe_experts(x, w_gate_up, w_down, topk_indices, topk_weights):
    """Routed expert GEMMs — exact InklingExperts.forward semantics
    (transformers models/inkling/modeling_inkling.py:315-339; the
    use_experts_implementation dispatch falls back to this original loop
    when no _experts_implementation is configured, integrations/moe.py:497-
    509 — the ref run's path). x is the FLATTENED token batch [T, hidden]
    (InklingMoE.forward flattens before the call, :422-423). w_gate_up
    [E, 2*ffn, hidden] holds DE-INTERLEAVED rows ([gates; ups] halves —
    the checkpoint's w13 stores interleaved [g0,u0,g1,u1,..] rows;
    transformers applies Interleave(dim=1) at load, conversion_mapping.py
    "inkling_mm_model", same de-interleave as vLLM inkling
    nvidia/moe.py:583-595); w_down [E, hidden, ffn]. Per hit expert in
    ASCENDING id order (expert_hit = nonzero of the expert-major mask,
    :323-327): fused gate/up GEMM -> chunk -> silu(gate)*up -> down GEMM
    -> * that (token, slot)'s routing weight AFTER down_proj (:336) ->
    index_add_ accumulation in the INPUT dtype (:321, :337). The ascending
    expert order fixes the bf16 rounding order; a token's slots hold
    distinct experts (topk positions, B2.8), so within-expert order cannot
    matter. Grouped-GEMM Triton form is Stage-3 work (D4)."""
    F = torch.nn.functional
    #1. accumulator in the input dtype, like the ref (:321)
    out = torch.zeros_like(x)
    #2. hit experts in ascending id order (:323-327)
    for e in torch.unique(topk_indices).tolist():
        #3. every (token, slot) this expert serves; batch their rows
        #   (:331-332 — index_add_ makes the within-expert order moot)
        token_idx, slot = torch.where(topk_indices == e)
        h = x[token_idx]
        #4. fused gate/up GEMM, chunk halves, silu * up, down GEMM
        #   (:333-335; act_fn = silu, config hidden_act)
        gate, up = F.linear(h, w_gate_up[e]).chunk(2, dim=-1)
        h = F.linear(F.silu(gate) * up, w_down[e])
        #5. routing weight AFTER down_proj, accumulate in the input dtype
        out.index_add_(
            0, token_idx, (h * topk_weights[token_idx, slot, None]).to(out.dtype))
    return out


def moe_shared(x, w_gate, w_up, w_down, gammas):
    """The n_shared=2 'sink' experts — exact InklingSharedExperts.forward
    semantics (modeling_inkling.py:394-405). x keeps its ORIGINAL shape
    (InklingMoE passes the pre-flatten residuals, :424); gammas
    [T, n_shared] are the router's shared-slot weights (B2.8). Weights are
    3D stacks: w_gate/w_up [n_shared, ffn, hidden] (checkpoint
    shared_w13_weight has interleaved rows -> Interleave(dim=1) +
    Chunk(dim=1), conversion_mapping.py "inkling_mm_model"), w_down
    [n_shared, hidden, ffn] (= shared_w2_weight). Unlike the routed path
    the gamma multiplies act_fn(gate)*up BEFORE down_proj (:401), and the
    n_shared expert outputs are summed in fp32 then downcast (:404)."""
    #1. broadcast tokens across the shared experts; gammas to [S, T, 1]
    shape = x.shape
    n_shared = w_gate.shape[0]
    h = x.reshape(1, -1, shape[-1]).expand(n_shared, -1, -1)
    g = gammas.reshape(-1, n_shared, 1).transpose(0, 1)
    #2. batched gate/up GEMMs (transpose(1, 2) is a stride view, :399-400);
    #   gamma scales the activation PRE-down (:401)
    gate = torch.bmm(h, w_gate.transpose(1, 2))
    up = torch.bmm(h, w_up.transpose(1, 2))
    act = torch.nn.functional.silu(gate) * up * g
    #3. down GEMM, fp32 sum over the shared experts, downcast (:402-405)
    down = torch.bmm(act, w_down.transpose(1, 2))
    return down.float().sum(dim=0).to(x.dtype).view(shape)


def moe(x, w_gate, bias, global_scale, w_gate_up, w_down, w_sh_gate,
        w_sh_up, w_sh_down, top_k, n_shared, route_scale):
    """Full MoE block: router -> routed experts + shared sink experts —
    exact InklingMoE.forward semantics (modeling_inkling.py:418-425):
    gate on the unflattened input, routed experts on the flattened tokens,
    shared experts on the ORIGINAL input, routed + shared added in that
    order (bf16)."""
    #1. route (B2.8), flatten tokens for the expert loop (:421-422)
    _, topk_weights, topk_indices, shared_gammas = moe_gate(
        x, w_gate, bias, global_scale, top_k, n_shared, route_scale)
    flat = x.view(-1, x.shape[-1])
    #2. routed experts on flat tokens, back to the input shape (:423)
    routed = moe_experts(flat, w_gate_up, w_down, topk_indices,
                         topk_weights).view(x.shape)
    #3. + shared experts on the pre-flatten residuals (:424)
    return routed + moe_shared(x, w_sh_gate, w_sh_up, w_sh_down,
                               shared_gammas)


def dense_mlp(x, w_gate, w_up, w_down, global_scale):
    """Dense-layer MLP (layers 0-1; dense_mlp_idx=2 is a COUNT of leading
    dense layers, census B1.2) — exact InklingMLP semantics (transformers
    models/inkling/modeling_inkling.py:285-299):
    down_proj(silu(gate_proj(x)) * up_proj(x)) * global_scale. The native
    module holds three SEPARATE bias-free linears (:291-293) — kept
    separate here, not one fused GEMM, so cuBLAS sees the module's exact
    shapes; global_scale is a LIVE bf16 scalar parameter (:295, :299;
    checkpoint values ~0.008 / 0.026 on layers 0 / 1, not 1.0) and is the
    module's LAST op. The whole module runs in the model dtype, bf16 (NOT
    in _keep_in_fp32_modules_strict :610 — only the sconvs are fp32).
    ffn = 24576: the raw config's dense_intermediate_size lands on
    config.intermediate_size (configuration_inkling.py:125-126; the raw
    intermediate_size 3072 is the EXPERT ffn). The checkpoint stores
    gate/up fused as w13_dn [2*ffn, hidden] with interleaved
    [g0,u0,g1,u1,..] rows; transformers converts via Interleave(dim=0) +
    Chunk(dim=0) -> gate_proj, up_proj (conversion_mapping.py
    "inkling_mm_model") — callers pass the two de-interleaved halves.
    Fused Triton form is Stage-3 work (D4)."""
    F = torch.nn.functional
    #1. gate/up/down chain with silu, no biases anywhere (:298)
    h = F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)
    #2. trailing live global_scale multiply, the module's last op (:299)
    return h * global_scale


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
