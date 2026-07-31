"""B2: the 66-layer graph. Order per layer: rmsnorm -> attention (global or
SWA by layer id; rel-bias + log scaling pre-softmax; sconv on K,V,attn-out)
-> residual -> rmsnorm -> MoE (sigmoid gate+bias, top-6, norm_after_topk,
route_scale 8, +2 shared sink experts; layers 0-1 = dense MLP) -> sconv on
moe-out -> residual. No RoPE anywhere."""
import os

import torch

from engine.pyengine.kernels.attn_global import attn_global
from engine.pyengine.kernels.attn_prefill import attn_prefill_fused
from engine.pyengine.kernels.attn_swa import attn_swa
from engine.pyengine.kernels.moe_gemm import moe_experts_packed

# exp-0013 kill switch: PYENGINE_FUSED_ATTN=0 restores the eager pooled
# attention chain unchanged (read once at import — must be constant for
# the life of the process so captured graphs match the eager warmups)
_FUSED_ATTN = os.environ.get("PYENGINE_FUSED_ATTN", "1") != "0"

# exp-0015b kill switch: PYENGINE_W4A4_DECODE=0 sends the decode paths back
# to the W4A16 Triton grouped GEMM (prefill stays W4A4 — that is exp-0015a's
# accepted variable). Read once at import for the same reason as above.
_W4A4_DECODE = os.environ.get("PYENGINE_W4A4_DECODE", "1") != "0"

# exp-0019 kill switch: PYENGINE_BF16_GROUPED=0 sends the ONE un-exported MoE
# layer (dense_mlp_idx = 2, bf16 routed experts) back to the reference
# per-hit-expert loop — and with it restores graphrun's eager island, since
# experts_capturable() below is the single predicate both sites read. Read
# once at import for the same reason as above: it must be constant for the
# life of the process so captured graphs match the eager warmups.
_BF16_GROUPED = os.environ.get("PYENGINE_BF16_GROUPED", "1") != "0"

# exp-0023 kill switch: PYENGINE_FUSED_PREFILL_ATTN=0 restores attn_prefill's
# eager materialized-score chain unchanged. Read once at import for the same
# reason as above (and prefill is not captured, but the two paths must not
# alternate within a process: their reduction orders differ, so mixing them
# would break same-schedule bitwise determinism).
_FUSED_PREFILL_ATTN = os.environ.get("PYENGINE_FUSED_PREFILL_ATTN",
                                     "1") != "0"

# exp-0025 kill switch: PYENGINE_FUSED_SCONV=0 restores sconv_prefill's
# eager fp32 conv1d chain unchanged (and with it the channel-major output
# layout it leaks into the residual stream). Read once at import for the
# same reason as above; the two paths must not alternate within a process
# (they are bitwise equal today — t_fused_sconv arm a — but the layout of
# what they hand downstream is not, and downstream kernels tile by it).
_FUSED_SCONV = os.environ.get("PYENGINE_FUSED_SCONV", "1") != "0"


def experts_capturable(w_gate_up):
    """True when this MoE layer's routed-expert GEMMs are legal inside
    CUDA-graph capture — i.e. grouped, sync-free and fixed-shape at fixed T.
    The NVFP4 layers 3-65 always are (loader.PackedExperts -> moe_gemm /
    moe_gemm_w4a4); layer 2's plain bf16 stacks are iff exp-0019's grouped
    bf16 twin is on, because the reference loop it otherwise takes does a
    per-traversal unique().tolist() device->host sync. graphrun reads this
    to decide which layers must be carved out as eager islands, so the two
    sites can never disagree about what got captured."""
    if hasattr(w_gate_up, "packed"):
        return True
    #1. the grouped bf16 twin is specialized to 128-multiple shapes (the
    #   checkpoint's hidden 6144 / ffn 3072 — its k-loop and n-tiles are
    #   unmasked); the toy stacks the bring-up tests build keep the
    #   reference loop, as does any future non-conforming layer
    return (_BF16_GROUPED and w_gate_up.dim() == 3
            and w_gate_up.dtype == torch.bfloat16
            and w_gate_up.shape[2] % 128 == 0
            and (w_gate_up.shape[1] // 2) % 128 == 0)


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
    kernel is Stage-3 work (D4).

    exp-0025: with PYENGINE_FUSED_SCONV=1 (default) and a 3-D CUDA input,
    #1-#3 run as ONE Triton kernel (kernels/sconv_prefill.py) that reads
    the bf16 input once, does the 4-tap dot and the own-input residual in
    fp32 registers, and stores one bf16 round into a CONTIGUOUS output.
    Same tap order, same fp32 arithmetic, so it is BITWISE equal to the
    chain below (t_fused_sconv arm a) — what changes is HBM traffic
    (~36 bytes/element -> 4) and the output LAYOUT: the eager form's
    `conv.transpose(1, 2) + xf` inherits the transposed operand's
    channel-major strides and hands them to every downstream consumer."""
    if _FUSED_SCONV and x.is_cuda and x.dim() == 3 and weight.dim() == 3 \
            and weight.shape[1] == 1 and weight.shape[0] == x.shape[2]:
        from engine.pyengine.kernels.sconv_prefill import sconv_prefill_fused
        return sconv_prefill_fused(x, weight)
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


def moe_experts(x, w_gate_up, w_down, topk_indices, topk_weights,
                phase="decode"):
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
    #0. NVFP4 layers (loader.PackedExperts): grouped GEMM directly on the
    #   packed bytes — one launch per matrix for ALL (token, slot) pairs,
    #   no per-expert bf16 materialization, no per-layer CPU sync
    #   (exp-0003; D13-envelope semantics, kernels/moe_gemm.py docstring)
    if hasattr(w_gate_up, "packed"):
        #0'. exp-0015a: PREFILL routes to the native-FP4-MMA W4A4 grouped
        #    GEMM (the reference's own numeric recipe for these layers —
        #    kernels/moe_gemm_w4a4.py docstring). exp-0015b: DECODE routes
        #    there too, including the CUDA-graph-captured batched step —
        #    the same kernel, same recipe, no phase distinction left. The
        #    graphed step's routed-expert GEMMs were 266.4 of its 322.7 ms
        #    (exp-0012 kernel table) because the W4A16 Triton kernel pays
        #    the exp-0010 convert-ALU floor per weight element; native FP4
        #    MMA deletes that pipe and leaves the decode MoE weight-traffic
        #    bound. Capture-safety, replay-vs-eager bitwise equality and
        #    same-schedule determinism are probed in tests/t_w4a4_capture.
        #    Kill switch PYENGINE_W4A4_DECODE=0 restores the W4A16 decode.
        #    Lazy import: the CUTLASS DSL stack loads on first use.
        if w_gate_up.input_amax is not None and (
                phase == "prefill" or _W4A4_DECODE):
            from engine.pyengine.kernels.moe_gemm_w4a4 import (
                moe_experts_w4a4)
            return moe_experts_w4a4(x, w_gate_up, w_down, topk_indices,
                                    topk_weights)
        return moe_experts_packed(x, w_gate_up, w_down, topk_indices,
                                  topk_weights)
    #0''. exp-0019: the ONE MoE layer the NVFP4 export skipped (layer
    #    dense_mlp_idx = 2, bf16 routed experts) takes the same grouped
    #    structure on plain bf16 stacks. Below this line lives the reference
    #    loop, whose torch.unique(...).tolist() is a device->host sync per
    #    traversal and whose ~2 x 200 launches serve 1-2 rows each; that sync
    #    is the reason exp-0012 could not capture layer 2 and carved it out
    #    as graphrun's eager island. The grouped twin is sync-free and
    #    fixed-shape, so the island disappears (see experts_capturable) and
    #    device 0's segment captures as ONE graph.
    #    Kill switch PYENGINE_BF16_GROUPED=0 restores this loop + the island.
    if experts_capturable(w_gate_up):
        from engine.pyengine.kernels.moe_gemm_bf16 import moe_experts_bf16
        return moe_experts_bf16(x, w_gate_up, w_down, topk_indices,
                                topk_weights)
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
        w_sh_up, w_sh_down, top_k, n_shared, route_scale, phase="decode"):
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
                         topk_weights, phase=phase).view(x.shape)
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
                 head_dim, is_global, alpha=None, n_floor=None, eps=1e-6,
                 state=None, states=None, lens=None, window=None):
    """One attention layer, PREFILL (no-cache) form — exact InklingAttention
    semantics (transformers models/inkling/modeling_inkling.py:217-282 with
    eager_attention_forward :157-182). Both layer shapes run through here:
    global (64Q/8KV heads, rel extent 1024, log scaling) and SWA (64Q/16KV,
    extent = window 512, no scaling) differ only in weights, mask and
    `is_global`. x [B, T, C] is the POST-input_layernorm hidden state;
    attn_mask is the additive eager mask (additive_causal_mask). Scaling is
    1/head_dim, not 1/sqrt, because q/k are per-head RMS-normalized
    (:197-198). No biases anywhere. `state` (a kv.LayerKV, batch-1 only)
    makes prefill POPULATE the layer's recurrent state — post-norm K +
    post-sconv V into the cache, raw pre-conv k/v tails into the sconv
    rings — with numerics untouched (same ops either way; B3.1).
    attn_decode is the cached single-token form.

    exp-0023: with PYENGINE_FUSED_PREFILL_ATTN=1 (default) and a mask
    present, #3-#8 run as ONE Triton kernel (kernels/attn_prefill.py) that
    never materializes a [B,H,T,T] tensor. It reproduces the mask from the
    POSITIONS — causal, plus `window` on SWA layers — so the caller's mask
    must be additive_causal_mask(q_positions, kv_positions, dtype, window),
    which is what layer_prefill already requires of it and what both
    in-tree builders construct; `window` comes down from mc.window (a
    caller that omits it falls back to the rel extent, which IS the window
    on SWA layers per rel_bias's docstring, and the kernel asserts the two
    agree). Numerics move by reduction order only — fp32 flash accumulation
    instead of a bf16-rounded score tensor, the B3.1/B3.3 class exp-0013
    established on the decode side, gated by the D13 envelope.

    `states` + `lens` (exp-0009 batched group prefill) is the
    multi-row form of `state`: rows are RIGHT-padded to a common T, so with
    the causal mask a real position never attends a pad (pads sit at future
    key positions) and row i's live values are exactly the [:lens[i]]
    slices — each row seeds its own LayerKV through the same per-seq
    seed/append calls."""
    F = torch.nn.functional
    B, T = x.shape[0], x.shape[1]
    n_heads, n_kv = wq.shape[0] // head_dim, wk.shape[0] // head_dim
    #1. projections (:228-231); K and V pass their window-4 sconv first
    q = F.linear(x, wq)
    k_raw = F.linear(x, wk)
    v_raw = F.linear(x, wv)
    k = sconv_prefill(k_raw, w_k_sconv)
    v = sconv_prefill(v_raw, w_v_sconv)
    r = F.linear(x, wr)
    #2. per-head q/k rmsnorm over head_dim, then [B,T,H,D] -> [B,H,T,D]
    #   (:233-235; v is not normalized)
    q = rmsnorm(q.view(B, T, n_heads, head_dim), w_q_norm, eps).transpose(1, 2)
    kn = rmsnorm(k.view(B, T, n_kv, head_dim), w_k_norm, eps)
    vv = v.view(B, T, n_kv, head_dim)
    k = kn.transpose(1, 2)
    v = vv.transpose(1, 2)
    #2b. populate the recurrent state: cache holds per-token values that
    #    never change once written (post-norm K, post-sconv V); the rings
    #    hold the last sconv_k-1 RAW inputs (B3.1)
    if state is not None:
        assert B == 1, "cache population is batch-1 in bring-up scope"
        state.k_ring.seed(k_raw[0])
        state.v_ring.seed(v_raw[0])
        state.cache.append(kn[0], vv[0])
    elif states is not None:
        for i, st in enumerate(states):
            st.k_ring.seed(k_raw[i, : lens[i]])
            st.v_ring.seed(v_raw[i, : lens[i]])
            st.cache.append(kn[i, : lens[i]], vv[i, : lens[i]])
    #3f. exp-0023 fused form: the rel-bias MIX (rel_bias #1) is all the
    #    kernel needs — it gathers by distance and applies the band, the
    #    causal/window mask, the softmax and the PV product itself, so
    #    nothing [B,H,T,T]-shaped is ever materialized and repeat_kv's
    #    copies disappear. tau rides on the mix instead of the gathered
    #    bias: the gather is a pure copy, so scaling before it is the same
    #    per-element fp32 multiply + bf16 round on every live pair, and 0
    #    stays 0 on the dead ones (apply_log_scaling's shapes are generic
    #    over the last two axes, [B,H,T,E] here instead of [B,H,T,K]).
    if _FUSED_PREFILL_ATTN and attn_mask is not None:
        rel = (r.view(B, T, n_heads, -1) @ rel_proj).transpose(1, 2)
        if is_global and n_floor is not None:
            q, rel = apply_log_scaling(q, rel, q_positions, alpha, n_floor)
        out = attn_prefill_fused(
            q, k, v, rel, q_positions, kv_positions,
            None if is_global else (window or rel_proj.shape[1]))
        return F.linear(out.reshape(B, T, n_heads * head_dim), wo)
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


def layer_prefill(x, w, mask, pos, mc, layer_idx, state=None, trace=None,
                  states=None, lens=None):
    """One full decoder layer, PREFILL form — exact InklingDecoderLayer
    op order (transformers models/inkling/modeling_inkling.py:566-591):
    input_layernorm -> attention (B2.6 global / B2.7 SWA) -> attn_sconv
    (module adds its OWN input back in fp32, B2.5; outer residual is
    separate) -> + residual (bf16) -> post_attention_layernorm -> MLP
    (dense for layer_idx < mc.dense_idx — a COUNT, census B1.2 — else MoE)
    -> mlp_sconv -> + residual. `w` maps fixed keys to converted-to-module-
    layout tensors (conversions per B2.5-B2.10; built by
    t_b2.load_layer_weights): attention wq/wk/wv/wr/wo/ksc/vsc/qn/kn/proj;
    norms attn_norm/mlp_norm (checkpoint attn_norm/mlp_norm ->
    input/post_attention_layernorm, conversion_mapping.py); layer sconvs
    attn_sconv/mlp_sconv; and either mlp_gate/mlp_up/mlp_down/mlp_gs
    (dense) or gate_w/gate_b/gate_gs/gu/w2/shg/shu/shd (MoE). mc is the
    verified ModelConfig; the caller's mask must match the layer type
    (window=mc.window on SWA layers). `state` (kv.LayerKV, batch-1)
    additionally populates the layer's KV cache + all four sconv rings —
    same numerics either way (B3.1). `states` + `lens` (exp-0009) is the
    multi-row form: right-padded rows, row i seeding states[i] from its
    [:lens[i]] slices (attn_prefill docstring — causality isolates real
    rows from pads). `trace` (a dict, tests only) records
    the attention-half residual stream ('x1') and the MLP input stream
    ('mlpin'), mirroring layer_decode's hook (B3.2) — references only,
    numerics untouched."""
    is_global = layer_idx in mc.global_layers
    #1. attention half: pre-norm, attention, attn_sconv, outer residual
    #   in the ref's order residual + hidden (:574-583)
    h = rmsnorm(x, w["attn_norm"], mc.rms_eps)
    h = attn_prefill(h, w["wq"], w["wk"], w["wv"], w["wr"], w["wo"],
                     w["ksc"], w["vsc"], w["qn"], w["kn"], w["proj"],
                     mask, pos, pos, mc.head_dim, is_global,
                     alpha=mc.log_alpha, n_floor=mc.log_floor,
                     eps=mc.rms_eps, state=state, states=states, lens=lens,
                     window=mc.window)
    if state is not None:
        state.attn_ring.seed(h[0])
    elif states is not None:
        for i, st in enumerate(states):
            st.attn_ring.seed(h[i, : lens[i]])
    h = sconv_prefill(h, w["attn_sconv"])
    x = x + h
    #2. MLP half: pre-norm, dense MLP or MoE block, mlp_sconv, residual
    #   (:585-591)
    h = rmsnorm(x, w["mlp_norm"], mc.rms_eps)
    if trace is not None:
        trace["x1"] = x
        trace["mlpin"] = h
    if layer_idx < mc.dense_idx:
        h = dense_mlp(h, w["mlp_gate"], w["mlp_up"], w["mlp_down"],
                      w["mlp_gs"])
    else:
        h = moe(h, w["gate_w"], w["gate_b"], w["gate_gs"], w["gu"],
                w["w2"], w["shg"], w["shu"], w["shd"], mc.topk,
                mc.n_shared, mc.route_scale, phase="prefill")
    if state is not None:
        state.mlp_ring.seed(h[0])
    elif states is not None:
        for i, st in enumerate(states):
            st.mlp_ring.seed(h[i, : lens[i]])
    h = sconv_prefill(h, w["mlp_sconv"])
    return x + h


def sconv_decode(x, weight, ring):
    """Window-4 sconv, DECODE form — one new token x [1, C] (the site's RAW
    input), history in `ring` (kv.SconvRing: the previous kernel-1 raw
    inputs, zeros before t=0). Same math as sconv_prefill at this position
    (causal_conv1d_update semantics, modeling_inkling.py:441-457: roll
    state, weight[..., -1] taps the current token): fp32 depthwise dot over
    the exact 4-token window via the SAME F.conv1d op (padding=0 emits
    exactly the one full-window output), own-input residual in fp32,
    downcast to the input dtype."""
    #1. full conv window (oldest first) from the ring; fp32 module math
    win = ring.step(x)
    wf = weight.float()
    #2. depthwise conv, one output per channel: [1, C, k] -> [1, C, 1]
    conv = torch.nn.functional.conv1d(
        win.float().t()[None], wf, bias=None, padding=0, groups=wf.shape[0])
    #3. own-input residual in fp32, then downcast to the input dtype
    return (conv[:, :, 0] + x.float()).to(x.dtype)


def attn_decode(x, wq, wk, wv, wr, wo, w_k_sconv, w_v_sconv, w_q_norm,
                w_k_norm, rel_proj, state, pos, head_dim, is_global,
                alpha=None, n_floor=None, eps=1e-6):
    """One attention layer, DECODE form — a single new token at absolute
    position `pos` (int), K/V history from state.cache (kv.LayerKV). Same
    ops as attn_prefill with Q=1: project, k/v raw through their sconv
    rings (sconv_decode), per-head q/k rmsnorm, append post-norm K +
    post-sconv V, gather the visible key set, rel bias at backward
    distances pos - kv_pos, tau round-trip on global layers, GQA repeat,
    bf16 scores * 1/head_dim + bias, fp32 softmax, @V, o_proj. NO additive
    mask: every gathered key is visible by construction — the cache only
    holds positions <= pos (causal), and the SWA ring holds exactly the
    window-512 allowed set (kv.RingKV); beyond-extent global keys get bias
    0 from rel_bias, same as prefill where the mask never hides them."""
    F = torch.nn.functional
    n_heads, n_kv = wq.shape[0] // head_dim, wk.shape[0] // head_dim
    #1. projections for the new token; k/v raw through their sconv rings
    q = F.linear(x, wq)
    k = sconv_decode(F.linear(x, wk), w_k_sconv, state.k_ring)
    v = sconv_decode(F.linear(x, wv), w_v_sconv, state.v_ring)
    r = F.linear(x, wr)
    #2. per-head q/k rmsnorm; append this token, then gather the key set
    q = rmsnorm(q.view(1, n_heads, head_dim), w_q_norm, eps)
    k = rmsnorm(k.view(1, n_kv, head_dim), w_k_norm, eps)
    state.cache.append(k, v.view(1, n_kv, head_dim))
    ck, cv, kv_pos = state.cache.gather()
    #3. rel bias for the (1-query, N-key) pair grid
    qpos = torch.tensor([pos], device=x.device)
    bias = rel_bias(r.view(1, 1, n_heads, -1), rel_proj, qpos, kv_pos)
    #4. [B=1, H, Q=1, D] query; tau round-trip on global layers (==1.0
    #   below the 128000 floor, applied unconditionally like the ref)
    qh = q.view(1, 1, n_heads, head_dim).transpose(1, 2)
    if is_global and n_floor is not None:
        qh, bias = apply_log_scaling(qh, bias, qpos, alpha, n_floor)
    #5. GQA repeat over the gathered keys: [N,n_kv,D] -> [1,n_heads,N,D]
    N = ck.shape[0]
    rep = n_heads // n_kv
    kh = ck.transpose(0, 1)[None, :, None].expand(
        1, n_kv, rep, N, head_dim).reshape(1, n_heads, N, head_dim)
    vh = cv.transpose(0, 1)[None, :, None].expand(
        1, n_kv, rep, N, head_dim).reshape(1, n_heads, N, head_dim)
    #6. bf16 scores * 1/head_dim + bias (same association order as prefill)
    attn = torch.matmul(qh, kh.transpose(2, 3)) * (1.0 / head_dim) + bias
    #7. fp32 softmax downcast, weighted values, output projection
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(qh.dtype)
    out = torch.matmul(attn, vh).transpose(1, 2).reshape(
        1, n_heads * head_dim)
    return F.linear(out, wo)


def layer_decode(x, w, state, pos, mc, layer_idx, trace=None):
    """One full decoder layer, DECODE form — x [1, hidden] is the new
    token's layer input at absolute position `pos`; `state` the layer's
    kv.LayerKV (populated by layer_prefill / earlier decode steps). Exact
    layer_prefill op order with the cached single-token attention and
    ring-state sconvs; the MLP half is position-independent so dense_mlp /
    moe are reused unchanged at T=1 (B2.9 note). `trace` (a dict, tests
    only) records the attention-half residual ('x1') and the MLP input
    ('mlpin') so t_b3 can gate the KV-fed attention half separately from
    the MoE half's bf16 routing-weight granularity."""
    is_global = layer_idx in mc.global_layers
    #1. attention half: pre-norm, cached attention, attn_sconv, residual
    h = rmsnorm(x, w["attn_norm"], mc.rms_eps)
    h = attn_decode(h, w["wq"], w["wk"], w["wv"], w["wr"], w["wo"],
                    w["ksc"], w["vsc"], w["qn"], w["kn"], w["proj"],
                    state, pos, mc.head_dim, is_global, alpha=mc.log_alpha,
                    n_floor=mc.log_floor, eps=mc.rms_eps)
    h = sconv_decode(h, w["attn_sconv"], state.attn_ring)
    x = x + h
    #2. MLP half: pre-norm, dense MLP or MoE block, mlp_sconv, residual
    h = rmsnorm(x, w["mlp_norm"], mc.rms_eps)
    if trace is not None:
        trace["x1"] = x
        trace["mlpin"] = h
    if layer_idx < mc.dense_idx:
        h = dense_mlp(h, w["mlp_gate"], w["mlp_up"], w["mlp_down"],
                      w["mlp_gs"])
    else:
        h = moe(h, w["gate_w"], w["gate_b"], w["gate_gs"], w["gu"],
                w["w2"], w["shg"], w["shu"], w["shd"], mc.topk,
                mc.n_shared, mc.route_scale)
    h = sconv_decode(h, w["mlp_sconv"], state.mlp_ring)
    return x + h


def sconv_decode_batch(x, weight, rings):
    """Window-4 sconv, BATCHED decode form — x [B, C] carries the same
    sconv site's raw input for B independent co-resident sequences;
    rings[b] is row b's kv.SconvRing. Per-row math is sconv_decode's
    exactly (fp32 module, own-input residual, weight[..., -1] taps the
    current token) but the depthwise conv over all B windows is ONE
    F.conv1d call instead of B. Ring rolls stay per-row (each ring is
    per-sequence state). Row order is the caller's batch order; bf16
    row-count invariance is NOT claimed (B3.1/B3.3) — the binding gates
    are the D13 envelope and same-schedule determinism (scheduler.py)."""
    #1. roll each row's ring; stack the full windows [B, k, C], oldest first
    win = torch.stack([rings[b].step(x[b:b + 1])
                       for b in range(x.shape[0])])
    wf = weight.float()
    #2. one depthwise conv over the batch: [B, C, k] -> [B, C, 1]
    conv = torch.nn.functional.conv1d(
        win.float().permute(0, 2, 1), wf, bias=None, padding=0,
        groups=wf.shape[0])
    #3. own-input residual in fp32, then downcast (sconv_decode #3)
    return (conv[:, :, 0] + x.float()).to(x.dtype)


def sconv_decode_batch_pooled(x, weight, tails, slots):
    """Window-4 sconv, POOLED batched decode form (exp-0008) — x [B, C]
    as sconv_decode_batch; tails [S, kernel-1, C] is the site's slot pool
    (kv.SconvPool), slots [B] the co-resident rows' slot ids. BITWISE
    equal to sconv_decode_batch on the same state: the [B, kernel, C]
    window holds the same values whether assembled by ONE gather + ONE
    cat (here) or B per-row cat/stack chains (there) — assembly is
    copying — and the depthwise conv + fp32 residual are the same ops on
    the same shapes in the same order. The rolled tails written back by
    ONE index_put_ are the same win[:, 1:] rows the per-ring rebind kept,
    through the storage the per-seq ring views alias (kv.SconvRing
    backing), so per-seq prefill/decode interleave stays coherent. slots
    are distinct by construction (one scheduler slot per sequence) and
    every index is a pure function of the schedule — same-schedule
    bitwise determinism holds."""
    #1. all B windows in one gather+cat; roll all tails in one write
    win = torch.cat([tails[slots], x[:, None, :]], dim=1)  # [B, k, C]
    tails[slots] = win[:, 1:]
    wf = weight.float()
    #2. one depthwise conv over the batch (sconv_decode_batch #2)
    conv = torch.nn.functional.conv1d(
        win.float().permute(0, 2, 1), wf, bias=None, padding=0,
        groups=wf.shape[0])
    #3. own-input residual in fp32, then downcast (sconv_decode #3)
    return (conv[:, :, 0] + x.float()).to(x.dtype)


def attn_decode_batch(x, wq, wk, wv, wr, wo, w_k_sconv, w_v_sconv, w_q_norm,
                      w_k_norm, rel_proj, states, pos, head_dim, is_global,
                      alpha=None, n_floor=None, eps=1e-6):
    """One attention layer, BATCHED decode form — x [B, hidden] holds B
    co-resident sequences' current tokens; states[b]/pos[b] are sequence
    b's kv.LayerKV and absolute position. Weight-shared work (q/k/v/r
    projections, k/v sconvs, per-head q/k rmsnorm, o_proj) runs as B-row
    batches so each weight tensor is read ONCE per layer per step; the
    KV-fed core (cache append/gather, rel bias, scores, softmax, @V) stays
    per-sequence — co-residents sit at different cache lengths. Per-row op
    order inside the core is attn_decode's exactly."""
    F = torch.nn.functional
    B = x.shape[0]
    n_heads, n_kv = wq.shape[0] // head_dim, wk.shape[0] // head_dim
    #1. shared-weight projections; k/v raw through their per-row rings
    q = F.linear(x, wq)
    k = sconv_decode_batch(F.linear(x, wk), w_k_sconv,
                           [s.k_ring for s in states])
    v = sconv_decode_batch(F.linear(x, wv), w_v_sconv,
                           [s.v_ring for s in states])
    r = F.linear(x, wr)
    #2. per-head q/k rmsnorm across the whole batch (rowwise op)
    q = rmsnorm(q.view(B, n_heads, head_dim), w_q_norm, eps)
    k = rmsnorm(k.view(B, n_kv, head_dim), w_k_norm, eps)
    v = v.view(B, n_kv, head_dim)
    #3. per-sequence attention core, batch order (= admission order)
    outs = []
    for b in range(B):
        st = states[b]
        st.cache.append(k[b:b + 1], v[b:b + 1])
        ck, cv, kv_pos = st.cache.gather()
        qpos = torch.tensor([pos[b]], device=x.device)
        bias = rel_bias(r[b].view(1, 1, n_heads, -1), rel_proj, qpos,
                        kv_pos)
        qh = q[b].view(1, 1, n_heads, head_dim).transpose(1, 2)
        if is_global and n_floor is not None:
            qh, bias = apply_log_scaling(qh, bias, qpos, alpha, n_floor)
        N = ck.shape[0]
        rep = n_heads // n_kv
        kh = ck.transpose(0, 1)[None, :, None].expand(
            1, n_kv, rep, N, head_dim).reshape(1, n_heads, N, head_dim)
        vh = cv.transpose(0, 1)[None, :, None].expand(
            1, n_kv, rep, N, head_dim).reshape(1, n_heads, N, head_dim)
        attn = torch.matmul(qh, kh.transpose(2, 3)) * (1.0 / head_dim) + bias
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(qh.dtype)
        outs.append(torch.matmul(attn, vh).transpose(1, 2).reshape(
            1, n_heads * head_dim))
    #4. one shared o_proj GEMM over the stacked rows
    return F.linear(torch.cat(outs, dim=0), wo)


def decode_batch_ctx(pos, slots, mc, device, page_size):
    """exp-0007: per-STEP, per-DEVICE indexing for the pooled batched
    decode attention core (attn_decode_batch_pooled) — built once and
    shared by every layer of a type on that device, so the per-layer
    enqueue cost is ~20 batched ops instead of ~20 x B per-row chains
    (exp-0005: attn_core = 1646 ms of the 1989 ms step at B=64, host-
    enqueue-bound). pos/slots are host int lists in batch-row order
    (= admission order); everything here is a pure function of the
    schedule, so same-schedule bitwise determinism holds."""
    B = len(pos)
    pos_t = torch.tensor(pos, device=device)
    n = pos_t + 1                        # post-append cache lengths
    #1. SWA ring geometry (kv.RingKV slot rule: position p sits at
    #   p % window): slot s holds p = base + ((s - base) mod window),
    #   base = max(0, n - window); live iff p < n. dist = q_pos - p is in
    #   [0, window) exactly on live slots, negative on dead ones.
    s = torch.arange(mc.window, device=device)[None, :]
    base = (n - mc.window).clamp_min(0)[:, None]
    p = base + ((s - base) % mc.window)
    #2. global padded-gather geometry: logical positions 0..Lg-1 with
    #   rows shorter than Lg masked; per-layer flat slots come from each
    #   layer's device page table (attn_decode_batch_pooled #3)
    Lg = max(pos) + 1
    j = torch.arange(Lg, device=device)
    return {"B": B, "pos": pos_t,
            "slots": torch.tensor(slots, device=device), "pos_host": pos,
            "ring_slot": pos_t % mc.window, "swa_valid": p < n[:, None],
            "swa_dist": pos_t[:, None] - p, "Lg": Lg,
            "g_pageidx": j // page_size, "g_off": j % page_size,
            "g_valid": j[None, :] < n[:, None],
            "g_dist": pos_t[:, None] - j[None, :]}


def decode_batch_ctx_static(pos_t, slots_t, Lg, mc, page_size):
    """exp-0012: decode_batch_ctx with STATIC shapes for CUDA-graph
    capture — pos/slots arrive as device int64 tensors (the graph
    runner's static input buffers, so replay sees fresh content at fixed
    addresses) and Lg is the padded key length (a monotone 512-bucket
    high-water mark, graphrun.GraphRunner), not max(pos)+1. Every entry
    is the same arithmetic as decode_batch_ctx on the same values —
    rows beyond a live position are dead by the SAME predicates (p < n /
    j < n) the dynamic ctx uses for short rows, so the padded columns
    are masked exactly like any other dead slot. Pure tensor ops on
    device data: no host reads, no torch.tensor(host_list) — legal
    inside stream capture. No "pos_host": the graph path hoists page
    allocation out of the layer walk (attn_decode_batch_pooled #3a)."""
    B = pos_t.shape[0]
    device = pos_t.device
    n = pos_t + 1
    #1. SWA ring geometry (decode_batch_ctx #1, unchanged math)
    s = torch.arange(mc.window, device=device)[None, :]
    base = (n - mc.window).clamp_min(0)[:, None]
    p = base + ((s - base) % mc.window)
    #2. global padded-gather geometry at the FIXED bucket length
    j = torch.arange(Lg, device=device)
    return {"B": B, "pos": pos_t, "slots": slots_t,
            "ring_slot": pos_t % mc.window, "swa_valid": p < n[:, None],
            "swa_dist": pos_t[:, None] - p, "Lg": Lg,
            "g_pageidx": j // page_size, "g_off": j % page_size,
            "g_valid": j[None, :] < n[:, None],
            "g_dist": pos_t[:, None] - j[None, :]}


def attn_decode_batch_pooled(x, wq, wk, wv, wr, wo, w_k_sconv, w_v_sconv,
                             w_q_norm, w_k_norm, rel_proj, states, pool,
                             ctx, head_dim, is_global, alpha=None,
                             n_floor=None, eps=1e-6, fused=None):
    """One attention layer, POOLED batched decode form (exp-0007) — the
    per-row Python core of attn_decode_batch collapsed into per-layer
    batched ops over the layer's batch pool (kv.SwaPool / kv.GlobalPool).
    Per-row op ORDER is attn_decode's exactly (project -> k/v sconv ->
    q/k rmsnorm -> append -> rel bias -> tau -> GQA scores -> fp32
    softmax -> @V -> o_proj); what changes is HOW rows are traversed:
    one broadcast matmul over [B, n_kv, rep, 1, L] instead of B loop
    iterations, dead key slots excluded by SELECTION (masked_fill with
    the eager mask constant — a stale pool row can hold any finite
    stale value, so overwrite, never add) exactly where the per-seq
    gather simply omitted them. bf16 kernel tiling differs from the
    per-row form, so per-token bits may drift within the D13 envelope
    (B3.1/B3.3 semantics); shapes and index tensors are pure functions
    of the schedule, so same-schedule bitwise determinism holds."""
    F = torch.nn.functional
    B = ctx["B"]
    n_heads, n_kv = wq.shape[0] // head_dim, wk.shape[0] // head_dim
    #1. shared-weight projections; k/v raw through the pooled sconv step
    #   (exp-0008, attack #3: one gather+cat+conv per site instead of B
    #   per-row ring chains — bitwise equal, sconv_decode_batch_pooled)
    q = F.linear(x, wq)
    k = sconv_decode_batch_pooled(F.linear(x, wk), w_k_sconv,
                                  pool.sconv.kt, ctx["slots"])
    v = sconv_decode_batch_pooled(F.linear(x, wv), w_v_sconv,
                                  pool.sconv.vt, ctx["slots"])
    r = F.linear(x, wr)
    #2. per-head q/k rmsnorm across the whole batch (rowwise op)
    q = rmsnorm(q.view(B, n_heads, head_dim), w_q_norm, eps)
    k = rmsnorm(k.view(B, n_kv, head_dim), w_k_norm, eps)
    v = v.view(B, n_kv, head_dim)
    slots = ctx["slots"]
    #3. batched append + batched key-set fetch from the layer pool
    if is_global:
        #3a. page allocation is host-side (page tables + free list are
        #    host state; ensure_pages mirrors new ids on-device), then
        #    ONE index_put_ appends all rows, ONE gather fetches the
        #    padded [B, Lg] key set — no .tolist() anywhere. The CUDA-
        #    graph path (exp-0012) passes states=None: it hoists page
        #    allocation to before graph replay (graphrun.step #1), so
        #    the captured region stays free of host-side work
        ps = pool.page_size
        if states is not None:
            for b, st in enumerate(states):
                st.cache.ensure_pages(ctx["pos_host"][b] // ps)
        table = pool.table_dev[slots]                      # [B, num_pages]
        flat_new = (table.gather(1, (ctx["pos"] // ps)[:, None])[:, 0] * ps
                    + ctx["pos"] % ps)
        pool.kp.view(-1, n_kv, head_dim)[flat_new] = k
        pool.vp.view(-1, n_kv, head_dim)[flat_new] = v
        valid, dist, L = ctx["g_valid"], ctx["g_dist"], ctx["Lg"]
        flat = table[:, ctx["g_pageidx"]] * ps + ctx["g_off"][None, :]
        flat = torch.where(valid, flat, torch.zeros_like(flat))
    else:
        pool.k[slots, ctx["ring_slot"]] = k
        pool.v[slots, ctx["ring_slot"]] = v
        valid, dist = ctx["swa_valid"], ctx["swa_dist"]
        L = pool.k.shape[1]
    if fused is None:
        fused = _FUSED_ATTN
    if fused:
        #4f. fused two-shape kernel path (exp-0013): the rel-logit mix
        #    (rel_bias #1) and the tau round-trip stay in torch — tiny,
        #    and the per-row scalar tau multiply commutes with the
        #    kernel's bias gather at identical rounding points (scale
        #    then round then gather == gather then scale then round,
        #    elementwise; masked lanes are 0 either way). ONE flash-
        #    decode kernel per layer then reads K/V IN PLACE from the
        #    pool — the [B, L, H, D] gather copies, matmul-contiguity
        #    copies and [B, H, 1, L] score/bias intermediates of the
        #    eager chain below are never materialized
        #    (kernels/attn_swa.py has the numerics contract)
        rl = torch.matmul(r.view(B, n_heads, -1), rel_proj)
        if is_global and n_floor is not None:
            tau = log_scale_tau(ctx["pos"], alpha, n_floor)
            q = (q.float() * tau[:, None, None]).to(q.dtype)
            rl = (rl.float() * tau[:, None, None]).to(rl.dtype)
        if is_global:
            out = attn_global(q.contiguous(), pool.kp, pool.vp,
                              flat.contiguous(), ctx["pos"],
                              rl.contiguous(), L)
        else:
            out = attn_swa(q.contiguous(), pool.k, pool.v, slots,
                           ctx["pos"], rl.contiguous())
        return F.linear(out.reshape(B, n_heads * head_dim), wo)
    if is_global:
        K = pool.kp.view(-1, n_kv, head_dim)[flat]         # [B, L, H, D]
        V = pool.vp.view(-1, n_kv, head_dim)[flat]
    else:
        K = pool.k[slots]                                  # [B, 512, H, D]
        V = pool.v[slots]
    #4. rel bias over the whole (row, key-slot) grid — rel_bias's exact
    #   math batched: mix profiles, gather at the clamped backward
    #   distance, zero out-of-band (dead slots have dist < 0)
    extent = rel_proj.shape[1]
    rel_logits = (r.view(B, 1, n_heads, -1) @ rel_proj).transpose(1, 2)
    gidx = dist.clamp(0, extent - 1)[:, None, None, :].expand(
        B, n_heads, 1, L)
    bias = rel_logits.gather(-1, gidx).masked_fill(
        ((dist < 0) | (dist >= extent))[:, None, None, :], 0.0)
    #5. tau round-trip on global layers (== 1.0 below the 128000 floor,
    #   applied unconditionally like the ref; apply_log_scaling's op
    #   order with per-row scalars)
    if is_global and n_floor is not None:
        tau = log_scale_tau(ctx["pos"], alpha, n_floor)
        q = (q.float() * tau[:, None, None]).to(q.dtype)
        bias = (bias.float() * tau[:, None, None, None]).to(bias.dtype)
    #6. GQA via views (head h -> kv group h // rep, the repeat_kv map):
    #   ONE broadcast matmul for scores, bias added in attn_decode's
    #   association order, dead slots overwritten with the eager mask
    #   constant, fp32 softmax, @V, o_proj
    rep = n_heads // n_kv
    qh = q.view(B, n_kv, rep, 1, head_dim)
    kh = K.permute(0, 2, 3, 1)[:, :, None]                 # [B,n_kv,1,D,L]
    attn = torch.matmul(qh, kh) * (1.0 / head_dim)
    attn = attn + bias.view(B, n_kv, rep, 1, L)
    attn = attn.masked_fill(~valid[:, None, None, None, :],
                            torch.finfo(attn.dtype).min)
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(qh.dtype)
    vh = V.permute(0, 2, 1, 3)[:, :, None]                 # [B,n_kv,1,L,D]
    out = torch.matmul(attn, vh).reshape(B, n_heads * head_dim)
    return F.linear(out, wo)


def layer_decode_batch(x, w, states, pos, mc, layer_idx, ctx=None,
                       pool=None):
    """One full decoder layer, BATCHED decode form — layer_decode's exact
    op order with the layer's weights read ONCE for all B rows: norms,
    projections and the MLP half run as B-row batches (dense_mlp / moe
    already take a flattened [T, hidden] token batch — the rows ride
    through natively, sharing expert weight reads across sequences); the
    attention core and the four sconv ring updates stay per-sequence —
    unless `ctx` + `pool` are given (exp-0007: the scheduler's pooled
    states), which routes attention through the per-layer batched core
    attn_decode_batch_pooled."""
    is_global = layer_idx in mc.global_layers
    #1. attention half: batched pre-norm + attention, batched attn_sconv
    h = rmsnorm(x, w["attn_norm"], mc.rms_eps)
    if ctx is not None:
        h = attn_decode_batch_pooled(
            h, w["wq"], w["wk"], w["wv"], w["wr"], w["wo"], w["ksc"],
            w["vsc"], w["qn"], w["kn"], w["proj"], states, pool, ctx,
            mc.head_dim, is_global, alpha=mc.log_alpha,
            n_floor=mc.log_floor, eps=mc.rms_eps)
        h = sconv_decode_batch_pooled(h, w["attn_sconv"], pool.sconv.at,
                                      ctx["slots"])
    else:
        h = attn_decode_batch(
            h, w["wq"], w["wk"], w["wv"], w["wr"], w["wo"], w["ksc"],
            w["vsc"], w["qn"], w["kn"], w["proj"], states, pos,
            mc.head_dim, is_global, alpha=mc.log_alpha,
            n_floor=mc.log_floor, eps=mc.rms_eps)
        h = sconv_decode_batch(h, w["attn_sconv"],
                               [s.attn_ring for s in states])
    x = x + h
    #2. MLP half: dense MLP / MoE consume the [B, hidden] token batch
    h = rmsnorm(x, w["mlp_norm"], mc.rms_eps)
    if layer_idx < mc.dense_idx:
        h = dense_mlp(h, w["mlp_gate"], w["mlp_up"], w["mlp_down"],
                      w["mlp_gs"])
    else:
        h = moe(h, w["gate_w"], w["gate_b"], w["gate_gs"], w["gu"],
                w["w2"], w["shg"], w["shu"], w["shd"], mc.topk,
                mc.n_shared, mc.route_scale)
    if ctx is not None:
        h = sconv_decode_batch_pooled(h, w["mlp_sconv"], pool.sconv.mt,
                                      ctx["slots"])
    else:
        h = sconv_decode_batch(h, w["mlp_sconv"],
                               [s.mlp_ring for s in states])
    return x + h


def generate_greedy(ids, n_new, layers, states, w_emb, w_en, w_fn, w_un, mc,
                    capture=None):
    """B3.2: batch-1 greedy decode loop. ids [T] int64 on w_emb's device;
    `layers` = the 66 per-layer weight dicts in layer_prefill layout
    (t_b2.load_layer_weights; MoE routed experts may be loader.PackedExperts
    — moe_experts only indexes them), possibly spread across GPUs (the
    hidden state hops with .to(); a copy is bitwise, so placement never
    changes numerics); `states` = one FRESH kv.LayerKV per layer on that
    layer's device. Prefill populates the recurrent state and yields token
    1 from the last-position logits; each further token embeds the previous
    one and runs layer_decode at absolute position T-1+s (B3.1). Greedy =
    argmax of the unpadded-vocab logits in fp32 (the B2.11 comparison
    form). No eos early-exit: the canonical workloads run ignore_eos (P1),
    and greedy stays well-defined past eos. Returns (tokens, margins):
    n_new generated ids and their top1-top2 fp32 logit gaps. `capture` (a
    list, tests only) additionally receives each step's full fp32 logits
    row on CPU — read-only copies, numerics untouched (B3.2)."""
    T = ids.shape[0]
    dt = w_emb.dtype
    #1. per-device prefill masks/positions, built once per GPU (the mask
    #   build is pure selection — identical on every device, B2.6/B2.7)
    per_dev = {}

    def _masks(dev):
        if dev not in per_dev:
            p = torch.arange(T, device=dev)
            per_dev[dev] = (additive_causal_mask(p, p, dt),
                           additive_causal_mask(p, p, dt, window=mc.window),
                           p)
        return per_dev[dev]

    tokens, margins = [], []

    def _pick(logit_row):
        #1. fp32 argmax + top-2 margin over the unpadded vocab (B2.11 form)
        lg = logit_row.float()
        top2 = torch.topk(lg, 2).values
        tokens.append(int(lg.argmax()))
        margins.append(float(top2[0] - top2[1]))
        if capture is not None:
            capture.append(lg.cpu())

    #2. prefill: embed -> 66 x layer_prefill(state=...) -> last-pos logits
    _, h = embed(ids[None], w_emb, w_en, eps=mc.rms_eps)
    for L in range(mc.num_layers):
        dev = layers[L]["attn_norm"].device
        full, win, p = _masks(dev)
        h = layer_prefill(h.to(dev), layers[L],
                          full if L in mc.global_layers else win, p, mc, L,
                          state=states[L])
    _pick(final_logits(h[:, -1], w_fn, w_un, mc.logits_mult,
                       mc.vocab_unpadded, eps=mc.rms_eps)[0])
    #3. decode loop: feed token s-1 at absolute position T-1+s
    for s in range(1, n_new):
        prev = torch.tensor([tokens[-1]], device=w_emb.device)
        _, h = embed(prev, w_emb, w_en, eps=mc.rms_eps)
        for L in range(mc.num_layers):
            h = layer_decode(h.to(layers[L]["attn_norm"].device), layers[L],
                             states[L], T - 1 + s, mc, L)
        _pick(final_logits(h, w_fn, w_un, mc.logits_mult, mc.vocab_unpadded,
                           eps=mc.rms_eps)[0])
    return tokens, margins


def final_logits(h, w_norm, w_unembed, mult, n_unpadded, eps=1e-6):
    """Final norm -> next-token logits, exact InklingForCausalLM semantics
    (modeling_inkling.py:783-789; the multimodal wrapper is identical,
    :1280-1286): model.norm rmsnorm, then the hidden state is DIVIDED by
    logits_mup_width_multiplier (bf16 division, not a reciprocal multiply),
    then the bias-free lm_head GEMM (checkpoint model.llm.unembed.weight ->
    lm_head.weight, conversion_mapping.py), then the padded vocab tail is
    sliced off at unpadded_vocab_size (200058 of 201024) before any
    sampling/argmax."""
    #1. final rmsnorm, then the mup width divide in the model dtype (:783)
    h = rmsnorm(h, w_norm, eps) / mult
    #2. unembed GEMM, slice the vocab padding (:786-789)
    return torch.nn.functional.linear(h, w_unembed)[..., :n_unpadded]
