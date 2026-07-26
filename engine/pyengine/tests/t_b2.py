"""B2 acceptance tests, one subcommand per PROGRESS.md item.
Run from repo root: python -m engine.pyengine.tests.t_b2 <subcommand>

`ref` (B2.1) generates the per-layer reference activations ONCE with the
NATIVE transformers Inkling implementation (transformers 5.14.1 ships it;
the checkpoint dir has no remote code) on the SAME checkpoint, tiny prompt,
layers {0,1,2,5,6}, saved under engine/pyengine/tests/ref/ and committed.
On later runs `ref` verifies the committed files (sha256 + manifest) instead
of regenerating — the refs are frozen once generated, like goldens.

Why not a naive from_pretrained: P2 probe run 3 (2026-07-24/25, cited in
scripts/generate_goldens.py) demonstrated transformers loads this NVFP4
checkpoint INCORRECTLY — the packed routed-expert weights of MoE layers 3-65
hit "Reinit due to size mismatch" (U8 half-width vs bf16 module shapes) and
are randomly re-initialized. Everything else (479-entry bf16 exclude list,
B1.3) loads exactly. So this test:
  1. truncates the model to layers 0-6 (a causal decoder: activations of
     layers {0,1,2,5,6} only depend on layers below them, so the slice is
     exact; full 66-layer bf16 experts would need ~1.8 TB and fit nowhere);
  2. loads with transformers' own code + weight-conversion pipeline
     (conversion_mapping.py "inkling_mm_model": w13 tensors pass
     Interleave(dim=1) — de-interleave [g0,u0,g1,u1,..] rows into
     [gates;ups] halves — before landing in gate_up_proj);
  3. repairs the 8 mis-loaded expert tensors (layers 3-6 x w13/w2) with
     loader.dequant_nvfp4 (B1.4: bit-exact vs vLLM dequantize_to_dtype)
     plus the same de-interleave, asserting from_pretrained's own
     loading-info that NOTHING ELSE was re-initialized;
  4. proves the de-interleave replication bit-exactly against layer 2's
     bf16 experts, which transformers' own Interleave op converted;
  5. runs one eager forward (attn_implementation="eager": the rel-pos bias
     only feeds the eager path; sconvs are fp32 per the model's
     _keep_in_fp32_modules_strict) and captures module-boundary activations.
"""
import hashlib
import json
import pathlib
import re
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from engine.pyengine import config as pcfg
from engine.pyengine import loader
from engine.pyengine import model as pmodel

MODEL_DIR = "/workspace/models/inkling-nvfp4"
REF_DIR = pathlib.Path(__file__).resolve().parent / "ref"
REF_TENSORS = REF_DIR / "b2_ref_activations.safetensors"
REF_META = REF_DIR / "b2_ref_meta.json"

PROMPT = "The capital of France is Paris, and the capital of Germany is"
REF_LAYERS = (0, 1, 2, 5, 6)   # PROGRESS.md B2 header
N_TRUNC = 7                    # keep layers 0..6 — covers every REF_LAYER
FIXUP_LAYERS = (3, 4, 5, 6)    # NVFP4-packed expert layers inside the slice
DEV = "cuda:0"                 # GPU 4 under CUDA_VISIBLE_DEVICES=4,5,6,7
MAX_REF_MIB = 64               # commit budget: refs must stay small
GiB = 1 << 30


def _deinterleave_rows(t):
    #1. exact mirror of transformers' Interleave(dim=0-of-2D) conversion op
    #   (core_model_loading.py:184-208): [x0,y0,x1,y1,..] rows -> [xs;ys].
    #   Applied per expert it equals the mapping's Interleave(dim=1) on the
    #   3D tensor. Same de-interleave as vLLM inkling nvidia/moe.py:583-595.
    r = t.shape[0]
    return t.reshape(r // 2, 2, *t.shape[1:]).transpose(0, 1).reshape(t.shape)


def _disk(idx, name):
    #1. read one whole tensor from its shard (CPU)
    with safe_open(str(idx.shard_path(name)), framework="pt") as f:
        return f.get_tensor(name)


def _disk_expert(idx, name, e):
    #1. read one expert's slice of a 3D tensor without loading all 256
    with safe_open(str(idx.shard_path(name)), framework="pt") as f:
        return f.get_slice(name)[e]


def t_ref():
    #1. frozen-refs fast path: if the committed files exist, verify them
    if REF_TENSORS.exists() or REF_META.exists():
        _verify_existing()
        return
    t0 = time.time()
    import transformers
    from transformers import AutoConfig, AutoTokenizer, InklingForConditionalGeneration
    torch.manual_seed(0)

    #2. checkpoint plumbing from the B1 loader (all fail-loud, all verified)
    idx = loader.build_shard_index(MODEL_DIR)
    hdr = loader.read_headers(idx)
    dm = loader.build_dtype_map(idx, hdr)
    mc = pcfg.load_verified(MODEL_DIR)

    #3. pre-flight: the 7-layer bf16 slice is ~156 GiB on ONE GPU — demand
    #   headroom up front instead of dying mid-load
    free, _ = torch.cuda.mem_get_info(torch.device(DEV))
    if free < 180 * GiB:
        raise SystemExit(f"[t_b2 ref] {DEV} has {free / GiB:.1f} GiB free, "
                         f"need 180 — GPU not idle?")

    #4. truncate the verified config to layers 0..6; keep per-layer types
    #   (layer 5 global/"hybrid", rest SWA; layers 0-1 dense, 2-6 sparse)
    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    tc = cfg.text_config
    assert tc.num_hidden_layers == mc.num_layers, tc.num_hidden_layers
    assert len(tc.layer_types) == mc.num_layers
    tc.layer_types = list(tc.layer_types[:N_TRUNC])
    tc.mlp_layer_types = list(tc.mlp_layer_types[:N_TRUNC])
    tc.num_hidden_layers = N_TRUNC
    want_types = ["hybrid" if i in mc.global_layers else "hybrid_sliding"
                  for i in range(N_TRUNC)]
    want_mlp = ["dense" if i < mc.dense_idx else "sparse" for i in range(N_TRUNC)]
    assert tc.layer_types == want_types, tc.layer_types
    assert tc.mlp_layer_types == want_mlp, tc.mlp_layer_types

    #5. load through transformers' own conversion pipeline; the 8 packed
    #   expert tensors (layers 3-6 x w13/w2) size-mismatch and re-init —
    #   expected, repaired in step 7
    model, info = InklingForConditionalGeneration.from_pretrained(
        MODEL_DIR, config=cfg, dtype=torch.bfloat16,
        attn_implementation="eager", device_map={"": 0},
        ignore_mismatched_sizes=True, output_loading_info=True)
    model.eval()
    lm = model.model.language_model
    t_load = time.time() - t0

    #6. reconcile loading-info EXACTLY: the mismatch set must be precisely
    #   the 8 packed expert tensors; no text-tower/lm_head key may be
    #   missing; no kept-layer key may be dropped except quant companions
    mismatched = {e[0] if isinstance(e, (tuple, list)) else e
                  for e in info["mismatched_keys"]}
    want_mm = {f"model.language_model.layers.{L}.mlp.experts.{p}"
               for L in FIXUP_LAYERS for p in ("gate_up_proj", "down_proj")}
    if mismatched != want_mm:
        raise SystemExit(f"[t_b2 ref] mismatched keys != expected 8 expert "
                         f"packs:\n  extra={sorted(mismatched - want_mm)}\n"
                         f"  absent={sorted(want_mm - mismatched)}")
    bad_missing = [k for k in info["missing_keys"]
                   if "language_model" in k or k.startswith("lm_head")]
    if bad_missing:
        raise SystemExit(f"[t_b2 ref] text-tower keys missing: {bad_missing[:8]}")
    pat = re.compile(r"language_model\.layers\.(\d+)\.")
    for k in info["unexpected_keys"]:
        m = pat.search(k)
        if m and int(m.group(1)) < N_TRUNC and not any(
                s in k for s in (".scale", ".input_amax", ".original_shape")):
            raise SystemExit(f"[t_b2 ref] kept-layer weight dropped: {k}")

    #7. repair layers 3-6 routed experts: B1.4 bit-exact dequant + the
    #   proven de-interleave, expert by expert (bounded workspace)
    fix_ratio = {}
    with torch.no_grad():
        for L in FIXUP_LAYERS:
            experts = lm.layers[L].mlp.experts
            for wname, pname, deint in (("w13_weight", "gate_up_proj", True),
                                        ("w2_weight", "down_proj", False)):
                base = f"model.llm.layers.{L}.mlp.experts.{wname}"
                packed, scale = _disk(idx, base), _disk(idx, base + ".scale")
                scale2 = _disk(idx, base + ".scale2")
                param = getattr(experts, pname)
                assert param.shape == (mc.n_experts, packed.shape[1],
                                       packed.shape[2] * 2), (base, param.shape)
                for e in range(mc.n_experts):
                    deq = loader.dequant_nvfp4(
                        packed[e].to(DEV), scale[e].to(DEV),
                        scale2[e].to(DEV), dm.group_size)
                    if deint:
                        deq = _deinterleave_rows(deq)
                    param.data[e].copy_(deq)
                #7a. ModelOpt invariant (B1.4): max|deq| == 6*448*scale2
                r0 = (param.data[0].abs().max().float()
                      / (6.0 * 448.0 * scale2[0].to(DEV))).item()
                fix_ratio[f"{L}.{wname}"] = round(r0, 6)
                if not 0.98 < r0 < 1.02:
                    raise SystemExit(f"[t_b2 ref] dequant invariant broken "
                                     f"{base}: ratio {r0}")

    #8. prove the load end-to-end with bit-exact spot checks vs disk:
    #   (a) layer-2 bf16 experts: transformers' OWN Interleave conversion
    #       must equal our _deinterleave_rows (justifies step 7); w2 rename
    b2 = "model.llm.layers.2.mlp.experts."
    for e in (0, mc.n_experts - 1):
        assert torch.equal(
            _deinterleave_rows(_disk_expert(idx, b2 + "w13_weight", e).to(DEV)),
            lm.layers[2].mlp.experts.gate_up_proj.data[e]), ("interleave", e)
        assert torch.equal(_disk_expert(idx, b2 + "w2_weight", e).to(DEV),
                           lm.layers[2].mlp.experts.down_proj.data[e]), ("w2", e)
    #   (b) dense layer 0: w13_dn -> Interleave+Chunk -> gate/up
    dd = _deinterleave_rows(_disk(idx, "model.llm.layers.0.mlp.w13_dn.weight").to(DEV))
    g13, u13 = dd.chunk(2, dim=0)
    assert torch.equal(g13, lm.layers[0].mlp.gate_proj.weight)
    assert torch.equal(u13, lm.layers[0].mlp.up_proj.weight)
    #   (c) plain renames: embed rows, layer-5 wq, fp32-upcast sconv
    with safe_open(str(idx.shard_path("model.llm.embed.weight")),
                   framework="pt") as f:
        rows = f.get_slice("model.llm.embed.weight")[:4]
    assert torch.equal(rows.to(DEV), lm.embed_tokens.weight[:4])
    assert torch.equal(_disk(idx, "model.llm.layers.5.attn.wq_du.weight").to(DEV),
                       lm.layers[5].self_attn.q_proj.weight)
    ksc = _disk(idx, "model.llm.layers.0.attn.k_sconv.weight")
    assert lm.layers[0].self_attn.k_sconv.conv1d.weight.dtype == torch.float32
    assert torch.equal(ksc.float().to(DEV),
                       lm.layers[0].self_attn.k_sconv.conv1d.weight)
    t_fix = time.time() - t0 - t_load

    #9. module-boundary capture hooks on layers {0,1,2,5,6} + embeddings.
    #   Decoder layers/mlp/sconv/experts get hidden_states positionally;
    #   self_attn gets it as a kwarg -> hooks read args[0] or the kwarg.
    caps = {}

    def grab(t):
        return t.detach().to("cpu").contiguous()

    def cap_io(key, save_in=True):
        def hook(mod, args, kwargs, out):
            if save_in:
                src = args[0] if args else kwargs["hidden_states"]
                caps[key + ".in"] = grab(src)
            caps[key + ".out"] = grab(out[0] if isinstance(out, tuple) else out)
        return hook

    def cap_gate(key):
        def hook(mod, args, kwargs, out):
            for n, t in zip(("routed_logits", "topk_weights",
                             "topk_indices", "shared_gammas"), out):
                caps[f"{key}.{n}"] = grab(t)
        return hook

    hooks = []

    def reg(mod, fn):
        hooks.append(mod.register_forward_hook(fn, with_kwargs=True))

    reg(lm.embed_tokens, cap_io("embed_tokens", save_in=False))
    reg(lm.embed_norm, cap_io("embed_norm", save_in=False))
    for L in REF_LAYERS:
        lay, k = lm.layers[L], f"layers.{L}"
        reg(lay, cap_io(k))
        reg(lay.input_layernorm, cap_io(f"{k}.input_layernorm", save_in=False))
        reg(lay.self_attn, cap_io(f"{k}.self_attn"))
        reg(lay.attn_sconv, cap_io(f"{k}.attn_sconv", save_in=False))
        reg(lay.post_attention_layernorm,
            cap_io(f"{k}.post_attention_layernorm", save_in=False))
        reg(lay.mlp, cap_io(f"{k}.mlp"))
        reg(lay.mlp_sconv, cap_io(f"{k}.mlp_sconv", save_in=False))
        if L in (0, 5):   # attention internals for the SWA + global ref items
            att = lay.self_attn
            reg(att.k_sconv, cap_io(f"{k}.self_attn.k_sconv"))
            reg(att.v_sconv, cap_io(f"{k}.self_attn.v_sconv"))
            reg(att.r_proj, cap_io(f"{k}.self_attn.r_proj", save_in=False))
            reg(att.q_norm, cap_io(f"{k}.self_attn.q_norm", save_in=False))
            reg(att.k_norm, cap_io(f"{k}.self_attn.k_norm", save_in=False))
            reg(att.rel_logits_proj,
                cap_io(f"{k}.self_attn.rel_logits_proj", save_in=False))
        if tc.mlp_layer_types[L] == "sparse":
            reg(lay.mlp.gate, cap_gate(f"{k}.mlp.gate"))
            reg(lay.mlp.experts, cap_io(f"{k}.mlp.experts"))
            reg(lay.mlp.shared_experts, cap_io(f"{k}.mlp.shared_experts"))

    #10. tokenize the tiny prompt and run ONE eager forward, no cache
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEV)
    with torch.no_grad():
        out = lm(input_ids=ids, use_cache=False)
    for h in hooks:
        h.remove()
    caps["input_ids"] = ids.cpu().contiguous()
    caps["final_norm.out"] = grab(out.last_hidden_state)

    #11. every float capture must be finite (garbage weights would show here)
    for n, t in caps.items():
        if t.is_floating_point() and not torch.isfinite(t.float()).all():
            raise SystemExit(f"[t_b2 ref] non-finite activation: {n}")

    #12. persist: small safetensors + manifest with provenance + sha256
    REF_DIR.mkdir(exist_ok=True)
    save_file(caps, str(REF_TENSORS))
    size_mib = REF_TENSORS.stat().st_size / (1 << 20)
    if size_mib > MAX_REF_MIB:
        raise SystemExit(f"[t_b2 ref] refs {size_mib:.1f} MiB > {MAX_REF_MIB}"
                         f" — not committable, shrink the capture set")
    meta = {
        "item": "B2.1",
        "prompt": PROMPT,
        "token_ids": ids[0].tolist(),
        "ref_layers": list(REF_LAYERS),
        "n_trunc_layers": N_TRUNC,
        "layer_types": list(tc.layer_types),
        "mlp_layer_types": list(tc.mlp_layer_types),
        "attn_implementation": "eager",
        "dtype": "bf16 weights/activations; sconv convs fp32 "
                 "(_keep_in_fp32_modules_strict); softmax fp32 (eager)",
        "fixup": {"layers": list(FIXUP_LAYERS),
                  "method": "loader.dequant_nvfp4 (B1.4 bit-exact vs vLLM) "
                            "+ Interleave(dim=1) de-interleave, proven vs "
                            "layer-2 bf16 experts",
                  "max_deq_over_6x448xscale2_expert0": fix_ratio},
        "loading_info": {"mismatched": sorted(mismatched),
                         "missing_total": len(info["missing_keys"]),
                         "unexpected_total": len(info["unexpected_keys"])},
        "versions": {"transformers": transformers.__version__,
                     "torch": torch.__version__,
                     "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(0)},
        "wall_s": {"load": round(t_load, 1), "fixup": round(t_fix, 1),
                   "total": round(time.time() - t0, 1)},
        "tensors": {n: [str(t.dtype).replace("torch.", ""), list(t.shape)]
                    for n, t in sorted(caps.items())},
        "sha256": hashlib.sha256(REF_TENSORS.read_bytes()).hexdigest(),
    }
    REF_META.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")

    #13. summary line = the test's green evidence
    print(f"ref ok: generated {len(caps)} tensors ({size_mib:.1f} MiB) for "
          f"layers {list(REF_LAYERS)}, prompt {ids.shape[1]} toks; native "
          f"transformers {transformers.__version__} eager, 7-layer slice on "
          f"{DEV}; fixup layers 3-6 experts dequant (invariant ratios "
          f"{sorted(set(fix_ratio.values()))}); spot checks bit-exact "
          f"(interleave/dense-chunk/embed/wq/fp32-sconv); all finite; "
          f"wall {time.time() - t0:.1f} s (load {t_load:.1f})")


def _load_refs():
    """Silent core of the ref verifier: every B2.2+ item loads the frozen
    refs through this, so a tampered/partial ref set fails ALL items."""
    #1. both files must exist together — a half-committed ref set is a bug
    if not (REF_TENSORS.exists() and REF_META.exists()):
        raise SystemExit(f"[t_b2 ref] partial refs: tensors="
                         f"{REF_TENSORS.exists()} meta={REF_META.exists()}"
                         f" — run item B2.1 first")
    meta = json.loads(REF_META.read_text())

    #2. content hash pins the frozen refs (same convention as goldens)
    sha = hashlib.sha256(REF_TENSORS.read_bytes()).hexdigest()
    if sha != meta["sha256"]:
        raise SystemExit(f"[t_b2 ref] sha256 mismatch: file {sha[:12]} vs "
                         f"meta {meta['sha256'][:12]} — refs tampered/stale")

    #3. manifest agreement: exact key set, dtype and shape per tensor
    tens = load_file(str(REF_TENSORS))
    manifest = meta["tensors"]
    if set(tens) != set(manifest):
        raise SystemExit(f"[t_b2 ref] key set mismatch: "
                         f"only-file={sorted(set(tens) - set(manifest))[:5]} "
                         f"only-meta={sorted(set(manifest) - set(tens))[:5]}")
    for n, t in tens.items():
        want = [str(t.dtype).replace("torch.", ""), list(t.shape)]
        if manifest[n] != want:
            raise SystemExit(f"[t_b2 ref] manifest mismatch {n}: "
                             f"{manifest[n]} vs {want}")
        if t.is_floating_point() and not torch.isfinite(t.float()).all():
            raise SystemExit(f"[t_b2 ref] non-finite ref tensor: {n}")
    return tens, meta, sha


def _verify_existing():
    #1. checks live in _load_refs; this wrapper prints the green evidence
    tens, meta, sha = _load_refs()
    print(f"ref ok: existing refs verified — {len(tens)} tensors, sha256 "
          f"{sha[:12]}, layers {meta['ref_layers']}, prompt "
          f"{len(meta['token_ids'])} toks (generated with transformers "
          f"{meta['versions']['transformers']}, {meta['wall_s']['total']} s)")


def _rel_err(got, want):
    #1. fp32 global L2 relative error — the B2 items' "rel err" gate
    g, w = got.float(), want.float()
    return ((g - w).norm() / w.norm()).item()


def t_embed():
    """B2.2: our embed + embed_norm (model.py) vs the frozen transformers
    refs, rel err < 1e-2 in bf16 on both module boundaries."""
    #1. frozen refs (sha-verified) give input_ids + expected outputs
    tens, meta, _ = _load_refs()
    ids = tens["input_ids"].to(DEV)
    want_lookup = tens["embed_tokens.out"].to(DEV)
    want_norm = tens["embed_norm.out"].to(DEV)

    #2. read the two weights straight from their shards (B1.1 index);
    #   shapes/dtypes pinned against the verified config
    idx = loader.build_shard_index(MODEL_DIR)
    mc = pcfg.load_verified(MODEL_DIR)
    w_e = _disk(idx, "model.llm.embed.weight").to(DEV)
    w_n = _disk(idx, "model.llm.embed_norm.weight").to(DEV)
    assert w_e.shape == (mc.vocab, mc.hidden), w_e.shape
    assert w_n.shape == (mc.hidden,), w_n.shape
    assert w_e.dtype == w_n.dtype == torch.bfloat16

    #3. our engine path (pure torch, exact InklingRMSNorm semantics)
    h, hn = pmodel.embed(ids, w_e, w_n, eps=mc.rms_eps)

    #4. gate: rel err < 1e-2 (item text) on lookup AND normed output;
    #   bit-equality reported as extra evidence (lookup is a row copy,
    #   norm replays the same fp32-then-bf16 ops on the same GPU)
    re_l, re_n = _rel_err(h, want_lookup), _rel_err(hn, want_norm)
    bit_l = torch.equal(h, want_lookup)
    bit_n = torch.equal(hn, want_norm)
    if not (re_l < 1e-2 and re_n < 1e-2):
        raise SystemExit(f"[t_b2 embed] rel err over 1e-2: "
                         f"lookup {re_l:.3e}, embed_norm {re_n:.3e}")
    max_n = (hn.float() - want_norm.float()).abs().max().item()

    #5. summary line = the test's green evidence
    print(f"embed ok: {ids.shape[1]} toks; lookup rel err {re_l:.1e} "
          f"(bit-exact {bit_l}), embed_norm rel err {re_n:.1e} < 1e-2 "
          f"(bit-exact {bit_n}, max|diff| {max_n:.1e}); "
          f"weights {tuple(w_e.shape)} bf16 from shards, eps {mc.rms_eps}")


def _ulp_bf16(a, b):
    """Exact bf16 ulp distance. Finite inputs only (callers gate finiteness).
    Sign-magnitude bit patterns map to a monotonic 'lexicographic' integer
    (i >= 0 -> i, else -32768 - i; both zeros -> 0), where adjacent
    representable bf16 values differ by exactly 1."""
    #1. reinterpret bits, widen so the subtraction cannot overflow
    ai = a.view(torch.int16).to(torch.int32)
    bi = b.view(torch.int16).to(torch.int32)
    #2. monotonic mapping, then integer distance = ulp count
    al = torch.where(ai >= 0, ai, -32768 - ai)
    bl = torch.where(bi >= 0, bi, -32768 - bi)
    return (al - bl).abs()


def t_rmsnorm():
    """B2.3: Triton rmsnorm kernel (kernels/rmsnorm.py) vs the torch
    reference (model.rmsnorm — proven bit-exact vs transformers in B2.2)
    on random bf16 tensors. Gate per case: elementwise <= 2 bf16 ulp AND
    fp32 global L2 rel err < 1e-3 — far inside the B2 bf16 budget (1e-2).
    Bit-exactness is NOT required: tl.sum's tree order differs from torch's
    reduction order by a few fp32 ulps in the variance; after the bf16
    downcast that is <= 1 ulp on xhat, <= 2 on the weight product (analysis
    in kernels/rmsnorm.py). Extra anchor: kernel output on the frozen embed
    lookup ref vs the frozen embed_norm.out (real checkpoint weights)."""
    from engine.pyengine.kernels import rmsnorm as krms
    torch.manual_seed(0)
    mc = pcfg.load_verified(MODEL_DIR)

    #1. random cases over the two real widths: hidden 6144 (BLOCK 8192,
    #   masked lanes) incl. decode-shaped, ref-prompt-shaped, batchy and
    #   odd row counts + scale extremes (1e-4 makes eps dominate the
    #   variance); head_dim 128 3D = q/k-norm shape (BLOCK 128, unmasked)
    cases = [((1, mc.hidden), 1.0), ((13, mc.hidden), 1.0),
             ((512, mc.hidden), 1.0), ((2048, mc.hidden), 1.0),
             ((257, mc.hidden), 1e-4), ((64, mc.hidden), 1e4),
             ((13, mc.g_q_heads, mc.head_dim), 1.0),
             ((5, mc.hidden), 0.0)]   # all-zero rows: var=0, eps path only
    tot_el = tot_exact = 0
    max_ulp = worst_rel = 0.0
    for shape, scale in cases:
        x = (torch.randn(shape, device=DEV, dtype=torch.float32)
             * scale).to(torch.bfloat16)
        w = torch.randn(shape[-1], device=DEV,
                        dtype=torch.float32).to(torch.bfloat16)
        want = pmodel.rmsnorm(x, w, eps=mc.rms_eps)
        got = krms.rmsnorm(x, w, eps=mc.rms_eps)
        #2. hard contract: same dtype/shape, finite everywhere
        assert got.dtype == want.dtype == torch.bfloat16, got.dtype
        assert got.shape == want.shape == x.shape, got.shape
        if not torch.isfinite(got.float()).all():
            raise SystemExit(f"[t_b2 rmsnorm] non-finite output {shape}")
        #3. gates: <= 2 bf16 ulp elementwise, global rel err < 1e-3
        ulp = _ulp_bf16(got, want)
        mu = ulp.max().item()
        rel = 0.0 if scale == 0.0 else _rel_err(got, want)
        if mu > 2 or rel >= 1e-3:
            raise SystemExit(f"[t_b2 rmsnorm] case {shape} scale {scale}: "
                             f"max ulp {mu}, rel err {rel:.3e}")
        if scale == 0.0 and not torch.equal(got, want):
            raise SystemExit("[t_b2 rmsnorm] zero-input case not bit-exact")
        tot_el += ulp.numel()
        tot_exact += (ulp == 0).sum().item()
        max_ulp = max(max_ulp, mu)
        worst_rel = max(worst_rel, rel)

    #4. real-weights anchor: frozen embed lookup -> kernel -> vs frozen
    #   embed_norm.out, same 1e-2 gate the torch path passed in B2.2
    tens, meta, _ = _load_refs()
    idx = loader.build_shard_index(MODEL_DIR)
    w_n = _disk(idx, "model.llm.embed_norm.weight").to(DEV)
    got_a = krms.rmsnorm(tens["embed_tokens.out"].to(DEV), w_n,
                         eps=mc.rms_eps)
    want_a = tens["embed_norm.out"].to(DEV)
    rel_a = _rel_err(got_a, want_a)
    ulp_a = _ulp_bf16(got_a, want_a).max().item()
    if rel_a >= 1e-2:
        raise SystemExit(f"[t_b2 rmsnorm] ref anchor rel err {rel_a:.3e}")

    #5. summary line = the test's green evidence
    print(f"rmsnorm ok: triton kernel vs torch ref, {len(cases)} random "
          f"cases (widths {{{mc.hidden},{mc.head_dim}}}, rows 1-2048, "
          f"scales 0/1e-4/1/1e4): {tot_el} elements, bit-exact "
          f"{tot_exact} ({tot_el - tot_exact} off), max {max_ulp:.0f} "
          f"ulp <= 2, "
          f"worst rel err {worst_rel:.1e} < 1e-3; frozen embed_norm anchor "
          f"rel err {rel_a:.1e} < 1e-2 (max {ulp_a:.0f} ulp)")


def t_relbias():
    """B2.4: relative-attention bias table + log scaling (model.py, pure
    torch) vs ref math. Gates:
      (a) frozen anchors — replay the captured r_proj.out of layers 0 (SWA,
          extent 512) and 5 (global, extent 1024) through our rel_bias with
          the real checkpoint proj banks, vs the captured rel_logits_proj.out
          (PRE-tau per the B2.1 note; tau==1.0 at 13 toks anyway): rel err
          < 1e-2 (B2 header budget), bit-exactness reported as evidence;
      (b) native-module cross-check — the ACTUAL transformers
          InklingRelativeLogits (the class that generated the refs) run
          in-process on real + random proj banks at synthetic long positions
          (offsets past the 128000 floor; distances spanning negative,
          in-band, and >= extent for both extents): torch.equal REQUIRED —
          identical math must give identical bits; out-of-band exactly zero;
      (c) log scaling — no non-trivial frozen anchor exists (13-tok prompt
          => tau==1.0, and every B2 prompt is tiny), so tau is gated against
          an independent float64 evaluation over the full context range plus
          exactness facts: tau==1.0 exactly through pos floor-1 (clamp arm,
          => inert across the whole 16K serving window), >1 and monotonic
          from pos==floor, and the fp32-multiply-then-downcast application
          op order (modeling_inkling.py:259-261) bit-checked on random
          bf16 Q/bias at positions past the floor."""
    import math
    from transformers.models.inkling.modeling_inkling import (
        InklingRelativeLogits)
    torch.manual_seed(0)
    tens, meta, _ = _load_refs()
    mc = pcfg.load_verified(MODEL_DIR)
    idx = loader.build_shard_index(MODEL_DIR)

    #1. frozen anchors: layers 0 + 5, real weights, real activations
    Q = tens["input_ids"].shape[1]
    pos = torch.arange(Q, device=DEV)
    projs, anchor_re, anchor_bit = {}, {}, {}
    for L in (0, 5):
        is_global = L in mc.global_layers
        extent = mc.rel_extent if is_global else mc.window
        proj = _disk(idx, f"model.llm.layers.{L}.attn.rel_logits_proj.proj"
                     ).to(DEV)
        assert proj.shape == (mc.d_rel, extent), proj.shape
        assert proj.dtype == torch.bfloat16, proj.dtype
        projs[extent] = proj
        heads = mc.g_q_heads if is_global else mc.s_q_heads
        rs = tens[f"layers.{L}.self_attn.r_proj.out"].to(DEV).view(
            1, Q, heads, mc.d_rel)
        got = pmodel.rel_bias(rs, proj, pos, pos)
        want = tens[f"layers.{L}.self_attn.rel_logits_proj.out"].to(DEV)
        assert got.shape == want.shape == (1, heads, Q, Q), got.shape
        #1a. future keys (distance < 0) must be EXACT zeros
        fut = (pos[:, None] - pos[None, :] < 0).view(1, 1, Q, Q)
        if not (got.masked_select(fut) == 0).all():
            raise SystemExit(f"[t_b2 relbias] layer {L}: nonzero future bias")
        re = _rel_err(got, want)
        if re >= 1e-2:
            raise SystemExit(f"[t_b2 relbias] layer {L} anchor rel err "
                             f"{re:.3e} >= 1e-2")
        anchor_re[L], anchor_bit[L] = re, torch.equal(got, want)

    #2. native-module cross-check at synthetic long positions. Cases give
    #   distances: ref-shaped -12..12 (negative + in-band), prefill@200k
    #   -15..1515 (all three zones for both extents), decode@300k 0..599
    #   (crosses extent 512 in decode shape, Q=1).
    cases = ((13, 0, 13, 0, "ref-shaped"),
             (16, 200000, 1516, 198500, "prefill-200k"),
             (1, 300000, 600, 299401, "decode-300k"))
    n_cross = el_cross = 0
    for extent, real_proj in sorted(projs.items()):
        for bank, pj in (("real", real_proj),
                         ("random", torch.randn(
                             mc.d_rel, extent, device=DEV,
                             dtype=torch.float32).to(torch.bfloat16))):
            #2a. the reference module itself, bf16 on the same GPU
            mod = InklingRelativeLogits(mc.d_rel, extent).to(
                device=DEV, dtype=torch.bfloat16)
            with torch.no_grad():
                mod.proj.copy_(pj)
            for qlen, qoff, klen, koff, tag in cases:
                qp = torch.arange(qlen, device=DEV) + qoff
                kp = torch.arange(klen, device=DEV) + koff
                rs = torch.randn(1, qlen, mc.g_q_heads, mc.d_rel,
                                 device=DEV,
                                 dtype=torch.float32).to(torch.bfloat16)
                with torch.no_grad():
                    want = mod(rs, qp, kp)
                got = pmodel.rel_bias(rs, pj, qp, kp)
                if not torch.equal(got, want):
                    raise SystemExit(f"[t_b2 relbias] not bit-exact vs "
                                     f"native module: extent {extent}, "
                                     f"{bank}, {tag}")
                #2b. band semantics: out-of-band exact zeros, in-band alive
                d = qp[:, None] - kp[None, :]
                oob = ((d < 0) | (d >= extent)).view(1, 1, qlen, klen)
                if oob.any() and not (got.masked_select(oob) == 0).all():
                    raise SystemExit(f"[t_b2 relbias] nonzero out-of-band: "
                                     f"extent {extent}, {bank}, {tag}")
                if not got.masked_select(~oob).any():
                    raise SystemExit(f"[t_b2 relbias] in-band all zero: "
                                     f"extent {extent}, {bank}, {tag}")
                n_cross += 1
                el_cross += got.numel()

    #3. log scaling. (a) exactness: tau == 1.0 for every position below the
    #   floor — covers the 13-tok ref prompt AND the whole 16K serve window
    ones = lambda t: torch.ones_like(t)                      # noqa: E731
    t13 = pmodel.log_scale_tau(pos, mc.log_alpha, mc.log_floor)
    serve = torch.arange(16384, device=DEV)
    t_serve = pmodel.log_scale_tau(serve, mc.log_alpha, mc.log_floor)
    if not (torch.equal(t13, ones(t13)) and
            torch.equal(t_serve, ones(t_serve))):
        raise SystemExit("[t_b2 relbias] tau != 1.0 below the floor")
    #3b. boundary + monotonicity around pos = floor-1 (eff == floor -> 1.0)
    bpos = torch.arange(mc.log_floor - 3, mc.log_floor + 3, device=DEV)
    tb = pmodel.log_scale_tau(bpos, mc.log_alpha, mc.log_floor)
    if not (torch.equal(tb[:3], ones(tb[:3])) and (tb[3:] > 1.0).all()
            and (tb.diff() >= 0).all()):
        raise SystemExit(f"[t_b2 relbias] tau boundary broken: {tb.tolist()}")
    #3c. independent float64 evaluation, log-spaced to the 1M context edge
    lpos = torch.unique(torch.cat([
        torch.logspace(0, math.log10(1048575), 4096,
                       device=DEV).to(torch.int64),
        bpos, torch.tensor([1048575], device=DEV)]))
    t32 = pmodel.log_scale_tau(lpos, mc.log_alpha, mc.log_floor)
    t64 = 1.0 + mc.log_alpha * torch.log(
        ((lpos + 1).double() / mc.log_floor).clamp(min=1.0))
    tau_rel = ((t32.double() - t64).abs() / t64).max().item()
    if tau_rel >= 1e-6:
        raise SystemExit(f"[t_b2 relbias] tau vs fp64: rel {tau_rel:.3e}")
    #3d. application op order on random bf16 Q/bias past the floor, checked
    #    against a literal transcription of modeling_inkling.py:259-261
    apos = torch.arange(5, device=DEV) + 250000
    qs = torch.randn(1, mc.g_q_heads, 5, mc.head_dim, device=DEV,
                     dtype=torch.float32).to(torch.bfloat16)
    bias = torch.randn(1, mc.g_q_heads, 5, 700, device=DEV,
                       dtype=torch.float32).to(torch.bfloat16)
    gq, gb = pmodel.apply_log_scaling(qs, bias, apos, mc.log_alpha,
                                      mc.log_floor)
    tau_t = (1.0 + mc.log_alpha * torch.log(
        ((apos + 1).float() / mc.log_floor).clamp(min=1.0))).view(1, 1, -1, 1)
    if not (torch.equal(gq, (qs.float() * tau_t).to(qs.dtype)) and
            torch.equal(gb, (bias.float() * tau_t).to(bias.dtype))):
        raise SystemExit("[t_b2 relbias] apply_log_scaling op order differs")
    if torch.equal(gq, qs):
        raise SystemExit("[t_b2 relbias] tau>1 case was a no-op on Q")

    #4. summary line = the test's green evidence
    print(f"relbias ok: frozen anchors L0(swa,ext512)/L5(global,ext1024) "
          f"rel err {anchor_re[0]:.1e}/{anchor_re[5]:.1e} < 1e-2 (bit-exact "
          f"{anchor_bit[0]}/{anchor_bit[5]}), future-dist zeros exact; "
          f"native-module cross-check bit-exact {n_cross}/{n_cross} cases "
          f"({el_cross} els; extents {{512,1024}} x real/random proj x "
          f"{{13tok, prefill@200k dist-15..1515, decode@300k}}), out-of-band "
          f"exact zeros; tau: ==1.0 exactly below floor 128000 (16K serve "
          f"window inert), floor boundary+monotonic ok, fp64 agreement "
          f"{tau_rel:.1e} < 1e-6 over {lpos.numel()} pts to ctx 1048575, "
          f"apply op-order bit-exact (modeling_inkling.py:259-261)")


def t_sconv():
    """B2.5: sconv (window-4, prefill form; model.sconv_prefill) matches ref
    on layer 0. Gates:
      (a) frozen anchors — all FOUR layer-0 sconvs (k/v_sconv C=2048,
          attn/mlp_sconv C=6144) replayed from their captured module inputs
          (k/v_sconv.in captured directly; attn_sconv's input IS
          self_attn.out and mlp_sconv's input IS mlp.out, per the decoder
          forward modeling_inkling.py:576-591) with the real checkpoint
          weights, vs the captured module outputs: rel err < 1e-2 (B2
          header budget), bit-exactness reported as evidence;
      (b) native-module cross-check — the ACTUAL transformers
          InklingShortConvolution run in-process (past_key_values=None,
          conv_mask=None — identical to the ref forward: batch-1 makes
          apply_mask_to_padding_states a no-op, :433), torch.equal REQUIRED,
          on the 4 real-weight frozen inputs AND random cases spanning
          T {1,3,4,13,600} x C {1024,2048,6144} x batch {1,2} (decode-shaped,
          shorter-than-kernel, exactly-kernel, ref-length, long);
      (c) structural facts — window/causality: perturbing token t changes
          ONLY outputs t..t+k-1 (bit-checked both sides); tap orientation:
          delta weight at the LAST tap gives out == 2x exactly (current
          token; cross-correlation, no flip) while delta at tap 0 reproduces
          x[t-3]+x[t] by manual indexing; zero input -> exact zero out (no
          bias); output dtype == input dtype."""
    from transformers.models.inkling.modeling_inkling import (
        InklingShortConvolution)
    torch.manual_seed(0)
    tens, meta, _ = _load_refs()
    mc = pcfg.load_verified(MODEL_DIR)
    idx = loader.build_shard_index(MODEL_DIR)
    k = mc.sconv_k

    #1. the four layer-0 sconvs: (tag, checkpoint weight, in-key, out-key).
    #   attn_sconv consumes self_attn.out, mlp_sconv consumes mlp.out
    #   (decoder forward modeling_inkling.py:576-591)
    L0 = "model.llm.layers.0."
    anchors = (
        ("k_sconv", L0 + "attn.k_sconv.weight",
         "layers.0.self_attn.k_sconv.in", "layers.0.self_attn.k_sconv.out"),
        ("v_sconv", L0 + "attn.v_sconv.weight",
         "layers.0.self_attn.v_sconv.in", "layers.0.self_attn.v_sconv.out"),
        ("attn_sconv", L0 + "attn_sconv.weight",
         "layers.0.self_attn.out", "layers.0.attn_sconv.out"),
        ("mlp_sconv", L0 + "mlp_sconv.weight",
         "layers.0.mlp.out", "layers.0.mlp_sconv.out"),
    )

    def native(w_f32, x):
        #2. the reference module itself, in-process, prefill path (no cache,
        #   no mask); default module dtype fp32 == _keep_in_fp32_modules_strict
        mod = InklingShortConvolution(w_f32.shape[0], k, layer_idx=0,
                                      conv_idx=2).to(DEV)
        assert mod.conv1d.bias is None
        assert mod.conv1d.weight.dtype == torch.float32
        with torch.no_grad():
            mod.conv1d.weight.copy_(w_f32)
            return mod(x, past_key_values=None, conv_mask=None)

    #3. gate (a)+(b) on real weights/activations: our function vs the frozen
    #   capture (rel err) AND vs the in-process native module (torch.equal)
    anchor_re, anchor_bit = {}, {}
    for tag, wname, ikey, okey in anchors:
        w = _disk(idx, wname).to(DEV)
        assert w.dtype == torch.bfloat16 and w.shape[1:] == (1, k), (tag, w.shape)
        x = tens[ikey].to(DEV)
        want = tens[okey].to(DEV)
        got = pmodel.sconv_prefill(x, w)
        assert got.dtype == x.dtype == torch.bfloat16 and got.shape == want.shape
        re = _rel_err(got, want)
        if re >= 1e-2:
            raise SystemExit(f"[t_b2 sconv] {tag} anchor rel err {re:.3e}"
                             f" >= 1e-2")
        anchor_re[tag], anchor_bit[tag] = re, torch.equal(got, want)
        if not torch.equal(got, native(w.float(), x)):
            raise SystemExit(f"[t_b2 sconv] {tag}: ours != native module "
                             f"on the frozen input")

    #4. gate (b) random cases: bf16-representable weights (checkpoint dtype)
    #   upcast to fp32, bf16 inputs; torch.equal required vs native module
    rand_cases = ((1, 1, 2048), (1, 3, 1024), (1, 4, 6144),
                  (2, 13, 2048), (1, 600, 6144))
    for B, T, C in rand_cases:
        w = torch.randn(C, 1, k, device=DEV).to(torch.bfloat16)
        x = torch.randn(B, T, C, device=DEV).to(torch.bfloat16)
        got = pmodel.sconv_prefill(x, w)
        if not torch.equal(got, native(w.float(), x)):
            raise SystemExit(f"[t_b2 sconv] random case B{B} T{T} C{C}: "
                             f"ours != native module")
        assert got.dtype == torch.bfloat16 and got.shape == x.shape

    #5. gate (c) structural facts, C=8 T=32 keeps them readable.
    #   (c1) window/causality: perturb token t -> only out[t..t+k-1] moves
    C, T, t = 8, 32, 11
    w = torch.randn(C, 1, k, device=DEV).to(torch.bfloat16)
    x1 = torch.randn(1, T, C, device=DEV).to(torch.bfloat16)
    x2 = x1.clone()
    x2[0, t] += 1.0
    o1, o2 = pmodel.sconv_prefill(x1, w), pmodel.sconv_prefill(x2, w)
    if not torch.equal(o1[:, :t], o2[:, :t]):
        raise SystemExit("[t_b2 sconv] non-causal: past outputs changed")
    if not torch.equal(o1[:, t + k:], o2[:, t + k:]):
        raise SystemExit(f"[t_b2 sconv] window > {k}: far-future outputs "
                         f"changed")
    if torch.equal(o1[0, t], o2[0, t]):
        raise SystemExit("[t_b2 sconv] perturbed token's own output frozen")
    #   (c2) tap orientation: delta at the LAST tap == identity conv
    #   -> conv(x)=x, +residual -> exactly 2x (fp32 doubling is exact)
    w_last = torch.zeros(C, 1, k, device=DEV)
    w_last[:, 0, -1] = 1.0
    if not torch.equal(pmodel.sconv_prefill(x1, w_last),
                       (x1.float() * 2).to(torch.bfloat16)):
        raise SystemExit("[t_b2 sconv] last tap is not the current token")
    #   delta at tap 0 -> conv(x)[t] = x[t-(k-1)], zeros before that
    w_first = torch.zeros(C, 1, k, device=DEV)
    w_first[:, 0, 0] = 1.0
    xf = x1.float()
    shifted = torch.cat([torch.zeros(1, k - 1, C, device=DEV),
                         xf[:, : T - (k - 1)]], dim=1)
    if not torch.equal(pmodel.sconv_prefill(x1, w_first),
                       (shifted + xf).to(torch.bfloat16)):
        raise SystemExit(f"[t_b2 sconv] tap 0 is not the token {k-1} back")
    #   (c3) zero input -> exact zero output (proves no bias term)
    if not (pmodel.sconv_prefill(torch.zeros(1, T, C, device=DEV,
                                             dtype=torch.bfloat16), w)
            == 0).all():
        raise SystemExit("[t_b2 sconv] zero input gave nonzero output")

    #6. summary line = the test's green evidence
    a = anchor_re
    print(f"sconv ok: layer-0 frozen anchors k/v/attn/mlp rel err "
          f"{a['k_sconv']:.1e}/{a['v_sconv']:.1e}/{a['attn_sconv']:.1e}/"
          f"{a['mlp_sconv']:.1e} < 1e-2 (bit-exact "
          f"{anchor_bit['k_sconv']}/{anchor_bit['v_sconv']}/"
          f"{anchor_bit['attn_sconv']}/{anchor_bit['mlp_sconv']}); "
          f"native-module cross-check bit-exact 4 real + "
          f"{len(rand_cases)} random cases (T {{1,3,4,13,600}} x "
          f"C {{1024,2048,6144}}); window-{k} causality bit-checked, "
          f"last-tap-is-current + tap-0 orientation exact, zero-in zero-out "
          f"(no bias), fp32 module math (modeling_inkling.py:500-542)")


def t_attn_global():
    """B2.6: global-attention layer (idx 5) forward, prefill/no-cache form
    (model.attn_prefill + model.additive_causal_mask) vs the frozen ref
    slice. Gates:
      (a) frozen anchors — replay from the captured self_attn.in with the
          real layer-5 checkpoint weights: full path vs self_attn.out (the
          o_proj output; attn_sconv sits OUTSIDE the module, decoder forward
          modeling_inkling.py:584) plus every captured internal boundary
          (k/v_proj -> k/v_sconv, r_proj, q/k_norm, rel_logits_proj):
          rel err < 1e-2 (B2 header budget), bit-exactness reported;
      (b) mask — our additive_causal_mask must equal transformers'
          create_causal_mask output bit-for-bit on the exact call the ref
          run made (eager form 0/finfo.min, attention_mask=None,
          position_ids=arange — InklingTextModel.forward :693-705);
      (c) native-module cross-check — the ACTUAL transformers
          InklingAttention (layer_idx 5: global, 64Q/8KV heads, extent 1024,
          log scaling armed with floor 128000 — tau==1.0 at every tested
          position per B2.4, but the fp32 round-trip runs) instantiated
          in-process in eval mode with the ref dtype policy (bf16 module,
          fp32 sconv convs per _keep_in_fp32_modules_strict :610),
          torch.equal REQUIRED, on the real frozen input + random inputs
          T {1,29,600} with real weights and T {4,13} (incl. batch 2) with
          random weights;
      (d) structural — causality through the full path: perturbing token t
          leaves outputs before t bit-identical and changes t onward."""
    from transformers import AutoConfig
    from transformers.masking_utils import create_causal_mask
    from transformers.models.inkling.modeling_inkling import InklingAttention
    torch.manual_seed(0)
    F = torch.nn.functional
    tens, meta, _ = _load_refs()
    mc = pcfg.load_verified(MODEL_DIR)
    idx = loader.build_shard_index(MODEL_DIR)
    L = 5

    #1. layer-5 attention weights from shards, shapes pinned vs config
    base = f"model.llm.layers.{L}.attn."
    names = {"wq": "wq_du.weight", "wk": "wk_dv.weight", "wv": "wv_dv.weight",
             "wr": "wr_du.weight", "wo": "wo_ud.weight",
             "ksc": "k_sconv.weight", "vsc": "v_sconv.weight",
             "qn": "q_norm.weight", "kn": "k_norm.weight",
             "proj": "rel_logits_proj.proj"}
    w = {key: _disk(idx, base + suffix).to(DEV)
         for key, suffix in names.items()}
    qd, kvd = mc.g_q_heads * mc.head_dim, mc.g_kv_heads * mc.head_dim
    shapes = {"wq": (qd, mc.hidden), "wk": (kvd, mc.hidden),
              "wv": (kvd, mc.hidden), "wr": (mc.g_q_heads * mc.d_rel,
                                             mc.hidden),
              "wo": (mc.hidden, qd), "ksc": (kvd, 1, mc.sconv_k),
              "vsc": (kvd, 1, mc.sconv_k), "qn": (mc.head_dim,),
              "kn": (mc.head_dim,), "proj": (mc.d_rel, mc.rel_extent)}
    for key, s in shapes.items():
        assert w[key].shape == s and w[key].dtype == torch.bfloat16, (
            key, w[key].shape, w[key].dtype)

    #2. real text config for the native module + mask builder, eager like
    #   the ref run; layer 5 must be a global ("hybrid") layer
    tc = AutoConfig.from_pretrained(MODEL_DIR,
                                    trust_remote_code=True).text_config
    tc._attn_implementation = "eager"
    assert tc.layer_types[L] == "hybrid", tc.layer_types[L]
    assert tc.log_scaling_n_floor == mc.log_floor

    #3. gate (b): our mask vs the native builder, bitwise, on the exact
    #   ref-run call; eager form = {0, finfo.min} only
    x = tens[f"layers.{L}.self_attn.in"].to(DEV)
    B, Q = x.shape[0], x.shape[1]
    pos = torch.arange(Q, device=DEV)
    mask = pmodel.additive_causal_mask(pos, pos, x.dtype)
    nat_mask = create_causal_mask(config=tc, inputs_embeds=x,
                                  attention_mask=None, past_key_values=None,
                                  position_ids=pos.unsqueeze(0))
    if not (isinstance(nat_mask, torch.Tensor)
            and torch.equal(mask, nat_mask)):
        raise SystemExit("[t_b2 attn_global] additive_causal_mask != native "
                         "create_causal_mask (eager)")
    fmin = torch.finfo(x.dtype).min
    assert mask.shape == (1, 1, Q, Q) and ((mask == 0) | (mask == fmin)).all()

    def ours(wd, xx, msk, p):
        #4. our full path (model.attn_prefill), global-layer arm
        return pmodel.attn_prefill(
            xx, wd["wq"], wd["wk"], wd["wv"], wd["wr"], wd["wo"], wd["ksc"],
            wd["vsc"], wd["qn"], wd["kn"], wd["proj"], msk, p, p,
            mc.head_dim, is_global=True, alpha=mc.log_alpha,
            n_floor=mc.log_floor, eps=mc.rms_eps)

    #5. gate (a): frozen anchors — every captured internal boundary,
    #   recomputed as a chain from self_attn.in, then the full path
    kin, vin = F.linear(x, w["wk"]), F.linear(x, w["wv"])
    ksc_out = pmodel.sconv_prefill(kin, w["ksc"])
    vsc_out = pmodel.sconv_prefill(vin, w["vsc"])
    r_out = F.linear(x, w["wr"])
    pfx = f"layers.{L}.self_attn."
    inter = {
        "k_sconv.in": kin, "v_sconv.in": vin,
        "k_sconv.out": ksc_out, "v_sconv.out": vsc_out,
        "r_proj.out": r_out,
        "q_norm.out": pmodel.rmsnorm(
            F.linear(x, w["wq"]).view(B, Q, mc.g_q_heads, mc.head_dim),
            w["qn"], mc.rms_eps),
        "k_norm.out": pmodel.rmsnorm(
            ksc_out.view(B, Q, mc.g_kv_heads, mc.head_dim),
            w["kn"], mc.rms_eps),
        "rel_logits_proj.out": pmodel.rel_bias(
            r_out.view(B, Q, mc.g_q_heads, mc.d_rel), w["proj"], pos, pos),
    }
    n_bit_inter = 0
    for name, got_i in inter.items():
        want_i = tens[pfx + name].to(DEV)
        assert got_i.shape == want_i.shape, (name, got_i.shape)
        re_i = _rel_err(got_i, want_i)
        if re_i >= 1e-2:
            raise SystemExit(f"[t_b2 attn_global] internal anchor {name} "
                             f"rel err {re_i:.3e} >= 1e-2")
        n_bit_inter += torch.equal(got_i, want_i)
    got = ours(w, x, mask, pos)
    want = tens[pfx + "out"].to(DEV)
    assert got.shape == want.shape == (B, Q, mc.hidden), got.shape
    assert got.dtype == torch.bfloat16
    re_full = _rel_err(got, want)
    if re_full >= 1e-2:
        raise SystemExit(f"[t_b2 attn_global] full-path anchor rel err "
                         f"{re_full:.3e} >= 1e-2")
    bit_full = torch.equal(got, want)

    def build_native(wd):
        #6. the reference module itself: construct on GPU, eval mode, ref
        #   dtype policy (bf16 module, fp32 sconv convs), real/random weights
        with torch.device(DEV):
            mod = InklingAttention(tc, L)
        mod.eval()
        pnames = {n for n, _ in mod.named_parameters()}
        assert pnames == {"q_proj.weight", "k_proj.weight", "v_proj.weight",
                          "r_proj.weight", "o_proj.weight", "q_norm.weight",
                          "k_norm.weight", "rel_logits_proj.proj",
                          "k_sconv.conv1d.weight",
                          "v_sconv.conv1d.weight"}, pnames
        for sub in (mod.q_proj, mod.k_proj, mod.v_proj, mod.r_proj,
                    mod.o_proj, mod.q_norm, mod.k_norm, mod.rel_logits_proj):
            sub.to(torch.bfloat16)
        assert mod.k_sconv.conv1d.weight.dtype == torch.float32
        assert mod.k_sconv.conv1d.bias is None
        with torch.no_grad():
            mod.q_proj.weight.copy_(wd["wq"])
            mod.k_proj.weight.copy_(wd["wk"])
            mod.v_proj.weight.copy_(wd["wv"])
            mod.r_proj.weight.copy_(wd["wr"])
            mod.o_proj.weight.copy_(wd["wo"])
            mod.q_norm.weight.copy_(wd["qn"])
            mod.k_norm.weight.copy_(wd["kn"])
            mod.rel_logits_proj.proj.copy_(wd["proj"])
            mod.k_sconv.conv1d.weight.copy_(wd["ksc"].float())
            mod.v_sconv.conv1d.weight.copy_(wd["vsc"].float())
        assert not mod.is_sliding and mod.sliding_window is None
        assert mod.scaling == 1.0 / mc.head_dim
        assert mod.rel_extent == mc.rel_extent
        return mod

    #7. gate (c): torch.equal vs the native module, real + random weights
    w_rand = {key: torch.randn(shapes[key], device=DEV).to(torch.bfloat16)
              for key in shapes}
    mods = {"real": build_native(w), "rand": build_native(w_rand)}

    def rnd(b, t):
        return torch.randn(b, t, mc.hidden, device=DEV).to(torch.bfloat16)

    cases = (("real", x, "ref-x"), ("real", rnd(1, 1), "T1"),
             ("real", rnd(1, 29), "T29"), ("real", rnd(1, 600), "T600"),
             ("rand", rnd(2, 4), "B2T4"), ("rand", rnd(1, 13), "T13"))
    n_cross = el_cross = 0
    for wkey, xx, tag in cases:
        T = xx.shape[1]
        p = torch.arange(T, device=DEV)
        m = pmodel.additive_causal_mask(p, p, xx.dtype)
        got_c = ours(w if wkey == "real" else w_rand, xx, m, p)
        with torch.no_grad():
            nat, _ = mods[wkey](hidden_states=xx, attention_mask=m,
                                conv_mask=None, past_key_values=None)
        if not torch.equal(got_c, nat):
            raise SystemExit(f"[t_b2 attn_global] not bit-exact vs native "
                             f"module: {wkey}/{tag}")
        n_cross += 1
        el_cross += got_c.numel()

    #8. gate (d): causality through the full path — perturb one token
    t = 7
    x2 = x.clone()
    x2[0, t] += 1.0
    o2 = ours(w, x2, mask, pos)
    if not torch.equal(got[:, :t], o2[:, :t]):
        raise SystemExit("[t_b2 attn_global] non-causal: perturbing token "
                         f"{t} changed earlier outputs")
    if torch.equal(got[:, t:], o2[:, t:]):
        raise SystemExit(f"[t_b2 attn_global] perturbing token {t} had no "
                         f"effect from {t} on")

    #9. summary line = the test's green evidence
    print(f"attn_global ok: layer-5 frozen anchors — full path rel err "
          f"{re_full:.1e} < 1e-2 (bit-exact {bit_full}), {len(inter)} "
          f"internals (k/v_proj+sconv, r_proj, q/k_norm, rel_bias) < 1e-2, "
          f"bit-exact {n_bit_inter}/{len(inter)}; mask bit-equal native "
          f"create_causal_mask (eager 0/finfo.min); native-module "
          f"cross-check bit-exact {n_cross}/{len(cases)} cases "
          f"({el_cross} els; real+random weights, T {{1,4,13,29,600}}, "
          f"batch 2); causality bit-checked through full path; tau==1.0 at "
          f"all tested positions (scaling armed, floor {mc.log_floor})")


def t_attn_swa():
    """B2.7: SWA-attention layer (idx 0: window 512, 64Q/16KV heads, rel
    extent = window 512, NO log scaling — modeling_inkling.py:190-196,:254)
    forward, prefill/no-cache form, vs the frozen ref slice. Same
    model.attn_prefill as B2.6 — the SWA arm differs only in weights, the
    window mask, and is_global=False. Gates:
      (a) frozen anchors — replay from the captured layer-0 self_attn.in
          with the real checkpoint weights: full path vs self_attn.out plus
          every captured internal boundary, rel err < 1e-2, bit-exactness
          reported;
      (b) mask — our additive_causal_mask(window=512) must equal
          transformers' create_sliding_window_causal_mask output bit-for-bit
          on the exact call the ref run made (InklingTextModel.forward
          :694-703; layer 0 reads the "sliding_attention" entry, :709);
      (c) native-module cross-check — the ACTUAL transformers
          InklingAttention(layer_idx 0) in-process, ref dtype policy,
          torch.equal REQUIRED, real + random weights, T {1, 13, 600} +
          batch 2 — T 600 CROSSES the 512 window on both weight sets;
      (d) causality — perturbing token t leaves outputs before t
          bit-identical;
      (e) window cutoff — at T 600, a perturbation at token t must be
          INVISIBLE to queries >= t + window + sconv_k - 1 (the k/v sconv
          smears token t into keys/values t..t+3; the last query seeing
          key t+3 is t+3+511) and visible at the direct window edge
          t + window - 1 — both sides bit-checked."""
    from transformers import AutoConfig
    from transformers.masking_utils import create_sliding_window_causal_mask
    from transformers.models.inkling.modeling_inkling import InklingAttention
    torch.manual_seed(0)
    F = torch.nn.functional
    tens, meta, _ = _load_refs()
    mc = pcfg.load_verified(MODEL_DIR)
    idx = loader.build_shard_index(MODEL_DIR)
    L = 0

    #1. layer-0 attention weights from shards, shapes pinned vs config —
    #   16 KV heads and the (d_rel, window) rel bank are the SWA signature
    base = f"model.llm.layers.{L}.attn."
    names = {"wq": "wq_du.weight", "wk": "wk_dv.weight", "wv": "wv_dv.weight",
             "wr": "wr_du.weight", "wo": "wo_ud.weight",
             "ksc": "k_sconv.weight", "vsc": "v_sconv.weight",
             "qn": "q_norm.weight", "kn": "k_norm.weight",
             "proj": "rel_logits_proj.proj"}
    w = {key: _disk(idx, base + suffix).to(DEV)
         for key, suffix in names.items()}
    qd, kvd = mc.s_q_heads * mc.head_dim, mc.s_kv_heads * mc.head_dim
    shapes = {"wq": (qd, mc.hidden), "wk": (kvd, mc.hidden),
              "wv": (kvd, mc.hidden), "wr": (mc.s_q_heads * mc.d_rel,
                                             mc.hidden),
              "wo": (mc.hidden, qd), "ksc": (kvd, 1, mc.sconv_k),
              "vsc": (kvd, 1, mc.sconv_k), "qn": (mc.head_dim,),
              "kn": (mc.head_dim,), "proj": (mc.d_rel, mc.window)}
    for key, s in shapes.items():
        assert w[key].shape == s and w[key].dtype == torch.bfloat16, (
            key, w[key].shape, w[key].dtype)

    #2. real text config for the native module + mask builder, eager like
    #   the ref run; layer 0 must be a sliding ("hybrid_sliding") layer
    tc = AutoConfig.from_pretrained(MODEL_DIR,
                                    trust_remote_code=True).text_config
    tc._attn_implementation = "eager"
    assert tc.layer_types[L] == "hybrid_sliding", tc.layer_types[L]
    assert tc.sliding_window_size == mc.window

    #3. gate (b): our window mask vs the native builder, bitwise, on the
    #   exact ref-run call; eager form = {0, finfo.min} only
    x = tens[f"layers.{L}.self_attn.in"].to(DEV)
    B, Q = x.shape[0], x.shape[1]
    pos = torch.arange(Q, device=DEV)
    mask = pmodel.additive_causal_mask(pos, pos, x.dtype, window=mc.window)
    nat_mask = create_sliding_window_causal_mask(
        config=tc, inputs_embeds=x, attention_mask=None,
        past_key_values=None, position_ids=pos.unsqueeze(0))
    if not (isinstance(nat_mask, torch.Tensor)
            and torch.equal(mask, nat_mask)):
        raise SystemExit("[t_b2 attn_swa] additive_causal_mask(window) != "
                         "native create_sliding_window_causal_mask (eager)")
    fmin = torch.finfo(x.dtype).min
    assert mask.shape == (1, 1, Q, Q) and ((mask == 0) | (mask == fmin)).all()

    def ours(wd, xx, msk, p):
        #4. our full path (model.attn_prefill), SWA arm: no log scaling
        return pmodel.attn_prefill(
            xx, wd["wq"], wd["wk"], wd["wv"], wd["wr"], wd["wo"], wd["ksc"],
            wd["vsc"], wd["qn"], wd["kn"], wd["proj"], msk, p, p,
            mc.head_dim, is_global=False, eps=mc.rms_eps)

    #5. gate (a): frozen anchors — every captured internal boundary,
    #   recomputed as a chain from self_attn.in, then the full path
    kin, vin = F.linear(x, w["wk"]), F.linear(x, w["wv"])
    ksc_out = pmodel.sconv_prefill(kin, w["ksc"])
    vsc_out = pmodel.sconv_prefill(vin, w["vsc"])
    r_out = F.linear(x, w["wr"])
    pfx = f"layers.{L}.self_attn."
    inter = {
        "k_sconv.in": kin, "v_sconv.in": vin,
        "k_sconv.out": ksc_out, "v_sconv.out": vsc_out,
        "r_proj.out": r_out,
        "q_norm.out": pmodel.rmsnorm(
            F.linear(x, w["wq"]).view(B, Q, mc.s_q_heads, mc.head_dim),
            w["qn"], mc.rms_eps),
        "k_norm.out": pmodel.rmsnorm(
            ksc_out.view(B, Q, mc.s_kv_heads, mc.head_dim),
            w["kn"], mc.rms_eps),
        "rel_logits_proj.out": pmodel.rel_bias(
            r_out.view(B, Q, mc.s_q_heads, mc.d_rel), w["proj"], pos, pos),
    }
    n_bit_inter = 0
    for name, got_i in inter.items():
        want_i = tens[pfx + name].to(DEV)
        assert got_i.shape == want_i.shape, (name, got_i.shape)
        re_i = _rel_err(got_i, want_i)
        if re_i >= 1e-2:
            raise SystemExit(f"[t_b2 attn_swa] internal anchor {name} "
                             f"rel err {re_i:.3e} >= 1e-2")
        n_bit_inter += torch.equal(got_i, want_i)
    got = ours(w, x, mask, pos)
    want = tens[pfx + "out"].to(DEV)
    assert got.shape == want.shape == (B, Q, mc.hidden), got.shape
    assert got.dtype == torch.bfloat16
    re_full = _rel_err(got, want)
    if re_full >= 1e-2:
        raise SystemExit(f"[t_b2 attn_swa] full-path anchor rel err "
                         f"{re_full:.3e} >= 1e-2")
    bit_full = torch.equal(got, want)

    def build_native(wd):
        #6. the reference module itself: construct on GPU, eval mode, ref
        #   dtype policy (bf16 module, fp32 sconv convs), real/random weights
        with torch.device(DEV):
            mod = InklingAttention(tc, L)
        mod.eval()
        pnames = {n for n, _ in mod.named_parameters()}
        assert pnames == {"q_proj.weight", "k_proj.weight", "v_proj.weight",
                          "r_proj.weight", "o_proj.weight", "q_norm.weight",
                          "k_norm.weight", "rel_logits_proj.proj",
                          "k_sconv.conv1d.weight",
                          "v_sconv.conv1d.weight"}, pnames
        for sub in (mod.q_proj, mod.k_proj, mod.v_proj, mod.r_proj,
                    mod.o_proj, mod.q_norm, mod.k_norm, mod.rel_logits_proj):
            sub.to(torch.bfloat16)
        assert mod.k_sconv.conv1d.weight.dtype == torch.float32
        assert mod.k_sconv.conv1d.bias is None
        with torch.no_grad():
            mod.q_proj.weight.copy_(wd["wq"])
            mod.k_proj.weight.copy_(wd["wk"])
            mod.v_proj.weight.copy_(wd["wv"])
            mod.r_proj.weight.copy_(wd["wr"])
            mod.o_proj.weight.copy_(wd["wo"])
            mod.q_norm.weight.copy_(wd["qn"])
            mod.k_norm.weight.copy_(wd["kn"])
            mod.rel_logits_proj.proj.copy_(wd["proj"])
            mod.k_sconv.conv1d.weight.copy_(wd["ksc"].float())
            mod.v_sconv.conv1d.weight.copy_(wd["vsc"].float())
        assert mod.is_sliding and mod.sliding_window == mc.window
        assert mod.scaling == 1.0 / mc.head_dim
        assert mod.rel_extent == mc.window
        assert mod.num_key_value_heads == mc.s_kv_heads
        return mod

    #7. gate (c): torch.equal vs the native module, real + random weights;
    #   T 600 > 512 exercises the beyond-window arm on both weight sets
    w_rand = {key: torch.randn(shapes[key], device=DEV).to(torch.bfloat16)
              for key in shapes}
    mods = {"real": build_native(w), "rand": build_native(w_rand)}

    def rnd(b, t):
        return torch.randn(b, t, mc.hidden, device=DEV).to(torch.bfloat16)

    x600 = rnd(1, 600)
    cases = (("real", x, "ref-x"), ("real", rnd(1, 1), "T1"),
             ("real", x600, "T600"), ("rand", rnd(2, 4), "B2T4"),
             ("rand", rnd(1, 13), "T13"), ("rand", rnd(1, 600), "T600"))
    n_cross = el_cross = 0
    for wkey, xx, tag in cases:
        T = xx.shape[1]
        p = torch.arange(T, device=DEV)
        m = pmodel.additive_causal_mask(p, p, xx.dtype, window=mc.window)
        got_c = ours(w if wkey == "real" else w_rand, xx, m, p)
        with torch.no_grad():
            nat, _ = mods[wkey](hidden_states=xx, attention_mask=m,
                                conv_mask=None, past_key_values=None)
        if not torch.equal(got_c, nat):
            raise SystemExit(f"[t_b2 attn_swa] not bit-exact vs native "
                             f"module: {wkey}/{tag}")
        n_cross += 1
        el_cross += got_c.numel()

    #8. gate (d): causality on the frozen input — perturb one token
    t = 7
    x2 = x.clone()
    x2[0, t] += 1.0
    o2 = ours(w, x2, mask, pos)
    if not torch.equal(got[:, :t], o2[:, :t]):
        raise SystemExit("[t_b2 attn_swa] non-causal: perturbing token "
                         f"{t} changed earlier outputs")
    if torch.equal(got[:, t:], o2[:, t:]):
        raise SystemExit(f"[t_b2 attn_swa] perturbing token {t} had no "
                         f"effect from {t} on")

    #9. gate (e): window cutoff at T 600 through the full path. Perturbing
    #   x[tw] changes keys/values tw..tw+sconv_k-1 (window-4 sconv smear,
    #   B2.5); the last query that can see key tw+3 sits at distance
    #   window-1 past it, so queries >= tw + window + sconv_k - 1 must be
    #   BIT-identical, while the direct window edge tw + window - 1 (its
    #   key tw changed in-window) must move
    tw = 37
    cut = tw + mc.window + mc.sconv_k - 1   # 37 + 512 + 3 = 552 < 600
    p600 = torch.arange(600, device=DEV)
    m600 = pmodel.additive_causal_mask(p600, p600, x600.dtype,
                                       window=mc.window)
    base_o = ours(w, x600, m600, p600)
    xp = x600.clone()
    xp[0, tw] += 1.0
    pert_o = ours(w, xp, m600, p600)
    col_diff = (base_o != pert_o).any(-1)[0]   # per query position
    if col_diff[:tw].any():
        raise SystemExit("[t_b2 attn_swa] non-causal at T600: queries "
                         f"before {tw} changed")
    if not (col_diff[tw] and col_diff[tw + mc.window - 1]):
        raise SystemExit("[t_b2 attn_swa] perturbation invisible at token "
                         f"{tw} or window edge {tw + mc.window - 1}")
    if col_diff[cut:].any():
        raise SystemExit(f"[t_b2 attn_swa] window leak: queries >= {cut} "
                         f"(= {tw} + window + sconv_k - 1) saw token {tw}")

    #10. summary line = the test's green evidence
    print(f"attn_swa ok: layer-0 frozen anchors — full path rel err "
          f"{re_full:.1e} < 1e-2 (bit-exact {bit_full}), {len(inter)} "
          f"internals (k/v_proj+sconv, r_proj, q/k_norm, rel_bias ext512) "
          f"< 1e-2, bit-exact {n_bit_inter}/{len(inter)}; window mask "
          f"bit-equal native create_sliding_window_causal_mask (eager "
          f"0/finfo.min); native-module cross-check bit-exact "
          f"{n_cross}/{len(cases)} cases ({el_cross} els; real+random "
          f"weights, T {{1,4,13,600}} incl. 600>window, batch 2); "
          f"causality bit-checked; window cutoff exact at T600 (edge "
          f"{tw}+{mc.window - 1} moves, queries >= {tw}+{mc.window}+"
          f"{mc.sconv_k}-1 bit-identical, k/v sconv smear accounted)")


def t_gate():
    """B2.8: MoE gate/router (model.moe_gate) vs the frozen refs + the
    native module. The router (InklingTopkRouter, modeling_inkling.py
    :342-377) scores 258 slots (256 routed + 2 shared) in one linear;
    CHOICE ranks sigmoid(routed)+bias, WEIGHTS are bias-free normalized
    sigmoids over the 6 chosen + 2 shared logits, * route_scale *
    global_scale. Gates:
      (a) frozen anchors — all three sparse ref layers {2,5,6}: replay the
          captured mlp.in (the gate's input, InklingMoE.forward :421) with
          the real checkpoint gate weights (f32 bias/global_scale downcast
          to bf16 exactly as the ref load did — router not in
          _keep_in_fp32_modules_strict :610; captures are bf16): topk
          INDICES torch.equal REQUIRED (item: exact); topk_weights,
          routed_logits, shared_gammas rel err < 1e-2 REQUIRED,
          bit-exactness reported;
      (b) native-module cross-check — the ACTUAL transformers
          InklingTopkRouter in-process, bf16, torch.equal REQUIRED on ALL
          FOUR outputs: real weights (frozen input + random T {1,257}) and
          random weights incl. bias and global_scale != 1 (batch 2, T 600);
      (c) structural — per token the chosen 6 are distinct, in-range, and
          really the 6 LARGEST sigmoid+bias scores (min chosen >= max
          unchosen, no topk call); an independent fp64 normalized-sigmoid
          evaluation (NO bias term) reproduces weights+gammas < 1e-2 —
          proving the bias steers choice only; a +100 bias spike on an
          unchosen expert forces it into every token's top-6; weights+gammas
          sum to route_scale*global_scale (softmax sums to 1 pre-scale)."""
    from transformers import AutoConfig
    from transformers.models.inkling.modeling_inkling import InklingTopkRouter
    torch.manual_seed(0)
    F = torch.nn.functional
    tens, meta, _ = _load_refs()
    mc = pcfg.load_verified(MODEL_DIR)
    idx = loader.build_shard_index(MODEL_DIR)
    names4 = ("routed_logits", "topk_weights", "topk_indices",
              "shared_gammas")

    #1. real gate weights for the three sparse ref layers, shapes/dtypes
    #   pinned; bias/global_scale downcast f32 -> bf16 like the ref load
    layers = (2, 5, 6)
    W, down_exact = {}, True
    for L in layers:
        base = f"model.llm.layers.{L}.mlp.gate."
        wg = _disk(idx, base + "weight").to(DEV)
        b32 = _disk(idx, base + "bias").to(DEV)
        gs32 = _disk(idx, base + "global_scale").to(DEV)
        assert wg.shape == (mc.n_experts + mc.n_shared, mc.hidden), wg.shape
        assert wg.dtype == torch.bfloat16, wg.dtype
        assert b32.shape == (mc.n_experts,) and b32.dtype == torch.float32
        assert gs32.shape == (1,) and gs32.dtype == torch.float32
        b, gs = b32.to(torch.bfloat16), gs32.to(torch.bfloat16)
        down_exact &= (torch.equal(b.float(), b32)
                       and torch.equal(gs.float(), gs32))
        W[L] = (wg, b, gs)

    def ours(xx, wd):
        #2. our full router (model.moe_gate)
        return pmodel.moe_gate(xx, *wd, mc.topk, mc.n_shared, mc.route_scale)

    #3. gate (a) + (c): frozen anchors and structural pins per layer
    anchor_re, n_bit = {}, 0
    worst_re64 = worst_sum = 0.0
    keep = {}
    for L in layers:
        wg, b, gs = W[L]
        x = tens[f"layers.{L}.mlp.in"].to(DEV)
        rl, tw, ti, sg = ours(x, W[L])
        keep[L] = (x, rl, tw, ti, sg)
        want = {n: tens[f"layers.{L}.mlp.gate.{n}"].to(DEV) for n in names4}
        assert ti.dtype == torch.int64, ti.dtype
        assert rl.dtype == tw.dtype == sg.dtype == torch.bfloat16
        for n, g in zip(names4, (rl, tw, ti, sg)):
            assert g.shape == want[n].shape, (L, n, g.shape)
        #3a. indices EXACT (the item's gate)
        if not torch.equal(ti, want["topk_indices"]):
            bad = (ti != want["topk_indices"]).any(-1).sum().item()
            raise SystemExit(f"[t_b2 gate] layer {L}: topk_indices differ "
                             f"from ref on {bad}/{ti.shape[0]} tokens")
        #3b. weights/logits/gammas < 1e-2, bit-exactness counted
        res = {}
        for n, g in (("routed_logits", rl), ("topk_weights", tw),
                     ("shared_gammas", sg)):
            res[n] = _rel_err(g, want[n])
            if res[n] >= 1e-2:
                raise SystemExit(f"[t_b2 gate] layer {L} {n} rel err "
                                 f"{res[n]:.3e} >= 1e-2")
            n_bit += torch.equal(g, want[n])
        anchor_re[L] = res["topk_weights"]
        #3c. structural: distinct, in-range, and truly the 6 largest
        #    sigmoid+bias scores (direct comparison, no topk call)
        assert ti.min() >= 0 and ti.max() < mc.n_experts
        if not (torch.sort(ti, dim=-1)[0].diff(dim=-1) > 0).all():
            raise SystemExit(f"[t_b2 gate] layer {L}: duplicate expert "
                             f"in a token's top-{mc.topk}")
        sfc = (F.linear(x.reshape(-1, mc.hidden), wg
                        ).sigmoid()[..., :-mc.n_shared] + b)
        chosen_min = sfc.gather(-1, ti).amin(-1)
        unchosen_max = sfc.scatter(-1, ti, float("-inf")).amax(-1)
        if not (chosen_min >= unchosen_max).all():
            raise SystemExit(f"[t_b2 gate] layer {L}: chosen set is not "
                             f"the {mc.topk} largest scores")
        #3d. independent fp64 normalized-sigmoid weights, NO bias term —
        #    different math path (no logsigmoid/logsumexp), proves the
        #    correction bias never enters the weights
        lg8 = torch.cat([rl.gather(-1, ti), F.linear(
            x.reshape(-1, mc.hidden), wg)[..., -mc.n_shared:]], -1).double()
        s8 = lg8.sigmoid()
        w64 = (s8 / s8.sum(-1, keepdim=True)
               * mc.route_scale * gs.double())
        re64 = _rel_err(torch.cat([tw, sg], -1), w64)
        if re64 >= 1e-2:
            raise SystemExit(f"[t_b2 gate] layer {L}: fp64 "
                             f"normalized-sigmoid weights differ {re64:.3e}")
        worst_re64 = max(worst_re64, re64)
        #3e. weights+gammas sum to route_scale*global_scale per token
        tot = torch.cat([tw, sg], -1).float().sum(-1)
        want_tot = mc.route_scale * gs.float()
        dev = ((tot - want_tot).abs() / want_tot).max().item()
        if dev >= 2e-2:
            raise SystemExit(f"[t_b2 gate] layer {L}: slot sum off by "
                             f"{dev:.3e} from route_scale*global_scale")
        worst_sum = max(worst_sum, dev)

    #4. gate (c) spike: +100 bias on an expert NO token chose must force it
    #   into every token's top-6 (bias steers choice)
    x2, _, _, ti0, _ = keep[2]
    absent = next(j for j in range(mc.n_experts) if not (ti0 == j).any())
    b_spike = W[2][1].clone()
    b_spike[absent] += 100.0
    _, _, ti_s, _ = ours(x2, (W[2][0], b_spike, W[2][2]))
    if not (ti_s == absent).any(-1).all():
        raise SystemExit(f"[t_b2 gate] +100 bias spike on expert {absent} "
                         f"did not force it into every top-{mc.topk}")

    #5. gate (b): the native module in-process, real config, bf16 like the
    #   ref run; torch.equal on all four outputs
    tc = AutoConfig.from_pretrained(MODEL_DIR,
                                    trust_remote_code=True).text_config
    assert tc.n_routed_experts == mc.n_experts
    assert tc.n_shared_experts == mc.n_shared
    assert tc.num_experts_per_tok == mc.topk
    assert tc.route_scale == mc.route_scale

    def build_native(wg, bb, gg):
        with torch.device(DEV):
            mod = InklingTopkRouter(tc)
        mod.eval()
        pn = {n for n, _ in mod.named_parameters()}
        assert pn == {"weight", "global_scale",
                      "e_score_correction_bias"}, pn
        mod.to(torch.bfloat16)
        with torch.no_grad():
            mod.weight.copy_(wg)
            mod.e_score_correction_bias.copy_(bb)
            mod.global_scale.copy_(gg)
        assert mod.top_k == mc.topk and mod.route_scale == mc.route_scale
        assert mod.n_total_experts == mc.n_experts + mc.n_shared
        return mod

    def rnd(*shape):
        return torch.randn(*shape, device=DEV).to(torch.bfloat16)

    w_rand = ((torch.randn(mc.n_experts + mc.n_shared, mc.hidden,
                           device=DEV) * 0.02).to(torch.bfloat16),
              (torch.randn(mc.n_experts, device=DEV) * 0.5
               ).to(torch.bfloat16),
              torch.tensor([1.625], device=DEV, dtype=torch.bfloat16))
    mods = {"real": build_native(*W[2]), "rand": build_native(*w_rand)}
    wsets = {"real": W[2], "rand": w_rand}
    cases = (("real", keep[2][0], "ref-x"), ("real", rnd(1, 1, mc.hidden),
             "T1"), ("real", rnd(1, 257, mc.hidden), "T257"),
             ("rand", rnd(2, 13, mc.hidden), "B2T13"),
             ("rand", rnd(1, 600, mc.hidden), "T600"))
    n_cross = 0
    for wkey, xx, tag in cases:
        got = ours(xx, wsets[wkey])
        with torch.no_grad():
            nat = mods[wkey](xx)
        for n, g, na in zip(names4, got, nat):
            if not torch.equal(g, na):
                raise SystemExit(f"[t_b2 gate] not bit-exact vs native "
                                 f"module: {wkey}/{tag}/{n}")
        n_cross += 1

    #6. summary line = the test's green evidence
    print(f"gate ok: frozen anchors layers {{2,5,6}} — topk_indices EXACT "
          f"3/3 layers ({3 * ti0.shape[0] * mc.topk} slots), "
          f"weights rel err {anchor_re[2]:.1e}/{anchor_re[5]:.1e}/"
          f"{anchor_re[6]:.1e} < 1e-2 (+ routed_logits/shared_gammas, "
          f"bit-exact {n_bit}/9); native-module cross-check bit-exact "
          f"{n_cross}/{len(cases)} cases x all 4 outputs (real+random "
          f"weights incl. global_scale!=1, T {{1,13,257,600}}, batch 2); "
          f"top-6 = 6 largest sigmoid+bias scores (distinct, in-range), "
          f"fp64 bias-free normalized-sigmoid weights agree "
          f"{worst_re64:.1e} (bias steers choice only; +100 spike forces "
          f"expert into all top-6), slot sums = route_scale*global_scale "
          f"(max dev {worst_sum:.1e}); router all-bf16 like ref (f32 "
          f"bias/global_scale downcast exact={down_exact})")


def t_moe():
    """B2.9: MoE expert GEMMs + 2 shared (sink) experts (model.moe_experts /
    moe_shared / moe) match the ref layer out. Gates:
      (a) frozen anchors — all three sparse ref layers {2,5,6}: replay the
          captured mlp.experts.in (flat tokens) with the CAPTURED router
          outputs through moe_experts vs mlp.experts.out; the captured
          shared_experts.in + shared_gammas through moe_shared vs
          shared_experts.out; and the whole block from mlp.in (real gate
          weights, bf16 like the ref load — B2.8) through model.moe vs
          mlp.out (the item's 'ref layer out'): rel err < 1e-2 each (B2
          header budget), bit-exactness reported; composition pinned
          bitwise (moe == routed.view + shared). Layer-2 experts load bf16
          straight from shards; layers 5/6 are NVFP4 -> loader.dequant_nvfp4
          (B1.4, bit-exact vs vLLM) + the Interleave(dim=1) de-interleave
          proven vs HF's own conversion on layer-2 bf16 experts in B2.1 —
          the same repair the ref run itself used. The routed/full anchors
          are NOT expected bit-exact: the ref run's from_pretrained
          dispatched InklingExperts to "grouped_mm" (torch._grouped_mm —
          modeling_utils.py:2100 defaults it when no kwarg is passed and
          _grouped_mm_can_dispatch :2113 passes), same math, different
          accumulation order; provenance is PINNED in (b);
      (b) native-module cross-check — the ACTUAL transformers InklingMoE
          (InklingTopkRouter + InklingExperts + InklingSharedExperts)
          in-process, all bf16 like the ref run (experts are NOT in
          _keep_in_fp32_modules_strict :610; standalone dispatch falls back
          to the eager loop, integrations/moe.py:497-509), torch.equal
          REQUIRED: FIRST the capture-provenance pin — the native experts
          under forced "grouped_mm" dispatch must reproduce the frozen
          layer-2 experts.out BIT-exactly (proving the (a) deltas are
          exactly the kernel difference) — then eager-dispatch cases: real
          layer-2 weights on the frozen mlp.in + random inputs (T 57,
          batch 2), and a small random config (hidden 512, 8 experts,
          ffn 96, same top-6/2-shared) on random weights (T 600, batch 2);
      (c) structural — crafted one-chooser-per-expert routing reproduces
          each token's single-expert chain (batch-1 GEMMs) bit-exactly with
          the routing weight applied AFTER down_proj; zero routing weights /
          zero gammas give exact zero outputs; perturbing an expert NO
          token chose leaves routed out bit-identical (sparsity honesty);
          joint slot permutation is a bitwise no-op; an independent fp64
          replay (per-token python loop, no index_add/bmm) of routed,
          shared, and the full block agrees < 1e-2."""
    import copy
    from transformers import AutoConfig
    from transformers.models.inkling.modeling_inkling import InklingMoE
    torch.manual_seed(0)
    F = torch.nn.functional
    tens, meta, _ = _load_refs()
    mc = pcfg.load_verified(MODEL_DIR)
    idx = loader.build_shard_index(MODEL_DIR)
    hdr = loader.read_headers(idx)
    dm = loader.build_dtype_map(idx, hdr)

    def deint3(t):
        #1. Interleave(dim=1) on a 3D expert stack == the per-expert row
        #   de-interleave _deinterleave_rows (proven bitwise vs HF's own
        #   Interleave on layer-2 experts in B2.1)
        E, r, c = t.shape
        return t.reshape(E, r // 2, 2, c).transpose(1, 2).reshape(E, r, c)

    def load_expert_weights(L):
        #2. full expert stacks on the GPU in the module layout: gate_up
        #   [256, 2*ffn, hidden] de-interleaved rows, down [256, hidden,
        #   ffn]. Layer 2 is the bf16-stored MoE layer; layers 5/6 are
        #   NVFP4 packs -> B1.4 dequant per expert (bounded workspace),
        #   exactly like the ref run's own repair (t_ref step 7)
        base = f"model.llm.layers.{L}.mlp.experts."
        if hdr[base + "w13_weight"][0] == "BF16":
            w13 = _disk(idx, base + "w13_weight").to(DEV)
            gu = deint3(w13)
            assert torch.equal(gu[1], _deinterleave_rows(w13[1]))
            del w13
            w2 = _disk(idx, base + "w2_weight").to(DEV)
        else:
            gu = torch.empty(mc.n_experts, 2 * mc.expert_ffn, mc.hidden,
                             dtype=torch.bfloat16, device=DEV)
            w2 = torch.empty(mc.n_experts, mc.hidden, mc.expert_ffn,
                             dtype=torch.bfloat16, device=DEV)
            for wname, dst, deint in (("w13_weight", gu, True),
                                      ("w2_weight", w2, False)):
                packed = _disk(idx, base + wname)
                scale = _disk(idx, base + wname + ".scale")
                scale2 = _disk(idx, base + wname + ".scale2").to(DEV)
                for e in range(mc.n_experts):
                    deq = loader.dequant_nvfp4(
                        packed[e].to(DEV), scale[e].to(DEV), scale2[e],
                        dm.group_size)
                    dst[e] = _deinterleave_rows(deq) if deint else deq
        assert gu.shape == (mc.n_experts, 2 * mc.expert_ffn, mc.hidden)
        assert w2.shape == (mc.n_experts, mc.hidden, mc.expert_ffn)
        assert gu.dtype == w2.dtype == torch.bfloat16
        return gu, w2

    def load_shared_weights(L):
        #3. shared sink experts: shared_w13 interleaved rows ->
        #   Interleave(dim=1) + Chunk(dim=1) -> gate/up [2, ffn, hidden];
        #   shared_w2 = down [2, hidden, ffn] (conversion_mapping.py
        #   "inkling_mm_model")
        base = f"model.llm.layers.{L}.mlp.shared_experts."
        w13 = _disk(idx, base + "shared_w13_weight").to(DEV)
        assert w13.shape == (mc.n_shared, 2 * mc.expert_ffn, mc.hidden)
        g, u = deint3(w13).chunk(2, dim=1)
        wd = _disk(idx, base + "shared_w2_weight").to(DEV)
        assert wd.shape == (mc.n_shared, mc.hidden, mc.expert_ffn)
        return g.contiguous(), u.contiguous(), wd

    #4. gate (a): frozen anchors, layers {2,5,6}; keep layer 2 resident
    #   for the native cross-check + structural pins, free 5/6 after use
    layers = (2, 5, 6)
    anchor_re, n_bit = {}, 0
    L2W, keep2 = None, None
    for L in layers:
        gu, w2 = load_expert_weights(L)
        shg, shu, shd = load_shared_weights(L)
        gb = f"model.llm.layers.{L}.mlp.gate."
        wg = _disk(idx, gb + "weight").to(DEV)
        b = _disk(idx, gb + "bias").to(DEV).to(torch.bfloat16)
        gs = _disk(idx, gb + "global_scale").to(DEV).to(torch.bfloat16)

        ti = tens[f"layers.{L}.mlp.gate.topk_indices"].to(DEV)
        tw = tens[f"layers.{L}.mlp.gate.topk_weights"].to(DEV)
        sg = tens[f"layers.{L}.mlp.gate.shared_gammas"].to(DEV)
        xe = tens[f"layers.{L}.mlp.experts.in"].to(DEV)
        xs = tens[f"layers.{L}.mlp.shared_experts.in"].to(DEV)
        xm = tens[f"layers.{L}.mlp.in"].to(DEV)
        #4a. data-flow sanity: experts consume the flattened mlp input,
        #    shared experts the unflattened residuals (InklingMoE :418-424)
        assert ti.dtype == torch.int64 and xe.shape == (xm.shape[1], mc.hidden)
        assert torch.equal(xe, xm.view(-1, mc.hidden))
        assert torch.equal(xs, xm)

        got_r = pmodel.moe_experts(xe, gu, w2, ti, tw)
        got_s = pmodel.moe_shared(xs, shg, shu, shd, sg)
        got_m = pmodel.moe(xm, wg, b, gs, gu, w2, shg, shu, shd,
                           mc.topk, mc.n_shared, mc.route_scale)
        assert got_m.dtype == torch.bfloat16 and got_m.shape == xm.shape
        res = {}
        for name, g, wkey in (("experts", got_r, f"layers.{L}.mlp.experts.out"),
                              ("shared", got_s,
                               f"layers.{L}.mlp.shared_experts.out"),
                              ("full", got_m, f"layers.{L}.mlp.out")):
            want = tens[wkey].to(DEV)
            assert g.shape == want.shape, (L, name, g.shape)
            res[name] = _rel_err(g, want)
            if res[name] >= 1e-2:
                raise SystemExit(f"[t_b2 moe] layer {L} {name} rel err "
                                 f"{res[name]:.3e} >= 1e-2")
            n_bit += torch.equal(g, want)
        anchor_re[L] = res
        #4b. composition pin: full block == routed.view + shared, bitwise
        if not torch.equal(got_m, got_r.view(xm.shape) + got_s):
            raise SystemExit(f"[t_b2 moe] layer {L}: moe() != "
                             f"routed + shared bitwise")
        if L == 2:
            L2W = {"wg": wg, "b": b, "gs": gs, "gu": gu, "w2": w2,
                   "shg": shg, "shu": shu, "shd": shd}
            keep2 = (xe, xs, xm, ti, tw, sg, got_r, got_s, got_m)
        else:
            del gu, w2, shg, shu, shd, wg, b, gs, got_r, got_s, got_m
            torch.cuda.empty_cache()

    #5. gate (b): the native InklingMoE in-process, all-bf16 like the ref
    tc = AutoConfig.from_pretrained(MODEL_DIR,
                                    trust_remote_code=True).text_config
    assert tc.hidden_act == "silu"
    assert tc.moe_intermediate_size == mc.expert_ffn
    assert tc.n_routed_experts == mc.n_experts

    def build_native(cfg, wd):
        #5a. construct under a bf16 default dtype (a fp32 InklingMoE at
        #    full size would waste 54 GiB), eval, copy all 8 params
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device(DEV):
                mod = InklingMoE(cfg)
        finally:
            torch.set_default_dtype(old)
        mod.eval()
        pn = {n for n, _ in mod.named_parameters()}
        assert pn == {"gate.weight", "gate.global_scale",
                      "gate.e_score_correction_bias",
                      "experts.gate_up_proj", "experts.down_proj",
                      "shared_experts.gate_proj", "shared_experts.up_proj",
                      "shared_experts.down_proj"}, pn
        with torch.no_grad():
            mod.gate.weight.copy_(wd["wg"])
            mod.gate.e_score_correction_bias.copy_(wd["b"])
            mod.gate.global_scale.copy_(wd["gs"])
            mod.experts.gate_up_proj.copy_(wd["gu"])
            mod.experts.down_proj.copy_(wd["w2"])
            mod.shared_experts.gate_proj.copy_(wd["shg"])
            mod.shared_experts.up_proj.copy_(wd["shu"])
            mod.shared_experts.down_proj.copy_(wd["shd"])
        for _, p in mod.named_parameters():
            assert p.dtype == torch.bfloat16
        return mod

    def ours(xx, wd):
        return pmodel.moe(xx, wd["wg"], wd["b"], wd["gs"], wd["gu"],
                          wd["w2"], wd["shg"], wd["shu"], wd["shd"],
                          mc.topk, mc.n_shared, mc.route_scale)

    def rnd(*shape):
        return torch.randn(*shape, device=DEV).to(torch.bfloat16)

    #5b. real layer-2 weights. First the capture-provenance pin: the ref
    #    run's experts ran under "grouped_mm" dispatch (modeling_utils.py
    #    :2100), so the native module forced onto that path must reproduce
    #    the frozen experts.out bit-for-bit — proving the routed anchor
    #    deltas in (a) are exactly the grouped_mm-vs-eager kernel
    #    difference, not a semantics gap
    mod_real = build_native(tc, L2W)
    tc._experts_implementation = "grouped_mm"
    with torch.no_grad():
        out_gmm = mod_real.experts(keep2[0], keep2[3], keep2[4])
    tc._experts_implementation = None
    if not torch.equal(out_gmm, tens["layers.2.mlp.experts.out"].to(DEV)):
        raise SystemExit("[t_b2 moe] native grouped_mm dispatch does NOT "
                         "reproduce the frozen experts capture — provenance "
                         "assumption broken")
    #    then the eager-dispatch cross-checks: frozen input + random shapes
    n_cross = 0
    for xx, tag in ((keep2[2], "ref-x"), (rnd(1, 57, mc.hidden), "T57"),
                    (rnd(2, 5, mc.hidden), "B2T5")):
        got_c = ours(xx, L2W)
        with torch.no_grad():
            nat = mod_real(xx)
        if not torch.equal(got_c, nat):
            raise SystemExit(f"[t_b2 moe] not bit-exact vs native "
                             f"InklingMoE: real/{tag}")
        n_cross += 1
    del mod_real
    torch.cuda.empty_cache()

    #5c. small random config: different shapes exercise the same code
    tcs = copy.deepcopy(tc)
    tcs.hidden_size, tcs.n_routed_experts = 512, 8
    tcs.intermediate_size = tcs.moe_intermediate_size = 96
    ws = {"wg": (torch.randn(8 + mc.n_shared, 512, device=DEV) * 0.05
                 ).to(torch.bfloat16),
          "b": (torch.randn(8, device=DEV) * 0.5).to(torch.bfloat16),
          "gs": torch.tensor([1.625], device=DEV, dtype=torch.bfloat16),
          "gu": (torch.randn(8, 192, 512, device=DEV) * 0.1
                 ).to(torch.bfloat16),
          "w2": (torch.randn(8, 512, 96, device=DEV) * 0.1
                 ).to(torch.bfloat16),
          "shg": (torch.randn(mc.n_shared, 96, 512, device=DEV) * 0.1
                  ).to(torch.bfloat16),
          "shu": (torch.randn(mc.n_shared, 96, 512, device=DEV) * 0.1
                  ).to(torch.bfloat16),
          "shd": (torch.randn(mc.n_shared, 512, 96, device=DEV) * 0.1
                  ).to(torch.bfloat16)}
    mod_small = build_native(tcs, ws)
    for xx, tag in ((torch.randn(1, 600, 512, device=DEV
                                 ).to(torch.bfloat16), "T600"),
                    (torch.randn(2, 13, 512, device=DEV
                                 ).to(torch.bfloat16), "B2T13")):
        got_c = ours(xx, ws)
        with torch.no_grad():
            nat = mod_small(xx)
        if not torch.equal(got_c, nat):
            raise SystemExit(f"[t_b2 moe] not bit-exact vs native "
                             f"InklingMoE: small/{tag}")
        n_cross += 1
    del mod_small

    #6. gate (c): structural pins on the layer-2 anchors
    xe, xs, xm, ti, tw, sg, got_r, got_s, got_m = keep2
    gu, w2 = L2W["gu"], L2W["w2"]
    shg, shu, shd = L2W["shg"], L2W["shu"], L2W["shd"]
    T = xe.shape[0]
    #6a. zero routing weights / zero gammas -> exact zero outputs
    if not (pmodel.moe_experts(xe, gu, w2, ti, torch.zeros_like(tw))
            == 0).all():
        raise SystemExit("[t_b2 moe] zero routing weights gave nonzero out")
    if not (pmodel.moe_shared(xs, shg, shu, shd, torch.zeros_like(sg))
            == 0).all():
        raise SystemExit("[t_b2 moe] zero gammas gave nonzero shared out")
    #6b. one-chooser-per-expert routing: expert 6t serves ONLY token t
    #    (batch-1 GEMM), weight 1.5 on slot 0, 0 elsewhere -> each row must
    #    equal the manual single-expert chain with the weight applied
    #    AFTER down_proj, bit-for-bit
    ti_pin = torch.arange(T * mc.topk, device=DEV,
                          dtype=torch.int64).view(T, mc.topk)
    tw_pin = torch.zeros(T, mc.topk, device=DEV, dtype=torch.bfloat16)
    tw_pin[:, 0] = 1.5
    got_pin = pmodel.moe_experts(xe, gu, w2, ti_pin, tw_pin)
    for t in (0, 7, T - 1):
        e = t * mc.topk
        g1, u1 = F.linear(xe[t:t + 1], gu[e]).chunk(2, dim=-1)
        exp = (F.linear(F.silu(g1) * u1, w2[e]) * tw_pin[t, 0]
               ).to(torch.bfloat16)
        if not torch.equal(got_pin[t:t + 1], exp):
            raise SystemExit(f"[t_b2 moe] one-chooser pin: token {t} != "
                             f"manual expert-{e} chain (weight after down)")
    #6c. perturbing an expert NO token chose must not move routed out
    hit = set(ti.flatten().tolist())
    absent = next(j for j in range(mc.n_experts) if j not in hit)
    saved = gu[absent].clone()
    with torch.no_grad():
        gu[absent] = torch.randn_like(gu[absent].float()).to(torch.bfloat16)
    if not torch.equal(pmodel.moe_experts(xe, gu, w2, ti, tw), got_r):
        raise SystemExit(f"[t_b2 moe] perturbing unchosen expert {absent} "
                         f"moved routed out")
    with torch.no_grad():
        gu[absent] = saved
    assert torch.equal(gu[absent], saved)
    #6d. joint slot permutation is a no-op (only (expert, weight) pairs
    #    matter, not slot positions)
    perm = torch.randperm(mc.topk, device=DEV)
    if not torch.equal(
            pmodel.moe_experts(xe, gu, w2, ti[:, perm], tw[:, perm]), got_r):
        raise SystemExit("[t_b2 moe] slot permutation changed routed out")
    #6e. independent fp64 replay: per-token loop, explicit weighted sum
    #    over the 6 chosen experts (weight after down), gammas before down
    #    on the shared pair, no index_add_/bmm anywhere
    x64 = xe.double()
    r64 = torch.zeros(T, mc.hidden, dtype=torch.float64, device=DEV)
    for t in range(T):
        for s in range(mc.topk):
            e = int(ti[t, s])
            g1 = gu[e, :mc.expert_ffn].double() @ x64[t]
            u1 = gu[e, mc.expert_ffn:].double() @ x64[t]
            r64[t] += tw[t, s].double() * (
                w2[e].double() @ (F.silu(g1) * u1))
    sh64 = torch.zeros_like(r64)
    xs64 = xs.double().reshape(-1, mc.hidden)
    for s in range(mc.n_shared):
        g1 = xs64 @ shg[s].double().T
        u1 = xs64 @ shu[s].double().T
        sh64 += (F.silu(g1) * u1 * sg[:, s].double()[:, None]
                 ) @ shd[s].double().T
    re_r64 = _rel_err(got_r, r64)
    re_s64 = _rel_err(got_s, sh64)
    re_m64 = _rel_err(got_m.view(-1, mc.hidden), r64 + sh64)
    if max(re_r64, re_s64, re_m64) >= 1e-2:
        raise SystemExit(f"[t_b2 moe] fp64 replay disagrees: routed "
                         f"{re_r64:.3e} shared {re_s64:.3e} full {re_m64:.3e}")

    #7. summary line = the test's green evidence
    a = anchor_re
    print(f"moe ok: frozen anchors layers {{2,5,6}} — experts/shared/"
          f"full-block rel err "
          f"{a[2]['experts']:.1e}/{a[2]['shared']:.1e}/{a[2]['full']:.1e} | "
          f"{a[5]['experts']:.1e}/{a[5]['shared']:.1e}/{a[5]['full']:.1e} | "
          f"{a[6]['experts']:.1e}/{a[6]['shared']:.1e}/{a[6]['full']:.1e} "
          f"< 1e-2 (bit-exact {n_bit}/9; composition routed+shared bitwise "
          f"3/3; layers 5/6 via B1.4 nvfp4 dequant + de-interleave); "
          f"capture provenance PINNED: native grouped_mm dispatch (the ref "
          f"run's path, modeling_utils.py:2100) reproduces frozen "
          f"experts.out bit-exactly; native InklingMoE eager cross-check "
          f"bit-exact {n_cross}/5 cases "
          f"(real layer-2 weights T {{13,57}} batch 2 + random small-config "
          f"T {{13,600}}); pins: one-chooser batch-1 chain bit-exact "
          f"(weight AFTER down_proj), zero-weights/zero-gammas exact zeros, "
          f"unchosen-expert perturbation invisible, slot-permutation no-op; "
          f"fp64 replay agrees routed/shared/full "
          f"{re_r64:.1e}/{re_s64:.1e}/{re_m64:.1e} < 1e-2")


def _todo(item):
    def f():
        raise SystemExit(f"[t_b2] not implemented — that is item {item}'s job")
    return f


def main():
    #1. dispatch on subcommand; unimplemented ones fail loud with their item
    cmds = {"ref": t_ref,
            "embed": t_embed, "rmsnorm": t_rmsnorm,
            "relbias": t_relbias, "sconv": t_sconv,
            "attn_global": t_attn_global, "attn_swa": t_attn_swa,
            "gate": t_gate, "moe": t_moe,
            "dense": _todo("B2.10"), "logits": _todo("B2.11")}
    usage = f"usage: python -m engine.pyengine.tests.t_b2 {{{'|'.join(cmds)}}}"
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        raise SystemExit(usage)
    cmds[sys.argv[1]]()


if __name__ == "__main__":
    main()
