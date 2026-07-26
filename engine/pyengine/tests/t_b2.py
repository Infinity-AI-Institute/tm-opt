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


def _todo(item):
    def f():
        raise SystemExit(f"[t_b2] not implemented — that is item {item}'s job")
    return f


def main():
    #1. dispatch on subcommand; unimplemented ones fail loud with their item
    cmds = {"ref": t_ref,
            "embed": t_embed, "rmsnorm": t_rmsnorm,
            "relbias": t_relbias, "sconv": _todo("B2.5"),
            "attn_global": _todo("B2.6"), "attn_swa": _todo("B2.7"),
            "gate": _todo("B2.8"), "moe": _todo("B2.9"),
            "dense": _todo("B2.10"), "logits": _todo("B2.11")}
    usage = f"usage: python -m engine.pyengine.tests.t_b2 {{{'|'.join(cmds)}}}"
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        raise SystemExit(usage)
    cmds[sys.argv[1]]()


if __name__ == "__main__":
    main()
