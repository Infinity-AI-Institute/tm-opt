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


def _todo(item):
    def f():
        raise SystemExit(f"[t_b2] not implemented — that is item {item}'s job")
    return f


def main():
    #1. dispatch on subcommand; unimplemented ones fail loud with their item
    cmds = {"ref": t_ref,
            "embed": t_embed, "rmsnorm": _todo("B2.3"),
            "relbias": _todo("B2.4"), "sconv": _todo("B2.5"),
            "attn_global": _todo("B2.6"), "attn_swa": _todo("B2.7"),
            "gate": _todo("B2.8"), "moe": _todo("B2.9"),
            "dense": _todo("B2.10"), "logits": _todo("B2.11")}
    usage = f"usage: python -m engine.pyengine.tests.t_b2 {{{'|'.join(cmds)}}}"
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        raise SystemExit(usage)
    cmds[sys.argv[1]]()


if __name__ == "__main__":
    main()
