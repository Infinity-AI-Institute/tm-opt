"""B1 acceptance tests, one subcommand per PROGRESS.md item.
Run from repo root: python -m engine.pyengine.tests.t_b1 <subcommand>"""
import json
import pathlib
import sys

from safetensors import safe_open

from engine.pyengine import config as pcfg
from engine.pyengine import loader

MODEL_DIR = "/workspace/models/inkling-nvfp4"


def t_index():
    #1. build the index (loader's own fail-loud checks run inside)
    idx = loader.build_shard_index(MODEL_DIR)

    #2. enumeration shape: 33 model shards + mtp.safetensors = 34 files
    assert len(idx.shard_files) == loader.N_MODEL_SHARDS + 1, idx.shard_files
    assert loader.MTP_SHARD in idx.shard_files

    #3. ground truth: each shard's safetensors header must list exactly the
    #   tensors the index assigns to it (catches stale index / dup entries)
    total = 0
    for fname in idx.shard_files:
        expected = set(idx.tensors_in_shard(fname))
        with safe_open(str(idx.model_dir / fname), framework="pt") as f:
            actual = set(f.keys())
        if actual != expected:
            raise SystemExit(
                f"[t_b1 index] {fname}: header vs index mismatch; "
                f"header-only={sorted(actual - expected)[:5]} "
                f"index-only={sorted(expected - actual)[:5]}")
        total += len(actual)
    assert total == len(idx.tensor_to_shard)

    #4. spot checks later items rely on: embed resolves; mtp shard non-empty
    assert idx.shard_path("model.llm.embed.weight").is_file()
    n_mtp = len(idx.tensors_in_shard(loader.MTP_SHARD))
    assert n_mtp > 0, "no tensors mapped to mtp.safetensors"

    #5. summary line = the test's green evidence
    print(f"shard index ok: {len(idx.shard_files)} files "
          f"({loader.N_MODEL_SHARDS} model + {loader.MTP_SHARD}), "
          f"{total} tensors mapped ({n_mtp} in mtp), headers match index")


def t_census():
    #1. inputs: shard index, per-tensor (dtype, shape) from headers, verified config
    idx = loader.build_shard_index(MODEL_DIR)
    meta = loader.read_headers(idx)
    mc = pcfg.load_verified(MODEL_DIR)
    H, HD, K = mc.hidden, mc.head_dim, mc.sconv_k

    #2. layer taxonomy from CONFIG, not from tensor names:
    #   - global/SWA: complement of config.json local_layer_ids must equal the
    #     derived {5,11,...,65} (11 global, 55 SWA)
    #   - dense: dense_mlp_idx is a COUNT of leading dense layers, i.e. layers
    #     with id < 2 are dense — NOT "layer 2 is dense". Semantics per vLLM
    #     inkling: vllm/models/inkling/nvidia/model.py:164
    #     (`layer_id < config.dense_mlp_idx`); confirmed by this census below.
    raw = json.loads((pathlib.Path(MODEL_DIR) / "config.json").read_text())
    local_ids = set(raw["text_config"]["local_layer_ids"])
    global_ids = set(range(mc.num_layers)) - local_ids
    assert global_ids == set(mc.global_layers), sorted(global_ids)
    assert len(local_ids) == 55 and len(global_ids) == 11
    dense_ids = set(range(mc.dense_idx))
    moe_ids = set(range(mc.num_layers)) - dense_ids

    #3. expected suffix -> (dtype, shape) for one layer, from config shapes
    def expect_layer(i, quantized):
        kvd = (mc.s_kv_heads if i in local_ids else mc.g_kv_heads) * HD
        e = {
            "attn_norm.weight": ("BF16", (H,)),
            "mlp_norm.weight": ("BF16", (H,)),
            "attn_sconv.weight": ("BF16", (H, 1, K)),
            "mlp_sconv.weight": ("BF16", (H, 1, K)),
            "attn.q_norm.weight": ("BF16", (HD,)),
            "attn.k_norm.weight": ("BF16", (HD,)),
            "attn.wq_du.weight": ("BF16", (mc.g_q_heads * HD, H)),
            "attn.wo_ud.weight": ("BF16", (H, mc.g_q_heads * HD)),
            "attn.wk_dv.weight": ("BF16", (kvd, H)),
            "attn.wv_dv.weight": ("BF16", (kvd, H)),
            "attn.k_sconv.weight": ("BF16", (kvd, 1, K)),
            "attn.v_sconv.weight": ("BF16", (kvd, 1, K)),
            #3a. wr_du is 1024 rows on BOTH layer types (== rel_extent; exact
            #    role of the 1024 is pinned down in B2.4, shape asserted here)
            "attn.wr_du.weight": ("BF16", (1024, H)),
            #3b. rel-logits table spans the reachable rel-pos range: window for
            #    SWA layers, rel_extent for global layers
            "attn.rel_logits_proj.proj": (
                "BF16", (mc.d_rel, mc.window if i in local_ids else mc.rel_extent)),
        }
        if i in dense_ids:
            e |= {
                "mlp.w13_dn.weight": ("BF16", (2 * mc.dense_ffn, H)),  #fused gate+up
                "mlp.w2_md.weight": ("BF16", (H, mc.dense_ffn)),
                "mlp.global_scale": ("BF16", (1,)),
            }
            return e
        E, F = mc.n_experts, mc.expert_ffn
        e |= {
            #3c. gate scores routed + shared experts (shared_expert_sink) -> 258 rows
            "mlp.gate.weight": ("BF16", (E + mc.n_shared, H)),
            "mlp.gate.bias": ("F32", (E,)),
            "mlp.gate.global_scale": ("F32", (1,)),
            "mlp.shared_experts.shared_w13_weight": ("BF16", (mc.n_shared, 2 * F, H)),
            "mlp.shared_experts.shared_w2_weight": ("BF16", (mc.n_shared, H, F)),
        }
        for w, (od, idim) in {"w13_weight": (2 * F, H), "w2_weight": (H, F)}.items():
            if quantized:
                #3d. NVFP4 pack: U8 holds 2 fp4/byte on the input dim; one
                #    F8_E4M3 block scale per 16 inputs; F32 per-expert scale2
                e |= {
                    f"mlp.experts.{w}": ("U8", (E, od, idim // 2)),
                    f"mlp.experts.{w}.scale": ("F8_E4M3", (E, od, idim // 16)),
                    f"mlp.experts.{w}.scale2": ("F32", (E,)),
                    f"mlp.experts.{w}.input_amax": ("BF16", (1,)),
                    f"mlp.experts.{w}.original_shape": ("I64", (3,)),
                }
            else:
                e |= {f"mlp.experts.{w}": ("BF16", (E, od, idim))}
        return e

    #4. exact per-layer comparison (no missing, no extra, dtype+shape equal);
    #   quantized-vs-bf16 form detected per layer, consistency enforced by the
    #   exact match (which layers SHOULD be quantized is B1.3's exclude-list job)
    n_layer_tensors, bf16_moe = 0, []
    for i in range(mc.num_layers):
        prefix = f"model.llm.layers.{i}."
        actual = {n[len(prefix):]: v for n, v in meta.items() if n.startswith(prefix)}
        quantized = f"{prefix}mlp.experts.w13_weight.scale" in meta
        expected = expect_layer(i, quantized)
        if actual != expected:
            miss = sorted(set(expected) - set(actual))[:5]
            extra = sorted(set(actual) - set(expected))[:5]
            diff = {k: (actual[k], expected[k])
                    for k in expected if k in actual and actual[k] != expected[k]}
            raise SystemExit(f"[t_b1 census] layer {i}: missing={miss} "
                             f"extra={extra} wrong={dict(list(diff.items())[:5])}")
        if i in moe_ids and not quantized:
            bf16_moe.append(i)
        n_layer_tensors += len(actual)

    #5. non-layer LLM tensors: exact set + shapes
    non_layer = {
        "model.llm.embed.weight": ("BF16", (mc.vocab, H)),
        "model.llm.embed_norm.weight": ("BF16", (H,)),
        "model.llm.norm.weight": ("BF16", (H,)),
        "model.llm.unembed.weight": ("BF16", (mc.vocab, H)),
    }
    for name, exp in non_layer.items():
        assert meta.get(name) == exp, (name, meta.get(name), exp)

    #6. MTP: exactly 8 draft layers x 20 tensors in mtp.safetensors
    #   (per-tensor shape checks land with the MTP implementation item)
    mtp_names = idx.tensors_in_shard(loader.MTP_SHARD)
    mtp_layers = {}
    for n in mtp_names:
        lid = int(n.split(".layers.")[1].split(".")[0])
        mtp_layers[lid] = mtp_layers.get(lid, 0) + 1
    assert sorted(mtp_layers) == list(range(mc.mtp_layers)), sorted(mtp_layers)
    assert all(c == 20 for c in mtp_layers.values()), mtp_layers

    #7. every remaining tensor must be multimodal (out of text-only v1 scope:
    #   counted so the partition covers the checkpoint, otherwise ignored)
    accounted = (set(non_layer) | set(mtp_names)
                 | {n for n in meta if n.startswith("model.llm.layers.")})
    mm = sorted(set(meta) - accounted)
    bad_mm = [n for n in mm if not n.startswith(("model.audio.", "model.visual."))]
    assert not bad_mm, bad_mm
    total = n_layer_tensors + len(non_layer) + len(mtp_names) + len(mm)
    assert total == len(meta), (total, len(meta))

    #8. summary line = the test's green evidence
    print(f"census ok: {mc.num_layers} layers = {len(dense_ids)} dense{sorted(dense_ids)} "
          f"(ffn {mc.dense_ffn}) + {len(moe_ids)} moe ({mc.n_experts}x ffn{mc.expert_ffn} "
          f"+{mc.n_shared} shared; {len(moe_ids) - len(bf16_moe)} nvfp4-packed, "
          f"{len(bf16_moe)} bf16 @{bf16_moe}); attn {len(global_ids)} global + "
          f"{len(local_ids)} swa (w{mc.window}, {mc.s_kv_heads}kv); all shapes match "
          f"config; totals {n_layer_tensors} layer + {len(non_layer)} llm + "
          f"{len(mm)} multimodal(ignored) + {len(mtp_names)} mtp "
          f"({mc.mtp_layers}x20) = {len(meta)}")


def t_dtypes():
    #1. inputs: index, headers, verified config; build_dtype_map's own
    #   exclude-list reconciliation (both directions, whole checkpoint,
    #   every exclude entry matched) runs fail-loud inside the builder
    idx = loader.build_shard_index(MODEL_DIR)
    meta = loader.read_headers(idx)
    mc = pcfg.load_verified(MODEL_DIR)
    dm = loader.build_dtype_map(idx, meta)

    #2. the item's named bf16 set, re-asserted independently of the builder:
    #   embeds / embed_norm / final norm / unembed + layer-0 attention
    for n in ("model.llm.embed.weight", "model.llm.embed_norm.weight",
              "model.llm.norm.weight", "model.llm.unembed.weight"):
        assert dm.plain.get(n) == "BF16" and dm.is_excluded(n), n
    l0_attn = [n for n in meta if n.startswith("model.llm.layers.0.attn")]
    assert l0_attn and all(dm.plain[n] == "BF16" for n in l0_attn)

    #3. stronger checkpoint fact (flagged in B1.2, grounded here by the
    #   exclude list): attention is bf16 on ALL 66 layers, not just layer 0
    attn = [n for n in meta
            if n.startswith("model.llm.layers.") and ".attn" in n]
    attn_layers = {int(n.split("layers.")[1].split(".")[0]) for n in attn}
    assert attn_layers == set(range(mc.num_layers))
    assert all(dm.plain[n] == "BF16" for n in attn)

    #4. NVFP4 set is EXACTLY routed-expert w13/w2 of MoE layers 3..65; the
    #   first MoE layer (id 2 = dense_idx) is the exclude list's one MoE
    #   exception and stays plain bf16
    expect_packed = {f"model.llm.layers.{i}.mlp.experts.{w}"
                     for i in range(mc.dense_idx + 1, mc.num_layers)
                     for w in ("w13_weight", "w2_weight")}
    assert dm.packed == expect_packed, (
        sorted(dm.packed ^ expect_packed)[:5])
    for w in ("w13_weight", "w2_weight"):
        n = f"model.llm.layers.{mc.dense_idx}.mlp.experts.{w}"
        assert dm.plain.get(n) == "BF16" and dm.is_excluded(n), n

    #5. plain dtype budget pins the whole partition: the only plain F32s are
    #   the 64 MoE gates' bias + global_scale, the rest is bf16; plain +
    #   5-tensor packs (base + 4 companions) must cover the checkpoint
    f32 = sorted(n for n, d in dm.plain.items() if d == "F32")
    n_moe = mc.num_layers - mc.dense_idx
    assert len(f32) == 2 * n_moe and all(
        n.endswith((".mlp.gate.bias", ".mlp.gate.global_scale"))
        for n in f32), f32[:5]
    assert set(dm.plain.values()) == {"BF16", "F32"}
    assert len(dm.plain) + 5 * len(dm.packed) == len(meta)

    #6. mtp.safetensors: outside the exclude list's reach (builder enforces),
    #   all 8x20 draft tensors load bf16
    mtp = [n for n in dm.plain if n.startswith("model.mtp.")]
    assert len(mtp) == mc.mtp_layers * 20
    assert all(dm.plain[n] == "BF16" for n in mtp)

    #7. summary line = the test's green evidence
    n_bf16 = sum(1 for d in dm.plain.values() if d == "BF16")
    print(f"dtypes ok: exclude list ({len(dm.exclude_modules)} literal entries, "
          f"all matched) reconciles exactly: nvfp4 = {len(dm.packed)} expert "
          f"weights (layers {mc.dense_idx + 1}-{mc.num_layers - 1} x w13/w2, "
          f"U8 + scale/scale2/amax/shape, group {dm.group_size}); plain "
          f"{len(dm.plain)} ({n_bf16} bf16 + {len(f32)} f32 gate bias/"
          f"global_scale); attn bf16 on ALL {mc.num_layers} layers; "
          f"layer-{mc.dense_idx} experts bf16; mtp {len(mtp)} bf16 "
          f"(outside exclude scope)")


def t_dequant():
    #1. inputs: index/headers/config/dtype map (their own checks run inside);
    #   the item's ONE expert weight = w13 of the first quantized MoE layer
    #   (layer 3; representative of all 126 packs, same layout by B1.3)
    import torch
    idx = loader.build_shard_index(MODEL_DIR)
    meta = loader.read_headers(idx)
    mc = pcfg.load_verified(MODEL_DIR)
    dm = loader.build_dtype_map(idx, meta)
    base = f"model.llm.layers.{mc.dense_idx + 1}.mlp.experts.w13_weight"
    assert base in dm.packed, base

    #2. read an expert slice (first/middle/last) of base + block scale, and
    #   the full tiny companions; tensors resolve shards independently (the
    #   base and its companions need not share a shard file)
    experts = [0, 7, mc.n_experts - 1]

    def read(name, sel=None):
        with safe_open(str(idx.shard_path(name)), framework="pt") as f:
            if sel is None:
                return f.get_tensor(name)
            sl = f.get_slice(name)
            return torch.cat([sl[i:i + 1] for i in sel])

    dev = "cuda:0"  # first visible device = physical GPU 4 (vLLM owns 0-3)
    u8 = read(base, experts).to(dev)
    sc = read(base + ".scale", experts).to(dev)
    s2 = read(base + ".scale2")[experts].to(dev)
    orig_shape = tuple(read(base + ".original_shape").tolist())

    #3. checkpoint's own confirmation of the pack geometry: original_shape
    #   gives the unpacked dims -> input dim really is 2 fp4 per byte
    E, od, pk = meta[base][1]
    assert orig_shape == (E, od, pk * 2), (orig_shape, meta[base][1])

    #4. our dequant (loader.dequant_nvfp4, pure torch): 3D per-expert-scale2
    #   arm and 2D scalar-scale2 arm
    ours = loader.dequant_nvfp4(u8, sc, s2, dm.group_size, torch.bfloat16)
    ours_2d = loader.dequant_nvfp4(u8[0], sc[0], s2[0], dm.group_size,
                                   torch.bfloat16)

    #5. vLLM's dequant of the SAME tensors — the reference the item names:
    #   dequantize_to_dtype (vllm/model_executor/layers/quantization/utils/
    #   nvfp4_emulation_utils.py:346) is vLLM's weight-dequant entry point
    #   (run_nvfp4_emulations calls it at :484-491); swizzle=False because
    #   checkpoint scales are linear (see loader.dequant_nvfp4 docstring)
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils \
        import dequantize_to_dtype
    ref = dequantize_to_dtype(u8, sc, s2, torch.bfloat16, dm.group_size,
                              swizzle=False)
    ref_2d = dequantize_to_dtype(u8[0], sc[0], s2[0], torch.bfloat16,
                                 dm.group_size, swizzle=False)

    #6. match must be exact: same f32 ops in the same association, so the
    #   bf16 results are bit-identical, not merely close
    for name, a, b in (("3d", ours, ref), ("2d", ours_2d, ref_2d),
                       ("2d-vs-3d[0]", ours_2d, ours[0])):
        if not torch.equal(a, b):
            d = (a.float() - b.float()).abs()
            raise SystemExit(f"[t_b1 dequant] {name} mismatch: "
                             f"{(a != b).sum().item()} elems differ, "
                             f"max abs diff {d.max().item():.3e}")

    #7. sanity anchored to the ModelOpt invariant scale2 = amax/(6*448):
    #   per expert, max|deq| <= 6*448*scale2 (block scales are clamped to
    #   fp8 max 448) and reaches ~amax (fp8 rounding slack); values finite,
    #   not degenerate-zero
    assert torch.isfinite(ours.float()).all()
    ratios = []
    for j, e in enumerate(experts):
        cap = 6.0 * 448.0 * float(s2[j])
        m = ours[j].float().abs().max().item()
        assert 0.5 * cap < m <= cap * (1 + 1e-4), (e, m, cap)
        ratios.append(m / cap)
    nz = (ours != 0).float().mean().item()
    assert nz > 0.5, f"degenerate dequant: only {nz:.1%} nonzero"

    #8. summary line = the test's green evidence
    print(f"dequant ok: {base} experts {experts} + 2D arm, shape "
          f"{tuple(ours.shape)}: bit-exact vs vLLM dequantize_to_dtype("
          f"swizzle=False) in bf16; e2m1 low-nibble-first, f8e4m3 block "
          f"scale /{dm.group_size} * f32 scale2 (amax/2688, no reciprocal); "
          f"max|deq|/(6*448*scale2) = {[f'{r:.3f}' for r in ratios]}, "
          f"nonzero {nz:.1%}")


def t_plan():
    #1. inputs: index/headers/config/dtype map, then the plan (builder's own
    #   fail-loud checks — total classification, divisibility, whole-head
    #   slices, pack coherence — run inside build_shard_plan)
    idx = loader.build_shard_index(MODEL_DIR)
    meta = loader.read_headers(idx)
    mc = pcfg.load_verified(MODEL_DIR)
    dm = loader.build_dtype_map(idx, meta)
    plan = loader.build_shard_plan(meta, dm, mc, tp=4)

    #2. total partition: every checkpoint tensor is placed; the skipped set
    #   is EXACTLY the multimodal family (text-only v1 scope), nothing else
    assert set(plan.placement) == set(meta)
    skipped = {n for n, (k, _) in plan.placement.items() if k == loader.SKIP}
    assert skipped == {n for n in meta
                       if n.startswith(("model.audio.", "model.visual."))}

    #3. head-parallel attention, re-derived from config independently of the
    #   builder: 16 Q heads/rank; 2 KV heads/rank global, 4 SWA; GQA groups
    #   rank-local; wo row-parallel on the same 16-head extent; per-channel
    #   K/V sconvs follow their KV heads
    HD = mc.head_dim
    assert mc.g_q_heads % plan.tp == 0 and mc.g_kv_heads % plan.tp == 0
    assert mc.s_q_heads % plan.tp == 0 and mc.s_kv_heads % plan.tp == 0
    q_pr = mc.g_q_heads // plan.tp
    kv_pr = {True: mc.s_kv_heads // plan.tp, False: mc.g_kv_heads // plan.tp}
    local = set(range(mc.num_layers)) - set(mc.global_layers)
    for i in range(mc.num_layers):
        p = f"model.llm.layers.{i}.attn."
        assert plan.placement[p + "wq_du.weight"] == (loader.SHARD, 0)
        assert meta[p + "wq_du.weight"][1][0] // plan.tp == q_pr * HD
        assert plan.placement[p + "wo_ud.weight"] == (loader.SHARD, 1)
        assert meta[p + "wo_ud.weight"][1][1] // plan.tp == q_pr * HD
        kpr = kv_pr[i in local]
        assert q_pr % kpr == 0, (i, q_pr, kpr)
        for w in ("wk_dv.weight", "wv_dv.weight",
                  "k_sconv.weight", "v_sconv.weight"):
            assert plan.placement[p + w] == (loader.SHARD, 0)
            assert meta[p + w][1][0] // plan.tp == kpr * HD, (p + w, kpr)
        for w in ("q_norm.weight", "k_norm.weight", "wr_du.weight",
                  "rel_logits_proj.proj"):
            assert plan.placement[p + w] == (loader.REPLICATE, -1)

    #4. expert-parallel MoE: routed experts (and, where packed, their scales
    #   — spot check; full coherence is builder-enforced) split 64/rank on
    #   the expert dim; gate replicated (every rank routes every token);
    #   shared experts + dense MLPs split on the FFN dim instead
    assert mc.n_experts % plan.tp == 0
    e_pr = mc.n_experts // plan.tp
    for i in range(mc.dense_idx, mc.num_layers):
        p = f"model.llm.layers.{i}.mlp."
        for w in ("experts.w13_weight", "experts.w2_weight"):
            assert plan.placement[p + w] == (loader.SHARD, 0)
            assert meta[p + w][1][0] == mc.n_experts
        for g in ("gate.weight", "gate.bias", "gate.global_scale"):
            assert plan.placement[p + g] == (loader.REPLICATE, -1)
        assert plan.placement[p + "shared_experts.shared_w13_weight"] == \
            (loader.SHARD, 1)
        assert plan.placement[p + "shared_experts.shared_w2_weight"] == \
            (loader.SHARD, 2)
    b = f"model.llm.layers.{mc.dense_idx + 1}.mlp.experts.w13_weight"
    assert plan.placement[b + ".scale"] == (loader.SHARD, 0)
    assert plan.placement[b + ".scale2"] == (loader.SHARD, 0)
    for i in range(mc.dense_idx):
        p = f"model.llm.layers.{i}.mlp."
        assert plan.placement[p + "w13_dn.weight"] == (loader.SHARD, 0)
        assert plan.placement[p + "w2_md.weight"] == (loader.SHARD, 1)

    #5. embed/unembed + final norms replicated (deliberate bring-up choice —
    #   local lookup + full-logits argmax, no vocab-parallel gather; the
    #   budget below proves it fits)
    for n in ("model.llm.embed.weight", "model.llm.embed_norm.weight",
              "model.llm.norm.weight", "model.llm.unembed.weight"):
        assert plan.placement[n] == (loader.REPLICATE, -1)

    #6. mtp placed by the same rules, so the budget already covers the
    #   MTP-ON config pair (D7): head-parallel attention, FFN-split dense
    #   MLP, row-parallel input_proj on its concat(embed,hidden) input dim
    for i in range(mc.mtp_layers):
        p = f"model.mtp.layers.{i}."
        assert plan.placement[p + "input_proj.weight"] == (loader.SHARD, 1)
        t = p + "transformer_block."
        assert plan.placement[t + "attn.wq_du.weight"] == (loader.SHARD, 0)
        assert plan.placement[t + "mlp.w13_dn.weight"] == (loader.SHARD, 0)
        assert plan.placement[t + "mlp.w2_md.weight"] == (loader.SHARD, 1)

    #7. byte budget from headers: partition reconciles to the checkpoint
    #   total (551.35 GiB, CLAUDE.md fact); per-GPU = replicated + sharded/4
    #   and MUST be <= 150 GiB (the item's bar)
    GiB = 2 ** 30
    rb = plan.rank_bytes(meta)
    total = sum(loader.nbytes(*meta[n]) for n in meta)
    assert rb[loader.SHARD] + rb[loader.REPLICATE] + rb[loader.SKIP] == total
    assert abs(total / GiB - 551.35) < 0.01, total / GiB
    per_gpu = rb["per_rank"]
    assert per_gpu == rb[loader.REPLICATE] + rb[loader.SHARD] // plan.tp
    assert per_gpu <= 150 * GiB, f"budget blown: {per_gpu / GiB:.2f} GiB"

    #8. summary line = the test's green evidence (per-GPU budget printed)
    emb = sum(loader.nbytes(*meta[n]) for n in
              ("model.llm.embed.weight", "model.llm.unembed.weight"))
    print(f"plan ok: tp={plan.tp}, head-parallel attn ({q_pr}q + "
          f"{kv_pr[False]}kv-global/{kv_pr[True]}kv-swa heads/rank) + "
          f"expert-parallel moe ({e_pr} experts/rank), ffn-split shared/dense/"
          f"mtp mlp; per-GPU {per_gpu / GiB:.2f} GiB <= 150 GiB budget "
          f"(sharded {rb[loader.SHARD] / plan.tp / GiB:.2f} + replicated "
          f"{rb[loader.REPLICATE] / GiB:.2f}, of which embed+unembed "
          f"{emb / GiB:.2f}); skipped multimodal {rb[loader.SKIP] / GiB:.2f} "
          f"GiB; checkpoint total {total / GiB:.2f} GiB")


def main():
    #1. dispatch on subcommand; unimplemented ones fail loud with their item id
    done = {"index": t_index, "census": t_census, "dtypes": t_dtypes,
            "dequant": t_dequant, "plan": t_plan}
    todo = {"load": "B1.6"}
    usage = f"usage: python -m engine.pyengine.tests.t_b1 {{{'|'.join([*done, *todo])}}}"
    if len(sys.argv) != 2 or sys.argv[1] not in {*done, *todo}:
        raise SystemExit(usage)
    if sys.argv[1] in todo:
        raise SystemExit(
            f"[t_b1] '{sys.argv[1]}' not implemented yet — PROGRESS item {todo[sys.argv[1]]}")
    done[sys.argv[1]]()


if __name__ == "__main__":
    main()
