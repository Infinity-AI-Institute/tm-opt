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


def main():
    #1. dispatch on subcommand; unimplemented ones fail loud with their item id
    done = {"index": t_index, "census": t_census}
    todo = {"dtypes": "B1.3", "dequant": "B1.4",
            "plan": "B1.5", "load": "B1.6"}
    usage = f"usage: python -m engine.pyengine.tests.t_b1 {{{'|'.join([*done, *todo])}}}"
    if len(sys.argv) != 2 or sys.argv[1] not in {*done, *todo}:
        raise SystemExit(usage)
    if sys.argv[1] in todo:
        raise SystemExit(
            f"[t_b1] '{sys.argv[1]}' not implemented yet — PROGRESS item {todo[sys.argv[1]]}")
    done[sys.argv[1]]()


if __name__ == "__main__":
    main()
