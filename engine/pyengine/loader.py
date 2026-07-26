"""B1: checkpoint -> GPU tensors. NVFP4 block-scale layout per vLLM modelopt
(see dequant_nvfp4 for the cited layout facts). TP=4 plan: attention
head-parallel, MoE expert-parallel (64 experts/GPU), embeddings replicated."""
#TODO(B1.5..B1.6): implemented item-by-item by the build loop.
import json
import pathlib
import re
import struct
from dataclasses import dataclass

import torch

N_MODEL_SHARDS = 33            # model-XXXXX-of-00033.safetensors (CLAUDE.md model facts)
MTP_SHARD = "mtp.safetensors"  # MTP draft layers, separate file, listed in the index


@dataclass(frozen=True)
class ShardIndex:
    """B1.1: tensor name -> shard file map covering the whole checkpoint."""
    model_dir: pathlib.Path
    tensor_to_shard: dict   # tensor name -> shard filename (relative to model_dir)
    shard_files: tuple      # sorted distinct shard filenames (33 model + mtp)

    def shard_path(self, tensor_name: str) -> pathlib.Path:
        #1. resolve a tensor name to the absolute path of the shard holding it
        return self.model_dir / self.tensor_to_shard[tensor_name]

    def tensors_in_shard(self, shard_file: str) -> list:
        #1. inverse lookup: every tensor name the index assigns to one shard
        return [n for n, f in self.tensor_to_shard.items() if f == shard_file]


def build_shard_index(model_dir: str) -> ShardIndex:
    #1. read the checkpoint's own index; its weight_map covers the 33 model
    #   shards AND mtp.safetensors (verified: 34 distinct files, 2056 tensors)
    root = pathlib.Path(model_dir)
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"[loader] index not found: {index_path}")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    if not weight_map:
        raise SystemExit("[loader] empty weight_map")

    #2. enumerate distinct shard files; fail loud on any unexpected shape
    shard_files = tuple(sorted(set(weight_map.values())))
    if MTP_SHARD not in shard_files:
        raise SystemExit(f"[loader] {MTP_SHARD} missing from weight_map")
    model_shards = [f for f in shard_files if f != MTP_SHARD]
    if len(model_shards) != N_MODEL_SHARDS:
        raise SystemExit(
            f"[loader] expected {N_MODEL_SHARDS} model shards, got {len(model_shards)}")
    pat = re.compile(r"model-\d{5}-of-00033\.safetensors")
    bad = [f for f in model_shards if not pat.fullmatch(f)]
    if bad:
        raise SystemExit(f"[loader] unexpected shard names: {bad}")

    #3. every mapped file must actually exist on disk
    missing = [f for f in shard_files if not (root / f).is_file()]
    if missing:
        raise SystemExit(f"[loader] shard files missing on disk: {missing}")

    #4. hand back the immutable map
    return ShardIndex(model_dir=root, tensor_to_shard=dict(weight_map),
                      shard_files=shard_files)


def read_headers(idx: ShardIndex) -> dict:
    """B1.2: tensor name -> (dtype str, shape tuple) for the whole checkpoint,
    from safetensors headers only (no tensor data is read)."""
    #1. safetensors format: 8-byte LE header length, then JSON header
    meta = {}
    for fname in idx.shard_files:
        with open(idx.model_dir / fname, "rb") as f:
            (hdr_len,) = struct.unpack("<Q", f.read(8))
            hdr = json.loads(f.read(hdr_len))
        hdr.pop("__metadata__", None)
        for name, ent in hdr.items():
            if name in meta:
                raise SystemExit(f"[loader] tensor in two shards: {name}")
            meta[name] = (ent["dtype"], tuple(ent["shape"]))

    #2. headers must cover exactly the tensors the index maps (B1.1 invariant)
    if set(meta) != set(idx.tensor_to_shard):
        only_hdr = sorted(set(meta) - set(idx.tensor_to_shard))[:5]
        only_idx = sorted(set(idx.tensor_to_shard) - set(meta))[:5]
        raise SystemExit(
            f"[loader] header/index mismatch; header-only={only_hdr} index-only={only_idx}")
    return meta


#B1.3: one NVFP4-packed weight = base U8 tensor (2 fp4/byte on the input dim)
#plus exactly these four companion tensors (modelopt export layout)
PACK_SUFFIXES = (".scale", ".scale2", ".input_amax", ".original_shape")


@dataclass(frozen=True)
class DtypeMap:
    """B1.3: per-tensor load precision for the whole checkpoint. `packed`
    holds NVFP4 base names (load U8 + 4 companions, dequant group
    `group_size`); `plain` maps every other tensor to its stored dtype."""
    group_size: int          # inputs per F8_E4M3 block scale (16)
    exclude_modules: tuple   # hf_quant_config.json exclude list, verbatim
    packed: frozenset        # NVFP4 base tensor names (companions implied)
    plain: dict              # every non-pack tensor name -> stored dtype str

    def is_excluded(self, name: str) -> bool:
        #1. a modelopt exclude entry covers the named module and everything
        #   under it — match at component boundary, not raw prefix (else
        #   "...5.attn" would swallow "...5.attn_norm"). Same verdict as the
        #   exact-match arm of vLLM ModelOptQuantConfigBase.is_layer_excluded
        #   (vllm/model_executor/layers/quantization/modelopt.py:145); its
        #   wildcard arm is unreachable here (builder rejects '*' entries).
        return any(name == e or name.startswith(e + ".")
                   for e in self.exclude_modules)

    def companions(self, base: str) -> tuple:
        #1. the four side tensors carried by one NVFP4-packed base weight
        return tuple(base + s for s in PACK_SUFFIXES)


def build_dtype_map(idx: ShardIndex, meta: dict) -> DtypeMap:
    """B1.3: classify every tensor as NVFP4-packed vs plain from the headers,
    then verify the split is EXACTLY what hf_quant_config.json's exclude list
    predicts. Checkpoint reality (proved here, not assumed): quantized =
    routed-expert w13/w2 of MoE layers 3-65 only; ALL attention is bf16
    (every layer, not just 0); mtp.safetensors is entirely bf16."""
    #1. quant recipe must be the one we build for: NVFP4, 16-input block
    #   scales, bf16 KV, literal (wildcard-free) exclude entries
    q = json.loads(
        (idx.model_dir / "hf_quant_config.json").read_text())["quantization"]
    recipe = (q.get("quant_algo"), q.get("group_size"),
              q.get("kv_cache_quant_algo"))
    if recipe != ("NVFP4", 16, "none"):
        raise SystemExit(f"[loader] unexpected quant recipe: {recipe}")
    wild = [e for e in q["exclude_modules"] if "*" in e]
    if wild:
        raise SystemExit(f"[loader] wildcard exclude entries: {wild[:3]}")

    #2. packs from headers: every U8 tensor is an NVFP4 base and must carry
    #   exactly the four companions, dtype + shape derived from the base
    #   (scale: one F8_E4M3 per group_size inputs; input dim = 2 * packed)
    packed = frozenset(n for n, (dt, _) in meta.items() if dt == "U8")
    in_pack = set()
    for b in packed:
        bs = meta[b][1]
        want = {
            b + ".scale": ("F8_E4M3", bs[:-1] + (bs[-1] * 2 // q["group_size"],)),
            b + ".scale2": ("F32", (bs[0],)),
            b + ".input_amax": ("BF16", (1,)),
            b + ".original_shape": ("I64", (len(bs),)),
        }
        got = {c: meta.get(c) for c in want}
        if got != want:
            raise SystemExit(f"[loader] bad NVFP4 pack {b}:\n"
                             f"  want {want}\n  got {got}")
        in_pack |= {b, *want}

    #3. companion suffixes may not appear outside a pack (an orphan scale
    #   would mean a quantized base this map failed to classify)
    orphans = [n for n in meta if n.endswith(PACK_SUFFIXES) and n not in in_pack]
    if orphans:
        raise SystemExit(f"[loader] orphan pack companions: {orphans[:5]}")

    #4. everything else loads at its stored dtype
    plain = {n: meta[n][0] for n in meta if n not in in_pack}
    dm = DtypeMap(group_size=q["group_size"],
                  exclude_modules=tuple(q["exclude_modules"]),
                  packed=packed, plain=plain)

    #5. reconcile vs the exclude list, both directions, whole checkpoint:
    #   (a) every exclude entry matches >=1 tensor (config/checkpoint drift);
    #   (b) model.llm.*: in-a-pack <=> NOT excluded, tensor by tensor;
    #   (c) multimodal: all excluded, all plain;
    #   (d) model.mtp.*: no exclude entry reaches mtp.safetensors — the
    #       modelopt export covered the main model only; all mtp is BF16.
    dead = [e for e in dm.exclude_modules
            if not any(n == e or n.startswith(e + ".") for n in meta)]
    if dead:
        raise SystemExit(f"[loader] exclude entries matching nothing: {dead[:5]}")
    for n in meta:
        if n.startswith("model.llm."):
            if (n in in_pack) == dm.is_excluded(n):
                raise SystemExit(
                    f"[loader] exclude-list mismatch on {n}: in_pack="
                    f"{n in in_pack} excluded={dm.is_excluded(n)}")
        elif n.startswith(("model.audio.", "model.visual.")):
            if n in in_pack or not dm.is_excluded(n):
                raise SystemExit(f"[loader] multimodal not excluded-plain: {n}")
        elif n.startswith("model.mtp."):
            if dm.is_excluded(n) or plain.get(n) != "BF16":
                raise SystemExit(f"[loader] mtp expectation broken: {n}")
        else:
            raise SystemExit(f"[loader] unclassified tensor family: {n}")
    return dm


#B1.4: E2M1 decode table, index = 3-bit magnitude (sign is bit 0x08). Values
#per vLLM kE2M1ToFloat (vllm/model_executor/layers/quantization/utils/
#nvfp4_emulation_utils.py:20-22).
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequant_nvfp4(packed, scale, scale2, group_size, out_dtype=torch.bfloat16):
    """B1.4: dequantize one NVFP4-packed weight (or an expert slice of one).

    Layout facts, each cited to the vLLM code that defines them (all paths
    under vllm/model_executor/layers/quantization/ unless noted):
    - byte j of the packed input dim holds elements 2j (LOW nibble) and
      2j+1 (HIGH nibble) — utils/nvfp4_emulation_utils.py:316-333
      (break_fp4_bytes: low first) and :101-108 (triton kernel interleave).
    - value = e2m1(nibble) * block_scale * scale2: one F8_E4M3 block scale
      per `group_size`(=16) inputs, times the per-expert F32 scale2.
      scale2 is ModelOpt's weight_scale_2 = amax/(6.0*448.0), applied
      DIRECTLY, no reciprocation — modelopt.py:1253-1262 (docstring) and
      utils/nvfp4_emulation_utils.py:391-397 (dequantize_to_dtype:
      tensor_sf * global_scale). The inkling loader maps checkpoint names
      .scale/.scale2 -> weight_scale/weight_scale_2 verbatim
      (vllm/models/inkling/nvidia/model.py:381-386, moe.py:578-582).
    - checkpoint block scales are LINEAR row-major (exact [*, rows, k/16]
      shape, dtype-map-verified; swizzling is a load-time transform for the
      cutlass kernels, modelopt.py:1873 expects "2D, not swizzled" on disk)
      -> dequantize_to_dtype(swizzle=False) semantics.

    Shapes: packed (*L, rows, k/2) u8; scale (*L, rows, k/group_size)
    f8e4m3; scale2 (*L,) f32 (0-d for a single 2D weight slice).
    """
    #1. contract checks — dtypes/shapes exactly as stored (census B1.2/B1.3)
    assert packed.dtype == torch.uint8, packed.dtype
    assert scale.dtype == torch.float8_e4m3fn, scale.dtype
    assert scale2.dtype == torch.float32, scale2.dtype
    *lead, rows, pk = packed.shape
    k = pk * 2
    assert k % group_size == 0
    assert tuple(scale.shape) == (*lead, rows, k // group_size), scale.shape
    assert tuple(scale2.shape) == tuple(lead), scale2.shape

    #2. unpack nibbles along the input dim, LOW nibble first (see citation)
    nib = torch.stack((packed & 0x0F, packed >> 4), dim=-1)
    nib = nib.reshape(*lead, rows, k)

    #3. decode E2M1: 3-bit magnitude via LUT, sign bit 0x08
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    val = lut[(nib & 0x07).long()]
    val = torch.where(nib & 0x08 != 0, -val, val)

    #4. effective f32 scale per block = f8 block scale * scale2 — same
    #   association as vLLM's kernel (e2m1 * (scale*global), utils/
    #   nvfp4_emulation_utils.py:88-89,104-105) so bf16 results are bit-equal
    s = scale.float() * scale2.reshape(*scale2.shape, 1, 1)

    #5. broadcast the block scale over its group_size inputs; cast once
    val = val.reshape(*lead, rows, k // group_size, group_size)
    out = (val * s.unsqueeze(-1)).reshape(*lead, rows, k)
    return out.to(out_dtype)
