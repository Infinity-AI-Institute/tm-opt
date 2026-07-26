"""B1: checkpoint -> GPU tensors. NVFP4 block-scale layout per vLLM modelopt
(cite file:line when implemented). TP=4 plan: attention head-parallel,
MoE expert-parallel (64 experts/GPU), embeddings replicated."""
#TODO(B1.2..B1.6): implemented item-by-item by the build loop.
import json
import pathlib
import re
from dataclasses import dataclass

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
