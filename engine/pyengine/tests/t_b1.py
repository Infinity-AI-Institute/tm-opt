"""B1 acceptance tests, one subcommand per PROGRESS.md item.
Run from repo root: python -m engine.pyengine.tests.t_b1 <subcommand>"""
import sys

from safetensors import safe_open

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


def main():
    #1. dispatch on subcommand; unimplemented ones fail loud with their item id
    done = {"index": t_index}
    todo = {"census": "B1.2", "dtypes": "B1.3", "dequant": "B1.4",
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
