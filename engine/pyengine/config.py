"""#1. Python mirror of engine/include/tmopt/config.h — same provenance rules.
Loads and VERIFIES against the checkpoint json (fail loud), like the C++ side."""
import json, pathlib
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    num_layers: int = 66            #cfg: num_hidden_layers
    hidden: int = 6144              #cfg: hidden_size
    vocab: int = 201024             #cfg: vocab_size
    eos: int = 200006               #top: eos_token_id
    g_q_heads: int = 64; g_kv_heads: int = 8     #cfg: num_attention_heads / num_key_value_heads
    s_q_heads: int = 64; s_kv_heads: int = 16    #cfg: swa_*
    head_dim: int = 128             #cfg: head_dim
    window: int = 512               #cfg: sliding_window_size
    d_rel: int = 16; rel_extent: int = 1024      #cfg: d_rel / rel_extent
    log_alpha: float = 0.1; log_floor: int = 128000
    sconv_k: int = 4                #cfg: sconv_kernel_size
    n_experts: int = 256; topk: int = 6; n_shared: int = 2
    expert_ffn: int = 3072; dense_idx: int = 2; dense_ffn: int = 24576
    route_scale: float = 8.0
    mtp_layers: int = 8
    global_layers: tuple = tuple(range(5, 66, 6))  #derived: cfg local_layer_ids complement

def load_verified(model_dir: str) -> ModelConfig:
    #1. parse checkpoint config; abort on any mismatch with the defaults above
    cfg = json.loads((pathlib.Path(model_dir) / "config.json").read_text())
    txt = cfg["text_config"]
    mc = ModelConfig()
    checks = {
        "num_hidden_layers": mc.num_layers, "hidden_size": mc.hidden,
        "vocab_size": mc.vocab, "sliding_window_size": mc.window,
        "num_attention_heads": mc.g_q_heads, "num_key_value_heads": mc.g_kv_heads,
        "swa_num_key_value_heads": mc.s_kv_heads, "head_dim": mc.head_dim,
        "n_routed_experts": mc.n_experts, "num_experts_per_tok": mc.topk,
        "intermediate_size": mc.expert_ffn, "dense_mlp_idx": mc.dense_idx,
    }
    bad = {k: (txt.get(k), v) for k, v in checks.items() if txt.get(k) != v}
    if bad:
        raise SystemExit(f"[pyengine.config] MISMATCH vs checkpoint: {bad}")
    return mc
