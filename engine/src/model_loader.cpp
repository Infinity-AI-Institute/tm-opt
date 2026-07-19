/*
 * @brief  Checkpoint config parsing + verification.
 *
 *         Design: config.h holds documented expectations; this file makes them
 *         enforceable. One macro per field keeps the check list flat, greppable,
 *         and trivially extendable — no reflection, no schema framework
 *         (deliberately: the field count is ~40 and static).
 *
 *         Dependency: nlohmann/json via CMake FetchContent (single vendored
 *         target, header-only). Chosen over a bespoke parser: config parsing
 *         is startup-only, correctness-critical, and not performance-relevant.
 */
#include <cstdio>      //for fprintf
#include <cstdlib>     //for abort
#include <fstream>     //for std::ifstream
#include <nlohmann/json.hpp>  //for json parsing (FetchContent target)

#include "tmopt/config.h"

namespace tmopt {

using json = nlohmann::json;

namespace {

json load_json(const std::string &path)
{
    //1. fail loudly on missing files — a partial checkpoint must not serve
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "[config] FATAL: cannot open %s\n", path.c_str());
        std::abort();
    }
    return json::parse(f);
}

//1. one flat verifier: compares expectation vs parsed value, records mismatch.
//   Works for numbers, bools, strings via json's operator==.
struct Verifier {
    int mismatches = 0;

    template <typename T>
    void check(const json &j, const char *key, const T &expected)
    {
        if (!j.contains(key)) {
            std::fprintf(stderr, "[config] MISSING  %-28s (expected %s)\n",
                         key, json(expected).dump().c_str());
            ++mismatches;
            return;
        }
        if (j.at(key) != json(expected)) {
            std::fprintf(stderr, "[config] MISMATCH %-28s checkpoint=%s header=%s\n",
                         key, j.at(key).dump().c_str(),
                         json(expected).dump().c_str());
            ++mismatches;
        }
    }
};

//2. bit-pack a layer-id list into a mask (layers < 128 supported; Inkling has 66)
void pack_mask(const json &ids, uint64_t mask[2])
{
    mask[0] = mask[1] = 0;
    for (uint32_t id : ids) mask[id / 64] |= (1ull << (id % 64));
}

}  //namespace

ModelConfig load_and_verify_model_config(const std::string &model_dir)
{
    ModelConfig mc;  //header defaults = documented expectations

    //1. parse the three checkpoint JSONs
    const json cfg  = load_json(model_dir + "/config.json");
    const json &txt = cfg.at("text_config");
    const json &mtp = cfg.at("mtp_config");
    const json quant = load_json(model_dir + "/hf_quant_config.json")
                           .at("quantization");

    //2. verify every cited field (order mirrors config.h)
    Verifier v;
    v.check(txt, "num_hidden_layers",       mc.num_layers);
    v.check(txt, "hidden_size",             mc.hidden_size);
    v.check(txt, "vocab_size",              mc.vocab_size);
    v.check(txt, "unpadded_vocab_size",     mc.unpadded_vocab);
    v.check(cfg, "eos_token_id",            mc.eos_token_id);
    v.check(txt, "rms_norm_eps",            mc.rms_norm_eps);
    v.check(txt, "use_embed_norm",          mc.use_embed_norm);

    v.check(txt, "num_attention_heads",     mc.g_num_q_heads);
    v.check(txt, "num_key_value_heads",     mc.g_num_kv_heads);
    v.check(txt, "swa_num_attention_heads", mc.swa_num_q_heads);
    v.check(txt, "swa_num_key_value_heads", mc.swa_num_kv_heads);
    v.check(txt, "head_dim",                mc.head_dim);
    v.check(txt, "swa_head_dim",            mc.head_dim);
    v.check(txt, "sliding_window_size",     mc.sliding_window);

    v.check(txt, "d_rel",                   mc.d_rel);
    v.check(txt, "rel_extent",              mc.rel_extent);
    v.check(txt, "log_scaling_alpha",       mc.log_scaling_alpha);
    v.check(txt, "log_scaling_n_floor",     mc.log_scaling_floor);
    v.check(txt, "q_bias",                  mc.q_bias);
    v.check(txt, "o_bias",                  mc.o_bias);

    v.check(txt, "use_sconv",               mc.use_sconv);
    v.check(txt, "sconv_kernel_size",       mc.sconv_kernel);

    v.check(txt, "n_routed_experts",        mc.n_routed_experts);
    v.check(txt, "num_experts_per_tok",     mc.experts_per_tok);
    v.check(txt, "n_shared_experts",        mc.n_shared_experts);
    v.check(txt, "shared_expert_sink",      mc.shared_expert_sink);
    v.check(txt, "intermediate_size",       mc.expert_ffn);
    v.check(txt, "route_scale",             mc.route_scale);
    v.check(txt, "use_gate_bias",           mc.use_gate_bias);
    v.check(txt, "gate_activation",         std::string("sigmoid"));
    v.check(txt, "norm_after_topk",         mc.norm_after_topk);
    v.check(txt, "use_global_scale",        mc.use_global_scale);
    v.check(txt, "dense_mlp_idx",           mc.dense_mlp_idx);
    v.check(txt, "dense_intermediate_size", mc.dense_ffn);

    v.check(mtp, "num_nextn_predict_layers", mc.mtp_layers);
    v.check(txt, "model_max_length",         mc.model_max_length);

    v.check(quant, "quant_algo",           std::string("NVFP4"));
    v.check(quant, "kv_cache_quant_algo",  std::string("none"));

    //3. derived fields: masks from the id lists (+ sanity on expected counts)
    pack_mask(txt.at("local_layer_ids"), mc.local_layer_mask);
    if (txt.at("local_layer_ids").size() != 55) {
        std::fprintf(stderr, "[config] MISMATCH local_layer_ids count=%zu header=55\n",
                     txt.at("local_layer_ids").size());
        ++v.mismatches;
    }
    uint64_t mtp_mask[2] = {0, 0};
    pack_mask(mtp.at("local_layer_ids"), mtp_mask);
    mc.mtp_local_mask = mtp_mask[0];

    //4. verdict: any mismatch is fatal — a wrong constant must not serve
    if (v.mismatches) {
        std::fprintf(stderr,
            "[config] FATAL: %d mismatch(es) between engine/include/tmopt/config.h "
            "and %s.\nFix the header (and any kernel assuming the old value), "
            "or you are serving a different model.\n",
            v.mismatches, model_dir.c_str());
        std::abort();
    }
    std::fprintf(stderr, "[config] verified: all fields match checkpoint at %s\n",
                 model_dir.c_str());
    return mc;
}

}  //namespace tmopt