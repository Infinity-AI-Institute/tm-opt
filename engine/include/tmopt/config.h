/*
 * @brief  Inkling model + runtime configuration.
 */
#pragma once
#include <cstdint>   
#include <string>    

namespace tmopt {

struct ModelConfig {
    //1. backbone
    uint32_t num_layers        = 66;      //cfg: num_hidden_layers
    uint32_t hidden_size       = 6144;    
    uint32_t vocab_size        = 201024;  
    uint32_t unpadded_vocab    = 200058;  
    uint32_t eos_token_id      = 200006;  
    float    rms_norm_eps      = 1e-6f;  
    bool     use_embed_norm    = true;    

    //2. attention — two layer types with DIFFERENT KV shapes
    uint64_t local_layer_mask[2] = {0, 0}; //derived: bit i set <=> i in
                                           //cfg: local_layer_ids (55 SWA layers;
                                           //globals = {5,11,...,65}, every 6th)
    uint32_t g_num_q_heads     = 64;      
    uint32_t g_num_kv_heads    = 8;      
    uint32_t swa_num_q_heads   = 64;      
    uint32_t swa_num_kv_heads  = 16;      
    uint32_t head_dim          = 128;     
    uint32_t sliding_window    = 512;     

    //3. relative attention (model has NO RoPE)
    uint32_t d_rel             = 16;      
    uint32_t rel_extent        = 1024;    
    float    log_scaling_alpha = 0.1f;   
    uint32_t log_scaling_floor = 128000;  
    bool     q_bias            = false;   
    bool     o_bias            = false;   

    //4. short convolution
    bool     use_sconv         = true;    
    uint32_t sconv_kernel      = 4;      

    //5. MoE (every layer except dense_mlp_idx)
    uint32_t n_routed_experts  = 256;     
    uint32_t experts_per_tok   = 6;      
    uint32_t n_shared_experts  = 2;      
    bool     shared_expert_sink= true;    
    uint32_t expert_ffn        = 3072;    
    float    route_scale       = 8.0f;   
    bool     use_gate_bias     = true;   
    bool     gate_sigmoid      = true;   
    bool     norm_after_topk   = true;    
    bool     use_global_scale  = true;    
    uint32_t dense_mlp_idx     = 2;      
    uint32_t dense_ffn         = 24576;  

    //6. MTP speculative decode (weights in mtp.safetensors)
    uint32_t mtp_layers        = 8;       //mtp: num_nextn_predict_layers
    uint64_t mtp_local_mask    = 0;       //derived from mtp: local_layer_ids

    //7. context
    uint32_t model_max_length  = 1048576; //cfg: model_max_length

    //quant: quant_algo=NVFP4, kv_cache_quant_algo=none (KV is bf16),
    //quant: exclude_modules kept bf16 by the loader (embeds, norms, unembed,
    //quant: layer-0 attention, multimodal encoders) — enforced in loader
};

struct RuntimeConfig {
    std::string model_dir;
    uint32_t port             = 8100;
    uint32_t replica_gpus     = 4;     //choice: 600GB NVFP4 / 269GB per B300
                                       //=> >=3 GPUs; 4 matches official vLLM
                                       //recipe TP=4 + KV headroom. Standing
                                       //A/B vs 8x1 in the ledger.
    uint32_t max_batch_tokens = 8192;  //choice: scheduler budget; harness-swept
    uint32_t kv_page_size     = 16;    //choice: paged-KV granularity (vLLM-
                                       //family default); harness-swept
    float    gpu_mem_fraction = 0.90f; //choice: canonical benchmark parameter —
                                       //MUST equal vLLM's (fairness constitution)
    bool     enable_mtp       = true;  //choice: model ships 8 MTP heads; vLLM
                                       //baseline uses them; parity of features
    uint32_t mtp_draft_len    = 4;     //choice: placeholder; sweep 1..8 via
                                       //harness before trusting
};

/*
 * @brief  Parse config.json / mtp_config / hf_quant_config from model_dir,
 *         verify EVERY ModelConfig field above against the parsed values,
 *         and abort with a field-by-field diff on any mismatch.
 * @param  model_dir : checkpoint directory (must contain config.json)
 * @return verified ModelConfig (masks populated from the id lists)
 */
ModelConfig load_and_verify_model_config(const std::string &model_dir);

}  //namespace tmopt
