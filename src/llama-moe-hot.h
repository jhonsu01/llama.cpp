#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <string>
#include <vector>

struct llama_model;

// Pinned "hot" MoE experts: the most-routed (layer, expert) pairs of a ranked profile are copied ONCE into VRAM
// and never evicted.  Each routed expert of a token runs where it lives: hits on the GPU (the hot copy), misses
// on the CPU (the host weights, as with --n-cpu-moe).  No PCIe traffic per token, unlike an LRU that has to fill
// every miss.  Activated by --moe-hot-profile + --moe-hot-mib on layers whose expert weights are on the host.
//
// Profile format: Strata's expert-profile (tools/make_profile.py): "STRP", u32 version, n_layer, n_expert,
// slots, n, then n x (u16 layer, u16 expert) ranked most-used first.
struct llama_moe_hot {
    enum kind { KIND_GATE = 0, KIND_UP = 1, KIND_DOWN = 2, N_KIND = 3 };
    // graph nodes the split path adds per layer (4 id/weight ops, cast/reshape, weights, 2 x expert ffn + adds)
    static constexpr int NODES_PER_LAYER = 96;
    // the split path is for decode-sized batches: the CPU side relies on the CPU mul_mat_id, which accepts an
    // expert twice in a token, and from 32 tokens a host-weight mul_mat_id is offloaded to CUDA, which does not
    static constexpr int MAX_TOKENS = 16;

    llama_moe_hot(const llama_model & model, const std::vector<ggml_backend_t> & gpu_backends,
                  const std::string & profile_path, int64_t budget_mib);
    ~llama_moe_hot();

    bool has(int il) const { return il >= 0 && il < (int) layer_idx.size() && layer_idx[il] >= 0; }
    int  n_layers() const { return (int) layers.size(); }
    int  width(int il) const { return layers[layer_idx[il]].m; }

    // call only when has(il)
    ggml_tensor * hot_get(int il, int kind) const { return layers[layer_idx[il]].hot[kind]; }

    // CPU side (run on the CPU backend; a few int32 per token):
    //   ids_cpu [n_used, T]: the expert id; a hit is redirected to an expert this token already reads on the
    //                        CPU (the weight there is zeroed), so the CPU reads only the misses
    //   hit     [n_used, T]: 1 where the expert is hot, else 0
    ggml_tensor * build_ids_cpu(ggml_context * ctx, ggml_tensor * selected, int il);
    ggml_tensor * build_hit    (ggml_context * ctx, ggml_tensor * selected, int il);
    // GPU side, width m = min(hot experts of the layer, n_used), every id DISTINCT within a token (CUDA's
    // mul_mat_id groups a token's rows by expert and keeps one row per pair):
    //   ids_gpu [m, T]   : the token's hits first, then hot slots it does not use
    //   w_gpu   [1, m, T]: the router weight of each hit, 0 for the fillers
    ggml_tensor * build_ids_gpu(ggml_context * ctx, ggml_tensor * selected, int il);
    ggml_tensor * build_w_gpu  (ggml_context * ctx, ggml_tensor * selected, ggml_tensor * weights, int il);

    void log_final_stats() const;

    struct per_layer;
    struct op_ud {
        per_layer * L;
        int mode;      // 0 = gpu ids, 1 = cpu ids, 2 = hit, 3 = gpu weights
    };
    struct per_layer {
        int il = -1;
        std::vector<int32_t> map;            // expert -> hot index, -1 if cold
        int32_t k = 0;                       // hot experts in this layer
        int32_t m = 0;                       // GPU-side width: min(k, n_used)
        ggml_tensor * hot[N_KIND] = {};      // [ne0, ne1, k] in VRAM
        int64_t n_ids = 0, n_hits = 0;       // counted by the hit op (one task)
        op_ud ud[4];
        const llama_moe_hot * owner = nullptr;
        bool is_first = false;               // the first hot layer logs the running totals
    };

private:
    ggml_context_ptr ctx;
    std::vector<ggml_backend_buffer_ptr> bufs;
    std::vector<per_layer> layers;   // sized once: op userdata points into it
    std::vector<int> layer_idx;      // model layer -> index into layers, -1 when not hot
    int n_moe_layers = 0;            // layers with host-side experts, hot or not
};
