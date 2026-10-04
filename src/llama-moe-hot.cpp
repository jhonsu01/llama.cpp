#include "llama-moe-hot.h"
#include "llama.h"

#include "llama-model.h"
#include "llama-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace {

struct profile_pair { uint16_t layer, expert; };

std::vector<profile_pair> read_profile(const std::string & path, uint32_t & n_layer, uint32_t & n_expert) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("llama_moe_hot: cannot open profile " + path);
    }
    char magic[4];
    uint32_t hdr[5]; // version, n_layer, n_expert, slots, n
    f.read(magic, 4);
    f.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
    if (!f || std::memcmp(magic, "STRP", 4) != 0) {
        throw std::runtime_error("llama_moe_hot: " + path + " is not a STRP expert profile");
    }
    n_layer  = hdr[1];
    n_expert = hdr[2];
    std::vector<profile_pair> pairs(hdr[4]);
    f.read(reinterpret_cast<char *>(pairs.data()), (std::streamsize) (pairs.size() * sizeof(profile_pair)));
    if (!f) {
        throw std::runtime_error("llama_moe_hot: " + path + " is truncated");
    }
    return pairs;
}

inline int32_t id_at(const ggml_tensor * a, int64_t i, int64_t t) {
    return *(const int32_t *) ((const char *) a->data + i * a->nb[0] + t * a->nb[1]);
}

// dst and a: i32 [n_used, n_tokens] (a may be a view of the router's argsort; dst is contiguous)
void hot_ids_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    auto * ud = (llama_moe_hot::op_ud *) userdata;
    llama_moe_hot::per_layer & L = *ud->L;
    const int64_t n_used = a->ne[0];
    const int64_t n_tok  = a->ne[1];
    int32_t * out = (int32_t *) dst->data;

    // a cold expert anywhere in the batch: the CPU target of a token whose experts are all hot
    int32_t batch_cold = -1;
    if (ud->mode == 1) {
        for (int64_t t = 0; t < n_tok && batch_cold < 0; t++) {
            for (int64_t i = 0; i < n_used; i++) {
                const int32_t e = id_at(a, i, t);
                if (L.map[e] < 0) { batch_cold = e; break; }
            }
        }
    }

    for (int64_t t = 0; t < n_tok; t++) {
        int32_t first_cold = -1;
        for (int64_t i = 0; i < n_used && first_cold < 0; i++) {
            const int32_t e = id_at(a, i, t);
            if (L.map[e] < 0) first_cold = e;
        }
        for (int64_t i = 0; i < n_used; i++) {
            const int32_t e = id_at(a, i, t);
            const int32_t h = L.map[e];
            if (ud->mode == 1) {
                out[t * n_used + i] = h < 0 ? e : (first_cold >= 0 ? first_cold : (batch_cold >= 0 ? batch_cold : e));
            } else {
                out[t * n_used + i] = h >= 0 ? 1 : 0;
                L.n_ids++;
                L.n_hits += h >= 0;
            }
        }
    }
    // every ~1000 decoded tokens, the running share of experts computed on the GPU (the first hot layer reports)
    if (ud->mode == 2 && L.owner != nullptr && L.is_first && L.n_ids / 10000 != (L.n_ids - n_tok * n_used) / 10000) {
        L.owner->log_final_stats();
    }
}

// the GPU side of one token: its hits (hot index, router slot) first, then hot slots it does not use, m in all;
// slot = -1 marks a filler.  Every hot index is distinct.
void gpu_row(const llama_moe_hot::per_layer & L, const ggml_tensor * sel, int64_t t,
             int32_t * hot_idx, int32_t * slot, std::vector<char> & used) {
    const int64_t n_used = sel->ne[0];
    std::fill(used.begin(), used.end(), 0);
    int32_t n = 0;
    for (int64_t i = 0; i < n_used && n < L.m; i++) {
        const int32_t h = L.map[id_at(sel, i, t)];
        if (h >= 0) { hot_idx[n] = h; slot[n] = (int32_t) i; used[h] = 1; n++; }
    }
    for (int32_t j = 0; j < L.k && n < L.m; j++) {
        if (!used[j]) { hot_idx[n] = j; slot[n] = -1; n++; }
    }
}

// custom_4d ops of the GPU side: src[0] = selected ids [n_used, T]; mode 3 also src[1] = weights [1, n_used, T]
void hot_gpu_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    auto * ud = (llama_moe_hot::op_ud *) userdata;
    const llama_moe_hot::per_layer & L = *ud->L;
    const ggml_tensor * sel = dst->src[0];
    const ggml_tensor * w   = dst->src[1];
    const int64_t n_tok = sel->ne[1];
    std::vector<int32_t> hot_idx(L.m), slot(L.m);
    std::vector<char> used(L.k);
    for (int64_t t = 0; t < n_tok; t++) {
        gpu_row(L, sel, t, hot_idx.data(), slot.data(), used);
        for (int32_t j = 0; j < L.m; j++) {
            if (ud->mode == 0) {
                ((int32_t *) dst->data)[t * L.m + j] = hot_idx[j];
            } else {
                ((float *) dst->data)[t * L.m + j] = slot[j] < 0 ? 0.0f :
                    *(const float *) ((const char *) w->data + slot[j] * w->nb[1] + t * w->nb[2]);
            }
        }
    }
}

} // namespace

llama_moe_hot::llama_moe_hot(const llama_model & model, const std::vector<ggml_backend_t> & gpu_backends,
                             const std::string & profile_path, int64_t budget_mib) {
    if (gpu_backends.empty()) {
        throw std::runtime_error("llama_moe_hot: no GPU backend");
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(gpu_backends[0]);

    const int64_t n_expert    = model.hparams.n_expert;
    const int     n_layer_tot = (int) model.hparams.n_layer();

    uint32_t p_layer = 0, p_expert = 0;
    const std::vector<profile_pair> ranked = read_profile(profile_path, p_layer, p_expert);
    if ((int64_t) p_expert != n_expert) {
        throw std::runtime_error("llama_moe_hot: profile has " + std::to_string(p_expert) + " experts per layer, "
                                 "the model " + std::to_string(n_expert));
    }

    // layers whose experts are on the host (--n-cpu-moe / -ot), with separate gate/up/down
    std::vector<bool> eligible(n_layer_tot, false);
    std::vector<size_t> pair_bytes(n_layer_tot, 0);
    for (int il = 0; il < n_layer_tot; il++) {
        const auto & ly = model.layers[il];
        if (!ly.ffn_gate_exps || !ly.ffn_up_exps || !ly.ffn_down_exps || ly.ffn_gate_up_exps) continue;
        if (!ly.ffn_up_exps->buffer || !ggml_backend_buffer_is_host(ly.ffn_up_exps->buffer)) continue;
        if (!ly.ffn_gate_exps->data || !ly.ffn_up_exps->data || !ly.ffn_down_exps->data) continue;
        eligible[il]   = true;
        n_moe_layers++;
        pair_bytes[il] = ly.ffn_gate_exps->nb[2] + ly.ffn_up_exps->nb[2] + ly.ffn_down_exps->nb[2];
    }

    // the most-routed pairs that fit the budget
    const size_t budget = (size_t) budget_mib << 20;
    size_t used = 0;
    std::vector<std::vector<int32_t>> chosen(n_layer_tot);
    int64_t n_pairs = 0;
    for (const auto & p : ranked) {
        if (p.layer >= n_layer_tot || p.expert >= n_expert || !eligible[p.layer]) continue;
        if (used + pair_bytes[p.layer] > budget) break;
        used += pair_bytes[p.layer];
        chosen[p.layer].push_back(p.expert);
        n_pairs++;
    }
    if (n_pairs == 0) {
        throw std::runtime_error("llama_moe_hot: no hot experts (no host-side expert layers, or the budget is "
                                 "below one expert)");
    }

    int n_hot_layers = 0;
    for (int il = 0; il < n_layer_tot; il++) n_hot_layers += !chosen[il].empty();
    layers = std::vector<per_layer>(n_hot_layers);
    layer_idx.assign(n_layer_tot, -1);
    ctx.reset(ggml_init({ (size_t) n_hot_layers * N_KIND * ggml_tensor_overhead(), nullptr, true }));

    int li = 0;
    for (int il = 0; il < n_layer_tot; il++) {
        if (chosen[il].empty()) continue;
        per_layer & L = layers[li];
        layer_idx[il] = li++;
        L.il = il;
        L.k  = (int32_t) chosen[il].size();
        L.map.assign(n_expert, -1);
        for (int32_t j = 0; j < L.k; j++) L.map[chosen[il][j]] = j;
        L.m  = (int32_t) std::min<int64_t>(L.k, model.hparams.n_expert_used());
        for (int m = 0; m < 4; m++) L.ud[m] = { &L, m };
        L.owner    = this;
        L.is_first = li == 1;

        const auto & ly = model.layers[il];
        ggml_tensor * src[N_KIND] = { ly.ffn_gate_exps, ly.ffn_up_exps, ly.ffn_down_exps };
        for (int k = 0; k < N_KIND; k++) {
            GGML_ASSERT(ggml_is_contiguous(src[k]));
            ggml_tensor * t = ggml_new_tensor_3d(ctx.get(), src[k]->type, src[k]->ne[0], src[k]->ne[1], L.k);
            // the backend's alloc size, not ggml_nbytes: CUDA's quantized kernels read the row padding past the
            // last block, and tensor_alloc runs the buffer's init_tensor (which clears that padding)
            ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, t));
            if (!b) {
                throw std::runtime_error("llama_moe_hot: VRAM allocation failed (lower --moe-hot-mib)");
            }
            ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            bufs.emplace_back(b);
            if (ggml_backend_tensor_alloc(b, t, ggml_backend_buffer_get_base(b)) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("llama_moe_hot: tensor_alloc failed");
            }
            const size_t eb = src[k]->nb[2];
            for (int32_t j = 0; j < L.k; j++) {
                ggml_backend_tensor_set(t, (const char *) src[k]->data + (size_t) chosen[il][j] * eb, j * eb, eb);
            }
            L.hot[k] = t;
        }
    }

    LLAMA_LOG_INFO("llama_moe_hot: %lld hot experts in %d layers, %.0f MiB of VRAM (profile %s)\n",
                   (long long) n_pairs, n_hot_layers, used / 1048576.0, profile_path.c_str());
}

llama_moe_hot::~llama_moe_hot() = default;

ggml_tensor * llama_moe_hot::build_ids_gpu(ggml_context * c, ggml_tensor * selected, int il) {
    per_layer & L = layers[layer_idx[il]];
    ggml_tensor * args[1] = { selected };
    return ggml_custom_4d(c, GGML_TYPE_I32, L.m, selected->ne[1], 1, 1, args, 1, hot_gpu_op, 1, &L.ud[0]);
}

ggml_tensor * llama_moe_hot::build_w_gpu(ggml_context * c, ggml_tensor * selected, ggml_tensor * weights, int il) {
    per_layer & L = layers[layer_idx[il]];
    ggml_tensor * args[2] = { selected, weights };
    return ggml_custom_4d(c, GGML_TYPE_F32, 1, L.m, selected->ne[1], 1, args, 2, hot_gpu_op, 1, &L.ud[3]);
}

ggml_tensor * llama_moe_hot::build_ids_cpu(ggml_context * c, ggml_tensor * selected, int il) {
    return ggml_map_custom1(c, selected, hot_ids_op, 1, &layers[layer_idx[il]].ud[1]);
}

ggml_tensor * llama_moe_hot::build_hit(ggml_context * c, ggml_tensor * selected, int il) {
    return ggml_map_custom1(c, selected, hot_ids_op, 1, &layers[layer_idx[il]].ud[2]);
}

void llama_moe_hot::log_final_stats() const {
    int64_t ids = 0, hits = 0;
    for (const auto & L : layers) {
        ids  += L.n_ids;
        hits += L.n_hits;
    }
    if (ids > 0) {
        // every MoE layer routes the same number of experts: the hot layers' mean stands for each of n_moe_layers
        const double all = (double) ids / layers.size() * n_moe_layers;
        LLAMA_LOG_INFO("llama_moe_hot: %lld routed experts in the %zu hot layers, %.2f%% on the GPU "
                       "(%.2f%% of all %d MoE layers)\n",
                       (long long) ids, layers.size(), 100.0 * hits / ids, 100.0 * hits / all, n_moe_layers);
    }
}
