// src/poly/flint.cpp - RecursiveSI Flint.  The reference is llama.cpp's Flint graph.
#include "flint.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr float kMinWeightSum = 6.103515625e-5f;
constexpr size_t kScoreFloats = size_t(16) << 20;

int64_t layer_value(const MetaValue& v, int il) {
    if (v.type != MetaType::ARRAY) return (int64_t) v.u;
    if ((size_t) il >= v.items.size()) throw std::runtime_error("a per-layer value ends before Flint's layers");
    return (int64_t) v.items[(size_t) il].u;
}

}  // namespace

void Flint::begin_pass() { blocks_.assign(layers_.size(), Block{}); }

void Flint::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported Flint model: ") + what);
    };
    auto must = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = find(key);
        need(v != nullptr, key);
        return *v;
    };

    n_head_ = shape_i("attention.head_count");
    n_kv_ = shape_i("attention.head_count_kv");
    dim_ = shape_i("attention.key_length");
    n_rot_ = dim_;
    window_ = shape_i("attention.sliding_window");
    dense_layers_ = shape_i("leading_dense_block_count");
    n_ff_ = shape_i("feed_forward_length");
    theta_scale_ = std::pow(shape_f("rope.freq_base"), -2.f / (float) n_rot_);

    need(n_head_ > 0 && n_head_ <= 256 && n_kv_ > 0 && n_head_ % n_kv_ == 0, "bad attention head counts");
    need(dim_ > 0 && dim_ % 4 == 0 && shape_i("attention.value_length") == dim_,
         "the attention's keys and values differ in length");
    need(n_rot_ % 2 == 0 && window_ > 0, "bad rotary width or sliding window");
    need(dense_layers_ >= 0 && dense_layers_ <= n_layer_ && n_ff_ > 0, "bad dense-layer geometry");
    need(shape_i("expert_gating_func") == 2, "expert routing is not sigmoid");
    const MetaValue* norm = find("expert_weights_norm");
    need(norm && norm->u != 0, "expert weights are not normalized");
    const MetaValue* scale = find("expert_weights_scale");
    need(!scale || std::fabs((float) scale->f - 1.f) < 1e-6f, "expert weight scale is not 1");

    const MetaValue& sw = must("attention.sliding_window_pattern");
    const MetaValue& rp = must("attention.rope_pattern");
    swa_.resize((size_t) n_layer_);
    rope_.resize((size_t) n_layer_);
    for (int il = 0; il < n_layer_; ++il) {
        swa_[(size_t) il] = layer_value(sw, il) != 0;
        rope_[(size_t) il] = layer_value(rp, il) != 0;
        need(swa_[(size_t) il] == rope_[(size_t) il], "RoPE and sliding-window patterns differ");
    }
    clamp_exp_.assign((size_t) n_layer_, INFINITY);
    clamp_shexp_ = clamp_exp_;
}

void Flint::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a) { return upload(tensor(il, n), blk(il, n), a, 1, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    B.swa = swa_[(size_t) il] != 0;
    B.rope = rope_[(size_t) il] != 0;
    L.fixed = true;
    L.attn_norm = f32("attn_norm.weight", n_embd_);
    B.wq = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * dim_);
    B.wk = mat("attn_k.weight", n_embd_, (int64_t) n_kv_ * dim_);
    B.wv = mat("attn_v.weight", n_embd_, (int64_t) n_kv_ * dim_);
    B.wo = mat("attn_output.weight", (int64_t) n_head_ * dim_, n_embd_);
    B.q_norm = f32("attn_q_norm.weight", dim_);
    B.k_norm = f32("attn_k_norm.weight", dim_);
    B.sinks = f32("attn_sinks.weight", n_head_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    if (il < dense_layers_) {
        B.ffn_gate = mat("ffn_gate.weight", n_embd_, n_ff_);
        B.ffn_up = mat("ffn_up.weight", n_embd_, n_ff_);
        B.ffn_down = mat("ffn_down.weight", n_ff_, n_embd_);
    } else {
        load_moe(il);
    }
}

int Flint::widest() const { return std::max(n_head_ * dim_, n_ff_); }

void Flint::alloc_caches() {
    ring_ = std::min(opt_.max_context, window_ + rows_);
    keep_ = std::min(window_, ring_);
    size_t rings = 0;
    for (const Block& B : blocks_)
        if (B.swa) rings += 2 * (size_t) ring_ * n_kv_ * dim_;
    uint16_t* kv = rings ? (uint16_t*) dmalloc(rings * sizeof(uint16_t), "Flint sliding attention cache") : nullptr;
    const size_t row = (size_t) n_kv_ * dim_;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (B.swa) {
            B.keys = kv; kv += (size_t) ring_ * row;
            B.vals = kv; kv += (size_t) ring_ * row;
        } else {
            B.keys = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
            B.vals = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

void Flint::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * n_head_ * dim_, kv = n * n_kv_ * dim_;
    qraw_ = fbuf(heads); q_ = fbuf(heads);
    kraw_ = fbuf(kv); k_ = fbuf(kv); v_ = fbuf(kv);
    att_ = fbuf(heads);
    score_ = fbuf((size_t) chunk_ * n_head_ * opt_.max_context);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);
}

void Flint::store(const float* x, uint16_t* cache, const Block& B, int64_t pos, int n) {
    const int width = n_kv_ * dim_;
    if (recording_) return store_f16_at(x, cache, width, at_, n, B.swa ? ring_ : 0, st_);
    if (!B.swa) return store_f16(x, cache, width, pos, n, st_);
    const int at = (int) (pos % ring_), first = std::min(n, ring_ - at);
    store_f16(x, cache, width, at, first, st_);
    if (first < n) store_f16(x + (size_t) first * width, cache, width, 0, n - first, st_);
}

void Flint::attn(Block& B, int64_t pos, int n) {
    const long long* at = recording_ ? at_ : nullptr;
    const int rot = B.rope ? n_rot_ : 0;
    lin(B.wq, xn_, qraw_, n, xq_);
    head_norm_rope(qraw_, dim_, fp(B.q_norm), q_, dim_, n_head_, rot, theta_scale_, pos, n, rms_eps_, st_, nullptr,
                   nullptr, 1, at);
    lin(B.wk, xn_, kraw_, n, xq_);
    head_norm_rope(kraw_, dim_, fp(B.k_norm), k_, dim_, n_kv_, rot, theta_scale_, pos, n, rms_eps_, st_, nullptr,
                   nullptr, 1, at);
    lin(B.wv, xn_, v_, n, xq_);
    store(k_, B.keys, B, pos, n);
    store(v_, B.vals, B, pos, n);
    const int window = B.swa ? window_ : 0, ring = B.swa ? ring_ : 0;
    const float scale = 1.f / std::sqrt((float) dim_);
    if (n >= kStageRows)
        attend_batch(blas_, q_, B.keys, B.vals, pos, n, score_, (size_t) chunk_ * n_head_ * opt_.max_context,
                     awork_, att_, dim_, n_head_, n_kv_, scale, st_, window, ring, false, {}, {}, nullptr, nullptr, 0,
                     fp(B.sinks));
    else
        attend(q_, B.keys, B.vals, pos, n, score_, chunk_, att_, dim_, n_head_, n_kv_, scale, st_, window, ring,
               false, at, at ? (B.swa ? std::min(window_ + n - 1, opt_.max_context) : room(n)) : 0, 0, {}, {},
               fp(B.sinks));
    lin(B.wo, att_, ao_, n);
}

void Flint::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    const int all = n * n_embd_;
    rms_norm(r_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, all, 1, st_);
    dbg("attn_in", il, xn_, n_embd_);
    attn(B, pos_, n);
    dbg("attn", il, ao_, n_embd_);
    axpy(ao_, 1.f, r_, all, true, st_);

    rms_norm(r_, fp(L.ffn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, all, 1, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (L.moe) {
        if (!batched(L, n)) moe_router(L, n);
        return;
    }
    lin(B.ffn_gate, xn_, g_, n, xq_);
    lin(B.ffn_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_, INFINITY, st_);
    lin(B.ffn_down, h_, fo_, n);
    axpy(fo_, 1.f, r_, all, true, st_);
}

void Flint::moe_shared(const Layer& L, int il, int n) {
    lin(L.sh_gate, xn_, g_, n, xq_);
    lin(L.sh_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_shexp_, clamp_shexp_[(size_t) il], st_);
    lin(L.sh_down, h_, fo_, n);
}

void Flint::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }

void Flint::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

void Flint::route(int, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    for (int e = 0; e < n_expert_; ++e) probs[e] = 1.f / (1.f + std::exp(-logits[e]));
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e) if (probs[e] > probs[best]) best = e;
        ids[k] = best; w[k] = probs[best]; sum += w[k]; probs[best] = -1.f;
    }
    const float denom = std::max(sum, kMinWeightSum);
    for (int k = 0; k < n_used_; ++k) w[k] /= denom;
}

bool Flint::route_card(int, int n, int32_t* ids, float* w) {
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, nullptr, nullptr, true, 0.f, ids, w, st_);
    return true;
}

size_t Flint::card_row_bytes() const {
    size_t bytes = 0;
    for (const Block& B : blocks_) if (!B.swa) bytes += 2 * (size_t) n_kv_ * dim_ * sizeof(uint16_t);
    return bytes;
}

void Flint::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context) throw std::logic_error("save_rows outside context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t row = (size_t) n_kv_ * dim_ * sizeof(uint16_t), bytes = (size_t) n * row;
    for (const Block& B : blocks_) if (!B.swa) for (const uint16_t* cache : {B.keys, B.vals}) {
        ck(cudaMemcpy(dst, (const uint8_t*) cache + (size_t) first * row, bytes, cudaMemcpyDeviceToHost), "save rows");
        dst += bytes;
    }
}

void Flint::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows outside saved context");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t row = (size_t) n_kv_ * dim_ * sizeof(uint16_t), bytes = (size_t) n * row;
    for (const Block& B : blocks_) if (!B.swa) for (uint16_t* cache : {B.keys, B.vals}) {
        ck(cudaMemcpy((uint8_t*) cache + (size_t) first * row, src + (size_t) (first - from) * row, bytes,
                      cudaMemcpyHostToDevice), "load rows");
        src += (size_t) count * row;
    }
}

template <class F> void Flint::window_runs(int64_t pos, F&& f) const {
    for (int64_t p = std::max<int64_t>(0, pos - keep_); p < pos;) {
        const int64_t at = p % ring_, n = std::min<int64_t>(pos - p, ring_ - at);
        f((size_t) (p - (pos - keep_)), (size_t) at, (size_t) n); p += n;
    }
}

size_t Flint::extra_bytes() const {
    size_t bytes = 0;
    for (const Block& B : blocks_) if (B.swa) bytes += 2 * (size_t) keep_ * n_kv_ * dim_ * sizeof(uint16_t);
    return bytes;
}

void Flint::save_layer(int il, int64_t pos, uint8_t* dst) {
    if (!blocks_[(size_t) il].swa) return;
    dst += state_n_ * sizeof(float);
    const size_t one = 2 * (size_t) keep_ * n_kv_ * dim_ * sizeof(uint16_t);
    for (int j = 0; j < il; ++j) if (blocks_[(size_t) j].swa) dst += one;
    const Block& B = blocks_[(size_t) il];
    const size_t row = (size_t) n_kv_ * dim_ * sizeof(uint16_t);
    for (const uint16_t* cache : {B.keys, B.vals}) {
        std::memset(dst, 0, (size_t) keep_ * row);
        window_runs(pos, [&](size_t slot, size_t at, size_t n) {
            ck(cudaMemcpy(dst + slot * row, (const uint8_t*) cache + at * row, n * row, cudaMemcpyDeviceToHost),
               "save state");
        });
        dst += (size_t) keep_ * row;
    }
}

void Flint::load_extra(const uint8_t* src, int64_t pos) {
    const size_t row = (size_t) n_kv_ * dim_ * sizeof(uint16_t);
    for (const Block& B : blocks_) if (B.swa) for (uint16_t* cache : {B.keys, B.vals}) {
        window_runs(pos, [&](size_t slot, size_t at, size_t n) {
            ck(cudaMemcpy((uint8_t*) cache + at * row, src + slot * row, n * row, cudaMemcpyHostToDevice),
               "load state");
        });
        src += (size_t) keep_ * row;
    }
}

int Flint::head_key(int il, int n) const { return swa_[(size_t) il] ? 0 : room_key(n); }

}  // namespace strata::poly
