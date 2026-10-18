#include "nemotronh.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace strata::poly {
namespace {
constexpr float kMinWeightSum = 6.103515625e-5f;
constexpr size_t kScoreFloats = size_t(16) << 20;
}

void NemotronH::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    n_mamba_ = n_attn_ = 0;
}

void NemotronH::load_geometry() {
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported Nemotron-H model: ") + what);
    };
    auto arr = [&](const char* key, int fallback) {
        std::vector<int> out((size_t) n_layer_, fallback);
        const MetaValue* v = meta_in(*gg_, key);
        if (!v) return out;
        if (v->type != MetaType::ARRAY || v->items.size() < (size_t) n_layer_)
            throw std::runtime_error(std::string("unsupported Nemotron-H model: incomplete ") + key + " array");
        for (int i = 0; i < n_layer_; ++i) out[(size_t) i] = (int) v->items[(size_t) i].u;
        return out;
    };

    n_layer_ = shape_i("block_count");
    n_embd_ = shape_i("embedding_length");
    const MetaValue* vocab = meta_in(*gg_, "vocab_size");
    const TensorInfo* te = gg_->find("token_embd.weight");
    n_vocab_ = vocab ? (int) vocab->u : te && te->shape.size() == 2 ? (int) te->shape[1] : 0;
    rms_eps_ = shape_f("attention.layer_norm_rms_epsilon");
    n_head_ = shape_i("attention.head_count");
    head_dim_ = n_embd_ / n_head_;
    kv_heads_ = arr("attention.head_count_kv", 0);
    used_ = arr("expert_used_count", 0);
    expert_ff_ = arr("expert_feed_forward_length", 0);
    n_kv_ = *std::max_element(kv_heads_.begin(), kv_heads_.end());
    n_used_ = *std::max_element(used_.begin(), used_.end());
    n_ff_exp_ = *std::max_element(expert_ff_.begin(), expert_ff_.end());
    n_expert_ = shape_i("expert_count");
    n_ff_shexp_ = shape_i("expert_shared_feed_forward_length");
    latent_width_ = shape_i("moe_latent_size");
    conv_width_ = shape_i("ssm.conv_kernel");
    inner_ = shape_i("ssm.inner_size");
    state_dim_ = shape_i("ssm.state_size");
    groups_ = shape_i("ssm.group_count");
    ssm_heads_ = shape_i("ssm.time_step_rank");
    xbc_ = inner_ + 2 * groups_ * state_dim_;
    in_proj_ = inner_ + xbc_ + ssm_heads_;
    const MetaValue* norm = meta_in(*gg_, "expert_weights_norm");
    const MetaValue* scale = meta_in(*gg_, "expert_weights_scale");
    w_norm_ = !norm || norm->u != 0;
    w_scale_ = scale ? (float) scale->num() : 1.f;

    need(n_layer_ > 0 && n_embd_ > 0 && n_embd_ % 32 == 0 && n_vocab_ > 0, "bad base geometry");
    need(n_head_ > 0 && n_embd_ % n_head_ == 0 && n_kv_ > 0 && n_head_ % n_kv_ == 0, "bad attention heads");
    need(conv_width_ == 4 && state_dim_ == 128 && inner_ % ssm_heads_ == 0 && ssm_heads_ % groups_ == 0,
         "unsupported Mamba-2 geometry");
    need(latent_width_ > 0 && n_used_ > 0 && n_used_ <= kMaxUsed && n_expert_ <= 512, "bad MoE geometry");
    draft_ = block_layer_ = false;
    block_ = nullptr;
    clamp_exp_.assign((size_t) n_layer_, INFINITY);
    clamp_shexp_ = clamp_exp_;
}

void NemotronH::load_nemotron_moe(Layer& L, Block& B, int il) {
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    const TensorInfo* up = tensor(il, "ffn_up_exps.weight");
    const TensorInfo* down = tensor(il, "ffn_down_exps.weight");
    if (!up || !down || up->shape.size() != 3 || down->shape.size() != 3)
        throw std::runtime_error(blk(il, "ffn_*_exps.weight") + " is missing or not rank 3");
    const int ne = (int) up->shape[2], ff = expert_ff_[(size_t) il];
    if ((int) up->shape[0] != latent_width_ || (int) up->shape[1] != ff || (int) down->shape[0] != ff ||
        (int) down->shape[1] != latent_width_ || (int) down->shape[2] != ne)
        throw std::runtime_error(blk(il, "ffn_*_exps.weight") + " has unexpected heterogeneous dimensions");
    L.moe = L.relu2 = true;
    L.expert_count = ne;
    L.expert_used = used_[(size_t) il];
    L.expert_ff = ff;
    L.expert_in = L.expert_out = latent_width_;
    L.gate_inp = mat("ffn_gate_inp.weight", n_embd_, ne);
    L.gu_type = up->type;
    L.gu_stride = L.gu_bytes = home(*up).payload(*up) / (size_t) ne;
    L.exp_gate = L.exp_up = home(*up).data(*up);
    L.d_type = down->type;
    L.d_bytes = home(*down).payload(*down) / (size_t) ne;
    L.exp_down = home(*down).data(*down);
    std::string err;
    if (!cpu_expert_types_ok((int) L.gu_type, (int) L.d_type, latent_width_, ff, err))
        throw std::runtime_error(blk(il, "ffn_*_exps.weight: ") + err);
    L.slot.assign((size_t) ne, nullptr);
    B.latent_down = mat("ffn_latent_down.weight", n_embd_, latent_width_);
    B.latent_up = mat("ffn_latent_up.weight", latent_width_, n_embd_);
    B.shared_up = mat("ffn_up_shexp.weight", n_embd_, n_ff_shexp_);
    B.shared_down = mat("ffn_down_shexp.weight", n_ff_shexp_, n_embd_);
    const TensorInfo* bias = tensor(il, "exp_probs_b.bias");
    if (!bias || bias->type != kTypeF32 || (int) bias->elements() != ne)
        throw std::runtime_error(blk(il, "exp_probs_b.bias") + " is missing or has the wrong shape");
    B.bias.resize((size_t) ne);
    std::memcpy(B.bias.data(), home(*bias).data(*bias), (size_t) ne * sizeof(float));
    B.router_bias = upload(bias, blk(il, "exp_probs_b.bias"), ne, 1, kF32);
    ++n_moe_;
}

void NemotronH::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.fixed = true;
    if (used_[(size_t) il] > 0) {
        B.kind = Moe;
        load_nemotron_moe(L, B, il);
    } else if (kv_heads_[(size_t) il] > 0) {
        B.kind = Attention;
        B.wq = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * head_dim_);
        B.wk = mat("attn_k.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        B.wv = mat("attn_v.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        B.wo = mat("attn_output.weight", (int64_t) n_head_ * head_dim_, n_embd_);
        ++n_attn_;
    } else {
        B.kind = Mamba;
        B.ssm_in = mat("ssm_in.weight", n_embd_, in_proj_);
        B.conv = f32("ssm_conv1d.weight", conv_width_, xbc_);
        B.conv_bias = f32("ssm_conv1d.bias", xbc_);
        B.dt_bias = f32("ssm_dt.bias", ssm_heads_);
        B.a = f32("ssm_a", 1, ssm_heads_);
        B.d = f32("ssm_d", 1, ssm_heads_);
        B.ssm_norm = f32("ssm_norm.weight", inner_ / groups_, groups_);
        B.wo = mat("ssm_out.weight", inner_, n_embd_);
        ++n_mamba_;
    }
}

size_t NemotronH::recurrent_floats() const {
    return (size_t) n_mamba_ * ((size_t) 3 * xbc_ + (size_t) inner_ * state_dim_);
}

void NemotronH::alloc_caches() {
    float* state = state_;
    const size_t kv_row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (B.kind == Mamba) {
            B.conv_state = state; state += (size_t) 3 * xbc_;
            B.ssm_state = state; state += (size_t) inner_ * state_dim_;
        } else if (B.kind == Attention) {
            B.keys = (uint16_t*) kv_alloc(kv_row, 1);
            B.vals = (uint16_t*) kv_alloc(kv_row, 1);
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

void NemotronH::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    proj_ = fbuf(n * in_proj_);
    conv_out_ = fbuf(n * xbc_);
    mamba_out_ = fbuf(n * inner_);
    q_ = fbuf(heads); k_ = fbuf(kv); v_ = fbuf(kv); att_ = fbuf(heads);
    score_n_ = std::max((size_t) chunk_ * n_head_ * opt_.max_context,
                        attend_split_floats(chunk_, n_head_, opt_.max_context, head_dim_));
    score_ = fbuf(score_n_);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);
    latent_ = fbuf(n * latent_width_);
    shared_ = fbuf(n * n_embd_);
}

void NemotronH::save_layer(int il, int64_t, uint8_t* dst) {
    const Block& B = blocks_[(size_t) il];
    if (!B.conv_state) return;
    const size_t n = (size_t) 3 * xbc_ + (size_t) inner_ * state_dim_;
    ck(cudaMemcpy(dst + (size_t) (B.conv_state - state_) * sizeof(float), B.conv_state, n * sizeof(float),
                  cudaMemcpyDeviceToHost), "save Nemotron Mamba state");
}

size_t NemotronH::card_row_bytes() const { return (size_t) n_attn_ * 2 * n_kv_ * head_dim_ * sizeof(uint16_t); }

void NemotronH::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context) throw std::logic_error("save rows outside context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    for (const Block& B : blocks_) if (B.keys) for (const uint16_t* p : {B.keys, B.vals}) {
        ck(cudaMemcpy(dst, (const uint8_t*) p + at, bytes, cudaMemcpyDeviceToHost), "save rows"); dst += bytes;
    }
}

void NemotronH::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load rows outside saved context");
    cover(first + n); ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row;
    const size_t bytes = (size_t) n * row, skip = (size_t) (first - from) * row, all = (size_t) count * row;
    for (const Block& B : blocks_) if (B.keys) for (uint16_t* p : {B.keys, B.vals}) {
        ck(cudaMemcpy((uint8_t*) p + at, src + skip, bytes, cudaMemcpyHostToDevice), "load rows"); src += all;
    }
}

void NemotronH::mamba(Block& B, int n) {
    lin(B.ssm_in, xn_, proj_, n, xq_);
    float* z = proj_;
    float* xbc = proj_ + inner_;
    float* dt = proj_ + inner_ + xbc_;
    conv4_silu(xbc, B.conv_state, fp(B.conv), conv_out_, xbc_, n, st_,
               snap_bits_ ? snap_at_ + (B.conv_state - state_) : nullptr, rec_n_, snap_bits_, fp(B.conv_bias));
    add_bias(fp(B.dt_bias), dt, ssm_heads_, n, st_);
    const float* b = conv_out_ + inner_;
    const float* c = b + groups_ * state_dim_;
    mamba2_step(conv_out_, dt, fp(B.a), b, c, fp(B.d), z, fp(B.ssm_norm), B.ssm_state, mamba_out_,
                inner_, ssm_heads_, groups_, state_dim_, n, rms_eps_, st_);
    lin(B.wo, mamba_out_, ao_, n);
}

void NemotronH::attention(Block& B, int64_t pos, int n) {
    const long long* at = recording_ ? at_ : nullptr;
    lin(B.wq, xn_, q_, n, xq_); lin(B.wk, xn_, k_, n, xq_); lin(B.wv, xn_, v_, n, xq_);
    if (at) { store_f16_at(k_, B.keys, n_kv_ * head_dim_, at, n, 0, st_); store_f16_at(v_, B.vals, n_kv_ * head_dim_, at, n, 0, st_); }
    else { store_f16(k_, B.keys, n_kv_ * head_dim_, pos, n, st_); store_f16(v_, B.vals, n_kv_ * head_dim_, pos, n, st_); }
    const float scale = 1.f / std::sqrt((float) head_dim_);
    if (n >= kStageRows)
        attend_batch(blas_, q_, B.keys, B.vals, pos, n, score_, (size_t) chunk_ * n_head_ * opt_.max_context,
                     awork_, att_, head_dim_, n_head_, n_kv_, scale, st_);
    else
        attend(q_, B.keys, B.vals, pos, n, score_, chunk_, att_, head_dim_, n_head_, n_kv_, scale, st_, 0, 0,
               false, at, at ? room(n) : 0, score_n_);
    lin(B.wo, att_, ao_, n);
}

int NemotronH::head_key(int il, int n) const { return blocks_[(size_t) il].kind == Attention ? room_key(n) : 0; }

void NemotronH::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    rms_norm(r_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    if (B.kind == Mamba) mamba(B, n);
    else if (B.kind == Attention) attention(B, pos_, n);
    else {
        lin(L.gate_inp, xn_, rl_, n, xq_);
        lin(B.shared_up, xn_, g_, n, xq_);
        relu2_rows(g_, h_, n_ff_shexp_, n, st_);
        lin(B.shared_down, h_, shared_, n);
        lin(B.latent_down, xn_, latent_, n, xq_);
        ck(cudaMemcpyAsync(xn_, latent_, (size_t) n * latent_width_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
           "Nemotron latent input");
        kernels::native_quantize_q8_1(xn_, xq_, latent_width_ * n, 1, st_);
        copy2(rl_, h_rl_, n * L.expert_count, xn_, h_x_, n * latent_width_, st_);
        return;
    }
    axpy(ao_, 1.f, r_, n * n_embd_, true, st_);
}

void NemotronH::moe_shared(const Layer&, int, int n) {
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * latent_width_ * sizeof(float), cs(st_)), "Nemotron experts");
}

void NemotronH::layer_tail(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    if (!L.moe) return;
    lin(B.latent_up, fo_, ao_, n);
    axpy(shared_, 1.f, ao_, n * n_embd_, true, st_);
    axpy(ao_, 1.f, r_, n * n_embd_, true, st_);
}

void NemotronH::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

void NemotronH::route(int il, const float* logits, int32_t, int* ids, float* w, float* tmp) const {
    const Layer& L = layers_[(size_t) il];
    const int ne = L.expert_count, nu = L.expert_used;
    const auto& bias = blocks_[(size_t) il].bias;
    float* pick = tmp + ne;
    for (int e = 0; e < ne; ++e) { tmp[e] = 1.f / (1.f + std::exp(-logits[e])); pick[e] = tmp[e] + bias[(size_t) e]; }
    float sum = 0.f;
    for (int k = 0; k < nu; ++k) {
        int best = 0; for (int e = 1; e < ne; ++e) if (pick[e] > pick[best]) best = e;
        pick[best] = -INFINITY; ids[k] = best; w[k] = tmp[best]; sum += w[k];
    }
    const float denom = w_norm_ ? std::max(sum, kMinWeightSum) : 1.f;
    for (int k = 0; k < nu; ++k) w[k] = w[k] / denom * w_scale_;
}

} // namespace strata::poly
