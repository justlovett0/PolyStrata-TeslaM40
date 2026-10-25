// src/poly/flint.hpp - RecursiveSI Flint hybrid sliding/full-attention MoE.
#pragma once

#include "engine.hpp"

namespace strata::poly {

class Flint final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override { return 0; }
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    size_t extra_bytes() const override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    int head_key(int il, int n) const override;

private:
    struct Block {
        bool swa = false, rope = false;
        Dev wq, wk, wv, wo, q_norm, k_norm, sinks;
        Dev ffn_gate, ffn_up, ffn_down;
        uint16_t* keys = nullptr;
        uint16_t* vals = nullptr;
    };

    void attn(Block& B, int64_t pos, int n);
    void store(const float* x, uint16_t* cache, const Block& B, int64_t pos, int n);
    template <class F> void window_runs(int64_t pos, F&& f) const;

    std::vector<Block> blocks_;
    std::vector<uint8_t> swa_, rope_;
    int n_head_ = 0, n_kv_ = 0, dim_ = 0, n_rot_ = 0;
    int window_ = 0, ring_ = 0, keep_ = 0, chunk_ = 1, dense_layers_ = 0;
    float theta_scale_ = 0.f;

    float *qraw_ = nullptr, *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *v_ = nullptr;
    float *att_ = nullptr, *score_ = nullptr, *awork_ = nullptr;
};

}  // namespace strata::poly
