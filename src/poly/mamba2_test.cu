// Nemotron-H's Mamba-2 scan on the GPU against the direct CPU recurrence.
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    constexpr int inner = 8, heads = 2, groups = 2, state_dim = 128, rows = 3;
    constexpr float eps = 1e-5f;
    std::vector<float> x(rows * inner), dt(rows * heads), a(heads), b(rows * groups * state_dim),
        c(b.size()), d(heads), z(x.size()), norm(inner), state(inner * state_dim);
    for (size_t i = 0; i < x.size(); ++i) x[i] = float(int(i % 11) - 5) / 9.f;
    for (size_t i = 0; i < z.size(); ++i) z[i] = float(int(i % 7) - 3) / 5.f;
    for (size_t i = 0; i < dt.size(); ++i) dt[i] = float(int(i) - 2) / 4.f;
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = float(int(i % 13) - 6) / 23.f;
        c[i] = float(int(i % 17) - 8) / 29.f;
    }
    for (size_t i = 0; i < state.size(); ++i) state[i] = float(int(i % 19) - 9) / 37.f;
    a = {-0.25f, -0.5f};
    d = {0.5f, -0.25f};
    for (int i = 0; i < inner; ++i) norm[i] = 0.75f + 0.03f * i;

    std::vector<float> want_state = state, want(rows * inner);
    const int head_width = inner / heads, group_width = inner / groups;
    for (int row = 0; row < rows; ++row) {
        for (int ch = 0; ch < inner; ++ch) {
            const int h = ch / head_width, g = h / (heads / groups);
            const float delta = std::log1p(std::exp(dt[row * heads + h]));
            const float decay = std::exp(delta * a[h]);
            float sum = 0.f;
            for (int s = 0; s < state_dim; ++s) {
                const size_t si = (size_t) ch * state_dim + s;
                const size_t bi = ((size_t) row * groups + g) * state_dim + s;
                want_state[si] = want_state[si] * decay + b[bi] * (x[row * inner + ch] * delta);
                sum += want_state[si] * c[bi];
            }
            const float zv = z[row * inner + ch];
            want[row * inner + ch] = (sum + x[row * inner + ch] * d[h]) * zv / (1.f + std::exp(-zv));
        }
        for (int g = 0; g < groups; ++g) {
            float ss = 0.f;
            for (int i = 0; i < group_width; ++i) {
                const float v = want[row * inner + g * group_width + i];
                ss += v * v;
            }
            const float scale = 1.f / std::sqrt(ss / group_width + eps);
            for (int i = 0; i < group_width; ++i)
                want[row * inner + g * group_width + i] *= scale * norm[g * group_width + i];
        }
    }

    const size_t total = x.size() + dt.size() + a.size() + b.size() + c.size() + d.size() + z.size() +
                         norm.size() + state.size() + want.size();
    float* base = nullptr;
    if (cudaMalloc(&base, total * sizeof(float)) != cudaSuccess) return 2;
    size_t at = 0;
    auto put = [&](const std::vector<float>& v) {
        float* p = base + at;
        cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);
        at += v.size();
        return p;
    };
    float *gx = put(x), *gdt = put(dt), *ga = put(a), *gb = put(b), *gc = put(c), *gd = put(d),
          *gz = put(z), *gn = put(norm), *gs = put(state), *go = base + at;
    strata::poly::mamba2_step(gx, gdt, ga, gb, gc, gd, gz, gn, gs, go,
                              inner, heads, groups, state_dim, rows, eps, nullptr);
    std::vector<float> got(want.size()), got_state(state.size());
    cudaMemcpy(got.data(), go, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(got_state.data(), gs, got_state.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(base);
    float worst = 0.f;
    for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::abs(got[i] - want[i]));
    for (size_t i = 0; i < got_state.size(); ++i) worst = std::max(worst, std::abs(got_state[i] - want_state[i]));
    if (worst > 2e-5f) {
        std::fprintf(stderr, "mamba2_step: worst absolute error %.9g\n", worst);
        return 1;
    }
    std::printf("mamba2_step: GPU recurrence and grouped norm match CPU (worst %.3g)\n", worst);
    return 0;
}
