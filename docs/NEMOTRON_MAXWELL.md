# Nemotron 3.5 Bolt on Maxwell

## Model and launcher

- GGUF: `nemotron_h_moe`, UD-Q4_K_XL, 88 layers, 4096 embedding, 131072 vocabulary.
- Run `run-nemotron-ud-q4_k_xl.bat`; it verifies all three shards, starts the API on port 8082, and opens the UI.
- The JSON uses a 32768-token context, prompt batch 1, decode window 1, and no speculative draft.
- `--vram-reserve-mib 1152` gives Strata 21888 MiB, exactly 95% of the M40's 23040 MiB.

## Backend changes

- Added a `nemotron_h_moe` loader for the model's hybrid Mamba-2 layers, 8 GQA layers, and heterogeneous MoE geometry.
- Added native SM 5.2 Mamba-2 recurrence, SiLU gating, grouped RMS norm, and biased causal convolution. Its hardware
  CPU/GPU test had a worst absolute difference of `5.96e-07`.
- Added CUDA and CPU `relu(x)^2` activation, latent down/up projections, shared experts, sigmoid routing with
  selection bias, normalized top-k weights, and weight scaling. The activation test matched 14/14 values exactly.
- Extended expert execution and caching for per-layer expert counts, selected counts, FFN widths, and separate
  expert input/output widths. Existing Swift and GLM model paths remain supported.
- Increased retained GGUF metadata-array geometry beyond 64 entries so all 88 layer descriptors are available.

## Hardware test

- The complete 50.2 GB model loaded on the Tesla M40 in 24.4 seconds.
- PolyStrata cached 3292 of 13824 routed experts in 14.4 GiB; observed process VRAM was 21879 MiB.
- `/v1/chat/completions` completed a real chat prompt with `OK` and a clean stop at about 10.6 generated tokens/s.
- The live server reported the configured 32768-token context and passed shard, health, model-list, and chat tests.
