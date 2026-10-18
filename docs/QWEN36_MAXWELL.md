# Qwen3.6 35B-A3B on the Tesla M40

## Launcher

- Run `run-qwen36-ud-q4_k_s.bat`; the web UI and OpenAI-compatible API use port 8083.
- The launcher validates the 21,388,319,008-byte GGUF, Maxwell engine, and extracted tokenizer before loading.
- The current JSON uses a 120,768-token context, a 2,048-token prompt batch, MTP speculative decoding, and `--tail-skip 6`.
- `--vram-reserve-mib 1152` gives Strata 21,888 MiB, exactly 95% of the M40's 23,040 MiB.

## Model preparation

- Architecture `qwen35moe` is handled by PolyStrata's existing Qwen3.5/Qwen3.6 backend.
- The embedded 248,320-token Qwen tokenizer, merges, special-token map, and chat template were extracted to
  `D:\polystrata-maxwell\PolyStrata-data\packs\qwen36-ud-q4_k_s\tokenizer`.
- The model carries one MTP draft layer, so no second draft download is required.
- The expert profile is learned and written beside the tokenizer after use.

## M40 validation

- Full production load: 31.6 seconds; 10,095 of 10,496 routed experts cached in 16.8 GiB.
- Observed Strata process memory: 21,803 to 21,804 MiB, below the 21,888 MiB ceiling.
- A chat request returned `OK` with a normal stop; generation reached 43 tokens/s with 61 of 72 draft tokens accepted.
- An 814-token request exercised the batched prompt path successfully; its reported compute throughput was
  214.7 prompt tokens/s and generation reached 52.5 tokens/s.

## Long-generation VRAM fix

- Attention graphs are captured lazily for 1K, 2K, 4K and larger position ranges. On CUDA 12.0 with the M40's
  528.89 Windows driver, graph executable storage exhausted the startup reserve near token 1,295.
- The M40 driver exposes no CUDA virtual memory management. If graph creation reports OOM, the engine now rebuilds
  the partial expert cache with 256 MiB more headroom and retries the same graph. Dense weights, KV cache and model
  state stay loaded; only lower-priority routed experts move from GPU cache to their existing CPU path.
- A cold first request generated 1,800 tokens at 44.6 tokens/s, then a second request returned `OK`. Observed total
  GPU use was 21,846 MiB and the Strata process held 21,804 MiB, both below the 21,888 MiB target.
