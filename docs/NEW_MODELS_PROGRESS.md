# New model compatibility progress (2026-10-25)

This is the continuation record for the four models added to the M40 machine: Qwen3.5 122B-A10B, LFM2.5
24B-A2B, DeepSeek V4.1 Flash 1017, and DiffusionGemma 26B-A4B. The installed GPU is a Tesla M40 24 GB
(`sm_52`), and the intended Strata ceiling remains 21,888 MiB with 1,152 MiB left for the runtime unless a
backend has a stricter requirement.

## Qwen3.5 122B-A10B Q4_K_M

- Model: `D:\models\Qwen35-122\Q4_K_M\Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf` (three shards,
  approximately 71.2 GB).
- Launcher/config: `run-qwen35-122b-q4_k_m.bat` and `strata-qwen35-122b-q4_k_m.json`; it is also present in
  `run-polystrata-m40.bat` with 32K recommended and 64K near-limit choices.
- Tokenizer extracted to `D:\polystrata-maxwell\PolyStrata-data\packs\qwen35-122b-q4_k_m\tokenizer`.
- A direct engine test generated a token on the M40. A later full server start reached API-ready at 32K and loaded
  2,414 of 12,288 routed experts into 12.8 GiB of expert VRAM.
- The final API response test was interrupted while the DiffusionGemma CUDA build saturated the same disk/CPU. It
  had completed its 24-token prompt and started generation, so loading and execution were confirmed, but no final
  coherent API response was recorded. Re-run this test with no compiler or large download contending for drive D:.
- The configured expert profile path does not exist yet. A representative profile/calibration run should be made;
  this is the main remaining performance task for a 71 GB routed model whose uncached experts come from system RAM.

## LFM2.5 24B-A2B UD-Q4_K_XL

- Model: `D:\models\LFM2.5-24B\LFM2.5-24B-A2B-UD-Q4_K_XL.gguf` (approximately 14.9 GB).
- Launcher/config: `run-lfm25-24b-q4_k_xl.bat` and `strata-lfm25-24b-q4_k_xl.json`; it is also in the unified
  launcher with 64K recommended and 128K near-limit choices.
- Tokenizer extracted to `D:\polystrata-maxwell\PolyStrata-data\packs\lfm25-24b-q4_k_xl\tokenizer`.
- Fully tested through `serve/server.py` and the OpenAI `/v1/chat/completions` API. All experts remained resident,
  the cache reported 100% hits, and the stopped answer was coherent at approximately 54 tok/s.
- This preset is complete for the tested text path. The model spends many tokens in reasoning; clients requesting
  very small `max_tokens` values may receive only `reasoning_content`. A future preset default such as
  `reasoning_budget_tokens: 256` is optional rather than a compatibility fix.

## DeepSeek V4.1 Flash 1017 GSQ-RCO IQ2_XS

- Backbone: `D:\models\deep4-1-1017-rco-2bxs\DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf`
  (four shards, approximately 155 GB).
- This architecture is `deepseek41`; it must not be aliased to the older `deepseek4` implementation.
- Imported the current DSV4.1, Maya, trellis/TQ CPU, packer, and metadata support. Added SM52 DP4A, half arithmetic,
  BF16/E4M3 conversion, grid-constant, and tensor-core-free fallbacks.
- `strata-dsv41.exe` compiled successfully for SM52 and both `--version` and `--help` smoke tests passed. The local
  output is `engine-dsv41-maxwell\strata-dsv41.exe`; rebuild with `build-dsv41-maxwell.bat` or the expanded
  `build-strata-maxwell.bat`.
- Launcher/config: `run-deepseek41-1017-iq2_xs.bat` and `strata-deepseek41-1017-iq2_xs.json`; the unified launcher
  calls `tools\prepare_deepseek41_launch.ps1` before starting it.
- Required EnGram download (110,595,290,816 bytes):

  ```powershell
  & "$env:USERPROFILE\.local\bin\hf.exe" download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf --local-dir "D:\models\deep4-1-1017-rco-2bxs"
  ```

- The ISTA GGUF lacks the token map needed by the EnGram packer. Download the first Unsloth Q8 metadata shard; its
  weights are referenced for metadata only and are not copied into the pack:

  ```powershell
  & "$env:USERPROFILE\.local\bin\hf.exe" download unsloth/DeepSeek-V4.1-Flash-1017-GGUF UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf --local-dir "D:\models\deep4-1-1017-rco-2bxs"
  ```

- Once both files finish, run `tools\prepare_deepseek41_launch.ps1`. It verifies the four backbone shards and exact
  EnGram size, then builds the zero-copy pack at `D:\models\deep4-1-1017-rco-2bxs\maya-ds4`.
- Full model load and inference are untested because the EnGram and metadata downloads were incomplete.

## DiffusionGemma 26B-A4B Q4_K_M

- Model: `D:\models\diffgemma26\diffusiongemma-26B-A4B-it-Q4_K_M.gguf` (15.64 GiB).
- This is `diffusion-gemma`, a 256-token block-diffusion architecture. The normal autoregressive `gemma4` and
  `strata-poly` paths are incompatible.
- Built the DiffusionGemma llama.cpp PR backend from `D:\polystrata-maxwell\llama-latest` for native SM52. The
  build disables CUDA graphs, flash attention, VMM, AVX2, FMA, F16C and BMI2. AVX2 initially caused Windows
  `0xC000001D` on this machine; the corrected AVX baseline starts normally.
- `tools\diffusiongemma\diffusion-gemma-visual-server.cpp` adds PolyStrata's resident `--serve`, `GEN`, `T`, and
  `DONE` protocol. `build-diffusiongemma-maxwell.bat` copies that source into the sibling llama checkout, builds it,
  and copies the executable/DLLs into the ignored `engine-diffusiongemma-maxwell` output directory.
- Tokenizer extraction succeeded at
  `D:\polystrata-maxwell\PolyStrata-data\packs\diffusiongemma-26b-a4b-q4_k_m\tokenizer` (262,144 vocabulary,
  514,906 merges).
- Hardware startup test passed: all 31 layers offloaded, 16,013.20 MiB of model data occupied CUDA memory, a 2,048
  token noncausal context reserved 632.02 MiB of CUDA compute memory, and the engine reported `READY 1792` after
  reserving one 256-token diffusion canvas.
- No `GEN` request was sent before work stopped. The protocol bridge therefore still needs one direct token test,
  followed by a full PolyStrata API test. A model JSON/BAT and unified-launcher entry also remain to be created.
- The upstream branch prints extremely verbose graph diagnostics. Quieting those logs is optional after functional
  inference passes.

## Suggested continuation order

1. Finish the two DeepSeek downloads, run the preparation script, then perform a one-token engine test and a short
   OpenAI API request.
2. Start the DiffusionGemma engine at `--max-context 2048`, send a small `GEN` line using tokens produced by the
   extracted tokenizer, validate `T`/`DONE`, then create its JSON/BAT and unified-launcher entry.
3. Re-run Qwen3.5 without concurrent compilation/download I/O, record a coherent stopped API answer, and generate a
   representative expert profile if the backend's profiling flow supports this architecture.
4. Re-run the unified launcher syntax/start checks for all new entries. Do not modify or add the local generated
   `llama-dense-m40-last-run.json`.
