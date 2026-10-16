# Tesla M40 (experimental work in progress)

This fork targets the Tesla M40's native CUDA compute capability 5.2. On the test
machine, `nvidia-smi` reports **Tesla M40 24GB**, **23,040 MiB**, and driver
**528.89**. CUDA 12.0 can generate `sm_52` code; CUDA 13 cannot. The normal
PolyStrata release binaries do not contain code for this card.

## Build contract

The opt-in build is `-DSTRATA_EXPERIMENTAL_SM52=ON` with
`-DSTRATA_ENABLE_CUDA=ON` and `-DCMAKE_CUDA_ARCHITECTURES=52`. It requires a
CUDA 12.x toolkit. Its separate build and installed engine folders are
`build-maxwell/` and `engine-maxwell/`. The flag also enables the existing
Pascal compatibility sources, including the software DP4A and FP32 BF16 GEMM
path. A native `sm_52` image is required; a PTX or cubin image for SM 6.x will
not execute on the M40.

Example from an MSVC 14.39 developer command prompt after installing CUDA
12.0, CMake, and Ninja. CUDA 12.0's `nvcc` needs `--use-local-env` with this
toolchain:

```bat
cmake -S . -B build-maxwell -G Ninja -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM52=ON -DCMAKE_CUDA_ARCHITECTURES=52 -DCMAKE_CUDA_FLAGS=--use-local-env -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe"
cmake --build build-maxwell --target strata-device strata -j 4
build-maxwell\strata-device.exe
```

`setup.py` also builds a separate `engine-maxwell/` and selects it on an M40.
No model is installed on this machine, so this is not yet a validated
inference recipe. The smallest model offered by this checkout is over 5 GB;
the machine owner has deferred larger downloads.

## When model weights are available

For the Strata path, a first candidate is the pinned
[Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf)
from `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` (22.9 GB). Place the complete file in a
directory of your choice, then from this checkout run:

```bat
START-HERE.bat --yes --family qwen36-strata --model UD-Q4_K_XL --gguf-dir "D:\path\to\model-directory" --context 8192 --no-start
```

`setup.py` checks the pinned file size and SHA-256, extracts the tokenizer,
and writes a model launch script. It must select `engine-maxwell/strata.exe`.
Start with one token and a short deterministic prompt, then repeat requests,
grow the context, and call the existing OpenAI-compatible API. Record output
tokens and GPU activity before claiming inference success. The script above has
not been run because no model is installed.

## Compatibility audit

| Source | SM 5.2 issue | Intended fallback | Difficulty / state |
| --- | --- | --- | --- |
| `CMakeLists.txt`, `src/core/device.cu` | Build and runtime floors were SM 6.0 even in the older GPU mode | Separate opt-in SM 5.2 flag and runtime floor | Low; full engine built and device probe ran |
| `include/strata/kernels/dp4a.hpp`; `src/kernels/cuda/iq_kernels.cu`, `native_mmvq.cu`, `s2_expert_grouped.cu` | Native `__dp4a` starts at SM 6.1 | Reuse the existing signed-byte `vmad` implementation for `< 610` | Medium; compiled, native expert parity passed |
| `include/strata/kernels/dp4a.hpp`, `src/kernels/cuda/verify_kernels.cu` | `__nanosleep` starts at SM 7.0 | Existing spin pause for `< 700` | Low; full engine compiled |
| `src/prefill/gemm.cu` | No native BF16 matrix arithmetic or tensor cores | Existing older-GPU conversion and cuBLAS fallback | Medium; BF16 GEMM parity passed on M40 |
| `src/kernels/cuda/qsa_select.cu`, `qsa_prompt_attn.cu`, `src/prefill/moe_fused*.cu` | TF32/FP16/INT8 MMA and `cp.async` are unavailable | Existing FP32 tiled or warp scores, older prompt attention, and unfused expert paths selected by device capability | Medium; dispatch audit and runtime pending |
| `src/kernels/cuda/qsa_prompt_attn.cu` | SM 5.2 has no `__hsub2` packed-half subtraction | Convert the two signed bytes through FP32, then pack exact FP16 values | Low; native `sm_52` translation unit compiled |
| `src/kernels/cuda/sampler.cu`, `qsa_select.cu`, `include/strata/kernels/pdl.hpp` | CUDA 12.0 has no `cudaLaunchKernelEx` (cluster/PDL launch), and SM 5.2 cannot run clusters | Compile out those launch branches and keep their existing ordinary-kernel paths | Low; full build passed |
| `src/kernels/cuda/native_mmvq.cu`, `dequant_bf16.cu` | Pascal tuning tables are measured for SM 6.0/6.1, not SM 5.2 | Default SIMT MMVQ and ordinary dequantization until M40 parity and timing pass | Medium; runtime pending |
| `src/prefill/gdn_rec_parity.cu`, `src/prefill/kernels.cu` | `cp.async` and high shared-memory variants require newer cards | Existing pre-SM 8.0 kernels; keep parity-only probes out of normal inference | Medium; GDN recurrence parity passed |
| `src/poly/kernels*.cu`, `src/poly/` | Second engine has its own quantized kernels and dispatch | Port after the original `strata` path generates valid tokens | High; not yet validated |
| `setup.py` | SM 5.2 was rejected and setup could choose incompatible CUDA 12 release files | Recognize 5.2, require a local CUDA 12 build in `engine-maxwell/` | Medium; local build and 25 setup tests passed |

The newer prompt attention parity program intentionally refuses SM 5.2; the
runtime chooses its older decode attention kernel there. The two fused prefill
tests skip SM 5.2 by design because their kernels require SM 8.0. Model
loading, end-to-end prefill, sampling, and server behavior remain untested.

The `kv_hybrid_parity` harness now treats that prompt-attention refusal as the
expected Maxwell dispatch and verifies its decode fallback against a host
reference. The MSVC 14.39 Release build of `sampler_parity` gave two
stack-layout-sensitive false results: adding mismatch diagnostics or compiling
only that host harness with `/Od` made all three sampler variants pass. The
production CUDA kernels remain optimized. The exact cause of the harness
behavior is unresolved, so a real-model sampling check is still required.

## Model compatibility on the M40

All entries below are **untested on SM 5.2**. File sizes and architectures are
from `docs/BACKENDS.md`; they do not establish hardware support.

| Family | Engine | M40 state |
| --- | --- | --- |
| Qwen3.8 Flash Next, Coder, Swift 1.5, Unsloth variants | `strata` | First validation target; untested |
| Qwen3.8 NVFP4 | `strata` | Newer GPU quantization path; unavailable pending a separate fallback |
| Qwen3.6 35B A3B (`qwen36-strata`) | `strata` | Untested |
| GLM 5.3 Flash | `strata-poly` | Untested; very large host RAM requirement |
| DeepSeek V4 Flash | `strata-poly` | Untested; very large host RAM requirement |
| LFM2.5 24B A2B | `strata-poly` | Untested |
| Qwen3.6 35B A3B, Ornith 1.5, Tiel Coder, MiniCPM-V 4.7 | `strata-poly` | Untested |
| Gemma 4 26B A4B | `strata-poly` | Untested |

## Verification record

| Check | Result |
| --- | --- |
| GPU identification | Tesla M40 24GB, 23,040 MiB, compute capability 5.2 |
| Driver | 528.89 |
| CUDA 12 compiler | CUDA 12.0.140 installed; `nvcc --list-gpu-code` includes `sm_52` |
| Native kernel | Passed on the M40 with CUDA 12.0.140 and MSVC 14.39; `cuobjdump --list-elf` shows `sm_52` cubins |
| Memory transfer and cuBLAS SGEMM | `tools/maxwell_smoke.cu` passed on the M40 (expected 2x2 product and round-trip transfer) |
| `strata-device` | Built and ran on the M40: compute capability 5.2, 24 SMs, 22.432 GiB visible VRAM; its cubin is `sm_52` |
| Full engine build | `strata.exe --families` ran from both `build-maxwell/` and `engine-maxwell/`, reporting `qwen4exp qwen35moe`; every inspected CUDA image contains `sm_52` cubins only |
| GPU parity | Attention selection and top-k, eight native expert format pairs plus Q5_1 and BF16 embedding, GDN recurrence, BF16 GEMM, wide/S2 dequantization, MMVQ, DMA, RoPE, KV stream/hybrid/quantization, routing, GEMV, and three sampler paths passed on the M40. Two SM 8.0 fused prefill tests skipped. The sampler host harness needs `/Od` on MSVC 14.39, as described above. |
| Setup routing | 25 Python unit tests passed; real M40 local engine build wrote `engine-maxwell/BUILD.json` with arch 52 and CUDA 12 |
| `setup.py --check` | Detects Tesla M40 24GB, SM 5.2, CUDA 12, 192 GB RAM, and AVX-only CPU; labels `strata-poly` families unported |
| Model load, coherent token generation, server API | Pending a supported model; none installed, and all offered models exceed the owner's 5 GB download limit |

## Change log

- `CMakeLists.txt`: adds the opt-in native SM 5.2 configuration and CUDA 12 guard.
- `src/core/device.cu`: admits SM 5.2 in the Maxwell build only.
- `include/strata/kernels/maxwell_compat.hpp`, `include/strata/kernels/pdl.hpp`, `src/kernels/cuda/sampler.cu`, `src/kernels/cuda/qsa_select.cu`: reuse the old-GPU arithmetic layer and guard cluster launch APIs absent from CUDA 12.0.
- `setup.py`: recognizes SM 5.2 and directs its local build to separate folders.
- `tools/maxwell_smoke.cu`: native kernel, transfer, and cuBLAS hardware check.
- `docs/MAXWELL.md`: records the audit, test status, and model matrix.
