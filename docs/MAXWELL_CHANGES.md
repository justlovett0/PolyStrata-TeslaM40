# Tesla M40: changes required for Strata and PolyStrata

Target: Tesla M40 24 GB, CUDA compute capability **5.2** (23,040 MiB VRAM),
Windows driver 528.89. This is an opt-in port; normal newer-GPU builds retain
their existing paths. See [MAXWELL.md](MAXWELL.md) for commands and the full
compatibility audit.

## 1. Native CUDA build

- `CMakeLists.txt` adds `STRATA_EXPERIMENTAL_SM52`. It requires CUDA enabled,
  CUDA **12.x**, and `CMAKE_CUDA_ARCHITECTURES=52`; CUDA 13 cannot emit M40 code.
  It also enables the existing older-GPU (`SM60`) sources and defines the
  Maxwell flag for architecture-specific branches.
- Builds use CUDA 12.0, MSVC 14.39, and `nvcc --use-local-env`. Separate
  `build-maxwell/`, `engine-maxwell/` (`strata.exe`), and
  `engine-poly-maxwell/` (`strata-poly.exe`) keep CUDA 13 release binaries out
  of the M40 launch path. Inspected CUDA images contain native `sm_52` cubins.

## 2. Device detection and setup

- `src/core/device.cu` lowers the runtime floor to SM 5.2 only for the opt-in
  build. `setup.py` recognizes the Tesla M40 and its measured VRAM, selects
  CUDA 12 plus the local `engine-maxwell/` build, and rejects incompatible
  release binaries. The usual `strata-poly` setup route is still guarded;
  GLM uses the separate local launcher below.

## 3. Unsupported instructions and arithmetic

- The M40 has no native `__dp4a`, `__nanosleep`, tensor cores, TF32, native
  BF16 matrix arithmetic, or `cp.async`. Maxwell builds reuse the older-GPU
  signed-byte `vmad` DP4A implementation and spin pause in
  `include/strata/kernels/dp4a.hpp`. Existing FP32/cuBLAS and SIMT fallbacks
  handle GEMM, dequantization, attention, and expert work where dispatched.
- `src/kernels/cuda/qsa_prompt_attn.cu` replaces unsupported packed-half
  `__hsub2` with FP32 subtraction followed by exact FP16 packing on SM 5.2.
  `include/strata/kernels/maxwell_compat.hpp`, `pdl.hpp`, `sampler.cu`, and
  `qsa_select.cu` avoid CUDA 12.0's missing `cudaLaunchKernelEx` cluster/PDL
  API and retain ordinary kernel launches.

## 4. PolyStrata GLM launch limit

- GLM's `kda_step_kernel` in `src/poly/kernels.cu` used 1,024 threads. At
  71 registers/thread, that exceeds the M40's 65,536-register block limit
  and fails the first inference request. The SM 5.2 branch caps the launch at
  **512 threads**, letting each thread process more values. Other GPUs keep
  the original 1,024-thread limit.

## 5. Models, server, and memory

- The Swift IQ3_S launcher uses the native `strata.exe`; the GLM 5.3 Flash
  UD-IQ3_XXS launcher uses native `strata-poly.exe`, four GGUF shards, an
  extracted tokenizer/expert profile, and the existing API server. GLM's
  `run-glm-ud-iq3_xxs.bat` and `strata-glm-ud-iq3_xxs.json` start port 8081
  with `--no-draft`. The first validation used an 8,192-token context; the
  local JSON now requests 180,192 tokens.
- GLM reserves 1,664 MiB because other runtime allocations also use VRAM.
  At the validated 8,192-token context, `nvidia-smi` showed **21,563 MiB used /
  1,406 MiB free** after an API request, below the requested 21,888 MiB (95%)
  use ceiling. Memory use at 180,192 tokens has not been measured here.

## 6. Verification and remaining scope

- On the M40: native CUDA kernel, transfer, and cuBLAS SGEMM smoke tests passed;
  relevant GPU parity tests and setup tests passed. Swift IQ3_S works for the
  machine owner. GLM UD-IQ3_XXS generated coherent API answers (`Paris`, `4`)
  through the clean SM 5.2 build. The measured GLM decode rate was about
  3.3 tokens/s for one short request.
- These results validate those **specific model variants**, not every
  PolyStrata family or quantization. Newer-GPU-only NVFP4/fused kernels and
  untested model paths still need their own fallbacks or validation.
