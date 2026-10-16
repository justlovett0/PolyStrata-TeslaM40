// Native SM 5.2 builds reuse the pre-Pascal DP4A and pause implementations.
// CUDA 12.0 has no cudaLaunchKernelEx. Cluster and PDL launch paths cannot run
// on Maxwell anyway; keep their ordinary-kernel fallbacks available.
#pragma once

#include <cuda_runtime.h>
#if defined(__CUDACC__) || defined(__HIPCC__)
#include "strata/kernels/dp4a.hpp"
#endif

#if !defined(__HIPCC__) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
#define STRATA_CUDA_LAUNCH_EX 1
#else
#define STRATA_CUDA_LAUNCH_EX 0
#endif
