// Standalone native sm_52 smoke test: allocation, transfer, kernel launch, and cuBLAS SGEMM.
// Compile with: nvcc -arch=sm_52 tools/maxwell_smoke.cu -lcublas -o build-maxwell/maxwell_smoke.exe
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>

__global__ void add_one(float* data) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 4) data[i] += 1.0f;
}

int main() {
    cudaDeviceProp gpu{};
    if (cudaGetDeviceProperties(&gpu, 0) != cudaSuccess || gpu.major != 5 || gpu.minor != 2) {
        std::fprintf(stderr, "Expected an SM 5.2 device\n");
        return 1;
    }
    float input[4] = {1.f, 2.f, 3.f, 4.f};
    float output[4] = {};
    float* device = nullptr;
    if (cudaMalloc(&device, sizeof(input)) != cudaSuccess ||
        cudaMemcpy(device, input, sizeof(input), cudaMemcpyHostToDevice) != cudaSuccess) return 2;
    add_one<<<1, 4>>>(device);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost) != cudaSuccess) return 3;
    for (int i = 0; i < 4; ++i) if (output[i] != input[i] + 1.f) return 4;

    // Column-major C = A * B, each 2x2.  The expected result is (7, 10, 15, 22).
    const float a[4] = {1.f, 2.f, 3.f, 4.f};
    const float b[4] = {1.f, 2.f, 3.f, 4.f};
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    cublasHandle_t handle = nullptr;
    if (cudaMalloc(&da, sizeof(a)) != cudaSuccess ||
        cudaMalloc(&db, sizeof(b)) != cudaSuccess ||
        cudaMalloc(&dc, sizeof(output)) != cudaSuccess ||
        cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice) != cudaSuccess ||
        cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 5;
    const float one = 1.f, zero = 0.f;
    if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2,
                    &one, da, 2, db, 2, &zero, dc, 2) != CUBLAS_STATUS_SUCCESS ||
        cudaMemcpy(output, dc, sizeof(output), cudaMemcpyDeviceToHost) != cudaSuccess) return 6;
    const float expected[4] = {7.f, 10.f, 15.f, 22.f};
    for (int i = 0; i < 4; ++i) if (std::fabs(output[i] - expected[i]) > 1e-5f) return 7;
    cublasDestroy(handle);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaFree(device);
    std::printf("PASS %s sm_%d%d: kernel, transfer, cuBLAS SGEMM\n", gpu.name, gpu.major, gpu.minor);
    return 0;
}
