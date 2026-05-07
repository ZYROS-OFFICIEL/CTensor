#include "ops_cuda_f32_fused.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cmath>
#include <stdexcept>
#include <vector>

#define BLOCK 256

namespace {

inline float* cuda_ptr(const Tensor& t) {
    return reinterpret_cast<float*>(t.impl->data->data.get()) + t.impl->offset;
}

inline void check_launch(const char* msg) {
#ifndef NDEBUG
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(msg) + " (sync): " + cudaGetErrorString(e));
#else
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
#endif
}

inline void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
}

inline size_t numel_of(const Tensor& t) {
    size_t n = 1;
    for (auto d : t.impl->shape) n *= d;
    return n;
}

inline Tensor alloc_cuda(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (auto d : shape) n *= d;
    Tensor out(shape, DType::Float32);
    out.impl->data = Storage::allocate(n, DType::Float32, Device(DeviceType::CUDA));
    return out;
}

inline Tensor to_cuda(const Tensor& t) {
    return t.device().is_cuda() ? t : t.to(Device(DeviceType::CUDA));
}

__device__ __forceinline__ float device_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float device_silu(float x) {
    return x * device_sigmoid(x);
}

__device__ __forceinline__ float device_gelu(float x) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    float inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}
}