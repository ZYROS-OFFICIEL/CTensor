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
    const float c = 0.7978845608028654f; 
    float inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}
template<typename Op>
__global__ void ternary_kernel(const float* __restrict__ a,
                               const float* __restrict__ b,
                               const float* __restrict__ c,
                               float*       __restrict__ out,
                               size_t n, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(a[i], b[i], c[i]);
}

template<typename Op>
__global__ void binary_fused_kernel(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float*       __restrict__ out,
                                    size_t n, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(a[i], b[i]);
}

template<typename Op>
__global__ void binary_scalar_kernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float*       __restrict__ out,
                                     size_t n, float scalar, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(a[i], b[i], scalar);
}

template<typename Op>
__global__ void unary_fused_kernel(const float* __restrict__ in,
                                   float*       __restrict__ out,
                                   size_t n, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(in[i]);
}

__global__ void scale_shift_kernel(const float* __restrict__ in,
                                   float*       __restrict__ out,
                                   size_t n, float scale, float shift) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * scale + shift;
}

__global__ void bias_add_relu_kernel(const float* __restrict__ x,
                                     const float* __restrict__ bias,
                                     float*       __restrict__ out,
                                     size_t D, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        float v = x[i] + bias[i % D];
        out[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void bias_add_gelu_kernel(const float* __restrict__ x,
                                     const float* __restrict__ bias,
                                     float*       __restrict__ out,
                                     size_t D, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = device_gelu(x[i] + bias[i % D]);
}
}