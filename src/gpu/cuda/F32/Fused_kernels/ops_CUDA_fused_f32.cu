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
__global__ void binary_fused_kernel(const float* __restrict__ a,const float* __restrict__ b,float* __restrict__ out,size_t n, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(a[i], b[i]);
}

template<typename Op>
__global__ void binary_scalar_kernel(const float* __restrict__ a, const float* __restrict__ b, float*  __restrict__ out, size_t n, float scalar, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(a[i], b[i], scalar);
}

template<typename Op>
__global__ void unary_fused_kernel(const float* __restrict__ in,float*  __restrict__ out,size_t n, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(in[i]);
}

__global__ void scale_shift_kernel(const float* __restrict__ in,   float*   __restrict__ out,   size_t n, float scale, float shift) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * scale + shift;
}

__global__ void bias_add_relu_kernel(const float* __restrict__ x, const float* __restrict__ bias, float*  __restrict__ out, size_t D, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        float v = x[i] + bias[i % D];
        out[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void bias_add_gelu_kernel(const float* __restrict__ x, const float* __restrict__ bias, float*  __restrict__ out, size_t D, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = device_gelu(x[i] + bias[i % D]);
}

template<int BLOCK_SZ>
__global__ void layer_norm_kernel(const float* __restrict__ x,const float* __restrict__ weight,const float* __restrict__ bias,float*       __restrict__ out,size_t D, float eps) {

    size_t row = blockIdx.x;
    const float* xrow = x   + row * D;
    float*       orow = out + row * D;

    using BlockReduce = cub::BlockReduce<float, BLOCK_SZ>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float smean, svar;

    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < D; i += BLOCK_SZ)
        sum += xrow[i];
    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) smean = sum / (float)D;
    __syncthreads();

    float var = 0.0f;
    for (size_t i = threadIdx.x; i < D; i += BLOCK_SZ) {
        float d = xrow[i] - smean;
        var += d * d;
    }
    var = BlockReduce(temp).Sum(var);
    if (threadIdx.x == 0) svar = var / (float)D;
    __syncthreads();

    float rstd = rsqrtf(svar + eps);
    for (size_t i = threadIdx.x; i < D; i += BLOCK_SZ)
        orow[i] = (xrow[i] - smean) * rstd * weight[i] + bias[i];
}

template<typename Op>
static Tensor launch_ternary(const Tensor& a, const Tensor& b, const Tensor& c,
                             const char* name, Op op) {
    if (a.impl->shape != b.impl->shape || a.impl->shape != c.impl->shape)
        throw std::runtime_error(std::string(name) + ": shape mismatch");
    size_t n = numel_of(a);
    Tensor out = alloc_cuda(a.impl->shape);
    auto ag = to_cuda(a), bg = to_cuda(b), cg = to_cuda(c);
    dim3 grid((unsigned)((n + BLOCK - 1) / BLOCK));
    ternary_kernel<<<grid, BLOCK>>>(cuda_ptr(ag), cuda_ptr(bg), cuda_ptr(cg),
                                   cuda_ptr(out), n, op);
    check_launch(name);
    return out;
}

template<typename Op>
static Tensor launch_binary(const Tensor& a, const Tensor& b,
                            const char* name, Op op) {
    if (a.impl->shape != b.impl->shape)
        throw std::runtime_error(std::string(name) + ": shape mismatch");
    size_t n = numel_of(a);
    Tensor out = alloc_cuda(a.impl->shape);
    auto ag = to_cuda(a), bg = to_cuda(b);
    dim3 grid((unsigned)((n + BLOCK - 1) / BLOCK));
    binary_fused_kernel<<<grid, BLOCK>>>(cuda_ptr(ag), cuda_ptr(bg),
                                        cuda_ptr(out), n, op);
    check_launch(name);
    return out;
}

template<typename Op>
static Tensor launch_unary(const Tensor& a, const char* name, Op op) {
    size_t n = numel_of(a);
    Tensor out = alloc_cuda(a.impl->shape);
    auto ag = to_cuda(a);
    dim3 grid((unsigned)((n + BLOCK - 1) / BLOCK));
    unary_fused_kernel<<<grid, BLOCK>>>(cuda_ptr(ag), cuda_ptr(out), n, op);
    check_launch(name);
    return out;
}

Tensor fma_cuda_f32(const Tensor& a, const Tensor& b, const Tensor& c) {
    return launch_ternary(a, b, c, "fma_cuda_f32",
        [] __device__(float x, float y, float z) { return x * y + z; });
}

Tensor fms_cuda_f32(const Tensor& a, const Tensor& b, const Tensor& c) {
    return launch_ternary(a, b, c, "fms_cuda_f32",
        [] __device__(float x, float y, float z) { return x * y - z; });
}

Tensor nfma_cuda_f32(const Tensor& a, const Tensor& b, const Tensor& c) {
    return launch_ternary(a, b, c, "nfma_cuda_f32",
        [] __device__(float x, float y, float z) { return -x * y + z; });
}

Tensor mul_add_cuda_f32(const Tensor& a, const Tensor& b, const Tensor& c) {
    return fma_cuda_f32(a, b, c);
}
Tensor add_scale_cuda_f32(const Tensor& a, const Tensor& b, float scale) {
    if (a.impl->shape != b.impl->shape)
        throw std::runtime_error("add_scale_cuda_f32: shape mismatch");
    size_t n = numel_of(a);
    Tensor out = alloc_cuda(a.impl->shape);
    auto ag = to_cuda(a), bg = to_cuda(b);
    dim3 grid((unsigned)((n + BLOCK - 1) / BLOCK));
    binary_scalar_kernel<<<grid, BLOCK>>>(cuda_ptr(ag), cuda_ptr(bg),
                                         cuda_ptr(out), n, scale,
        [] __device__(float x, float y, float s) { return (x + y) * s; });
    check_launch("add_scale_cuda_f32");
    return out;
}

Tensor add_relu_cuda_f32(const Tensor& a, const Tensor& b) {
    return launch_binary(a, b, "add_relu_cuda_f32",
        [] __device__(float x, float y) { return fmaxf(x + y, 0.0f); });
}

Tensor add_sigmoid_cuda_f32(const Tensor& a, const Tensor& b) {
    return launch_binary(a, b, "add_sigmoid_cuda_f32",
        [] __device__(float x, float y) {
            return 1.0f / (1.0f + expf(-(x + y)));
        });
}

Tensor add_tanh_cuda_f32(const Tensor& a, const Tensor& b) {
    return launch_binary(a, b, "add_tanh_cuda_f32",
        [] __device__(float x, float y) { return tanhf(x + y); });
}

Tensor add_exp_cuda_f32(const Tensor& a, const Tensor& b) {
    return launch_binary(a, b, "add_exp_cuda_f32",
        [] __device__(float x, float y) { return expf(x + y); });
}

Tensor add_ln_cuda_f32(const Tensor& a, const Tensor& b) {
    return launch_binary(a, b, "add_ln_cuda_f32",
        [] __device__(float x, float y) { return logf(x + y); });
}

Tensor swiglu_cuda_f32(const Tensor& a, const Tensor& b) {
    return launch_binary(a, b, "swiglu_cuda_f32",
        [] __device__(float x, float y) {
            float sig = 1.0f / (1.0f + expf(-x));
            return x * sig * y;   // silu(x) * y
        });
}


Tensor exp_neg_cuda_f32(const Tensor& a) {
    return launch_unary(a, "exp_neg_cuda_f32",
        [] __device__(float x) { return expf(-x); });
}

Tensor ln_relu_cuda_f32(const Tensor& a) {
    return launch_unary(a, "ln_relu_cuda_f32",
        [] __device__(float x) { return fmaxf(logf(x), 0.0f); });
}

Tensor sigmoid_ln_cuda_f32(const Tensor& a) {
    return launch_unary(a, "sigmoid_ln_cuda_f32",
        [] __device__(float x) {
            return logf(1.0f / (1.0f + expf(-x)));
        });
}

Tensor silu_cuda_f32(const Tensor& a) {
    return launch_unary(a, "silu_cuda_f32",
        [] __device__(float x) {
            return x / (1.0f + expf(-x));
        });
}

Tensor gelu_cuda_f32(const Tensor& a) {
    return launch_unary(a, "gelu_cuda_f32",
        [] __device__(float x) {
            const float c = 0.7978845608028654f;
            return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
        });
}

Tensor scale_shift_cuda_f32(const Tensor& x, float scale, float shift) {
    size_t n = numel_of(x);
    Tensor out = alloc_cuda(x.impl->shape);
    auto xg = to_cuda(x);
    dim3 grid((unsigned)((n + BLOCK - 1) / BLOCK));
    scale_shift_kernel<<<grid, BLOCK>>>(cuda_ptr(xg), cuda_ptr(out), n, scale, shift);
    check_launch("scale_shift_cuda_f32");
    return out;
}

}