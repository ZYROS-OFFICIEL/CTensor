#include "ops_cuda_fused_int64.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cmath>
#include <stdexcept>
#include <vector>

#define BLOCK 256

namespace {

inline std::int64_t* cuda_ptr(const Tensor& t) {
    return reinterpret_cast<std::int64_t*>(t.impl->data->data.get()) + t.impl->offset;
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
    Tensor out(shape, DType::Int64);
    out.impl->data = Storage::allocate(n, DType::Int64, Device(DeviceType::CUDA));
    return out;
}

inline Tensor to_cuda(const Tensor& t) {
    return t.device().is_cuda() ? t : t.to(Device(DeviceType::CUDA));
}

__device__ __forceinline__ std::int64_t device_sigmoid(std::int64_t x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ std::int64_t device_silu(std::int64_t x) {
    return x * device_sigmoid(x);
}

__device__ __forceinline__ std::int64_t device_gelu(std::int64_t x) {
    const std::int64_t c = 0.7978845608028654f; 
    std::int64_t inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}
template<typename Op>
__global__ void ternary_kernel(const std::int64_t* __restrict__ a,
                               const std::int64_t* __restrict__ b,
                               const std::int64_t* __restrict__ c,
                               std::int64_t*       __restrict__ out,
                               size_t n, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(a[i], b[i], c[i]);
}

template<typename Op>
__global__ void binary_fused_kernel(const std::int64_t* __restrict__ a,const std::int64_t* __restrict__ b,std::int64_t* __restrict__ out,size_t n, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(a[i], b[i]);
}

template<typename Op>
__global__ void binary_scalar_kernel(const std::int64_t* __restrict__ a, const std::int64_t* __restrict__ b, std::int64_t*  __restrict__ out, size_t n, std::int64_t scalar, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(a[i], b[i], scalar);
}

template<typename Op>
__global__ void unary_fused_kernel(const std::int64_t* __restrict__ in,std::int64_t*  __restrict__ out,size_t n, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(in[i]);
}

__global__ void scale_shift_kernel(const std::int64_t* __restrict__ in,   std::int64_t*   __restrict__ out,   size_t n, std::int64_t scale, std::int64_t shift) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * scale + shift;
}

__global__ void bias_add_relu_kernel(const std::int64_t* __restrict__ x, const std::int64_t* __restrict__ bias, std::int64_t*  __restrict__ out, size_t D, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        std::int64_t v = x[i] + bias[i % D];
        out[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void bias_add_gelu_kernel(const std::int64_t* __restrict__ x, const std::int64_t* __restrict__ bias, std::int64_t*  __restrict__ out, size_t D, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = device_gelu(x[i] + bias[i % D]);
}

template<int BLOCK_SZ>
__global__ void layer_norm_kernel(const std::int64_t* __restrict__ x,const std::int64_t* __restrict__ weight,const std::int64_t* __restrict__ bias,std::int64_t*       __restrict__ out,size_t D, std::int64_t eps) {

    size_t row = blockIdx.x;
    const std::int64_t* xrow = x   + row * D;
    std::int64_t*       orow = out + row * D;

    using BlockReduce = cub::BlockReduce<std::int64_t, BLOCK_SZ>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ std::int64_t smean, svar;

    std::int64_t sum = 0.0f;
    for (size_t i = threadIdx.x; i < D; i += BLOCK_SZ)
        sum += xrow[i];
    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) smean = sum / (std::int64_t)D;
    __syncthreads();

    std::int64_t var = 0.0f;
    for (size_t i = threadIdx.x; i < D; i += BLOCK_SZ) {
        std::int64_t d = xrow[i] - smean;
        var += d * d;
    }
    var = BlockReduce(temp).Sum(var);
    if (threadIdx.x == 0) svar = var / (std::int64_t)D;
    __syncthreads();

    std::int64_t rstd = rsqrtf(svar + eps);
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

Tensor fma_cuda_int64(const Tensor& a, const Tensor& b, const Tensor& c) {
    return launch_ternary(a, b, c, "fma_cuda_int64",
        [] __device__(std::int64_t x, std::int64_t y, std::int64_t z) { return x * y + z; });
}

Tensor fms_cuda_int64(const Tensor& a, const Tensor& b, const Tensor& c) {
    return launch_ternary(a, b, c, "fms_cuda_int64",
        [] __device__(std::int64_t x, std::int64_t y, std::int64_t z) { return x * y - z; });
}

Tensor nfma_cuda_int64(const Tensor& a, const Tensor& b, const Tensor& c) {
    return launch_ternary(a, b, c, "nfma_cuda_int64",
        [] __device__(std::int64_t x, std::int64_t y, std::int64_t z) { return -x * y + z; });
}

Tensor mul_add_cuda_int64(const Tensor& a, const Tensor& b, const Tensor& c) {
    return fma_cuda_int64(a, b, c);
}
Tensor add_scale_cuda_int64(const Tensor& a, const Tensor& b, std::int64_t scale) {
    if (a.impl->shape != b.impl->shape)
        throw std::runtime_error("add_scale_cuda_int64: shape mismatch");
    size_t n = numel_of(a);
    Tensor out = alloc_cuda(a.impl->shape);
    auto ag = to_cuda(a), bg = to_cuda(b);
    dim3 grid((unsigned)((n + BLOCK - 1) / BLOCK));
    binary_scalar_kernel<<<grid, BLOCK>>>(cuda_ptr(ag), cuda_ptr(bg),
                                         cuda_ptr(out), n, scale,
        [] __device__(std::int64_t x, std::int64_t y, std::int64_t s) { return (x + y) * s; });
    check_launch("add_scale_cuda_int64");
    return out;
}

Tensor add_relu_cuda_int64(const Tensor& a, const Tensor& b) {
    return launch_binary(a, b, "add_relu_cuda_int64",
        [] __device__(std::int64_t x, std::int64_t y) { return fmaxf(x + y, 0.0f); });
}

Tensor add_sigmoid_cuda_int64(const Tensor& a, const Tensor& b) {
    return launch_binary(a, b, "add_sigmoid_cuda_int64",
        [] __device__(std::int64_t x, std::int64_t y) {
            return 1.0f / (1.0f + expf(-(x + y)));
        });
}

Tensor add_tanh_cuda_int64(const Tensor& a, const Tensor& b) {
    return launch_binary(a, b, "add_tanh_cuda_int64",
        [] __device__(std::int64_t x, std::int64_t y) { return tanhf(x + y); });
}

Tensor add_exp_cuda_int64(const Tensor& a, const Tensor& b) {
    return launch_binary(a, b, "add_exp_cuda_int64",
        [] __device__(std::int64_t x, std::int64_t y) { return expf(x + y); });
}

Tensor add_ln_cuda_int64(const Tensor& a, const Tensor& b) {
    return launch_binary(a, b, "add_ln_cuda_int64",
        [] __device__(std::int64_t x, std::int64_t y) { return logf(x + y); });
}

Tensor swiglu_cuda_int64(const Tensor& a, const Tensor& b) {
    return launch_binary(a, b, "swiglu_cuda_int64",
        [] __device__(std::int64_t x, std::int64_t y) {
            std::int64_t sig = 1.0f / (1.0f + expf(-x));
            return x * sig * y;   // silu(x) * y
        });
}


Tensor exp_neg_cuda_int64(const Tensor& a) {
    return launch_unary(a, "exp_neg_cuda_int64",
        [] __device__(std::int64_t x) { return expf(-x); });
}

Tensor ln_relu_cuda_int64(const Tensor& a) {
    return launch_unary(a, "ln_relu_cuda_int64",
        [] __device__(std::int64_t x) { return fmaxf(logf(x), 0.0f); });
}

Tensor sigmoid_ln_cuda_int64(const Tensor& a) {
    return launch_unary(a, "sigmoid_ln_cuda_int64",
        [] __device__(std::int64_t x) {
            return logf(1.0f / (1.0f + expf(-x)));
        });
}

Tensor silu_cuda_int64(const Tensor& a) {
    return launch_unary(a, "silu_cuda_int64",
        [] __device__(std::int64_t x) {
            return x / (1.0f + expf(-x));
        });
}

Tensor gelu_cuda_int64(const Tensor& a) {
    return launch_unary(a, "gelu_cuda_int64",
        [] __device__(std::int64_t x) {
            const std::int64_t c = 0.7978845608028654f;
            return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
        });
}

Tensor scale_shift_cuda_int64(const Tensor& x, std::int64_t scale, std::int64_t shift) {
    size_t n = numel_of(x);
    Tensor out = alloc_cuda(x.impl->shape);
    auto xg = to_cuda(x);
    dim3 grid((unsigned)((n + BLOCK - 1) / BLOCK));
    scale_shift_kernel<<<grid, BLOCK>>>(cuda_ptr(xg), cuda_ptr(out), n, scale, shift);
    check_launch("scale_shift_cuda_int64");
    return out;
}

}