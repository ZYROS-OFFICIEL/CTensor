// ops.cpp
#include "ops.h"
#include "tensor.h"
#include <stdexcept>
#include <cmath>
#include <functional>
#include <omp.h>
#include <type_traits>
#include <cstring> // memcpy
#include <iostream>

#if defined(__GNUC__) || defined(__clang__)
  #if defined(__x86_64__) || defined(__i386__)
    #define HAS_BUILTIN_CPU_SUPPORTS 1
  #endif
#endif

// ----------------------------- CPU feature detection -----------------------------
static bool cpu_supports(const char* feat) {
#if defined(HAS_BUILTIN_CPU_SUPPORTS)
    // Available on GCC/Clang for x86/x64
    return __builtin_cpu_supports(feat);
#else
    (void)feat;
    return false;
#endif
}

static bool has_avx2() {
    return cpu_supports("avx2");
}
static bool has_avx512f() {
    return cpu_supports("avx512f");
}

// ----------------------------- Helpers -----------------------------
inline void check_same_shape_or_throw(const Tensor& a, const Tensor& b) {
    auto as = a.shape();
    auto bs = b.shape();
    if (as != bs) throw std::invalid_argument("shape mismatch in binary op");
}

inline size_t dtype_size_bytes(DType dt) {
    return dtype_size(dt);
}

inline bool is_cpu_device(const Tensor& t) {
    if (!t.impl) throw std::runtime_error("Empty tensor in is_cpu_device");
    return (t.impl->data->device.type == DeviceType::CPU);
}

inline bool is_cuda_device(const Tensor& t) {
    if (!t.impl) throw std::runtime_error("Empty tensor in is_cuda_device");
    return (t.impl->data->device.type == DeviceType::CUDA);
}

// ----------------------------- Generic scalar/OpenMP kernels -----------------------------

// Binary elementwise kernel for two tensors (contiguous)
template<typename T, typename BinaryOp>
static void binary_elementwise_contig(const T* a, const T* b, T* out, size_t n, BinaryOp op) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        out[i] = op(a[i], b[i]);
    }
}

// Binary elementwise fallback: handles dtype and contiguity by using contiguous() if needed
template<typename T, typename BinaryOp>
static Tensor binary_elementwise_fallback(const Tensor& A, const Tensor& B, BinaryOp op, DType out_dtype) {
    // Ensure shapes match handled by caller
    Tensor a = A;
    Tensor b = B;
    if (!a.is_contiguous()) a = a.contiguous();
    if (!b.is_contiguous()) b = b.contiguous();

    size_t n = a.numel();
    Tensor out(a.shape(), out_dtype, a.requires_grad());
    T* out_ptr = reinterpret_cast<T*>(out.impl->data->data.get());
    const T* a_ptr = reinterpret_cast<const T*>(a.impl->data->data.get()) + a.impl->offset;
    const T* b_ptr = reinterpret_cast<const T*>(b.impl->data->data.get()) + b.impl->offset;

    binary_elementwise_contig<T>(a_ptr, b_ptr, out_ptr, n, op);
    return out;
}

// Unary elementwise kernel (scalar)
template<typename T, typename UnaryOp>
static Tensor unary_elementwise_fallback(const Tensor& A, UnaryOp op, DType out_dtype) {
    Tensor a = A.is_contiguous() ? A : A.contiguous();
    size_t n = a.numel();
    Tensor out(a.shape(), out_dtype, a.requires_grad());
    const T* a_ptr = reinterpret_cast<const T*>(a.impl->data->data.get()) + a.impl->offset;
    T* out_ptr = reinterpret_cast<T*>(out.impl->data->data.get());

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = op(a_ptr[i]);
    }
    return out;
}

// Reduction sum (flattened)
template<typename T>
static Tensor reduction_sum_fallback(const Tensor& A, int dim, DType out_dtype) {
    Tensor a = A.is_contiguous() ? A : A.contiguous();
    size_t n = a.numel();
    if (dim == -1 || a.impl->ndim == 0) {
        // total sum -> scalar tensor
        double total = 0.0;
        const T* ptr = reinterpret_cast<const T*>(a.impl->data->data.get()) + a.impl->offset;
        #pragma omp parallel for reduction(+ : total)
        for (size_t i = 0; i < n; ++i) total += static_cast<double>(ptr[i]);
        Tensor out({}, out_dtype, false);
        write_scalar_at(out.impl->data->data.get(), 0, out_dtype, total);
        return out;
    } else {
        // simple generic implementation: reduce specified dim
        // Build output shape
        auto shape = a.shape();
        if (dim < 0 || (size_t)dim >= shape.size()) throw std::invalid_argument("invalid reduce dim");
        std::vector<size_t> out_shape = shape;
        size_t reduce_size = out_shape[dim];
        out_shape.erase(out_shape.begin() + dim);
        if (out_shape.empty()) out_shape = {1}; // keep at least scalar

        Tensor out(out_shape, out_dtype, false);
        size_t out_n = out.numel();
        // We'll iterate over output positions and sum along dimension
        // Precompute strides
        const size_t* in_strides = a.impl->strides.data();
        const size_t* in_shape = a.impl->shape.data();

        #pragma omp parallel for
        for (size_t flat = 0; flat < out_n; ++flat) {
            // map flat -> coords for output
            size_t tmp = flat;
            size_t in_idx = a.impl->offset;
            for (int d = (int)out_shape.size()-1; d >= 0; --d) {
                size_t coord = tmp % out_shape[d];
                tmp /= out_shape[d];
                int in_d = (d < (int)dim) ? d : d + 1;
                in_idx += coord * in_strides[in_d];
            }
            double acc = 0.0;
            const T* base_ptr = reinterpret_cast<const T*>(a.impl->data->data.get());
            for (size_t r = 0; r < reduce_size; ++r) {
                acc += static_cast<double>(base_ptr[in_idx + r * in_strides[dim]]);
            }
            write_scalar_at(out.impl->data->data.get(), flat, out_dtype, acc);
        }
        return out;
    }
}
