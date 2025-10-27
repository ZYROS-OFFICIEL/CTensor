// simd_elementwise_verified.h
#pragma once
#include <immintrin.h>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cmath>
#include "tensor1.h"

// ---- Operation functors ----
struct AddOp {
    template<typename T> static T scalar(T a, T b) { return a + b; }
    static __m256d simd_pd(__m256d a, __m256d b) { return _mm256_add_pd(a,b); } // double
    static __m256  simd_ps(__m256 a, __m256 b)     { return _mm256_add_ps(a,b); } // float
    static __m256i simd_epi32(__m256i a, __m256i b){ return _mm256_add_epi32(a,b); } // int32
};
struct SubOp {
    template<typename T> static T scalar(T a, T b) { return a - b; }
    static __m256d simd_pd(__m256d a, __m256d b) { return _mm256_sub_pd(a,b); }
    static __m256  simd_ps(__m256 a, __m256 b)   { return _mm256_sub_ps(a,b); }
    static __m256i simd_epi32(__m256i a, __m256i b){ return _mm256_sub_epi32(a,b); }
};
struct MulOp {
    template<typename T> static T scalar(T a, T b) { return a * b; }
    static __m256d simd_pd(__m256d a, __m256d b) { return _mm256_mul_pd(a,b); }
    static __m256  simd_ps(__m256 a, __m256 b)   { return _mm256_mul_ps(a,b); }
    static __m256i simd_epi32(__m256i a, __m256i b){ return _mm256_mullo_epi32(a,b); } // low 32 bits
};
struct DivOp {
    template<typename T> static T scalar(T a, T b) { return a / b; }
    static __m256d simd_pd(__m256d a, __m256d b) { return _mm256_div_pd(a,b); }
    static __m256  simd_ps(__m256 a, __m256 b)   { return _mm256_div_ps(a,b); }

    // Integer SIMD divide: convert to float, div, convert back (note: different rounding semantics)
    static __m256i simd_epi32(__m256i a, __m256i b) {
        __m256 af = _mm256_cvtepi32_ps(a);
        __m256 bf = _mm256_cvtepi32_ps(b);
        __m256 cf = _mm256_div_ps(af, bf);
        return _mm256_cvtps_epi32(cf); // rounds according to current mode (usually trunc/round-to-nearest)
    }
};
struct PowOp {
    template<typename T> static T scalar(T a, T b) { return std::pow(a,b); }
    // No portable AVX2 SIMD pow here. Fall back to scalar for correctness.
};

// ---- Generic SIMD elementwise ----
template <typename Op>
Tensor tensor_elementwise_simd(const Tensor& a_, const Tensor& b_, Backend backend) {
    if (!a_.impl || !b_.impl) throw std::runtime_error("Null tensor");

    size_t ndim = std::max(a_.impl->ndim, b_.impl->ndim);
    Tensor a = pad_to_ndim(a_, ndim);
    Tensor b = pad_to_ndim(b_, ndim);

    if (!broadcastable(a.shape(), b.shape()))
        throw std::runtime_error("Shapes not compatible for broadcasting");

    std::vector<size_t> result_shape = broadcast_shape(a.shape(), b.shape());
    Tensor result(result_shape, a._dtype());

    size_t n = result.numel_();
    bool a_contig = (a.impl->ndim == 1 || a.impl->strides[a.impl->ndim - 1] == 1);
    bool b_contig = (b.impl->ndim == 1 || b.impl->strides[b.impl->ndim - 1] == 1);

    size_t width = simd_width(a._dtype(), backend); // elements per vector
    size_t i = 0;

    if (backend != Backend::SCALAR) {
        if (a._dtype() == DType::Double64) {
            double* pa = reinterpret_cast<double*>(a.impl->storage->data.get());
            double* pb = reinterpret_cast<double*>(b.impl->storage->data.get());
            double* po = reinterpret_cast<double*>(result.impl->storage->data.get());

            const size_t lane = 256 / 64; // 4 doubles per __m256d
            for (; i + lane <= n; i += lane) {
                __m256d va = a_contig
                    ? _mm256_loadu_pd(pa + i)
                    : _mm256_set_pd(
                        static_cast<double>(read_scalar_at(pa, i + 3, a._dtype())),
                        static_cast<double>(read_scalar_at(pa, i + 2, a._dtype())),
                        static_cast<double>(read_scalar_at(pa, i + 1, a._dtype())),
                        static_cast<double>(read_scalar_at(pa, i    , a._dtype()))
                      );

                __m256d vb;
                if (b_contig) vb = _mm256_loadu_pd(pb + i);
                else if (b.numel() == 1) vb = _mm256_set1_pd(read_scalar_at(pb, 0, b._dtype()));
                else vb = _mm256_set_pd(
                        static_cast<double>(read_scalar_at(pb, i + 3, b._dtype())),
                        static_cast<double>(read_scalar_at(pb, i + 2, b._dtype())),
                        static_cast<double>(read_scalar_at(pb, i + 1, b._dtype())),
                        static_cast<double>(read_scalar_at(pb, i    , b._dtype()))
                      );

                __m256d vc = Op::simd_pd(va, vb);
                _mm256_storeu_pd(po + i, vc);
            }
        }
        else if (a._dtype() == DType::Float32) {
            float* pa = reinterpret_cast<float*>(a.impl->storage->data.get());
            float* pb = reinterpret_cast<float*>(b.impl->storage->data.get());
            float* po = reinterpret_cast<float*>(result.impl->storage->data.get());

            const size_t lane = 256 / 32; // 8 floats per __m256
            for (; i + lane <= n; i += lane) {
                __m256 va = a_contig
                    ? _mm256_loadu_ps(pa + i)
                    : _mm256_set_ps(
                        static_cast<float>(read_scalar_at(pa, i + 7, a._dtype())),
                        static_cast<float>(read_scalar_at(pa, i + 6, a._dtype())),
                        static_cast<float>(read_scalar_at(pa, i + 5, a._dtype())),
                        static_cast<float>(read_scalar_at(pa, i + 4, a._dtype())),
                        static_cast<float>(read_scalar_at(pa, i + 3, a._dtype())),
                        static_cast<float>(read_scalar_at(pa, i + 2, a._dtype())),
                        static_cast<float>(read_scalar_at(pa, i + 1, a._dtype())),
                        static_cast<float>(read_scalar_at(pa, i     , a._dtype()))
                      );

                __m256 vb;
                if (b_contig) vb = _mm256_loadu_ps(pb + i);
                else if (b.numel() == 1) vb = _mm256_set1_ps(static_cast<float>(read_scalar_at(pb, 0, b._dtype())));
                else vb = _mm256_set_ps(
                        static_cast<float>(read_scalar_at(pb, i + 7, b._dtype())),
                        static_cast<float>(read_scalar_at(pb, i + 6, b._dtype())),
                        static_cast<float>(read_scalar_at(pb, i + 5, b._dtype())),
                        static_cast<float>(read_scalar_at(pb, i + 4, b._dtype())),
                        static_cast<float>(read_scalar_at(pb, i + 3, b._dtype())),
                        static_cast<float>(read_scalar_at(pb, i + 2, b._dtype())),
                        static_cast<float>(read_scalar_at(pb, i + 1, b._dtype())),
                        static_cast<float>(read_scalar_at(pb, i     , b._dtype()))
                      );

                __m256 vc = Op::simd_ps(va, vb);
                _mm256_storeu_ps(po + i, vc);
            }
        }
        else if (a._dtype() == DType::Int32) {
            int* pa = reinterpret_cast<int*>(a.impl->storage->data.get());
            int* pb = reinterpret_cast<int*>(b.impl->storage->data.get());
            int* po = reinterpret_cast<int*>(result.impl->storage->data.get());

            const size_t lane = 256 / 32; // 8 ints per __m256i
            for (; i + lane <= n; i += lane) {
                __m256i va = a_contig
                    ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pa + i))
                    : _mm256_set_epi32(
                        static_cast<int>(std::lrint(read_scalar_at(pa, i + 7, a._dtype()))),
                        static_cast<int>(std::lrint(read_scalar_at(pa, i + 6, a._dtype()))),
                        static_cast<int>(std::lrint(read_scalar_at(pa, i + 5, a._dtype()))),
                        static_cast<int>(std::lrint(read_scalar_at(pa, i + 4, a._dtype()))),
                        static_cast<int>(std::lrint(read_scalar_at(pa, i + 3, a._dtype()))),
                        static_cast<int>(std::lrint(read_scalar_at(pa, i + 2, a._dtype()))),
                        static_cast<int>(std::lrint(read_scalar_at(pa, i + 1, a._dtype()))),
                        static_cast<int>(std::lrint(read_scalar_at(pa, i    , a._dtype())))
                      );

                __m256i vb;
                if (b_contig) vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb + i));
                else if (b.numel() == 1) vb = _mm256_set1_epi32(static_cast<int>(std::lrint(read_scalar_at(pb, 0, b._dtype()))));
                else vb = _mm256_set_epi32(
                        static_cast<int>(std::lrint(read_scalar_at(pb, i + 7, b._dtype()))),
                        static_cast<int>(std::lrint(read_scalar_at(pb, i + 6, b._dtype()))),
                        static_cast<int>(std::lrint(read_scalar_at(pb, i + 5, b._dtype()))),
                        static_cast<int>(std::lrint(read_scalar_at(pb, i + 4, b._dtype()))),
                        static_cast<int>(std::lrint(read_scalar_at(pb, i + 3, b._dtype()))),
                        static_cast<int>(std::lrint(read_scalar_at(pb, i + 2, b._dtype()))),
                        static_cast<int>(std::lrint(read_scalar_at(pb, i + 1, b._dtype()))),
                        static_cast<int>(std::lrint(read_scalar_at(pb, i    , b._dtype())))
                      );

                __m256i vc = Op::simd_epi32(va, vb);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(po + i), vc);
            }
        }
    }

    // scalar fallback for leftovers or non-SIMD cases
    for (; i < n; ++i) {
        double va_s = read_scalar_at(a.impl->storage->data.get(), i, a._dtype());
        double vb_s = read_scalar_at(b.impl->storage->data.get(), i, b._dtype());
        write_scalar_at(result.impl->storage->data.get(), i, result._dtype(), Op::scalar(va_s, vb_s));
    }

    return result;
}

// ---- convenience operators that pick default backend ----
// implement or call your CPU-detection function to get a default backend
inline Backend get_default_backend(); // forward-declare; implement elsewhere

inline Tensor operator+(const Tensor& a, const Tensor& b) { return tensor_elementwise_simd<AddOp>(a, b, get_default_backend()); }
inline Tensor operator-(const Tensor& a, const Tensor& b) { return tensor_elementwise_simd<SubOp>(a, b, get_default_backend()); }
inline Tensor operator*(const Tensor& a, const Tensor& b) { return tensor_elementwise_simd<MulOp>(a, b, get_default_backend()); }
inline Tensor operator/(const Tensor& a, const Tensor& b) { return tensor_elementwise_simd<DivOp>(a, b, get_default_backend()); }

// pow: scalar fallback implementation (no AVX2 vector pow here)
inline Tensor pow_(const Tensor& a, const Tensor& b) {
    return tensor_elementwise_simd<PowOp>(a, b, Backend::SCALAR);
}
