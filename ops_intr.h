#include <immintrin.h>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include "tensor1.h"

// operation functor templates
struct AddOp {
    template<typename T>
    static T scalar(T a, T b) { return a + b; }

    static __m256d simd(__m256d a, __m256d b) { return _mm256_add_pd(a, b); }
    static __m256  simd(__m256 a, __m256 b)   { return _mm256_add_ps(a, b); }
    static __m256i simd_epi32(__m256i a, __m256i b) { return _mm256_add_epi32(a, b); }
};

// template for generic elementwise ops
template <typename Op>
Tensor tensor_elementwise(const Tensor& a_, const Tensor& b_, Backend backend) {
    if (!a_.impl || !b_.impl)
        throw std::runtime_error("Null tensor");

    // pad tensors to same ndim
    size_t ndim = std::max(a_.impl->ndim, b_.impl->ndim);
    Tensor a = pad_to_ndim(a_, ndim);
    Tensor b = pad_to_ndim(b_, ndim);

    // compute broadcasted shape
    std::vector<size_t> result_shape = broadcast_batch_shape_from_vectors(a.shape(), b.shape());
    Tensor result(result_shape, a._dtype());

    size_t n = result.numel_();
    bool a_contig = (a.impl->ndim == 1 || a.impl->strides[a.impl->ndim - 1] == 1);
    bool b_contig = (b.impl->ndim == 1 || b.impl->strides[b.impl->ndim - 1] == 1);

    double* out = reinterpret_cast<double*>(result.impl->storage->data.get());
    double* p_a = reinterpret_cast<double*>(a.impl->storage->data.get());
    double* p_b = reinterpret_cast<double*>(b.impl->storage->data.get());

    size_t width = simd_width(a._dtype(), backend);
    size_t i = 0;

    if (backend != Backend::SCALAR) {
        if (a._dtype() == DType::Double64) {
            for (; i + width <= n; i += width) {
                __m256d va = _mm256_loadu_pd(p_a + (a_contig ? i : 0));
                __m256d vb = _mm256_loadu_pd(p_b + (b_contig ? i : 0));
                __m256d vc = Op::simd(va, vb);
                _mm256_storeu_pd(out + i, vc);
            }
        } else if (a._dtype() == DType::Float32) {
            __m256* pa = reinterpret_cast<__m256*>(p_a);
            __m256* pb = reinterpret_cast<__m256*>(p_b);
            __m256* po = reinterpret_cast<__m256*>(out);
            for (; i + width <= n; i += width) {
                __m256 va = _mm256_loadu_ps(reinterpret_cast<float*>(p_a) + i);
                __m256 vb = _mm256_loadu_ps(reinterpret_cast<float*>(p_b) + i);
                __m256 vc = Op::simd(va, vb);
                _mm256_storeu_ps(reinterpret_cast<float*>(out) + i, vc);
            }
        } else if (a._dtype() == DType::Int32) {
            __m256i* pa = reinterpret_cast<__m256i*>(p_a);
            __m256i* pb = reinterpret_cast<__m256i*>(p_b);
            __m256i* po = reinterpret_cast<__m256i*>(out);
            for (; i + width <= n; i += width) {
                __m256i va = _mm256_loadu_si256(pa + i/width);
                __m256i vb = _mm256_loadu_si256(pb + i/width);
                __m256i vc = Op::simd_epi32(va, vb);
                _mm256_storeu_si256(po + i/width, vc);
            }
        }
    }

    // remaining scalar loop
    for (; i < n; ++i) {
        double va_s = a_contig ? p_a[i] : read_scalar_at(p_a, i, a._dtype());
        double vb_s = b_contig ? p_b[i] : read_scalar_at(p_b, i, b._dtype());
        write_scalar_at(out, i, result._dtype(), Op::scalar(va_s, vb_s));
    }

    return result;
}
