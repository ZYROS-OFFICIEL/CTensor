#pragma once
#include "tensor.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>

// Check for AVX2 support at compile time to avoid errors on non-AVX machines
#if defined(__AVX2__)

// ========================================================================
//                        Binary Operations (Float32)
// ========================================================================

Tensor add_avx2_f32(const Tensor& a, const Tensor& b);
Tensor sub_avx2_f32(const Tensor& a, const Tensor& b);
Tensor mul_avx2_f32(const Tensor& a, const Tensor& b);
Tensor div_avx2_f32(const Tensor& a, const Tensor& b);
Tensor pow_avx2_f32(const Tensor& a, const Tensor& b); // Requires specialized impl or scalar loop
Tensor matmul_avx2_f32(const Tensor& A, const Tensor& B);

// ========================================================================
//                        Comparisons (Float32)
// ========================================================================

Tensor lt_avx2_f32(const Tensor& a, const Tensor& b);
Tensor le_avx2_f32(const Tensor& a, const Tensor& b);
Tensor gt_avx2_f32(const Tensor& a, const Tensor& b);
Tensor ge_avx2_f32(const Tensor& a, const Tensor& b);
Tensor eq_avx2_f32(const Tensor& a, const Tensor& b);
Tensor ne_avx2_f32(const Tensor& a, const Tensor& b);

// ========================================================================
//                        Unary Operations (Float32)
// ========================================================================

Tensor abs_avx2_f32(const Tensor& a);
Tensor sqrt_avx2_f32(const Tensor& a);
Tensor relu_avx2_f32(const Tensor& a);
Tensor ln_avx2_f32(const Tensor& a);
Tensor exp_avx2_f32(const Tensor& a);
Tensor sin_avx2_f32(const Tensor& a);
Tensor asin_avx2_f32(const Tensor& a);
Tensor cos_avx2_f32(const Tensor& a);
Tensor acos_avx2_f32(const Tensor& a);
Tensor tan_avx2_f32(const Tensor& a);
Tensor atan_avx2_f32(const Tensor& a);
Tensor tanh_avx2_f32(const Tensor& a);
Tensor sinh_avx2_f32(const Tensor& a);
Tensor cosh_avx2_f32(const Tensor& a);
Tensor sigmoid_avx2_f32(const Tensor& a);
Tensor softplus_avx2_f32(const Tensor& a);

// ========================================================================
//                        Reductions (Float32)
// ========================================================================

Tensor sum_avx2_f32(const Tensor& t, int dim = -1);
Tensor mean_avx2_f32(const Tensor& t, int dim = -1);
Tensor max_avx2_f32(const Tensor& t, int dim = -1);
Tensor min_avx2_f32(const Tensor& t, int dim = -1);

#endif // __AVX2__