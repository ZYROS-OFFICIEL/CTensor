#pragma once
#include "tensor.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>


// ========================================================================
//                        Binary Operations (Double64)
// ========================================================================

Tensor add_avx512_d64(const Tensor& a, const Tensor& b);
Tensor sub_avx512_d64(const Tensor& a, const Tensor& b);
Tensor mul_avx512_d64(const Tensor& a, const Tensor& b);
Tensor div_avx512_d64(const Tensor& a, const Tensor& b);
Tensor pow_avx512_d64(const Tensor& a, const Tensor& b); // Requires specialized impl or scalar loop
Tensor matmul_avx512_d64(const Tensor& A, const Tensor& B);

// ========================================================================
//                        Comparisons (Double64)
// ========================================================================

Tensor lt_avx512_d64(const Tensor& a, const Tensor& b);
Tensor le_avx512_d64(const Tensor& a, const Tensor& b);
Tensor gt_avx512_d64(const Tensor& a, const Tensor& b);
Tensor ge_avx512_d64(const Tensor& a, const Tensor& b);
Tensor eq_avx512_d64(const Tensor& a, const Tensor& b);
Tensor ne_avx512_d64(const Tensor& a, const Tensor& b);

// ========================================================================
//                        Unary Operations (Double64)
// ========================================================================

Tensor abs_avx512_d64(const Tensor& a);
Tensor sqrt_avx512_d64(const Tensor& a);
Tensor relu_avx512_d64(const Tensor& a);
Tensor ln_avx512_d64(const Tensor& a);
Tensor exp_avx512_d64(const Tensor& a);
Tensor sin_avx512_d64(const Tensor& a);
Tensor asin_avx512_d64(const Tensor& a);
Tensor cos_avx512_d64(const Tensor& a);
Tensor acos_avx512_d64(const Tensor& a);
Tensor tan_avx512_d64(const Tensor& a);
Tensor atan_avx512_d64(const Tensor& a);
Tensor tanh_avx512_d64(const Tensor& a);
Tensor sinh_avx512_d64(const Tensor& a);
Tensor cosh_avx512_d64(const Tensor& a);
Tensor sigmoid_avx512_d64(const Tensor& a);
Tensor softplus_avx512_d64(const Tensor& a);

// ========================================================================
//                        Reductions (Double64)
// ========================================================================

Tensor sum_avx512_d64(const Tensor& t, int dim = -1);
Tensor mean_avx512_d64(const Tensor& t, int dim = -1);
Tensor max_avx512_d64(const Tensor& t, int dim = -1);
Tensor min_avx512_d64(const Tensor& t, int dim = -1);
