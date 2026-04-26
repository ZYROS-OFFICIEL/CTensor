#pragma once
#include "core/tensor.h"
#include "core/ops_dispatch.h"

#ifdef USE_CUDA

Tensor add_cuda   (const Tensor& a, const Tensor& b);
Tensor sub_cuda   (const Tensor& a, const Tensor& b);
Tensor mul_cuda   (const Tensor& a, const Tensor& b);
Tensor div_cuda   (const Tensor& a, const Tensor& b);
Tensor pow_cuda   (const Tensor& a, const Tensor& b);
Tensor matmul_cuda(const Tensor& A, const Tensor& B);

Tensor lt_cuda(const Tensor& a, const Tensor& b);
Tensor le_cuda(const Tensor& a, const Tensor& b);
Tensor gt_cuda(const Tensor& a, const Tensor& b);
Tensor ge_cuda(const Tensor& a, const Tensor& b);
Tensor eq_cuda(const Tensor& a, const Tensor& b);
Tensor ne_cuda(const Tensor& a, const Tensor& b);

Tensor abs_cuda     (const Tensor& a);
Tensor sqrt_cuda    (const Tensor& a);
Tensor relu_cuda    (const Tensor& a);
Tensor ln_cuda      (const Tensor& a);
Tensor exp_cuda     (const Tensor& a);
Tensor sin_cuda     (const Tensor& a);
Tensor asin_cuda    (const Tensor& a);
Tensor cos_cuda     (const Tensor& a);
Tensor acos_cuda    (const Tensor& a);
Tensor tan_cuda     (const Tensor& a);
Tensor atan_cuda    (const Tensor& a);
Tensor tanh_cuda    (const Tensor& a);
Tensor sinh_cuda    (const Tensor& a);
Tensor cosh_cuda    (const Tensor& a);
Tensor sigmoid_cuda (const Tensor& a);
Tensor softplus_cuda(const Tensor& a);

Tensor sum_cuda (const Tensor& t, int dim = -1);
Tensor mean_cuda(const Tensor& t, int dim = -1);
Tensor max_cuda (const Tensor& t, int dim = -1);
Tensor min_cuda (const Tensor& t, int dim = -1);

#endif