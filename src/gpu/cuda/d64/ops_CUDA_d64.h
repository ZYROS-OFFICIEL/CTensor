#pragma once
#include "tensor.h"

#ifdef USE_CUDA

Tensor add_cuda_d64(const Tensor& a, const Tensor& b);
Tensor sub_cuda_d64(const Tensor& a, const Tensor& b);
Tensor mul_cuda_d64(const Tensor& a, const Tensor& b);
Tensor div_cuda_d64(const Tensor& a, const Tensor& b);
Tensor pow_cuda_d64(const Tensor& a, const Tensor& b);
Tensor matmul_cuda_d64(const Tensor& A, const Tensor& B);

Tensor lt_cuda_d64(const Tensor& a, const Tensor& b);
Tensor le_cuda_d64(const Tensor& a, const Tensor& b);
Tensor gt_cuda_d64(const Tensor& a, const Tensor& b);
Tensor ge_cuda_d64(const Tensor& a, const Tensor& b);
Tensor eq_cuda_d64(const Tensor& a, const Tensor& b);
Tensor ne_cuda_d64(const Tensor& a, const Tensor& b);

Tensor abs_cuda_d64(const Tensor& a);
Tensor sqrt_cuda_d64(const Tensor& a);
Tensor relu_cuda_d64(const Tensor& a);
Tensor ln_cuda_d64(const Tensor& a);
Tensor exp_cuda_d64(const Tensor& a);
Tensor sin_cuda_d64(const Tensor& a);
Tensor asin_cuda_d64(const Tensor& a);
Tensor cos_cuda_d64(const Tensor& a);
Tensor acos_cuda_d64(const Tensor& a);
Tensor tan_cuda_d64(const Tensor& a);
Tensor atan_cuda_d64(const Tensor& a);
Tensor tanh_cuda_d64(const Tensor& a);
Tensor sinh_cuda_d64(const Tensor& a);
Tensor cosh_cuda_d64(const Tensor& a);
Tensor sigmoid_cuda_d64(const Tensor& a);
Tensor softplus_cuda_d64(const Tensor& a);

Tensor sum_cuda_d64(const Tensor& t, int dim = -1);
Tensor mean_cuda_d64(const Tensor& t, int dim = -1);
Tensor max_cuda_d64(const Tensor& t, int dim = -1);
Tensor min_cuda_d64(const Tensor& t, int dim = -1);

#endif