#pragma once
#include "tensor.h"

#ifdef USE_CUDA

Tensor add_cuda_f32(const Tensor& a, const Tensor& b);
Tensor sub_cuda_f32(const Tensor& a, const Tensor& b);
Tensor mul_cuda_f32(const Tensor& a, const Tensor& b);
Tensor div_cuda_f32(const Tensor& a, const Tensor& b);
Tensor pow_cuda_f32(const Tensor& a, const Tensor& b);
Tensor matmul_cuda_f32(const Tensor& A, const Tensor& B);

Tensor lt_cuda_f32(const Tensor& a, const Tensor& b);
Tensor le_cuda_f32(const Tensor& a, const Tensor& b);
Tensor gt_cuda_f32(const Tensor& a, const Tensor& b);
Tensor ge_cuda_f32(const Tensor& a, const Tensor& b);
Tensor eq_cuda_f32(const Tensor& a, const Tensor& b);
Tensor ne_cuda_f32(const Tensor& a, const Tensor& b);

Tensor abs_cuda_f32(const Tensor& a);
Tensor sqrt_cuda_f32(const Tensor& a);
Tensor relu_cuda_f32(const Tensor& a);
Tensor ln_cuda_f32(const Tensor& a);
Tensor exp_cuda_f32(const Tensor& a);
Tensor sin_cuda_f32(const Tensor& a);
Tensor asin_cuda_f32(const Tensor& a);
Tensor cos_cuda_f32(const Tensor& a);
Tensor acos_cuda_f32(const Tensor& a);
Tensor tan_cuda_f32(const Tensor& a);
Tensor atan_cuda_f32(const Tensor& a);
Tensor tanh_cuda_f32(const Tensor& a);
Tensor sinh_cuda_f32(const Tensor& a);
Tensor cosh_cuda_f32(const Tensor& a);
Tensor sigmoid_cuda_f32(const Tensor& a);
Tensor softplus_cuda_f32(const Tensor& a);

Tensor sum_cuda_f32(const Tensor& t, int dim = -1);
Tensor mean_cuda_f32(const Tensor& t, int dim = -1);
Tensor max_cuda_f32(const Tensor& t, int dim = -1);
Tensor min_cuda_f32(const Tensor& t, int dim = -1);

#endif