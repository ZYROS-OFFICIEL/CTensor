#pragma once
#include "tensor.h"

#ifdef USE_CUDA


Tensor fma_cuda_d64   (const Tensor& a, const Tensor& b, const Tensor& c);
Tensor fms_cuda_d64   (const Tensor& a, const Tensor& b, const Tensor& c);
Tensor nfma_cuda_d64  (const Tensor& a, const Tensor& b, const Tensor& c);
Tensor mul_add_cuda_d64(const Tensor& a, const Tensor& b, const Tensor& c);

Tensor add_scale_cuda_d64  (const Tensor& a, const Tensor& b, float scale);
Tensor add_relu_cuda_d64   (const Tensor& a, const Tensor& b);
Tensor add_sigmoid_cuda_d64(const Tensor& a, const Tensor& b);
Tensor add_tanh_cuda_d64   (const Tensor& a, const Tensor& b);
Tensor add_exp_cuda_d64    (const Tensor& a, const Tensor& b);
Tensor add_ln_cuda_d64     (const Tensor& a, const Tensor& b);
Tensor swiglu_cuda_d64     (const Tensor& a, const Tensor& b);

Tensor exp_neg_cuda_d64    (const Tensor& a);
Tensor ln_relu_cuda_d64    (const Tensor& a);
Tensor sigmoid_ln_cuda_d64 (const Tensor& a);
Tensor silu_cuda_d64       (const Tensor& a);
Tensor gelu_cuda_d64       (const Tensor& a);

Tensor scale_shift_cuda_d64(const Tensor& x, float scale, float shift);

Tensor layer_norm_cuda_d64  (const Tensor& x, const Tensor& weight,
                              const Tensor& bias, float eps = 1e-5f);
Tensor bias_add_relu_cuda_d64(const Tensor& x, const Tensor& bias);
Tensor bias_add_gelu_cuda_d64(const Tensor& x, const Tensor& bias);

#endif 