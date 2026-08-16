#pragma once
#include "tensor.h"
#include <cuda_fp16>
#ifdef USE_CUDA


Tensor fma_cuda_f16   (const Tensor& a, const Tensor& b, const Tensor& c);
Tensor fms_cuda_f16   (const Tensor& a, const Tensor& b, const Tensor& c);
Tensor nfma_cuda_f16  (const Tensor& a, const Tensor& b, const Tensor& c);
Tensor mul_add_cuda_f16(const Tensor& a, const Tensor& b, const Tensor& c);

Tensor add_scale_cuda_f16  (const Tensor& a, const Tensor& b, __half scale);
Tensor add_relu_cuda_f16   (const Tensor& a, const Tensor& b);
Tensor add_sigmoid_cuda_f16(const Tensor& a, const Tensor& b);
Tensor add_tanh_cuda_f16   (const Tensor& a, const Tensor& b);
Tensor add_exp_cuda_f16    (const Tensor& a, const Tensor& b);
Tensor add_ln_cuda_f16     (const Tensor& a, const Tensor& b);
Tensor swiglu_cuda_f16     (const Tensor& a, const Tensor& b);

Tensor exp_neg_cuda_f16    (const Tensor& a);
Tensor ln_relu_cuda_f16    (const Tensor& a);
Tensor sigmoid_ln_cuda_f16 (const Tensor& a);
Tensor silu_cuda_f16       (const Tensor& a);
Tensor gelu_cuda_f16       (const Tensor& a);

Tensor scale_shift_cuda_f16(const Tensor& x, __half scale, __half shift);

Tensor layer_norm_cuda_f16  (const Tensor& x, const Tensor& weight,
                              const Tensor& bias, __half eps = 1e-5f);
Tensor bias_add_relu_cuda_f16(const Tensor& x, const Tensor& bias);
Tensor bias_add_gelu_cuda_f16(const Tensor& x, const Tensor& bias);

#endif 