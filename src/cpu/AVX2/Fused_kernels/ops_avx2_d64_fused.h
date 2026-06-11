#pragma once
#include "core.h"
#include <immintrin.h>
#include <stdexcept>

//  Naming convention: <op1>_<op2>_avx2_d64
//  All ops follow the mathematical order of arguments:
//    fma(a,b,c)  = a*b + c
//    fms(a,b,c)  = a*b - c
//    add_relu(a,b) = relu(a+b)


Tensor fma_avx2_d64(const Tensor& a, const Tensor& b, const Tensor& c); 
Tensor fms_avx2_d64(const Tensor& a, const Tensor& b, const Tensor& c);
Tensor nfma_avx2_d64(const Tensor& a, const Tensor& b, const Tensor& c); 
Tensor add_scale_avx2_d64(const Tensor& a, const Tensor& b, double scale); 

Tensor add_relu_avx2_d64(const Tensor& a, const Tensor& b);   
Tensor add_sigmoid_avx2_d64(const Tensor& a, const Tensor& b);  
Tensor add_tanh_avx2_d64(const Tensor& a, const Tensor& b);   
Tensor mul_add_avx2_d64(const Tensor& a, const Tensor& b, const Tensor& c); 
Tensor add_exp_avx2_d64(const Tensor& a, const Tensor& b);   
Tensor add_ln_avx2_d64(const Tensor& a, const Tensor& b);   

Tensor exp_neg_avx2_d64(const Tensor& a);                     
Tensor ln_relu_avx2_d64(const Tensor& a);                     
Tensor sigmoid_ln_avx2_d64(const Tensor& a);                     

Tensor silu_avx2_d64(const Tensor& a);

Tensor gelu_avx2_d64(const Tensor& a);

Tensor swiglu_avx2_d64(const Tensor& a, const Tensor& b);

Tensor layer_norm_avx2_d64(const Tensor& x,const Tensor& weight,const Tensor& bias,double eps = 1e-5f);

Tensor bias_add_relu_avx2_d64(const Tensor& x, const Tensor& bias);

Tensor bias_add_gelu_avx2_d64(const Tensor& x, const Tensor& bias);

Tensor scale_shift_avx2_d64(const Tensor& x, double scale, double shift);