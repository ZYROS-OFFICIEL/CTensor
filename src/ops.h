#pragma once
#include "tensor.h"
#include <vector>
#include <string>

// Namespace for Operations to keep global scope clean
namespace Ops {

// ========================================================================
//                              BINARY OPS
// ========================================================================
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor pow(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);

// ========================================================================
//                              SCALAR OPS
// ========================================================================
Tensor add_scalar(const Tensor& a, double val);
Tensor sub_scalar(const Tensor& a, double val);
Tensor sub_scalar_rev(double val, const Tensor& a); // scalar - tensor
Tensor mul_scalar(const Tensor& a, double val);
Tensor div_scalar(const Tensor& a, double val);
Tensor div_scalar_rev(double val, const Tensor& a); // scalar / tensor
Tensor pow_scalar(const Tensor& a, double val);
Tensor pow_scalar_rev(double val, const Tensor& a); // scalar ^ tensor

// ========================================================================
//                              UNARY OPS
// ========================================================================
Tensor abs(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor sqrt(const Tensor& a);
Tensor sin(const Tensor& a);
Tensor cos(const Tensor& a);
Tensor tan(const Tensor& a);
Tensor asin(const Tensor& a);
Tensor acos(const Tensor& a);
Tensor atan(const Tensor& a);
Tensor sinh(const Tensor& a);
Tensor cosh(const Tensor& a);
Tensor tanh(const Tensor& a);
Tensor sigmoid(const Tensor& a);
Tensor relu(const Tensor& a);

// ========================================================================
//                              REDUCTIONS
// ========================================================================
Tensor sum(const Tensor& a, int dim = -1);
Tensor mean(const Tensor& a, int dim = -1);
Tensor max(const Tensor& a, int dim = -1);
Tensor min(const Tensor& a, int dim = -1);

// ========================================================================
//                              COMPARISONS
// ========================================================================
Tensor lt(const Tensor& a, const Tensor& b);
Tensor le(const Tensor& a, const Tensor& b);
Tensor gt(const Tensor& a, const Tensor& b);
Tensor ge(const Tensor& a, const Tensor& b);
Tensor eq(const Tensor& a, const Tensor& b);
Tensor neq(const Tensor& a, const Tensor& b);

// ... scalar comparisons ...
Tensor lt(const Tensor& a, double b);
Tensor le(const Tensor& a, double b);
Tensor gt(const Tensor& a, double b);
Tensor ge(const Tensor& a, double b);
Tensor eq(const Tensor& a, double b);
Tensor neq(const Tensor& a, double b);

} 
//------------------ Utilities -------------------------------------------
Tensor cat_mp(const std::vector<Tensor>& tensors, size_t dim);

//------------------ Operator Overloads ----------------------------------

// These overloads now map to the _mp versions.
// Ensure you are NOT including ops1.h at the same time as this file 
// in the same compilation unit to avoid ambiguous overloads.

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator+(const Tensor& a, double scalar);
Tensor operator+(double scalar, const Tensor& a);
Tensor operator-(const Tensor& a, const Tensor& b);
inline Tensor operator-(const Tensor& a, double scalar) { return sub_scalar_mp(a, scalar); }
inline Tensor operator-(double scalar, const Tensor& a) { return sub_afterscalar_mp(scalar, a); }
inline Tensor operator-(const Tensor& a) { return mult_scalar_mp(a, -1.0); }
Tensor operator*(const Tensor& a, const Tensor& b);
inline Tensor operator*(const Tensor& a, double scalar) { return mult_scalar_mp(a, scalar); }
inline Tensor operator*(double scalar, const Tensor& a) { return mult_scalar_mp(a, scalar); }
Tensor operator/(const Tensor& a, const Tensor& b);
inline Tensor operator/(const Tensor& a, double scalar) { return div_scalar_mp(a, scalar); }
inline Tensor operator/(double scalar, const Tensor& a) { return scalar_div_mp(scalar, a); }
Tensor operator^(const Tensor& a, const Tensor& b);
inline Tensor operator^(const Tensor& a, double scalar) { return pow_scalar_mp(a, scalar); }
inline Tensor operator^(double scalar, const Tensor& a) { return scalar_pow_mp(scalar, a); }

// Scalar comparisons
inline Tensor operator<(const Tensor& a, double b) 
inline Tensor operator<=(const Tensor& a, double b)
inline Tensor operator>(const Tensor& a, double b)
inline Tensor operator>=(const Tensor& a, double b)
inline Tensor operator==(const Tensor& a, double b)
inline Tensor operator!=(const Tensor& a, double b) 

// Two tensor comparisons
inline Tensor operator<(const Tensor& a, const Tensor& b) 
inline Tensor operator<=(const Tensor& a, const Tensor& b) 
inline Tensor operator>(const Tensor& a, const Tensor& b) 
inline Tensor operator>=(const Tensor& a, const Tensor& b) 
inline Tensor operator==(const Tensor& a, const Tensor& b) 
inline Tensor operator!=(const Tensor& a, const Tensor& b) 