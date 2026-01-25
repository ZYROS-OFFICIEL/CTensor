#pragma once
#include "tensor.h"
#include <vector>

// -------------------------------------------------------------
//                Traffic Controller API
// -------------------------------------------------------------

// Binary Ops
Tensor add(const Tensor &a, const Tensor &b);
Tensor sub(const Tensor &a, const Tensor &b);
Tensor mul(const Tensor &a, const Tensor &b);
Tensor div(const Tensor &a, const Tensor &b);
Tensor pow(const Tensor &a, const Tensor &b);
Tensor matmul(const Tensor &a, const Tensor &b);


// Scalar Ops
Tensor add_scalar(const Tensor &a, double scalar);
Tensor sub_scalar(const Tensor &a, double scalar);
Tensor sub_scalar_rev(double scalar, const Tensor &a);
Tensor mul_scalar(const Tensor &a, double scalar);
Tensor div_scalar(const Tensor &a, double scalar);
Tensor div_scalar_rev(double scalar, const Tensor &a);
Tensor pow_scalar(const Tensor &a, double scalar);
Tensor pow_scalar_rev(double scalar, const Tensor &a);

// Comparisons
Tensor lt(const Tensor &a, const Tensor &b);
Tensor le(const Tensor &a, const Tensor &b);
Tensor gt(const Tensor &a, const Tensor &b);
Tensor ge(const Tensor &a, const Tensor &b);
Tensor eq(const Tensor &a, const Tensor &b);
Tensor ne(const Tensor &a, const Tensor &b);

Tensor lt(const Tensor &a, double b);
Tensor le(const Tensor &a, double b);
Tensor gt(const Tensor &a, double b);
Tensor ge(const Tensor &a, double b);
Tensor eq(const Tensor &a, double b);
Tensor ne(const Tensor &a, double b);

// Unary Ops
Tensor abs(const Tensor &a);
Tensor log(const Tensor &a);
Tensor ln(const Tensor &a);
Tensor exp(const Tensor &a);
Tensor sqrt(const Tensor &a);
Tensor sin(const Tensor &a);
Tensor cos(const Tensor &a);
Tensor tan(const Tensor &a);
Tensor asin(const Tensor &a);
Tensor acos(const Tensor &a);
Tensor atan(const Tensor &a);
Tensor sinh(const Tensor &a);
Tensor cosh(const Tensor &a);
Tensor tanh(const Tensor &a);
Tensor sigmoid(const Tensor &a);
Tensor relu(const Tensor &a);
Tensor Relu(const Tensor &a);
Tensor softplus(const Tensor &a);

// Reductions
Tensor sum(const Tensor &a, int dim = -1);
Tensor mean(const Tensor &a, int dim = -1);
Tensor max(const Tensor &a, int dim = -1);
Tensor min(const Tensor &a, int dim = -1);

// Utils
Tensor cat(const std::vector<Tensor>& tensors, size_t dim);


//------------------ Operator Overloads ----------------------------------

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator+(const Tensor& a, double scalar);
Tensor operator+(double scalar, const Tensor& a);
Tensor operator-(const Tensor& a, const Tensor& b);
inline Tensor operator-(const Tensor& a, double scalar) { return sub_scalar(a, scalar); }
inline Tensor operator-(double scalar, const Tensor& a) { return sub_scalar_rev(scalar, a); }
inline Tensor operator-(const Tensor& a) { return mul_scalar(a, -1.0); }
Tensor operator*(const Tensor& a, const Tensor& b);
inline Tensor operator*(const Tensor& a, double scalar) { return mul_scalar(a, scalar); }
inline Tensor operator*(double scalar, const Tensor& a) { return mul_scalar(a, scalar); }
Tensor operator/(const Tensor& a, const Tensor& b);
inline Tensor operator/(const Tensor& a, double scalar) { return sub_scalar(a, scalar); }
inline Tensor operator/(double scalar, const Tensor& a) { return div_scalar_rev(scalar, a); } // Fixed rev
Tensor operator^(const Tensor& a, const Tensor& b);
inline Tensor operator^(const Tensor& a, double scalar) { return pow_scalar(a, scalar); }
inline Tensor operator^(double scalar, const Tensor& a) { return pow_scalar_rev(scalar, a); }

// Scalar comparisons
inline Tensor operator<(const Tensor& a, double b) { return lt(a, b); }
inline Tensor operator<=(const Tensor& a, double b) { return le(a, b); }
inline Tensor operator>(const Tensor& a, double b) { return gt(a, b); }
inline Tensor operator>=(const Tensor& a, double b) { return ge(a, b); }
inline Tensor operator==(const Tensor& a, double b) { return eq(a, b); }
inline Tensor operator!=(const Tensor& a, double b) { return ne(a, b); }

// Two tensor comparisons
inline Tensor operator<(const Tensor& a, const Tensor& b) { return lt(a, b); }
inline Tensor operator<=(const Tensor& a, const Tensor& b) { return le(a, b); }
inline Tensor operator>(const Tensor& a, const Tensor& b) { return gt(a, b); }
inline Tensor operator>=(const Tensor& a, const Tensor& b) { return ge(a, b); }
inline Tensor operator==(const Tensor& a, const Tensor& b) { return eq(a, b); }
inline Tensor operator!=(const Tensor& a, const Tensor& b) { return ne(a, b); }