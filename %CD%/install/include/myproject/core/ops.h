#pragma once
#include "tensor.h"
#include <vector>
#include <string>

// ========================================================================
//                              OPS NAMESPACE
// ========================================================================
// This namespace contains the functional API (e.g., Ops::add(a, b))
// The dispatcher logic (Device/Type/Arch) will be hidden behind these functions.

namespace Ops {

    // --------------------------------------------------------------------
    // 1. Binary Math Operations
    // --------------------------------------------------------------------
    Tensor add(const Tensor& a, const Tensor& b);
    Tensor sub(const Tensor& a, const Tensor& b);
    Tensor mul(const Tensor& a, const Tensor& b);
    Tensor div(const Tensor& a, const Tensor& b);
    Tensor pow(const Tensor& a, const Tensor& b);
    Tensor matmul(const Tensor& a, const Tensor& b);

    // --------------------------------------------------------------------
    // 2. Scalar Math Operations
    // --------------------------------------------------------------------
    // Support for both: Tensor + Scalar  AND  Scalar + Tensor
    Tensor add_scalar(const Tensor& a, double val);
    
    Tensor sub_scalar(const Tensor& a, double val);      // Tensor - Scalar
    Tensor sub_scalar_rev(double val, const Tensor& a);  // Scalar - Tensor
    
    Tensor mul_scalar(const Tensor& a, double val);
    
    Tensor div_scalar(const Tensor& a, double val);      // Tensor / Scalar
    Tensor div_scalar_rev(double val, const Tensor& a);  // Scalar / Tensor
    
    Tensor pow_scalar(const Tensor& a, double val);      // Tensor ^ Scalar
    Tensor pow_scalar_rev(double val, const Tensor& a);  // Scalar ^ Tensor

    // --------------------------------------------------------------------
    // 3. Unary / Activation Operations
    // --------------------------------------------------------------------
    Tensor abs(const Tensor& a);
    Tensor neg(const Tensor& a); // -a
    Tensor exp(const Tensor& a);
    Tensor log(const Tensor& a); // Natural logarithm (ln)
    Tensor sqrt(const Tensor& a);
    
    // Trigonometric
    Tensor sin(const Tensor& a);
    Tensor cos(const Tensor& a);
    Tensor tan(const Tensor& a);
    Tensor asin(const Tensor& a);
    Tensor acos(const Tensor& a);
    Tensor atan(const Tensor& a);
    Tensor sinh(const Tensor& a);
    Tensor cosh(const Tensor& a);
    Tensor tanh(const Tensor& a);

    // Activations
    Tensor sigmoid(const Tensor& a);
    Tensor relu(const Tensor& a);
    Tensor softplus(const Tensor& a);

    // --------------------------------------------------------------------
    // 4. Reduction Operations
    // --------------------------------------------------------------------
    // dim = -1 implies reduce all (flattened)
    Tensor sum(const Tensor& a, int dim = -1);
    Tensor mean(const Tensor& a, int dim = -1);
    Tensor max(const Tensor& a, int dim = -1);
    Tensor min(const Tensor& a, int dim = -1);

    // --------------------------------------------------------------------
    // 5. Comparison Operations
    // --------------------------------------------------------------------
    // Returns a Tensor (usually mask 0.0/1.0 or Bool dtype)
    
    // Tensor vs Tensor
    Tensor lt(const Tensor& a, const Tensor& b); // <
    Tensor le(const Tensor& a, const Tensor& b); // <=
    Tensor gt(const Tensor& a, const Tensor& b); // >
    Tensor ge(const Tensor& a, const Tensor& b); // >=
    Tensor eq(const Tensor& a, const Tensor& b); // ==
    Tensor neq(const Tensor& a, const Tensor& b);// !=

    // Tensor vs Scalar
    Tensor lt(const Tensor& a, double b);
    Tensor le(const Tensor& a, double b);
    Tensor gt(const Tensor& a, double b);
    Tensor ge(const Tensor& a, double b);
    Tensor eq(const Tensor& a, double b);
    Tensor neq(const Tensor& a, double b);

    // --------------------------------------------------------------------
    // 6. Utility Operations
    // --------------------------------------------------------------------
    Tensor cat(const std::vector<Tensor>& tensors, size_t dim);

} // namespace Ops


// ========================================================================
//                        OPERATOR OVERLOADS
// ========================================================================
// These provide the convenient syntax (a + b) by calling Ops::add(a, b).
// They handle all combinations of Tensor and Scalar.
