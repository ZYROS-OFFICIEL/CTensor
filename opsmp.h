#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include "tensor1.h"
#include <immintrin.h>
#include <functional>
//------------------ Helpers --------------------------------------------
//Verify requires_grad
Tensor setup_autograd(const Tensor& out, const std::string& op, const std::vector<Tensor>& parents);

// Compute result shape for elementwise binary op (a and b already padded)
static std::vector<size_t> compute_result_shape_padded(const Tensor& a, const Tensor& b);

// Check if two shapes are broadcastable
static bool broadcastable(const std::vector<size_t>& a, const std::vector<size_t>& b);

// Compute broadcasted shape of two tensors
static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b);

// Compute broadcast shape for batch dimensions (helper for matmul)
// --- helper: broadcast batch shapes ---
static std::vector<size_t> broadcast_batch_shape_from_vectors(
    const std::vector<size_t>& a,
    const std::vector<size_t>& b) ;

    //------------------ Tensor Operations (Multi-Threaded) -----------------
// Binary Ops
Tensor add_mp(const Tensor& a, const Tensor& b);
Tensor diff_mp(const Tensor& a, const Tensor& b);
Tensor mult_mp(const Tensor& a, const Tensor& b);
Tensor div_mp(const Tensor& a, const Tensor& b);
Tensor pow_mp(const Tensor& a, const Tensor& b);
Tensor matmul_mp(const Tensor& A, const Tensor& B);

// Scalar Ops
Tensor add_scalar_mp(const Tensor& a, double scalar);
Tensor sub_scalar_mp(const Tensor& a, double scalar);
Tensor sub_afterscalar_mp(double scalar, const Tensor& a);
Tensor mult_scalar_mp(const Tensor& a, double scalar);
Tensor div_scalar_mp(const Tensor& a, double scalar);
Tensor scalar_div_mp(double scalar, const Tensor& a);
Tensor pow_scalar_mp(const Tensor& a, double scalar);
Tensor scalar_pow_mp(double scalar, const Tensor& a);

// Unary Ops
Tensor abs_mp(const Tensor& A);
Tensor ln_mp(const Tensor& a);
Tensor exp_mp(const Tensor& a);
Tensor sqrt_mp(const Tensor& a);
Tensor sin_mp(const Tensor& a);
Tensor asin_mp(const Tensor& a);
Tensor cos_mp(const Tensor& a);
Tensor acos_mp(const Tensor& a);
Tensor tan_mp(const Tensor& a);
Tensor atan_mp(const Tensor& a);
Tensor tanh_mp(const Tensor& a);
Tensor sinh_mp(const Tensor& a);
Tensor cosh_mp(const Tensor& a);
Tensor sigmoid_mp(const Tensor& a);
Tensor Relu_mp(const Tensor& a);
Tensor softplus_mp(const Tensor& a);

//------------------ Reduction Operations --------------------------------

Tensor sum_mp(const Tensor& t, int dim = -1);
Tensor mean_mp(const Tensor& t, int dim = -1);
Tensor max_mp(const Tensor& t, int dim = -1); // Placeholder if you implement it
Tensor min_mp(const Tensor& t, int dim = -1); // Placeholder

//------------------ Element-wise Comparisons ----------------------------
Tensor lt_mp(const Tensor& a, double b);
Tensor le_mp(const Tensor& a, double b);
Tensor gt_mp(const Tensor& a, double b);
Tensor ge_mp(const Tensor& a, double b);
Tensor eq_mp(const Tensor& a, double b);
Tensor neq_mp(const Tensor& a, double b);

Tensor lt_mp(const Tensor& a, const Tensor& b);
Tensor le_mp(const Tensor& a, const Tensor& b);
Tensor gt_mp(const Tensor& a, const Tensor& b);
Tensor ge_mp(const Tensor& a, const Tensor& b);
Tensor eq_mp(const Tensor& a, const Tensor& b);
Tensor ne_mp(const Tensor& a, const Tensor& b);

//------------------ Utilities -------------------------------------------
Tensor cat_mp(const std::vector<Tensor>& tensors, size_t dim);

//------------------ Operator Overloads (Mapping to _mp) -----------------
// CAUTION: You cannot redefine global operators if ops1.h is also included.
// If you want these to REPLACE ops1, you must not include ops1.h.
// If you want them to coexist, you usually can't overload operators twice for the same types.
// I will comment them out here. You should use explicit function calls or
// replace ops1 entirely.

/*
Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator+(const Tensor& a, double scalar);
Tensor operator+(double scalar, const Tensor& a);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, double scalar);
Tensor operator*(double scalar, const Tensor& a);
Tensor operator/(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, double scalar);
Tensor operator/(double scalar, const Tensor& a);
Tensor operator^(const Tensor& a, const Tensor& b);
Tensor operator^(const Tensor& a, double scalar);
Tensor operator^(double scalar, const Tensor& a);
*/