#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include "tensor1.h"
#include <immintrin.h>

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

//------------------ Tensor Operations ----------------------------------

Tensor add(const Tensor& a, const Tensor& b);
Tensor add_(const Tensor& a, const Tensor& b);
Tensor diff_(const Tensor& a, const Tensor& b);
Tensor mult_(const Tensor& a, const Tensor& b);
Tensor div_(const Tensor& a, const Tensor& b);
Tensor pow_(const Tensor& a, const Tensor& b);
Tensor matmul_(const Tensor& A, const Tensor& B);

//------------------ Reduction Operations --------------------------------

Tensor sum(const Tensor& t, int dim = -1);
Tensor mean(const Tensor& t, int dim = -1);
Tensor max(const Tensor& t, int dim = -1);
Tensor min(const Tensor& t, int dim = -1);

//------------------ Other Tensor Utilities -------------------------------

static Tensor cat(const std::vector<Tensor>& tensors, size_t dim);

//------------------ Operator Overloads ----------------------------------

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);
Tensor operator^(const Tensor& a, const Tensor& b);
