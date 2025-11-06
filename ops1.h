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

//------------------ Tensor Operations ----------------------------------
Tensor apply_scalar_op(const Tensor& a,double scalar,std::function<double(double, double)> forward_op,std::function<std::shared_ptr<GradFn>(const Tensor&, double)> grad_fn_ctorensor );
Tensor add(const Tensor& a, const Tensor& b);
Tensor add_(const Tensor& a, const Tensor& b);
Tensor add_scalar(const Tensor& a, double scalar);
Tensor diff_(const Tensor& a, const Tensor& b);
Tensor sub_scalar(const Tensor& a, double scalar);
Tensor sub_afterscalar(double scalar ,const Tensor& a );
Tensor mult_(const Tensor& a, const Tensor& b);
Tensor mult_scalar(const Tensor& a, double scalar);
Tensor div_(const Tensor& a, const Tensor& b);
Tensor div_scalar(const Tensor& a, double scalar);
Tensor scalar_div(double scalar, const Tensor& a);
Tensor pow_(const Tensor& a, const Tensor& b);
Tensor pow_scalar(const Tensor& a, double scalar);
Tensor scalar_pow(double scalar, const Tensor& a);
Tensor matmul_(const Tensor& A, const Tensor& B);

Tensor ln_(const Tensor& a_);
Tensor exp_(const Tensor& a_);
Tensor sqrt_(const Tensor& a_);
Tensor sin_(const Tensor& a_);
Tensor asin_(const Tensor& a_);
Tensor cos_(const Tensor& a_);
Tensor acos_(const Tensor& a_);
Tensor tan_(const Tensor& a_);
Tensor atan_(const Tensor& a_);
Tensor tanh_(const Tensor& a_);
Tensor sigmoid_(const Tensor& a_);
Tensor Relu_(const Tensor& a_);
Tensor softplus_(const Tensor& a_);

//------------------ Reduction Operations --------------------------------

Tensor sum(const Tensor& t, int dim = -1);
Tensor mean(const Tensor& t, int dim = -1);
Tensor max(const Tensor& t, int dim = -1);
Tensor min(const Tensor& t, int dim = -1);

//------------------ Other Tensor Utilities -------------------------------

static Tensor cat(const std::vector<Tensor>& tensors, size_t dim);

//------------------ Operator Overloads ----------------------------------

inline Tensor operator+(const Tensor& a, const Tensor& b);
inline Tensor operator+(const Tensor& a, double scalar);
inline Tensor operator+(double scalar,const Tensor& a );
inline Tensor operator-(const Tensor& a, const Tensor& b);
inline Tensor operator-(const Tensor& a, double scalar) { return sub_scalar(a, scalar); }
inline Tensor operator-(double scalar, const Tensor& a) { return sub_afterscalar(scalar, a); }
inline Tensor operator-(const Tensor& a) { return mult_scalar(a, -1.0); }
inline Tensor operator*(const Tensor& a, const Tensor& b);
inline Tensor operator*(const Tensor& a, double scalar);
inline Tensor operator*(double scalar,const Tensor& a );
inline Tensor operator/(const Tensor& a, const Tensor& b);
inline Tensor operator/(const Tensor& a, double scalar) { return div_scalar(a, scalar); }
inline Tensor operator/(double scalar, const Tensor& a) { return scalar_div(scalar, a); }
inline Tensor operator^(const Tensor& a, const Tensor& b);
inline Tensor operator^(const Tensor& a, double scalar) { return pow_scalar(a, scalar); }
inline Tensor operator^(double scalar, const Tensor& a) { return scalar_pow(scalar, a); }
// --- Compound assignment operators ---

// Addition
inline Tensor& operator+=(Tensor& a, const Tensor& b) {
    a = add_(a, b);
    return a;
}
inline Tensor& operator+=(Tensor& a, double scalar) {
    a = add_scalar(a, scalar);
    return a;
}

// Subtraction
inline Tensor& operator-=(Tensor& a, const Tensor& b) {
    a = diff_(a, b);
    return a;
}
inline Tensor& operator-=(Tensor& a, double scalar) {
    a = sub_scalar(a, scalar);
    return a;
}

// Multiplication
inline Tensor& operator*=(Tensor& a, const Tensor& b) {
    a = mult_(a, b);
    return a;
}
inline Tensor& operator*=(Tensor& a, double scalar) {
    a = mult_scalar(a, scalar);
    return a;
}

// Division
inline Tensor& operator/=(Tensor& a, const Tensor& b) {
    a = div_(a, b);
    return a;
}
inline Tensor& operator/=(Tensor& a, double scalar) {
    a = div_scalar(a, scalar);
    return a;
}

// Power
inline Tensor& operator^=(Tensor& a, const Tensor& b) {
    a = pow_(a, b);
    return a;
}
inline Tensor& operator^=(Tensor& a, double scalar) {
    a = pow_scalar(a, scalar);
    return a;
}
