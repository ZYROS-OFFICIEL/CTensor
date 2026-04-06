#include "neuralnet/norm/norm.h"


static Tensor global_norm(const Tensor& t, double p) {
    Tensor flat = t.flatten(); // Shape: [numel]
    
    if (p == 1.0) {
        return sum(abs(flat), 0);
    } 
    else if (p == 2.0) {
        return sqrt(sum(pow_scalar(flat, 2.0), 0));
    } 
    else if (p == std::numeric_limits<double>::infinity()) {
        return max(abs(flat), 0);
    } 
    else if (p == -std::numeric_limits<double>::infinity()) {
        return min(abs(flat), 0);
    } 
    else if (p == 0.0) {
        // L0 norm: count non-zero elements
        return sum(ne(flat, 0.0).astype(DType::Float32), 0); 
    }
    else {
        return pow_scalar(sum(pow_scalar(abs(flat), p), 0), 1.0 / p);
    }
}

Tensor norm(const Tensor& t, double p, int dim, bool keepdim) {
    if (!t.impl) throw std::runtime_error("norm: empty tensor");

    Tensor res;
    if (dim == -1 && t.shape().size() > 1) {
        res = global_norm(t, p);
    } else {
        int actual_dim = dim < 0 ? dim + (int)t.shape().size() : dim;
        
        if (p == 1.0) {
            res = sum(abs(t), actual_dim);
        } else if (p == 2.0) {
            res = sqrt(sum(pow_scalar(t, 2.0), actual_dim));
        } else if (p == std::numeric_limits<double>::infinity()) {
            res = max(abs(t), actual_dim);
        } else if (p == -std::numeric_limits<double>::infinity()) {
            res = min(abs(t), actual_dim);
        } else if (p == 0.0) {
            res = sum(ne(t, 0.0).astype(DType::Float32), actual_dim);
        } else {
            res = pow_scalar(sum(pow_scalar(abs(t), p), actual_dim), 1.0 / p);
        }

        if (keepdim) {
            res = res.unsqueeze((size_t)actual_dim);
        }
    }
    return res;
}

Tensor infinity_norm(const Tensor& t, int dim, bool keepdim) {
    return norm(t, std::numeric_limits<double>::infinity(), dim, keepdim);
}

Tensor zero_norm(const Tensor& t, int dim, bool keepdim) {
    return norm(t, 0.0, dim, keepdim);
}


Tensor Lp_Norm(const Tensor& t, double p, int dim, double eps) {
    Tensor n = norm(t, p, dim, true); 
    
    Tensor denom = add_scalar(n, eps);
    return div(t, denom);
}

