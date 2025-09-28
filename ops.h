#include <vector>
#include <stdexcept>
#include "tensors.h"

std::vector<size_t> add_shape(const Tensor& a, const Tensor& b) {
    size_t ndim = std::max(a.ndim, b.ndim);
    std::vector<size_t> shape(ndim, 1);

    for (int i = 0; i < ndim; i++) {
        size_t dim_a = (i >= ndim - a.ndim) ? a.shape[i - (ndim - a.ndim)] : 1;
        size_t dim_b = (i >= ndim - b.ndim) ? b.shape[i - (ndim - b.ndim)] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1)
            throw std::runtime_error("Shapes not compatible for broadcasting");

        shape[i] = std::max(dim_a, dim_b);
    }
    return shape;
}

Tensor add_(const Tensor& a, const Tensor& b) {
    std::vector<size_t> result_shape = add_shape(a, b);
    Tensor result(result_shape);

    size_t n = result.numel();
    std::vector<size_t> idx(result_shape.size(), 0);

    for (size_t flat = 0; flat < n; flat++) {
        // Compute multi-dimensional index
        size_t rem = flat;
        for (int i = result_shape.size() - 1; i >= 0; i--) {
            idx[i] = rem % result_shape[i];
            rem /= result_shape[i];
        }

        // Compute indices in a and b
        size_t index_a = 0, index_b = 0;
        for (int i = 0; i < result_shape.size(); i++) {
            size_t dim_a = (i >= result_shape.size() - a.ndim) ? a.shape[i - (result_shape.size() - a.ndim)] : 1;
            size_t dim_b = (i >= result_shape.size() - b.ndim) ? b.shape[i - (result_shape.size() - b.ndim)] : 1;

            size_t idx_a = (dim_a == 1) ? 0 : idx[i];
            size_t idx_b = (dim_b == 1) ? 0 : idx[i];

            if (i < a.ndim) index_a += idx_a * a.strides[i - (result_shape.size() - a.ndim)];
            if (i < b.ndim) index_b += idx_b * b.strides[i - (result_shape.size() - b.ndim)];
        }

        result.data[flat] = a.data[index_a] + b.data[index_b];
    }

    return result;
}


Tensor mul_(const Tensor& a, const Tensor& b) {
    std::vector<size_t> result_shape = add_shape(a, b);
    Tensor result(result_shape);

    size_t n = result.numel();
    std::vector<size_t> idx(result_shape.size(), 0);

    for (size_t flat = 0; flat < n; flat++) {
        size_t rem = flat;
        for (int i = result_shape.size() - 1; i >= 0; i--) {
            idx[i] = rem % result_shape[i];
            rem /= result_shape[i];
        }

        // Compute indices in a and b
        size_t index_a = 0, index_b = 0;
        for (int i = 0; i < result_shape.size(); i++) {
            size_t dim_a = (i >= result_shape.size() - a.ndim) ? a.shape[i - (result_shape.size() - a.ndim)] : 1;
            size_t dim_b = (i >= result_shape.size() - b.ndim) ? b.shape[i - (result_shape.size() - b.ndim)] : 1;

            size_t idx_a = (dim_a == 1) ? 0 : idx[i];
            size_t idx_b = (dim_b == 1) ? 0 : idx[i];

            if (i < a.ndim) index_a += idx_a * a.strides[i - (result_shape.size() - a.ndim)];
            if (i < b.ndim) index_b += idx_b * b.strides[i - (result_shape.size() - b.ndim)];
        }

        result.data[flat] = a.data[index_a] * b.data[index_b];
    }

    return result;
}


Tensor operator+(const Tensor& a, const Tensor& b) {
    return add_(a,b); 
}
Tensor operator*(const Tensor& a, const Tensor& b) {
    return mul_(a,b); 
}