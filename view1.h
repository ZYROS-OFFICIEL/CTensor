#pragma once
#include "tensor1.h"
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <memory>

// ===== Tensor View and Utility Operations =====
class Tensor {
public:
    // ----- Permutation -----
    Tensor permute(const std::vector<size_t>& dims) const;

    // ----- Creation -----
    static Tensor arange(double start, double end, double step = 1.0, DType dtype = DType::Float32);

    // ----- Reshape and View Operations -----
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor select(size_t dim, size_t index) const;
    Tensor squeeze() const;
    Tensor unsqueeze(size_t dim) const;
    Tensor flatten() const;

    // ----- Type Conversion -----
    Tensor astype(DType new_dtype) const;
    void to_(DType new_dtype);

    // ----- In-place Operations -----
    Tensor& t_();
};
