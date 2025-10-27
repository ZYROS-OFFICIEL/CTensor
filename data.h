#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <stdexcept>
#include "tensor1.h"

// ---------- flat vector -> tensor ----------
template<typename T>
Tensor from_flat_vector(const std::vector<T>& data, const std::vector<size_t>& shape, DType dtype = DType::Float32, bool requires_grad = false) {
    // compute expected size
    size_t expected = 1;
    for (auto d : shape) expected *= d;
    if (expected != data.size()) throw std::invalid_argument("from_flat_vector: shape does not match data.size()");

    Tensor out(shape, dtype, requires_grad);
    for (size_t i = 0; i < expected; ++i) {
        // use write_scalar_at to handle dtype conversions
        write_scalar_at(out.impl->storage->data.get(), i, out.impl->dtype, static_cast<double>(data[i]));
    }
    return out;
}
// ---------- raw pointer -> tensor ----------
template<typename T>
Tensor from_raw_ptr(const T* ptr, size_t count, const std::vector<size_t>& shape, DType dtype = DType::Float32, bool requires_grad = false) {
    size_t expected = 1;
    for (auto d : shape) expected *= d;
    if (expected != count) throw std::invalid_argument("from_raw_ptr: shape does not match count");

    Tensor out(shape, dtype, requires_grad);
    for (size_t i = 0; i < expected; ++i) {
        write_scalar_at(out.impl->storage->data.get(), i, out.impl->dtype, static_cast<double>(ptr[i]));
    }
    return out;
}
// ---------- 2D nested vector -> tensor ----------
template<typename T>
Tensor from_2d_vector(const std::vector<std::vector<T>>& v2, DType dtype = DType::Float32, bool requires_grad = false) {
    size_t rows = v2.size();
    size_t cols = (rows == 0) ? 0 : v2[0].size();
    for (size_t r = 0; r < rows; ++r) {
        if (v2[r].size() != cols) throw std::invalid_argument("from_2d_vector: ragged rows not supported");
    }
    Tensor out({rows, cols}, dtype, requires_grad);
    size_t idx = 0;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            write_scalar_at(out.impl->storage->data.get(), idx++, out.impl->dtype, static_cast<double>(v2[r][c]));
        }
    }
    return out;
}
