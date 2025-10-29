#pragma once
#include <vector>
#include <string>
#include "tensor1.h"

// ---------- flat vector -> tensor ----------
template<typename T>
Tensor from_flat_vector(const std::vector<T>& data,
                        const std::vector<size_t>& shape,
                        DType dtype = DType::Float32,
                        bool requires_grad = false);

// ---------- raw pointer -> tensor ----------
template<typename T>
Tensor from_raw_ptr(const T* ptr,
                    size_t count,
                    const std::vector<size_t>& shape,
                    DType dtype = DType::Float32,
                    bool requires_grad = false);

// ---------- 2D nested vector -> tensor ----------
template<typename T>
Tensor from_2d_vector(const std::vector<std::vector<T>>& v2,
                      DType dtype = DType::Float32,
                      bool requires_grad = false);

// ---------- 3D nested vector -> tensor ----------
template<typename T>
Tensor from_3d_vector(const std::vector<std::vector<std::vector<T>>>& v3,
                      DType dtype = DType::Float32,
                      bool requires_grad = false);

// ---------- CSV (numeric) -> 2D tensor ----------
Tensor from_csv(const std::string& filename,
                DType dtype = DType::Float32,
                bool has_header = false,
                char sep = ',');

// ---------- binary file (.bin) -> tensor ----------
Tensor from_binary(const std::string& filename,
                   const std::vector<size_t>& shape,
                   DType dtype,
                   bool requires_grad = false);

// ---------- NumPy .npy file -> tensor ----------
Tensor from_npy(const std::string& filename, bool requires_grad = false);

// ---------- tensor -> image ----------
namespace tensorio {
    void to_image(const Tensor& t, const std::string& path);
    Tensor from_image(const std::string& path, DType dtype = DType::Float32);
}
