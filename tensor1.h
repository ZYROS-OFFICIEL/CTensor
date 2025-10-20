#pragma once
#include <iostream>
#include <cstddef>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <cassert>
#include <memory>

enum class DType { Float32, Int32, Double64 };

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return sizeof(float);
        case DType::Int32:   return sizeof(int);
        case DType::Double64:return sizeof(double);
    }
    return sizeof(float);
}
// read/write helpers convert via double (safe, simple)
inline double read_scalar_at(const void* data, size_t idx, DType dt) {
    switch (dt) {
        case DType::Float32:  return static_cast<double>( static_cast<const float*>(data)[idx] );
        case DType::Int32:    return static_cast<double>( static_cast<const int*>(data)[idx] );
        case DType::Double64: return static_cast<double>( static_cast<const double*>(data)[idx] );
    }
    return 0.0;
}

inline void write_scalar_at(void* data, size_t idx, DType dt, double val) {
    switch (dt) {
        case DType::Float32:  static_cast<float*>(data)[idx]  = static_cast<float>(val); break;
        case DType::Int32:    static_cast<int*>(data)[idx]    = static_cast<int>(std::lrint(val)); break;
        case DType::Double64: static_cast<double*>(data)[idx] = static_cast<double>(val); break;
    }
}

struct Storage {
    std::shared_ptr<void> data;
    std::shared_ptr<void> grad;
    size_t size = 0;

    static std::shared_ptr<Storage> allocate(size_t n, DType dt, bool requires_grad = false) {
        auto s = std::make_shared<Storage>();
        s->size = n * dtype_size(dt);

        // allocate data
        void* p = std::malloc(s->size);
        if (!p && s->size) throw std::bad_alloc();
        std::memset(p, 0, s->size);
        s->data = std::shared_ptr<void>(p, std::free);

        // optional grad
        if (requires_grad) {
            void* g = std::malloc(s->size);
            if (!g && s->size) throw std::bad_alloc();
            std::memset(g, 0, s->size);
            s->grad = std::shared_ptr<void>(g, std::free);
        } else {
            s->grad = nullptr;
        }

        return s;
    }
};

struct Tensorimpl {
    std::shared_ptr<Storage> storage;
    size_t offset = 0;
    size_t ndim = 0;
    size_t* shape = nullptr;
    size_t* strides = nullptr;
    bool requires_grad = false;
    DType dtype = DType::Float32;

    Tensorimpl(const std::vector<size_t>& shape_, DType dtype_ = DType::Float32, bool requires_grad_ = false)
        : offset(0), ndim(shape_.size()), requires_grad(requires_grad_), dtype(dtype_)
    {
        // allocate shape & strides
        shape = static_cast<size_t*>(std::malloc(ndim * sizeof(size_t)));
        strides = static_cast<size_t*>(std::malloc(ndim * sizeof(size_t)));
        if ((!shape && ndim) || (!strides && ndim)) {
            std::free(shape); std::free(strides);
            throw std::bad_alloc();
        }

        for (size_t i = 0; i < ndim; ++i) shape[i] = shape_[i];

        if (ndim > 0) {
            strides[ndim - 1] = 1;
            for (int i = (int)ndim - 2; i >= 0; --i)
                strides[i] = strides[i + 1] * shape[i + 1];
        }

        size_t numel = 1;
        for (auto v : shape_) numel *= v;

        storage = Storage::allocate(numel, dtype, requires_grad);
    }

    ~Tensorimpl() {
        std::free(shape);
        std::free(strides);
    }
};
struct Tensor{
    std::shared_ptr<Tensorimpl> impl;

    // --- Default constructor ---
    Tensor() = default;

    // --- Primary constructor (create a new impl) ---
    Tensor(const std::vector<size_t>& shape_,
           DType dtype_ = DType::Float32,
           bool requires_grad_ = false)
        : impl(std::make_shared<Tensorimpl>(shape_, dtype_, requires_grad_))
    {}

    // --- Copy constructor ---
    Tensor(const Tensor& other) = default;

    // --- Move constructor ---
    Tensor(Tensor&& other) noexcept = default;

    // --- Copy assignment ---
    Tensor& operator=(const Tensor& other) = default;

    // --- Move assignment ---
    Tensor& operator=(Tensor&& other) noexcept = default;

    // --- Destructor ---
    ~Tensor() = default;
}