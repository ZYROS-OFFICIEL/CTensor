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
    Tensorimpl* impl;

    Tensor() : impl(nullptr) {}

    Tensor(const std::vector<size_t>& shape_, DType dtype_ = DType::Float32, bool requires_grad_ = false) {
        impl = new Tensorimpl(shape_, dtype_, requires_grad_);
    }

    ~Tensor() {
        delete impl;
    }
}