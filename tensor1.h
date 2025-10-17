#pragma once
#include <iostream>
#include <cstddef>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdlib>      // malloc/free
#include <ctime>        // time for rand seed
#include <stdexcept>    // exceptions
#include <cassert>

using namespace std;
enum class DType { Float32, Int32, Double64 };
static size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return sizeof(float);
        case DType::Int32:   return sizeof(int);
        case DType::Double64:return sizeof(double);
    }
    return sizeof(float);
}

struct Storage{
    std::shared_ptr<void> data;
    size_t size;
    static shared_ptr<void> allocate(size_t n, DType dt){
        size = n * dtype_size(dt);
        data = std::shared_ptr<void>(malloc(size), free);
        if(!data && size) throw std::bad_alloc();
        memset(data.get(), 0, size);
        return std::shared_ptr<void>(p, std::free);
    }
}
struct Tensorimpl{
    std::shared_ptr<Storage> storage;
    size_t offset;
    size_t ndim;
    size_t* shape;
    size_t* strides;
    bool requires_grad;
    DType dtype;
    Tensorimpl(const std::vector<size_t>& shape_, DType dtype_ = DType::Float32, bool requires_grad_ = false)
        : offset(0), ndim(shape_.size()), shape(nullptr),
          strides(nullptr), requires_grad(requires_grad_), dtype(dtype_)
    {
        // allocate shape & strides
        shape = static_cast<size_t*>(malloc(ndim * sizeof(size_t)));
        strides = static_cast<size_t*>(malloc(ndim * sizeof(size_t)));
        if ((ndim && !shape) || (ndim && !strides)) {
            free(shape); free(strides);
            throw std::bad_alloc();
        }
        for (size_t i = 0; i < ndim; ++i) shape[i] = shape_[i];

        if (ndim > 0) {
            strides[ndim - 1] = 1;
            for (int i = (int)ndim - 2; i >= 0; --i)
                strides[i] = strides[i + 1] * shape[i + 1];
        }

        size_t numel = numel_();
        storage = std::make_shared<Storage>(Storage::allocate(numel, dtype));
    }
}