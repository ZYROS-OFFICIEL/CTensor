#include "tensor1.h"
#include "autograd.h"
#include "data.h"
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <omp.h>
#include <immintrin.h> 

// ======================================================================================
//                                  DISPATCHER UTILS
// ======================================================================================

#define DISPATCH_CASE(ENUM, TYPE, ...) \
    case ENUM: { \
        using scalar_t = TYPE; \
        __VA_ARGS__(); \
        break; \
    }

#define DISPATCH_ALL_TYPES(DTYPE, NAME, ...) \
    switch (DTYPE) { \
        DISPATCH_CASE(DType::Float32,  float,    __VA_ARGS__) \
        DISPATCH_CASE(DType::Int32,    int32_t,  __VA_ARGS__) \
        DISPATCH_CASE(DType::Double64, double,   __VA_ARGS__) \
        DISPATCH_CASE(DType::UInt8,    uint8_t,  __VA_ARGS__) \
        DISPATCH_CASE(DType::Int8,     int8_t,   __VA_ARGS__) \
        DISPATCH_CASE(DType::Int16,    int16_t,  __VA_ARGS__) \
        DISPATCH_CASE(DType::Int64,    int64_t,  __VA_ARGS__) \
        DISPATCH_CASE(DType::Bool,     bool,     __VA_ARGS__) \
        default: throw std::runtime_error(std::string(NAME) + ": unsupported dtype"); \
    }

// ======================================================================================
//                                  STORAGE & TENSORIMPL
// ======================================================================================

std::shared_ptr<Storage> Storage::allocate(size_t n, DType dt, bool requires_grad) {
    auto s = std::make_shared<Storage>();
    size_t nbytes = n * dtype_size(dt); 
    s->size = n; 

    void* p = std::malloc(nbytes); 
    if (!p && nbytes) throw std::bad_alloc(); 
    std::memset(p, 0, nbytes); 
    s->data = std::shared_ptr<void>(p, std::free);

    if (requires_grad) {
        void* g = std::malloc(nbytes); 
        if (!g && nbytes) throw std::bad_alloc(); 
        std::memset(g, 0, nbytes); 
        s->grad = std::shared_ptr<void>(g, std::free);
    } else {
        s->grad = nullptr;
    }
    return s;
}

Tensorimpl::Tensorimpl(const std::vector<size_t>& shape_, DType dtype_, bool requires_grad_)
    : offset(0), ndim(shape_.size()), requires_grad(requires_grad_), dtype(dtype_) 
{
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

Tensorimpl::Tensorimpl(std::shared_ptr<Storage> storage_, size_t offset_, const std::vector<size_t>& shape_, const std::vector<size_t>& strides_, DType dtype_, bool requires_grad_)
    : storage(std::move(storage_)), offset(offset_), ndim(shape_.size()), requires_grad(requires_grad_), dtype(dtype_)
{
    shape = static_cast<size_t*>(std::malloc(ndim * sizeof(size_t)));
    strides = static_cast<size_t*>(std::malloc(ndim * sizeof(size_t)));
    if ((!shape && ndim) || (!strides && ndim)) {
        std::free(shape); std::free(strides);
        throw std::bad_alloc();
    }
    for (size_t i = 0; i < ndim; ++i) {
        shape[i] = shape_[i];
        strides[i] = strides_[i];
    }
}

Tensorimpl::~Tensorimpl() {
    std::free(shape);
    std::free(strides);
}

// ======================================================================================
//                                  TENSOR KERNELS
// ======================================================================================

// --- Contiguous Kernel ---
template <typename T>
void contiguous_kernel(const void* src, void* dst, size_t n, 
                       size_t ndim, const size_t* shape, const size_t* strides, size_t offset) {
    const T* s = (const T*)src;
    T* d = (T*)dst;
    
    // We can rely on compiler autovectorization here since types are known!
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        size_t temp = i;
        size_t src_idx = offset;
        for (int dim = (int)ndim - 1; dim >= 0; --dim) {
            size_t sz = shape[dim];
            size_t coord = temp % sz;
            temp /= sz;
            src_idx += coord * strides[dim];
        }
        d[i] = s[src_idx];
    }
}

// --- Fill Kernel ---
template <typename T>
void fill_kernel(void* data, size_t n, double value) {
    T* ptr = (T*)data;
    T v = static_cast<T>(value);
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        ptr[i] = v;
    }
}

// --- Vector Init Kernel ---
template <typename T>
void vector_init_kernel(void* data, size_t n, const std::vector<double>& vals) {
    T* ptr = (T*)data;
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        ptr[i] = static_cast<T>(vals[i]);
    }
}