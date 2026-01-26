#include "tensor.h"
#include <cstdlib>
#include <cstring>
#include <random>
#include <iostream>
#include <omp.h>

#ifdef _WIN32
    #include <malloc.h>
    #define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
    #define aligned_free _aligned_free
#else
    #define aligned_free free
#endif

// ----------------- Storage Implementation -----------------

std::shared_ptr<Storage> Storage::allocate(size_t n, DType dt, Device dev) {
    auto s = std::make_shared<Storage>();
    s->size = n;
    s->device = dev;
    
    size_t nbytes = n * dtype_size(dt);
    
    if (dev.is_cpu()) {
        if (nbytes == 0) {
            s->data = nullptr;
        } else {
            // Allocate 64-byte aligned memory for AVX-512
            void* ptr = nullptr;
            #ifdef _WIN32
                ptr = _aligned_malloc(nbytes, 64);
            #else
                if (posix_memalign(&ptr, 64, nbytes) != 0) ptr = nullptr;
            #endif
            
            if (!ptr) throw std::runtime_error("Memory allocation failed");
            
            // Custom deleter for shared_ptr
            s->data = std::shared_ptr<void>(ptr, [](void* p) { 
                #ifdef _WIN32
                    _aligned_free(p);
                #else
                    free(p);
                #endif
            });
        }
    } else {
        throw std::runtime_error("CUDA/GPU not implemented in this build");
    }
    
    return s;
}

// ----------------- Helper: Strides -----------------
static SmallVector<size_t, 5> calc_strides(const SmallVector<size_t, 5>& shape) {
    if (shape.empty()) return {};
    std::vector<size_t> temp(shape.size());
    size_t current = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        temp[i] = current;
        current *= shape[i];
    }
    return SmallVector<size_t, 5>(temp);
}

// ----------------- Tensorimpl Constructors -----------------

Tensorimpl::Tensorimpl(const std::vector<size_t>& shape_, DType dtype_, bool requires_grad_, Device dev_)
    : shape(shape_), dtype(dtype_), requires_grad(requires_grad_), ndim(shape_.size()) 
{
    strides = calc_strides(shape);
    size_t total_el = 1;
    for (auto s : shape) total_el *= s;
    data = Storage::allocate(total_el, dtype, dev_);
}

Tensorimpl::Tensorimpl(std::shared_ptr<Storage> storage_,
                       size_t offset_,
                       const SmallVector<size_t, 5>& shape_,
                       const SmallVector<size_t, 5>& strides_,
                       DType dtype_,
                       bool requires_grad_)
    : data(storage_), offset(offset_), shape(shape_), strides(strides_), 
      dtype(dtype_), requires_grad(requires_grad_), ndim(shape_.size()) {}

// ----------------- Tensor Implementation -----------------

Tensor::Tensor(const std::vector<size_t>& shape_, DType dtype_, bool requires_grad_) {
    Tensorimpl* raw = new Tensorimpl(shape_, dtype_, requires_grad_, Device(DeviceType::CPU));
    impl = intrusive_ptr<Tensorimpl>(raw); 
}

size_t Tensor::numel() const {
    if(!impl) return 0;
    size_t n = 1;
    for(size_t i=0; i<impl->ndim; ++i) n *= impl->shape[i];
    return n;
}

std::vector<size_t> Tensor::shape() const {
    if(!impl) return {};
    return impl->shape.to_vector();
}

Device Tensor::device() const { 
    return impl ? impl->data->device : Device(DeviceType::CPU); 
}

Tensor::Proxy Tensor::operator[](size_t i) {
    if (!impl) throw std::runtime_error("Invalid tensor");
    if (impl->ndim == 0) throw std::out_of_range("Cannot index scalar");
    if (i >= impl->shape[0]) throw std::out_of_range("Index out of bounds");
    return Proxy(impl, impl->offset + i * impl->strides[0], 1);
}

Tensor::ConstProxy Tensor::operator[](size_t i) const {
    if (!impl) throw std::runtime_error("Invalid tensor");
    if (impl->ndim == 0) throw std::out_of_range("Cannot index scalar");
    if (i >= impl->shape[0]) throw std::out_of_range("Index out of bounds");
    return ConstProxy(impl, impl->offset + i * impl->strides[0], 1);
}

// ----------------- Tensor Factories -----------------

Tensor Tensor::empty(const std::vector<size_t>& shape_, DType dt, bool requires_grad) {
    return Tensor(shape_, dt, requires_grad);
}

Tensor Tensor::zeros(const std::vector<size_t>& shape_, DType dt, bool requires_grad) {
    Tensor t(shape_, dt, requires_grad);
    size_t n = t.numel();
    size_t nb = n * dtype_size(dt);
    if (nb > 0 && t.impl->data->data) {
        std::memset(t.impl->data->data.get(), 0, nb);
    }
    return t;
}

Tensor Tensor::ones(const std::vector<size_t>& shape_, DType dt, bool requires_grad) {
    return full(shape_, 1.0, dt, requires_grad);
}

Tensor Tensor::full(const std::vector<size_t>& shape_, double value, DType dt, bool requires_grad) {
    Tensor t(shape_, dt, requires_grad);
    size_t n = t.numel();
    auto* ptr = t.impl->data->data.get();
    
    // Parallel fill
    #pragma omp parallel for
    for(size_t i=0; i<n; ++i) {
        write_scalar_at(ptr, i, dt, value);
    }
    return t;
}

Tensor Tensor::rand(const std::vector<size_t>& shape_, DType dt, bool requires_grad) {
    Tensor t(shape_, dt, requires_grad);
    size_t n = t.numel();
    auto* ptr = t.impl->data->data.get();
    
    // Thread-safe random generation
    #pragma omp parallel
    {
        std::mt19937 rng(std::random_device{}() + omp_get_thread_num());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        #pragma omp for
        for(size_t i=0; i<n; ++i) {
            write_scalar_at(ptr, i, dt, dist(rng));
        }
    }
    return t;
}

Tensor Tensor::from_vector(const std::vector<double>& data, const std::vector<size_t>& shape, DType dtype, bool requires_grad) {
    Tensor t(shape, dtype, requires_grad);
    if (t.numel() != data.size()) throw std::runtime_error("Size mismatch in from_vector");
    
    size_t n = data.size();
    auto* ptr = t.impl->data->data.get();
    
    #pragma omp parallel for
    for(size_t i=0; i<n; ++i) {
        write_scalar_at(ptr, i, dtype, data[i]);
    }
    return t;
}

// ----------------- Contiguity & Reshape -----------------

bool Tensor::is_contiguous() const {
    if (!impl) return false;
    size_t z = 1;
    for (int i = (int)impl->ndim - 1; i >= 0; --i) {
        if (impl->strides[i] != z) return false;
        z *= impl->shape[i];
    }
    return true;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    
    Tensor out(shape(), _dtype(), false); // New packed tensor
    out.requires_grad_(requires_grad());
    
    // Generic copy with stride logic
    // Flatten loop using index mapping
    size_t n = numel();
    auto* out_ptr = out.impl->data->data.get();
    auto* in_ptr = impl->data->data.get();
    size_t in_offset = impl->offset;
    DType dt = _dtype();
    
    const size_t* shape_ptr = impl->shape.data();
    const size_t* stride_ptr = impl->strides.data();
    size_t ndim = impl->ndim;

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        size_t temp = i;
        size_t current_idx = in_offset;
        for (int d = (int)ndim - 1; d >= 0; --d) {
            size_t sz = shape_ptr[d];
            size_t coord = temp % sz;
            temp /= sz;
            current_idx += coord * stride_ptr[d];
        }
        double val = read_scalar_at(in_ptr, current_idx, dt);
        write_scalar_at(out_ptr, i, dt, val);
    }
    return out;
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t n = numel();
    size_t new_n = 1;
    bool has_minus = false;
    int minus_idx = -1;
    
    std::vector<size_t> final_shape = new_shape;
    for(size_t i=0; i<new_shape.size(); ++i) {
        if (new_shape[i] == (size_t)-1) {
            if (has_minus) throw std::runtime_error("Only one dim can be -1");
            has_minus = true;
            minus_idx = i;
        } else {
            new_n *= new_shape[i];
        }
    }
    
    if (has_minus) {
        if (n % new_n != 0) throw std::runtime_error("Invalid reshape");
        final_shape[minus_idx] = n / new_n;
    } else {
        if (n != new_n) throw std::runtime_error("Numel mismatch");
    }

    if (is_contiguous()) {
        // Create view
        Tensorimpl* view = new Tensorimpl(
            impl->data, impl->offset, 
            SmallVector<size_t, 5>(final_shape), 
            calc_strides(SmallVector<size_t, 5>(final_shape)), 
            impl->dtype, impl->requires_grad
        );
        Tensor t;
        t.impl = intrusive_ptr<Tensorimpl>(view);
        return t;
    } else {
        return contiguous().reshape(final_shape);
    }
}

Tensor Tensor::permute(const std::vector<size_t>& dims) const {
    if (dims.size() != impl->ndim) throw std::runtime_error("Permute dims mismatch");
    
    SmallVector<size_t, 5> new_shape;
    SmallVector<size_t, 5> new_strides;
    
    for(size_t d : dims) {
        if (d >= impl->ndim) throw std::out_of_range("Permute dim OOB");
        new_shape.push_back(impl->shape[d]);
        new_strides.push_back(impl->strides[d]);
    }
    
    Tensorimpl* view = new Tensorimpl(
        impl->data, impl->offset, new_shape, new_strides, impl->dtype, impl->requires_grad
    );
    Tensor t;
    t.impl = intrusive_ptr<Tensorimpl>(view);
    return t;
}

void Tensor::zero_grad() {
    if (impl && impl->grad) {
        size_t nbytes = numel() * dtype_bytes();
        if (impl->grad->data && impl->grad->data->data) {
             std::memset(impl->grad->data->data.get(), 0, nbytes);
        }
    }
}

Tensor Tensor::clone() const {
    return contiguous(); // Contiguous creates a copy
}

Tensor Tensor::detach() const {
    Tensor t = *this; // Share data
    // Create new impl that points to same data but no grad history
    Tensorimpl* det = new Tensorimpl(
        impl->data, impl->offset, impl->shape, impl->strides, impl->dtype, false
    );
    Tensor out;
    out.impl = intrusive_ptr<Tensorimpl>(det);
    return out;
}