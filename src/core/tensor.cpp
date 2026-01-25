#include "tensor.h"
#include "dispatch.h" 
#include "autograd.h" 
#include "data.h"     
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <omp.h>
#include <map>
#include <mutex>
#include <iostream>
#include <functional>

// ======================================================================================
//                              1. CACHING ALLOCATOR & 4. PADDING
// ======================================================================================

class MemoryPool {
private:
    std::map<size_t, std::vector<void*>> cache;
    std::mutex pool_mutex;
    
    static const size_t ALIGNMENT = 64;

public:
    static MemoryPool& instance() {
        static MemoryPool pool;
        return pool;
    }

    void* allocate(size_t nbytes) {
        // OPTIMIZATION 4: SIMD-Width Padding
        // Always round up to multiple of 64 bytes to ensure tail-free vectorization.
        if (nbytes == 0) return nullptr;
        
        size_t padded_size = (nbytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

        std::lock_guard<std::mutex> lock(pool_mutex);
        
        auto it = cache.find(padded_size);
        if (it != cache.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            return ptr;
        }

        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(padded_size, ALIGNMENT);
#else
        if (posix_memalign(&ptr, ALIGNMENT, padded_size) != 0) {
            ptr = nullptr;
        }
#endif
        if (!ptr) throw std::bad_alloc();
        return ptr;
    }

    void deallocate(void* ptr, size_t nbytes) {
        if (!ptr) return;
        
        // We must return using the PADDED size key
        size_t padded_size = (nbytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        
        std::lock_guard<std::mutex> lock(pool_mutex);
        cache[padded_size].push_back(ptr);
    }
};

std::shared_ptr<Storage> Storage::allocate(size_t n, DType dt, bool requires_grad, Device dev) {
    auto s = std::make_shared<Storage>();
    s->size = n;
    s->device = dev;

    size_t nbytes = n * dtype_size(dt);
    void* p = MemoryPool::instance().allocate(nbytes);
    
    if (nbytes > 0 && p) {
        std::memset(p, 0, nbytes);
    }
    
    // Deleter captures nbytes to calculate padded size correctly in deallocate
    s->data = std::shared_ptr<void>(p, [nbytes](void* ptr) {
        MemoryPool::instance().deallocate(ptr, nbytes);
    });
    
    return s;
}

// ======================================================================================
//                              5. SMALL TENSOR OPTIMIZATION (SBO)
// ======================================================================================

template<typename F>
inline void parallel_for(size_t n, F func) {
    if (n < 1024) {
        for (size_t i = 0; i < n; ++i) {
            func(i);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            func(i);
        }
    }
}

// ======================================================================================
//                              4. DISPATCH TABLE REGISTRY
// ======================================================================================

struct OpKey {
    std::string op_name;
    DType dtype;
    bool operator<(const OpKey& other) const {
        if (op_name != other.op_name) return op_name < other.op_name;
        return dtype < other.dtype;
    }
};

class KernelRegistry {
    std::map<OpKey, void*> registry;
public:
    static KernelRegistry& get() {
        static KernelRegistry instance;
        return instance;
    }
    void register_kernel(const std::string& name, DType dt, void* fn) {
        registry[{name, dt}] = fn;
    }
    void* get_kernel(const std::string& name, DType dt) {
        auto it = registry.find({name, dt});
        if (it == registry.end()) return nullptr;
        return it->second;
    }
};

#define REGISTER_KERNEL(NAME, TYPE, DTYPE_ENUM, FN) \
    KernelRegistry::get().register_kernel(NAME, DTYPE_ENUM, (void*)FN<TYPE>);

// ======================================================================================
//                                  TENSOR KERNELS (Optimized)
// ======================================================================================

template <typename T>
void fill_kernel_impl(void* data, size_t n, double value) {
    T* ptr = (T*)data;
    T v = static_cast<T>(value);
    parallel_for(n, [=](size_t i) {
        ptr[i] = v;
    });
}

// --- OPTIMIZATION 3: STRIDED POINTER ITERATOR (Zero Mod/Div) ---
// This kernel replaces the expensive modulus and division in the inner loop
// with an incremental counter method ("Dimensional Carry-Over").
template <typename T>
void contiguous_kernel_impl(const void* src, void* dst, size_t n, 
                       size_t ndim, const size_t* shape, const size_t* strides, size_t offset) {
    const T* s = (const T*)src;
    T* d = (T*)dst;

    if (ndim == 0) { // Scalar
        d[0] = s[offset];
        return;
    }

    #pragma omp parallel
    {
        // 1. Thread Partitioning
        size_t num_threads = omp_get_num_threads();
        size_t tid = omp_get_thread_num();
        size_t chunk = (n + num_threads - 1) / num_threads;
        size_t start = tid * chunk;
        size_t end = std::min(start + chunk, n);

        if (start < end) {
            // 2. Initial Setup (The ONLY time we use div/mod)
            // We need to calculate the multi-dim coordinates for the 'start' index
            // to know where the pointer begins.
            std::vector<size_t> coords(ndim); // Using vector locally is fine, cached
            size_t temp = start;
            size_t src_idx = offset;
            
            for (int dim = (int)ndim - 1; dim >= 0; --dim) {
                size_t sz = shape[dim];
                size_t c = temp % sz;
                temp /= sz;
                coords[dim] = c;
                src_idx += c * strides[dim];
            }
            
            // 3. The Hot Loop (Pure Pointer Arithmetic)
            for (size_t i = start; i < end; ++i) {
                d[i] = s[src_idx]; // Copy

                // Increment logic: Advance lowest dim, check overflow, carry over.
                for (int dim = (int)ndim - 1; dim >= 0; --dim) {
                    // Simple increment
                    coords[dim]++;
                    src_idx += strides[dim];

                    // Check bounds
                    if (coords[dim] < shape[dim]) {
                        // No overflow, we are done with increment
                        break; 
                    } else {
                        // Overflow! Reset this dim to 0 and loop to increment next dim (carry)
                        coords[dim] = 0;
                        // Backtrack pointer: We added strides[dim] * shape[dim] effectively over the loop
                        // but strictly we just added strides[dim] causing overflow.
                        // We need to return to index 0 of this dimension.
                        // We are currently at index 'shape[dim]', we want index '0'.
                        // Distance back is shape[dim] * strides[dim].
                        src_idx -= shape[dim] * strides[dim];
                    }
                }
            }
        }
    }
}

bool _registry_initialized = false;
void init_kernels() {
    if (_registry_initialized) return;
    
    REGISTER_KERNEL("fill", float, DType::Float32, fill_kernel_impl);
    REGISTER_KERNEL("contiguous", float, DType::Float32, contiguous_kernel_impl);
    
    REGISTER_KERNEL("fill", int32_t, DType::Int32, fill_kernel_impl);
    REGISTER_KERNEL("contiguous", int32_t, DType::Int32, contiguous_kernel_impl);

    REGISTER_KERNEL("fill", double, DType::Double64, fill_kernel_impl);
    REGISTER_KERNEL("contiguous", double, DType::Double64, contiguous_kernel_impl);
    
    _registry_initialized = true;
}

// ======================================================================================
//                                  TENSOR METHODS
// ======================================================================================

Device Tensor::device() const {
    if (!impl) throw std::runtime_error("Tensor is empty/undefined");
    return impl->data->device;
}

bool Tensor::is_contiguous() const {
    if (!impl) return false;
    if (impl->ndim == 0) return true;
    size_t expected_stride = 1;
    for (int i = (int)impl->ndim - 1; i >= 0; --i) {
        if (impl->shape[i] != 1) { 
            if (impl->strides[i] != expected_stride) return false;
            expected_stride *= impl->shape[i];
        }
    }
    return true;
}

Tensor Tensor::contiguous() const {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (is_contiguous()) return *this; 

    init_kernels(); 

    Tensor out(impl->shape.to_vector(), _dtype(), impl->requires_grad);
    size_t n = numel();
    
    using ContiguousFn = void(*)(const void*, void*, size_t, size_t, const size_t*, const size_t*, size_t);
    void* fn_ptr = KernelRegistry::get().get_kernel("contiguous", impl->dtype);

    if (fn_ptr) {
        auto func = reinterpret_cast<ContiguousFn>(fn_ptr);
        func(impl->data->data.get(),
             out.impl->data->data.get(),
             n, impl->ndim,
             impl->shape.data(),
             impl->strides.data(),
             impl->offset);
    } else {
         DISPATCH_ALL_TYPES(impl->dtype, "contiguous_fallback", [&] {
            contiguous_kernel_impl<scalar_t>(
                impl->data->data.get(),
                out.impl->data->data.get(),
                n, impl->ndim, impl->shape.data(), impl->strides.data(), impl->offset
            );
        });
    }
    
    return out;
}

Tensor Tensor::clone() const {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (is_contiguous()) {
        Tensor out(impl->shape.to_vector(), _dtype(), impl->requires_grad);
        size_t bytes = numel() * dtype_size(_dtype());
        char* src_bytes = (char*)impl->data->data.get() + impl->offset * dtype_size(_dtype());
        std::memcpy(out.impl->data->data.get(), src_bytes, bytes);
        return out;
    } else {
        return contiguous(); 
    }
}

void Tensor::print_shape() const {
    if (!impl) { std::cout << "()\n"; return; }
    std::cout << "(";
    for (size_t i = 0; i < impl->ndim; ++i) {
        std::cout << impl->shape[i];
        if (i < impl->ndim - 1) std::cout << ", ";
    }
    std::cout << ")\n";
}

Tensor Tensor::ones(const std::vector<size_t>& shape_, DType dt, bool requires_grad_){
    init_kernels();
    Tensor t(shape_, dt, requires_grad_);
    
    using FillFn = void(*)(void*, size_t, double);
    void* fn_ptr = KernelRegistry::get().get_kernel("fill", dt);

    if (fn_ptr) {
        reinterpret_cast<FillFn>(fn_ptr)(t.impl->data->data.get(), t.numel(), 1.0);
    } else {
        DISPATCH_ALL_TYPES(dt, "ones", [&] {
            fill_kernel_impl<scalar_t>(t.impl->data->data.get(), t.numel(), 1.0);
        });
    }
    return t;
}

Tensor Tensor::zeros(const std::vector<size_t>& shape_, DType dt, bool requires_grad_){
    Tensor t(shape_, dt, requires_grad_);
    return t;
}

Tensor Tensor::full(const std::vector<size_t>& shape_, double value, DType dt, bool requires_grad_){
    init_kernels();
    Tensor t(shape_, dt, requires_grad_);

    using FillFn = void(*)(void*, size_t, double);
    void* fn_ptr = KernelRegistry::get().get_kernel("fill", dt);
    
    if (fn_ptr) {
        reinterpret_cast<FillFn>(fn_ptr)(t.impl->data->data.get(), t.numel(), value);
    } else {
        DISPATCH_ALL_TYPES(dt, "full", [&] {
            fill_kernel_impl<scalar_t>(t.impl->data->data.get(), t.numel(), value);
        });
    }
    return t;
}

Tensor Tensor::rand(const std::vector<size_t>& shape_, DType dt, bool requires_grad_) {
    Tensor t(shape_, dt, requires_grad_);
    size_t n = t.numel_();
    
    DISPATCH_ALL_TYPES(dt, "rand", [&] {
        scalar_t* ptr = (scalar_t*)t.impl->data->data.get();
        parallel_for(n, [=](size_t i) {
             ptr[i] = static_cast<scalar_t>(static_cast<double>(std::rand()) / RAND_MAX);
        });
    });
    return t;
}

Tensor Tensor::empty(const std::vector<size_t>& shape_, DType dt, bool requires_grad_){
    Tensor t(shape_, dt, requires_grad_);
    return t;
}

Tensor Tensor::from_vector(const std::vector<double>& data, const std::vector<size_t>& shape, DType dtype, bool requires_grad)
{
    size_t n = 1;
    for (auto s : shape) n *= s;
    if (data.size() != n) throw std::invalid_argument("from_vector size mismatch");
    
    Tensor t(shape, dtype, requires_grad);
    DISPATCH_ALL_TYPES(dtype, "from_vector", [&] {
        using T = scalar_t;
        T* ptr = (T*)t.impl->data->data.get();
        parallel_for(n, [&](size_t i) {
             ptr[i] = static_cast<T>(data[i]);
        });
    });
    return t;
}

Tensor Tensor::detach() const {
    Tensor out = *this;
    if (out.impl) {
        // Deep copy of metadata (impl), shallow copy of storage
        Tensorimpl* raw = new Tensorimpl(*out.impl); 
        out.impl = intrusive_ptr<Tensorimpl>(raw);
        out.impl->requires_grad = false;
        out.impl->grad_fn = nullptr;
    }
    return out;
}

Tensor Tensor::detach() {
    return const_cast<const Tensor*>(this)->detach();
}

Tensor& Tensor::requires_grad_(bool b) {
    if (!impl) throw std::runtime_error("requires_grad_: undefined tensor");
    impl->requires_grad = b;
    return *this;
}

Tensor Tensor::astype(DType new_dtype) const {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (new_dtype == impl->dtype) return Tensor(*this);
    
    Tensor out(impl->shape.to_vector(), new_dtype, impl->requires_grad);
    size_t n = numel_();
    
    DISPATCH_ALL_TYPES(impl->dtype, "astype_src", [&] {
        using src_t = scalar_t;
        const src_t* s_ptr = (const src_t*)impl->data->data.get();
        size_t offset_base = impl->offset;
        
        DISPATCH_ALL_TYPES(new_dtype, "astype_dst", [&] {
            using dst_t = scalar_t;
            dst_t* d_ptr = (dst_t*)out.impl->data->data.get();
            
            if (is_contiguous()) {
                parallel_for(n, [=](size_t i) {
                    d_ptr[i] = static_cast<dst_t>(s_ptr[offset_base + i]);
                });
            } else {
                size_t ndim = impl->ndim;
                const size_t* shape_ptr = impl->shape.data();
                const size_t* stride_ptr = impl->strides.data();
                
                parallel_for(n, [=](size_t i) {
                    size_t temp = i;
                    size_t src_idx = offset_base;
                    for (int d = (int)ndim - 1; d >= 0; --d) {
                        size_t sz = shape_ptr[d];
                        size_t coord = temp % sz;
                        temp /= sz;
                        src_idx += coord * stride_ptr[d];
                    }
                    d_ptr[i] = static_cast<dst_t>(s_ptr[src_idx]);
                });
            }
        });
    });
    
    return out;
}

void Tensor::to_(DType new_dtype) {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (new_dtype == impl->dtype) return;
    Tensor converted = this->astype(new_dtype);
    this->impl = converted.impl;
}

Tensor& Tensor::t_() {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (impl->ndim < 2)
        throw std::invalid_argument("t_: tensor must have at least 2 dimensions");
    
    // In-place modification of the intrusive_ptr shared object?
    // WARNING: Modifying 'impl' in place affects all other views sharing this impl!
    // Standard PyTorch behavior for .t_() is in-place on the Tensor object, but not the storage?
    // Actually .t_() is in-place. If we share the impl, we break others.
    // However, our design shares impl. To do this safely, we should check ref_count.
    // For now, we modify implementation directly as requested.
    
    // Note: SVO swap
    std::swap(impl->shape[impl->ndim - 2], impl->shape[impl->ndim - 1]);
    std::swap(impl->strides[impl->ndim - 2], impl->strides[impl->ndim - 1]);
    return *this;
}

Tensor Tensor::permute(const std::vector<size_t>& dims) const {
    if (!impl) throw std::runtime_error("permute: tensor has no implementation");
    if (dims.size() != impl->ndim) throw std::invalid_argument("permute: dims size must match ndim.");

    std::vector<bool> seen(impl->ndim, false);
    for (auto d : dims) {
        if (d >= impl->ndim || seen[d]) throw std::invalid_argument("permute: invalid dim.");
        seen[d] = true;
    }

    SmallVector<size_t, 5> new_shape;
    SmallVector<size_t, 5> new_strides;
    for (size_t i = 0; i < impl->ndim; ++i) {
        new_shape.push_back(impl->shape[dims[i]]);
        new_strides.push_back(impl->strides[dims[i]]);
    }

    Tensor out;
    Tensorimpl* raw = new Tensorimpl(
        impl->data, 
        impl->offset, 
        new_shape, 
        new_strides, 
        impl->dtype, 
        impl->requires_grad
    );
    out.impl = intrusive_ptr<Tensorimpl>(raw);
    return out;
}

Tensor Tensor::arange(double start, double end, double step, DType dtype) {
    if (step == 0.0) throw std::invalid_argument("step must be non-zero");
    std::vector<double> vals;
    if (step > 0) {
        for (double x = start; x < end; x += step) vals.push_back(x);
    } else {
        for (double x = start; x > end; x += step) vals.push_back(x);
    }
    Tensor t({vals.size()}, dtype, false);
    
    DISPATCH_ALL_TYPES(dtype, "arange", [&] {
        scalar_t* ptr = (scalar_t*)t.impl->data->data.get();
        size_t n = vals.size();
        parallel_for(n, [&](size_t i) {
            ptr[i] = static_cast<scalar_t>(vals[i]);
        });
    });
    return t;
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    if (!impl) throw std::runtime_error("Empty tensor");
    size_t old_n = numel();
    size_t new_n = 1;
    for (auto v: new_shape) new_n *= v;
    if (old_n != new_n) throw std::invalid_argument("reshape: number of elements mismatch");
    
    if (!is_contiguous()) {
        Tensor contig = this->contiguous();
        return contig.reshape(new_shape);
    }

    SmallVector<size_t, 5> nsh(new_shape); // Conv vector to SVO
    SmallVector<size_t, 5> nst;
    
    // Strides calc
    size_t current = 1;
    if (!nsh.empty()) {
        nst.push_back(1); // placeholder, we need size
        // Actually simpler:
        std::vector<size_t> temp_st(nsh.size());
        temp_st.back() = 1;
        for (int i = (int)nsh.size()-2; i >= 0; --i) {
            temp_st[i] = temp_st[i+1] * nsh[i+1];
        }
        nst = SmallVector<size_t, 5>(temp_st);
    }
    
    Tensor out;
    Tensorimpl* raw = new Tensorimpl(impl->data, impl->offset, nsh, nst, impl->dtype, impl->requires_grad);
    out.impl = intrusive_ptr<Tensorimpl>(raw);
    return out;
}

Tensor Tensor::select(size_t dim, size_t index) const {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (dim >= impl->ndim) throw std::out_of_range("select: dim out of range");
    if (index >= impl->shape[dim]) throw std::out_of_range("select: index out of range");
    
    SmallVector<size_t, 5> nsh;
    SmallVector<size_t, 5> nst;
    for (size_t i = 0; i < impl->ndim; ++i) {
        if (i == dim) continue;
        nsh.push_back(impl->shape[i]);
        nst.push_back(impl->strides[i]);
    }
    
    size_t noffset = impl->offset + index * impl->strides[dim];
    
    Tensor out;
    Tensorimpl* raw = new Tensorimpl(impl->data, noffset, nsh, nst, impl->dtype, impl->requires_grad);
    out.impl = intrusive_ptr<Tensorimpl>(raw);
    return out;
}

Tensor Tensor::squeeze() const {
    if (!impl) throw std::runtime_error("Empty tensor");
    SmallVector<size_t, 5> nsh;
    SmallVector<size_t, 5> nst;
    for (size_t i = 0; i < impl->ndim; ++i) {
        if (impl->shape[i] == 1) continue;
        nsh.push_back(impl->shape[i]);
        nst.push_back(impl->strides[i]);
    }
    if (nsh.empty()) { nsh.push_back(1); nst.push_back(1); }
    
    Tensor out;
    Tensorimpl* raw = new Tensorimpl(impl->data, impl->offset, nsh, nst, impl->dtype, impl->requires_grad);
    out.impl = intrusive_ptr<Tensorimpl>(raw);
    return out;
}

Tensor Tensor::unsqueeze(size_t dim) const {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (dim > impl->ndim) throw std::out_of_range("unsqueeze: dim out of range");
    
    // Construct new shape/stride manually in SVO
    SmallVector<size_t, 5> nsh;
    SmallVector<size_t, 5> nst;
    
    for (size_t i = 0; i < dim; ++i) {
        nsh.push_back(impl->shape[i]);
        nst.push_back(impl->strides[i]);
    }
    nsh.push_back(1);
    size_t new_stride = (dim < impl->ndim) ? impl->strides[dim] : 1;
    nst.push_back(new_stride);
    for (size_t i = dim; i < impl->ndim; ++i) {
        nsh.push_back(impl->shape[i]);
        nst.push_back(impl->strides[i]);
    }

    Tensor out;
    Tensorimpl* raw = new Tensorimpl(impl->data, impl->offset, nsh, nst, impl->dtype, impl->requires_grad);
    out.impl = intrusive_ptr<Tensorimpl>(raw);
    return out;
}

Tensor Tensor::flatten() const {
    // SVO init with initializer list
    SmallVector<size_t, 5> nsh = { numel() };
    SmallVector<size_t, 5> nst = { 1 };
    
    if(!is_contiguous()) {
        return contiguous().flatten();
    }

    Tensor out;
    Tensorimpl* raw = new Tensorimpl(impl->data, impl->offset, nsh, nst, impl->dtype, impl->requires_grad);
    out.impl = intrusive_ptr<Tensorimpl>(raw);
    return out;
}

void Tensor::save_image(const std::string& path) const {
    // tensorio::to_image(*this, path);
}

void Tensor::backward() {
    extern void backward(Tensor&); 
    backward(*this);
}

// Gather with Dispatcher
Tensor Tensor::gather(const Tensor& index, size_t dim) const{
    const Tensor& input = *this;
    
    if (!input.impl || !index.impl) throw std::runtime_error("gather: input or index tensor is empty");
    if (input.impl->ndim != index.impl->ndim) throw std::runtime_error("gather: input and index must have same number of dimensions");

    size_t ndim = input.impl->ndim;
    for (size_t i = 0; i < ndim; ++i) {
        if (i != dim && input.impl->shape[i] != index.impl->shape[i])
            throw std::runtime_error("gather: shape mismatch");
    }

    std::vector<size_t> out_shape = index.shape();
    Tensor out(out_shape, input._dtype(), input.requires_grad());
    size_t n = out.numel();

    DISPATCH_ALL_TYPES(input._dtype(), "gather_in", [&] {
        using in_t = scalar_t;
        const in_t* in_data = (const in_t*)input.impl->data->data.get();
        in_t* out_data = (in_t*)out.impl->data->data.get();
        
        DISPATCH_ALL_TYPES(index._dtype(), "gather_idx", [&] {
            using idx_t = scalar_t;
            const idx_t* idx_data = (const idx_t*)index.impl->data->data.get();
            
            const size_t* out_strides = out.impl->strides.data();
            const size_t* idx_strides = index.impl->strides.data();
            const size_t* in_strides  = input.impl->strides.data();
            const size_t* out_shape_p = out.impl->shape.data();
            size_t idx_offset_base = index.impl->offset;
            size_t in_offset_base = input.impl->offset;

            // parallel_for here still uses modulo because gather is random access 
            // and the iterator logic is complex for non-contiguous gather indices.
            // For now, we leave this as SBO-only optimization.
            parallel_for(n, [=](size_t flat) {
                size_t rem = flat;
                size_t idx_offset = idx_offset_base;
                size_t in_offset_partial = in_offset_base;
                
                for (int d = (int)ndim - 1; d >= 0; --d) {
                    size_t sz = out_shape_p[d];
                    size_t coord = rem % sz;
                    rem /= sz;
                    idx_offset += coord * idx_strides[d];
                    if (d != (int)dim) {
                        in_offset_partial += coord * in_strides[d];
                    }
                }

                size_t gather_idx = static_cast<size_t>(idx_data[idx_offset]);
                size_t in_final_offset = in_offset_partial + gather_idx * in_strides[dim];

                out_data[flat] = in_data[in_final_offset];
            });
        });
    });

    return out;
}