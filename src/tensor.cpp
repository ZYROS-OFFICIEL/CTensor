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
// ======================================================================================
//                                  TENSOR METHODS
// ======================================================================================

Tensor::Tensor(const std::vector<size_t>& shape_, DType dtype_, bool requires_grad_)
    : impl(std::make_shared<Tensorimpl>(shape_, dtype_, requires_grad_))
{}

size_t Tensor::numel() const {
    if (!impl) return 0;
    size_t n = 1;
    for (size_t i = 0; i < impl->ndim; ++i) n *= impl->shape[i];
    return n;
}

std::vector<size_t> Tensor::shape() const {
    if (!impl) return {};
    return std::vector<size_t>(impl->shape, impl->shape + impl->ndim);
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

    Tensor out(shape(), _dtype(), impl->requires_grad);
    size_t n = numel();
    
    // Call the dispatcher! 
    // This generates 8 different versions of contiguous_kernel, one for each type.
    DISPATCH_ALL_TYPES(impl->dtype, "contiguous", [&] {
        contiguous_kernel<scalar_t>(
            impl->storage->data.get(),
            out.impl->storage->data.get(),
            n,
            impl->ndim,
            impl->shape,
            impl->strides,
            impl->offset
        );
    });
    
    return out;
}

Tensor Tensor::clone() const {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (is_contiguous()) {
        Tensor out(shape(), _dtype(), impl->requires_grad);
        size_t bytes = numel() * dtype_size(_dtype());
        char* src_bytes = (char*)impl->storage->data.get() + impl->offset * dtype_size(_dtype());
        std::memcpy(out.impl->storage->data.get(), src_bytes, bytes);
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

Tensor::Proxy Tensor::operator[](size_t i) {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (impl->ndim == 0) throw std::out_of_range("Tensor has no dimensions");
    if (i >= impl->shape[0]) throw std::out_of_range("Index out of bounds");
    return Proxy(impl, impl->offset + i * impl->strides[0], 1);
}

Tensor::ConstProxy Tensor::operator[](size_t i) const {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (impl->ndim == 0) throw std::out_of_range("Tensor has no dimensions");
    if (i >= impl->shape[0]) throw std::out_of_range("Index out of bounds");
    return ConstProxy(impl, impl->offset + i * impl->strides[0], 1);
}

// --- Constructors using Dispatcher ---

Tensor Tensor::ones(const std::vector<size_t>& shape_, DType dt, bool requires_grad_){
    Tensor t(shape_, dt, requires_grad_);
    DISPATCH_ALL_TYPES(dt, "ones", [&] {
        fill_kernel<scalar_t>(t.impl->storage->data.get(), t.numel(), 1.0);
    });
    return t;
}

Tensor Tensor::zeros(const std::vector<size_t>& shape_, DType dt, bool requires_grad_){
    Tensor t(shape_, dt, requires_grad_);
    return t;
}

Tensor Tensor::full(const std::vector<size_t>& shape_, double value, DType dt, bool requires_grad_){
    Tensor t(shape_, dt, requires_grad_);
    DISPATCH_ALL_TYPES(dt, "full", [&] {
        fill_kernel<scalar_t>(t.impl->storage->data.get(), t.numel(), value);
    });
    return t;
}

Tensor Tensor::rand(const std::vector<size_t>& shape_, DType dt, bool requires_grad_) {
    Tensor t(shape_, dt, requires_grad_);
    size_t n = t.numel_();
    
    // Dispatch rand. Note: rand() returns int/float depending on impl, we map to scalar_t
    DISPATCH_ALL_TYPES(dt, "rand", [&] {
        scalar_t* ptr = (scalar_t*)t.impl->storage->data.get();
        for(size_t i=0; i<n; ++i) {
            ptr[i] = static_cast<scalar_t>(static_cast<double>(std::rand()) / RAND_MAX);
        }
    });
    return t;
}

Tensor Tensor::empty(const std::vector<size_t>& shape_, DType dt, bool requires_grad_){
    Tensor t(shape_,dt,requires_grad_);
    return t;
}

Tensor Tensor::from_vector(const std::vector<double>& data,const std::vector<size_t>& shape,DType dtype, bool requires_grad)
{
    size_t n = 1;
    for (auto s : shape) n *= s;
    if (data.size() != n) throw std::invalid_argument("from_vector size mismatch");
    
    Tensor t(shape, dtype, requires_grad);
    DISPATCH_ALL_TYPES(dtype, "from_vector", [&] {
        vector_init_kernel<scalar_t>(t.impl->storage->data.get(), n, data);
    });
    return t;
}

Tensor Tensor::detach() const {
    Tensor out = *this;
    if (out.impl) {
        out.impl = std::make_shared<Tensorimpl>(*out.impl);
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

// --- Type Casting with Dispatcher ---
Tensor Tensor::astype(DType new_dtype) const {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (new_dtype == impl->dtype) return Tensor(*this);
    
    Tensor out(shape(), new_dtype, impl->requires_grad);
    size_t n = numel_();
    
    // Double dispatch: Switch on SRC type, then switch on DST type
    DISPATCH_ALL_TYPES(impl->dtype, "astype_src", [&] {
        using src_t = scalar_t;
        const src_t* s_ptr = (const src_t*)impl->storage->data.get();
        size_t offset_base = impl->offset;
        
        DISPATCH_ALL_TYPES(new_dtype, "astype_dst", [&] {
            using dst_t = scalar_t;
            dst_t* d_ptr = (dst_t*)out.impl->storage->data.get();
            
            if (is_contiguous()) {
                // Fast Linear Cast
                #pragma omp parallel for
                for (size_t i = 0; i < n; ++i) {
                    d_ptr[i] = static_cast<dst_t>(s_ptr[offset_base + i]);
                }
            } else {
                // Strided Cast
                size_t ndim = impl->ndim;
                const size_t* shape_ptr = impl->shape;
                const size_t* stride_ptr = impl->strides;
                
                #pragma omp parallel for
                for (size_t i = 0; i < n; ++i) {
                    size_t temp = i;
                    size_t src_idx = offset_base;
                    for (int d = (int)ndim - 1; d >= 0; --d) {
                        size_t sz = shape_ptr[d];
                        size_t coord = temp % sz;
                        temp /= sz;
                        src_idx += coord * stride_ptr[d];
                    }
                    d_ptr[i] = static_cast<dst_t>(s_ptr[src_idx]);
                }
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

    std::vector<size_t> new_shape(impl->ndim);
    std::vector<size_t> new_strides(impl->ndim);
    for (size_t i = 0; i < impl->ndim; ++i) {
        new_shape[i] = impl->shape[dims[i]];
        new_strides[i] = impl->strides[dims[i]];
    }

    auto new_storage = std::make_shared<Storage>();
    new_storage->data = impl->storage->data; 
    new_storage->grad = nullptr;             
    new_storage->size = impl->storage->size;

    Tensor out;
    out.impl = std::make_shared<Tensorimpl>(
        new_storage, impl->offset, new_shape, new_strides, impl->dtype, impl->requires_grad);

    if (impl->requires_grad) {
        out.impl->grad_fn = std::make_shared<GradPermute>(*this, dims);
    }
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
        scalar_t* ptr = (scalar_t*)t.impl->storage->data.get();
        size_t n = vals.size();
        for (size_t i = 0; i < n; ++i) ptr[i] = static_cast<scalar_t>(vals[i]);
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

    std::vector<size_t> nst(new_shape.size());
    if (!new_shape.empty()) {
        nst.back() = 1;
        for (int i = (int)new_shape.size()-2; i >= 0; --i)
            nst[i] = nst[i+1] * new_shape[i+1];
    }
    
    Tensor out;
    out.impl = std::make_shared<Tensorimpl>(impl->storage, impl->offset, new_shape, nst, impl->dtype, impl->requires_grad);
    
    if (impl->requires_grad) {
        std::vector<size_t> old_shape_vec(impl->shape, impl->shape + impl->ndim);
        out.impl->grad_fn = std::make_shared<GradReshape>(*this, old_shape_vec);
    }
    return out;
}

Tensor Tensor::select(size_t dim, size_t index) const {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (dim >= impl->ndim) throw std::out_of_range("select: dim out of range");
    if (index >= impl->shape[dim]) throw std::out_of_range("select: index out of range");
    std::vector<size_t> nsh;
    std::vector<size_t> nst;
    for (size_t i = 0; i < impl->ndim; ++i) {
        if (i == dim) continue;
        nsh.push_back(impl->shape[i]);
        nst.push_back(impl->strides[i]);
    }
    size_t noffset = impl->offset + index * impl->strides[dim];
    Tensor out;
    out.impl = std::make_shared<Tensorimpl>(impl->storage, noffset, nsh, nst, impl->dtype, impl->requires_grad);
    return out;
}

Tensor Tensor::squeeze() const {
    if (!impl) throw std::runtime_error("Empty tensor");
    std::vector<size_t> nsh;
    std::vector<size_t> nst;
    for (size_t i = 0; i < impl->ndim; ++i) {
        if (impl->shape[i] == 1) continue;
        nsh.push_back(impl->shape[i]);
        nst.push_back(impl->strides[i]);
    }
    if (nsh.empty()) { nsh.push_back(1); nst.push_back(1); }
    Tensor out;
    out.impl = std::make_shared<Tensorimpl>(impl->storage, impl->offset, nsh, nst, impl->dtype, impl->requires_grad);
    return out;
}

Tensor Tensor::unsqueeze(size_t dim) const {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (dim > impl->ndim) throw std::out_of_range("unsqueeze: dim out of range");
    std::vector<size_t> nsh = shape();
    nsh.insert(nsh.begin() + dim, 1);
    std::vector<size_t> nst(nsh.size());
    nst.back() = 1;
    for (int i = (int)nst.size()-2; i >= 0; --i) nst[i] = nst[i+1] * nsh[i+1];
    Tensor out;
    out.impl = std::make_shared<Tensorimpl>(impl->storage, impl->offset, nsh, nst, impl->dtype, impl->requires_grad);
    return out;
}

Tensor Tensor::flatten() const {
    std::vector<size_t> nsh = { numel() };
    std::vector<size_t> nst = { 1 };
    Tensor out;
    out.impl = std::make_shared<Tensorimpl>(impl->storage, impl->offset, nsh, nst, impl->dtype, impl->requires_grad);
    return out;
}

void Tensor::save_image(const std::string& path) const {
    tensorio::to_image(*this, path);
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

    // Raw pointers for speed via Dispatch
    DISPATCH_ALL_TYPES(input._dtype(), "gather_in", [&] {
        using in_t = scalar_t;
        const in_t* in_data = (const in_t*)input.impl->storage->data.get();
        in_t* out_data = (in_t*)out.impl->storage->data.get();
        
        DISPATCH_ALL_TYPES(index._dtype(), "gather_idx", [&] {
            using idx_t = scalar_t;
            const idx_t* idx_data = (const idx_t*)index.impl->storage->data.get();
            
            // Capture pointers to stacks for OMP
            const size_t* out_strides = out.impl->strides;
            const size_t* idx_strides = index.impl->strides;
            const size_t* in_strides  = input.impl->strides;
            const size_t* out_shape_p = out.impl->shape;
            size_t idx_offset_base = index.impl->offset;
            size_t in_offset_base = input.impl->offset;

            #pragma omp parallel for
            for (size_t flat = 0; flat < n; ++flat) {
                size_t rem = flat;
                size_t idx_offset = idx_offset_base;
                size_t in_offset_partial = in_offset_base;
                
                for (int d = (int)ndim - 1; d >= 0; --d) {
                    size_t coord = rem % out_shape_p[d];
                    rem /= out_shape_p[d];
                    idx_offset += coord * idx_strides[d];
                    if (d != (int)dim) {
                        in_offset_partial += coord * in_strides[d];
                    }
                }

                // Read index (safe cast)
                size_t gather_idx = static_cast<size_t>(idx_data[idx_offset]);
                size_t in_final_offset = in_offset_partial + gather_idx * in_strides[dim];

                out_data[flat] = in_data[in_final_offset];
            }
        });
    });

    return out;
}