#pragma once
#include <cstddef>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <cmath>

// ----------------- dtype helpers -----------------
enum class DType { Float32, Int32, Double64 };

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return sizeof(float);
        case DType::Int32:   return sizeof(int);
        case DType::Double64:return sizeof(double);
    }
    return sizeof(float);
}
// general SIMD width utility
inline size_t simd_width(DType dt, bool avx512 = false) {
    size_t bits = avx512 ? 512 : 256;   // AVX2 = 256, AVX-512 = 512
    size_t bytes = dtype_size(dt);      // size of one element
    return bits / (bytes * 8);          // elements per SIMD register
}
enum class Backend { SCALAR, AVX2, AVX512 };
inline Backend select_backend(bool avx512_supported, bool avx2_supported) {
    if (avx512_supported) return Backend::AVX512;
    if (avx2_supported)   return Backend::AVX2;
    return Backend::SCALAR;
}
// general SIMD width from backend
inline size_t simd_width(DType dt, Backend backend) {
    switch (backend) {
        case Backend::SCALAR: return 1;
        case Backend::AVX2:   return simd_width(dt, false);
        case Backend::AVX512: return simd_width(dt, true);
    }
    return 1;
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
// ----------------- forward for autograd node -----------------
struct GradFn; // defined in autograd.cpp / autograd.h

// ----------------- Storage -----------------
struct Storage {
    std::shared_ptr<void> data;
    std::shared_ptr<void> grad;
    size_t size = 0;

    // allocate implemented in tensor1.cpp
    static std::shared_ptr<Storage> allocate(size_t n, DType dt, bool requires_grad = false);
};

// ----------------- low-level impl -----------------
struct Tensorimpl {
    std::shared_ptr<Storage> storage;
    size_t offset = 0;
    size_t ndim = 0;
    size_t* shape = nullptr;
    size_t* strides = nullptr;
    bool requires_grad = false;
    DType dtype = DType::Float32;

    // constructors / destructor implemented in tensor1.cpp
    Tensorimpl(const std::vector<size_t>& shape_, DType dtype_ = DType::Float32, bool requires_grad_ = false);
    Tensorimpl(std::shared_ptr<Storage> storage_,
               size_t offset_,
               const std::vector<size_t>& shape_,
               const std::vector<size_t>& strides_,
               DType dtype_,
               bool requires_grad_);
    ~Tensorimpl();
};

// ----------------- Tensor (public API) -----------------
struct Tensor {
    std::shared_ptr<Tensorimpl> impl;

    // constructors / destructor - implement in tensor1.cpp
    Tensor() = default;
    Tensor(const std::vector<size_t>& shape_, DType dtype_ = DType::Float32, bool requires_grad_ = false);
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;
    ~Tensor() = default;

    // Basic accessors (can be inline wrappers)
    size_t numel() const;
    size_t numel_() const { return numel(); }
    std::vector<size_t> shape() const;
    inline DType _dtype() const;
    inline size_t dtype_bytes() const;
    bool requires_grad() const;

    // data read / write helpers (small wrappers)
    inline double read_scalar(size_t idx) const;
    inline void write_scalar(size_t idx, double val);

    // convenience constructors (implement in .cpp)
    static Tensor ones(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false);
    static Tensor zeros(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false);
    static Tensor full(const std::vector<size_t>& shape_, double value, DType dt = DType::Float32, bool requires_grad_ = false);
    static Tensor rand(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false);
    static Tensor empty(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false);
    static Tensor from_vector(const std::vector<double>& data,const std::vector<size_t>& shape,DType dtype = DType::Float32,bool requires_grad = false);
    // clone (deep copy)
    Tensor clone() const;

    // indexing proxy (must remain in header because it's templated)
    template <bool Writable>
    struct ProxyBase {
        using TDataPtr = std::conditional_t<Writable, void*, const void*>;
        std::shared_ptr<Tensorimpl> impl;
        size_t offset;
        size_t depth;

        ProxyBase(std::shared_ptr<Tensorimpl> impl_, size_t off, size_t dp = 0)
            : impl(std::move(impl_)), offset(off), depth(dp) {}

        ProxyBase operator[](size_t i) const;
        operator double() const;
        template <bool W = Writable, typename = std::enable_if_t<W>>
        ProxyBase& operator=(double val);
        template <bool W = Writable, typename T, typename = std::enable_if_t<W>>
        ProxyBase& operator=(T val);
    };

    using Proxy = ProxyBase<true>;
    using ConstProxy = ProxyBase<false>;

    Proxy operator[](size_t i);
    ConstProxy operator[](size_t i) const;

    // shape ops (declarations - implement in .cpp)
    Tensor astype(DType new_dtype) const;
    void to_(DType new_dtype);
    Tensor t_(); // in-place transpose
    Tensor permute(const std::vector<size_t>& dims) const;
    static Tensor arange(double start, double end, double step, DType dtype);
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor select(size_t dim, size_t index) const;
    Tensor squeeze() const;
    Tensor unsqueeze(size_t dim) const;
    Tensor flatten() const;

    // I/O helpers: implemented in data.cpp, NOT in tensor1.cpp
    static Tensor from_image(const std::string& path, DType dt = DType::Float32);
    void save_image(const std::string& path) const;

    // autograd hook (pointer to grad function) - can be set by ops
    std::shared_ptr<GradFn> grad_fn;

    // backward: you may implement a convenience backward() that calls a free function in autograd.cpp
    void backward(); // declared; implement in autograd.cpp or tensor1.cpp
};


// small inline wrappers that use impl - forward-declare to .cpp for safety
inline DType Tensor::_dtype() const {
    if (!impl) throw std::runtime_error("Tensor is empty");
    return impl->dtype;
}

inline size_t Tensor::dtype_bytes() const {
    if (!impl) throw std::runtime_error("Tensor is empty");
    return dtype_size(impl->dtype);
}

inline double Tensor::read_scalar(size_t idx) const {
    if (!impl) throw std::runtime_error("Tensor is empty");
    return read_scalar_at(impl->storage->data.get(), idx, impl->dtype);
}

inline void Tensor::write_scalar(size_t idx, double val) {
    if (!impl) throw std::runtime_error("Tensor is empty");
    write_scalar_at(impl->storage->data.get(), idx, impl->dtype, val);
}

// Proxy template member definitions (must be in header)
template <bool Writable>
typename Tensor::template ProxyBase<Writable> Tensor::ProxyBase<Writable>::operator[](size_t i) const {
    if (!impl) throw std::runtime_error("Invalid tensor");
    if (depth >= impl->ndim) throw std::out_of_range("Too many indices");
    if (i >= impl->shape[depth]) throw std::out_of_range("Index out of bounds");
    size_t new_offset = offset + i * impl->strides[depth];
    return ProxyBase(impl, new_offset, depth + 1);
}

template <bool Writable>
Tensor::ProxyBase<Writable>::operator double() const {
    if (!impl) throw std::runtime_error("Invalid tensor");
    if (depth != impl->ndim) throw std::out_of_range("Not at leaf index");
    return read_scalar_at(impl->storage->data.get(), offset, impl->dtype);
}

template <bool Writable>
template <bool W, typename>
typename Tensor::template ProxyBase<Writable>& Tensor::ProxyBase<Writable>::operator=(double val) {
    static_assert(Writable, "assignment only enabled for writable proxy");
    if (!impl) throw std::runtime_error("Invalid tensor");
    if (depth != impl->ndim) throw std::out_of_range("Not at leaf index");
    write_scalar_at(impl->storage->data.get(), offset, impl->dtype, val);
    return *this;
}

template <bool Writable>
template <bool W, typename T, typename>
typename Tensor::template ProxyBase<Writable>& Tensor::ProxyBase<Writable>::operator=(T val) {
    return operator=(static_cast<double>(val));
}

inline Tensor::Proxy Tensor::operator[](size_t i) {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (impl->ndim == 0) throw std::out_of_range("Tensor has no dimensions");
    return Proxy(impl, i * impl->strides[0], 1);
}

inline Tensor::ConstProxy Tensor::operator[](size_t i) const {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (impl->ndim == 0) throw std::out_of_range("Tensor has no dimensions");
    return ConstProxy(impl, i * impl->strides[0], 1);
}
// ---------- Helper: compute linear index in original tensor for a given multi-index ----------
// 'orig' has possibly fewer dims than padded_idx.size(); left-pad with 1s.
inline size_t linear_index_from_padded(const Tensor& orig, const std::vector<size_t>& padded_idx) {
    size_t offset = 0;
    size_t pad = padded_idx.size() - orig.impl->ndim;
    for (size_t i = 0; i < orig.impl->ndim; ++i) {
        size_t idx = padded_idx[pad + i];
        size_t dim = orig.impl->shape[i];
        size_t use_idx = (dim == 1 ? 0 : idx);
        offset += use_idx * orig.impl->strides[i];
    }
    return offset + orig.impl->offset;
}


// ---------- pad_to_ndim: expand tensor to target ndim by padding dimensions ----------
inline Tensor pad_to_ndim(const Tensor& t, size_t target_ndim) {
    if (t.impl->ndim == target_ndim) return Tensor(t);  // copy
    if (t.impl->ndim > target_ndim)
        throw std::runtime_error("pad_to_ndim: target_ndim smaller than tensor ndim");

    // Left-pad shape with 1s
    std::vector<size_t> new_shape(target_ndim, 1);
    for (size_t i = 0; i < t.impl->ndim; ++i)
        new_shape[target_ndim - t.impl->ndim + i] = t.shape()[i];

    Tensor result(new_shape, t.impl->dtype, t.impl->requires_grad);
    size_t N = result.numel_();
    std::vector<size_t> idx(target_ndim, 0);

    // Iterate over every element in result, compute its source index from t
    for (size_t flat = 0; flat < N; ++flat) {
        size_t rem = flat;
        for (int d = static_cast<int>(target_ndim) - 1; d >= 0; --d) {
            idx[d] = rem % new_shape[d];
            rem /= new_shape[d];
        }
        size_t src_idx = linear_index_from_padded(t, idx);
        double v = read_scalar_at(t.impl->storage->data.get(), src_idx, t.impl->dtype);
        write_scalar_at(result.impl->storage->data.get(), flat, result.impl->dtype, v);
    }
    return result;
}



// ---------- global helper: broadcast shape from two tensors ----------
inline std::vector<size_t> broadcast_batch_shape_from_vectors(const std::vector<size_t>& a,
                                                              const std::vector<size_t>& b) {
    size_t na = a.size(), nb = b.size();
    size_t n = std::max(na, nb);
    std::vector<size_t> result(n, 1);

    for (size_t i = 0; i < n; ++i) {
        size_t da = (i < n - na) ? 1 : a[i - (n - na)];
        size_t db = (i < n - nb) ? 1 : b[i - (n - nb)];

        if (da != db && da != 1 && db != 1)
            throw std::invalid_argument("broadcast_batch_shape_from_vectors: incompatible shapes");

        result[i] = std::max(da, db);
    }

    return result;
}