#pragma once
#include <cstddef>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <cmath>
#include <iostream>

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

// read/write helpers (small, header-safe)
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

// forward for autograd node
struct GradFn; // defined in autograd.cpp / autograd.h (forward only)

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
    std::shared_ptr<GradFn> grad_fn;

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
    Tensor(const size_t* shape_ptr, size_t ndim, DType dtype, bool requires_grad): Tensor(std::vector<size_t>(shape_ptr, shape_ptr + ndim), dtype, requires_grad) {}

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
    static Tensor from_vector(const std::vector<double>& data, const std::vector<size_t>& shape, DType dtype = DType::Float32, bool requires_grad = false);

    // clone (deep copy)
    Tensor clone() const;
    //Detach from computation graph (no grad_fn, no grad buffer)
    Tensor detach() const ;
    // ---------------- Templated Proxy ----------------
    template <bool Writable>
    struct ProxyBase {
        using TDataPtr = std::conditional_t<Writable, void*, const void*>;
        std::shared_ptr<Tensorimpl> impl;
        size_t offset;
        size_t depth;

        ProxyBase(std::shared_ptr<Tensorimpl> impl_, size_t off, size_t dp = 0)
            : impl(std::move(impl_)), offset(off), depth(dp) {}

        ProxyBase operator[](size_t i) const {
            if (!impl) throw std::runtime_error("Invalid tensor");
            if (depth >= impl->ndim) throw std::out_of_range("Too many indices");
            if (i >= impl->shape[depth]) throw std::out_of_range("Index out of bounds");
            size_t new_offset = offset + i * impl->strides[depth];
            return ProxyBase(impl, new_offset, depth + 1);
        }

        // Convert to double (read access)
        operator double() const {
            if (!impl) throw std::runtime_error("Invalid tensor");
            if (depth != impl->ndim) throw std::out_of_range("Not at leaf index");
            return read_scalar_at(impl->storage->data.get(), offset, impl->dtype);
        }

        // Only enabled for writable proxies
        template <bool W = Writable, typename = std::enable_if_t<W>>
        ProxyBase& operator=(double val) {
            if (!impl) throw std::runtime_error("Invalid tensor");
            if (depth != impl->ndim) throw std::out_of_range("Not at leaf index");
            write_scalar_at(impl->storage->data.get(), offset, impl->dtype, val);
            return *this;
        }

        template <bool W = Writable, typename T, typename = std::enable_if_t<W>>
        ProxyBase& operator=(T val) {
            return operator=(static_cast<double>(val));
        }
    };

    using Proxy = ProxyBase<true>;
    using ConstProxy = ProxyBase<false>;

    Proxy operator[](size_t i);
    ConstProxy operator[](size_t i) const;

    // shape ops (declarations - implement in .cpp)
    Tensor astype(DType new_dtype) const;
    void to_(DType new_dtype);
    Tensor& t_(); // in-place transpose
    Tensor permute(const std::vector<size_t>& dims) const;
    static Tensor arange(double start, double end, double step, DType dtype);
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor select(size_t dim, size_t index) const;
    Tensor squeeze() const;
    Tensor unsqueeze(size_t dim) const;
    Tensor flatten() const;
    void print_shape() const;
    // I/O helpers: implemented in data.cpp, NOT in tensor1.cpp
    static Tensor from_image(const std::string& path, DType dt = DType::Float32);
    void save_image(const std::string& path) const;

    Tensor gather(const Tensor& index, size_t dim=1);
    // autograd hook (pointer to grad function) - can be set by ops
    std::shared_ptr<GradFn> grad_fn;

    // backward: you may implement a convenience backward() that calls a free function in autograd.cpp
    Tensor detach() ;
    Tensor& requires_grad_(bool b);
    void backward(); // declared; implement in autograd.cpp or tensor1.cpp
};

// ---------- inline small wrappers that are safe in header ----------
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
inline bool Tensor::requires_grad() const {
    if (!impl) throw std::runtime_error("Tensor is empty");
    return impl->requires_grad;
}
// helper: compute linear index in original tensor for a given multi-index (padded left with 1s)
// 'orig' may have fewer dims than padded_idx.size(); we treat orig as left-padded with 1s.
inline size_t linear_index_from_padded(const Tensor& orig, const std::vector<size_t>& padded_idx) {
    if (!orig.impl) throw std::runtime_error("linear_index_from_padded: orig undefined");
    size_t pad = 0;
    if (padded_idx.size() < orig.impl->ndim) 
        throw std::invalid_argument("linear_index_from_padded: padded_idx smaller than orig.ndim");

    pad = padded_idx.size() - orig.impl->ndim;
    size_t offset = orig.impl->offset;
    for (size_t i = 0; i < orig.impl->ndim; ++i) {
        size_t idx = padded_idx[pad + i];
        size_t dim = orig.impl->shape[i];
        // if original dim == 1, broadcasted index maps to 0
        size_t use_idx = (dim == 1 ? 0 : idx);
        offset += use_idx * orig.impl->strides[i];
    }
    return offset;
}

// pad_to_ndim: expand tensor to target_ndim by left-padding dimensions of size 1
inline Tensor pad_to_ndim(const Tensor& t, size_t target_ndim) {
    if (!t.impl) throw std::runtime_error("pad_to_ndim: tensor undefined");
    size_t src_nd = t.impl->ndim;
    if (src_nd == target_ndim) return Tensor(t); // copy
    if (src_nd > target_ndim) 
        throw std::invalid_argument("pad_to_ndim: target_ndim smaller than tensor ndim");

    // Left-pad shape with 1s
    std::vector<size_t> new_shape(target_ndim, 1);
    for (size_t i = 0; i < src_nd; ++i)
        new_shape[target_ndim - src_nd + i] = t.impl->shape[i];

    // Create result tensor with same dtype and requires_grad flag
    Tensor result(new_shape, t.impl->dtype, t.impl->requires_grad);

    // Number of elements in result
    size_t N = result.numel_();
    std::vector<size_t> idx(target_ndim, 0);

    // Iterate flat over result, compute multi-index, map to src linear index and copy value
    for (size_t flat = 0; flat < N; ++flat) {
        size_t rem = flat;
        for (int d = (int)target_ndim - 1; d >= 0; --d) {
            idx[d] = rem % new_shape[d];
            rem /= new_shape[d];
        }
        // compute source linear index inside t (handles left-padding with ones and broadcasting)
        size_t src_linear = linear_index_from_padded(t, idx);

        double v = read_scalar_at(t.impl->storage->data.get(), src_linear, t.impl->dtype);
        write_scalar_at(result.impl->storage->data.get(), flat, result.impl->dtype, v);
    }

    // If source had a grad buffer and we want to preserve grad in result, copy it too.
    // (This mirrors copying data; gradients are optional — uncomment if desired.)
    if (t.impl->requires_grad && t.impl->storage->grad) {
        // allocate grad in result storage if not already (Storage::allocate created it based on requires_grad flag)
        // We wrote into result.impl->storage->data above; now copy gradients similarly (if result has grad buffer).
        if (!result.impl->storage->grad) {
            // result was created with same requires_grad flag so this should normally be present.
            // If not, skip or allocate as necessary.
        } else {
            // copy elementwise using same mapping
            for (size_t flat = 0; flat < N; ++flat) {
                size_t rem = flat;
                for (int d = (int)target_ndim - 1; d >= 0; --d) {
                    idx[d] = rem % new_shape[d];
                    rem /= new_shape[d];
                }
                size_t src_linear = linear_index_from_padded(t, idx);
                double gv = read_scalar_at(t.impl->storage->grad.get(), src_linear, t.impl->dtype);
                write_scalar_at(result.impl->storage->grad.get(), flat, result.impl->dtype, gv);
            }
        }
    }

    return result;
}
// simple flat print (for debugging) — prints as doubles
inline void print_t(const Tensor& t) {
    size_t n = t.numel_();
    std::cout << "[";
    for (size_t i = 0; i < n; i++) {
        double v = read_scalar_at(t.impl->storage->data.get(), i, t.impl->dtype);
        std::cout << v;
        if (i != n - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
}
// Helper to print a tensor (works for any shape)
inline void print_tensor(const Tensor& t, const std::string& name="Tensor") {
    std::cout << name << " shape: [";
    for (size_t i = 0; i < t.impl->ndim; ++i) {
        std::cout << t.impl->shape[i];
        if (i + 1 < t.impl->ndim) std::cout << ", ";
    }
    std::cout << "]\n";

    size_t n = t.numel();
    std::cout << name << " values: ";
    for (size_t i = 0; i < n; ++i) {
        std::cout << t.read_scalar(i) << " ";
    }
    std::cout << "\n\n";
}

static void print_(const Tensor& t, std::vector<size_t>& idx, size_t dim) {
    std::cout << "{";
    size_t dim_size = t.impl->shape[dim];
    for (size_t i = 0; i < dim_size; ++i) {
        idx[dim] = i;
        if (dim + 1 == t.impl->ndim) {
            // compute flat offset
            size_t offset = 0;
            for (size_t k = 0; k < t.impl->ndim; ++k) offset += idx[k] * t.impl->strides[k];
            double v = read_scalar_at(t.impl->storage->data.get(), offset, t.impl->dtype);
            // print nicely depending on dtype
            if (t.impl->dtype == DType::Int32) {
                long long iv = static_cast<long long>(std::lrint(v));
                std::cout << iv;
            } else {
                // for floats/doubles print value as-is
                std::cout << v;
            }
        } else {
            // recurse to next dimension
            print_recursive_braces(t, idx, dim + 1);
        }
        if (i + 1 != dim_size) std::cout << ", ";
    }
    std::cout << "}";
}

