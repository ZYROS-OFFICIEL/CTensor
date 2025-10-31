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
    static Tensor from_vector(const std::vector<double>& data, const std::vector<size_t>& shape, DType dtype = DType::Float32, bool requires_grad = false);

    // clone (deep copy)
    Tensor clone() const;

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

    // autograd hook (pointer to grad function) - can be set by ops
    std::shared_ptr<GradFn> grad_fn;

    // backward: you may implement a convenience backward() that calls a free function in autograd.cpp
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