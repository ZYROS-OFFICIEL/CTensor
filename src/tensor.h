#pragma once
#include "device.h"
#include <cstddef>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <cmath>
#include <iostream>
#include <cstdint>

// ----------------- DType System -----------------
enum class DType { 
    Float32, 
    Int32, 
    Double64,
    // --- New Types ---
    UInt8,   // Byte / Unsigned Char (Image data)
    Int8,    // Char
    Int16,   // Short
    Int64,   // Long (Indices)
    Bool,    // Boolean (Masks)
    Float16  // Half precision (storage only usually)
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return sizeof(float);
        case DType::Int32:   return sizeof(int32_t);
        case DType::Double64:return sizeof(double);
        case DType::UInt8:   return sizeof(uint8_t);
        case DType::Int8:    return sizeof(int8_t);
        case DType::Int16:   return sizeof(int16_t);
        case DType::Int64:   return sizeof(int64_t);
        case DType::Bool:    return sizeof(bool); 
        case DType::Float16: return 2; 
    }
    return sizeof(float);
}

// read/write helpers (Expanded to handle all types safely)
inline double read_scalar_at(const void* data, size_t idx, DType dt) {
    switch (dt) {
        case DType::Float32:  return static_cast<double>( static_cast<const float*>(data)[idx] );
        case DType::Int32:    return static_cast<double>( static_cast<const int32_t*>(data)[idx] );
        case DType::Double64: return static_cast<double>( static_cast<const double*>(data)[idx] );
        case DType::UInt8:    return static_cast<double>( static_cast<const uint8_t*>(data)[idx] );
        case DType::Int8:     return static_cast<double>( static_cast<const int8_t*>(data)[idx] );
        case DType::Int16:    return static_cast<double>( static_cast<const int16_t*>(data)[idx] );
        case DType::Int64:    return static_cast<double>( static_cast<const int64_t*>(data)[idx] );
        case DType::Bool:     return static_cast<double>( static_cast<const bool*>(data)[idx] );
        default: return 0.0;
    }
}

inline void write_scalar_at(void* data, size_t idx, DType dt, double val) {
    switch (dt) {
        case DType::Float32:  static_cast<float*>(data)[idx]   = static_cast<float>(val); break;
        case DType::Int32:    static_cast<int32_t*>(data)[idx] = static_cast<int32_t>(val); break;
        case DType::Double64: static_cast<double*>(data)[idx]  = static_cast<double>(val); break;
        case DType::UInt8:    static_cast<uint8_t*>(data)[idx] = static_cast<uint8_t>(val); break;
        case DType::Int8:     static_cast<int8_t*>(data)[idx]  = static_cast<int8_t>(val); break;
        case DType::Int16:    static_cast<int16_t*>(data)[idx] = static_cast<int16_t>(val); break;
        case DType::Int64:    static_cast<int64_t*>(data)[idx] = static_cast<int64_t>(val); break;
        case DType::Bool:     static_cast<bool*>(data)[idx]    = (val != 0.0); break;
        default: break;
    }
}

// forward for autograd node
struct GradFn; 

// ----------------- Storage -----------------
struct Storage {
    std::shared_ptr<void> data;
    std::shared_ptr<void> grad;
    size_t size = 0;
    Device device;
    static std::shared_ptr<Storage> allocate(size_t n, DType dt, bool requires_grad = false,Device dev = Device(DeviceType::CPU));
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

    Tensorimpl(const std::vector<size_t>& shape_, DType dtype_, bool requires_grad_, Device dev_ );
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

    Device device() const;
    Tensor to(Device target_device); 

    Tensor() = default;
    Tensor(const std::vector<size_t>& shape_, DType dtype_ = DType::Float32, bool requires_grad_ = false);
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;
    Tensor(const size_t* shape_ptr, size_t ndim, DType dtype, bool requires_grad): Tensor(std::vector<size_t>(shape_ptr, shape_ptr + ndim), dtype, requires_grad) {}

    ~Tensor() = default;

    size_t numel() const;
    size_t numel_() const { return numel(); }
    std::vector<size_t> shape() const;
    inline DType _dtype() const;
    inline size_t dtype_bytes() const;
    bool requires_grad() const;

    // --- Contiguity Helpers ---
    bool is_contiguous() const;
    Tensor contiguous() const; 

    // data read / write helpers
    inline double read_scalar(size_t idx) const;
    inline void write_scalar(size_t idx, double val);

    // convenience constructors
    static Tensor ones(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false);
    static Tensor zeros(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false);
    static Tensor full(const std::vector<size_t>& shape_, double value, DType dt = DType::Float32, bool requires_grad_ = false);
    static Tensor rand(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false);
    static Tensor empty(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false);
    static Tensor from_vector(const std::vector<double>& data, const std::vector<size_t>& shape, DType dtype = DType::Float32, bool requires_grad = false);

    Tensor clone() const;
    Tensor detach() const;
    Tensor detach(); 
    Tensor& requires_grad_(bool b);

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

        operator double() const {
            if (!impl) throw std::runtime_error("Invalid tensor");
            if (depth != impl->ndim) throw std::out_of_range("Not at leaf index");
            return read_scalar_at(impl->storage->data.get(), offset, impl->dtype);
        }

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

    // shape ops
    Tensor astype(DType new_dtype) const;
    void to_(DType new_dtype);
    Tensor& t_(); 
    Tensor permute(const std::vector<size_t>& dims) const;
    static Tensor arange(double start, double end, double step, DType dtype);
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor select(size_t dim, size_t index) const;
    Tensor squeeze() const;
    Tensor unsqueeze(size_t dim) const;
    Tensor flatten() const;
    void print_shape() const;
    
    static Tensor from_image(const std::string& path, DType dt = DType::Float32);
    void save_image(const std::string& path) const;

    Tensor gather(const Tensor& index, size_t dim=1) const;
    std::shared_ptr<GradFn> grad_fn;

    void backward(); 
};


#ifdef USE_CUDA
    #include <cuda_runtime.h>
#endif


// ---------- inline small wrappers ----------
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