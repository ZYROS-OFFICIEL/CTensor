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
#include <numeric> // For std::accumulate

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
    size_t size = 0;
    Device device;
    static std::shared_ptr<Storage> allocate(size_t n, DType dt, bool requires_grad = false, Device dev = Device(DeviceType::CPU));
};

// ----------------- Strides Helper -----------------
inline std::vector<size_t> calc_strides(const std::vector<size_t>& shape) {
    if (shape.empty()) return {};
    std::vector<size_t> strides(shape.size());
    size_t current = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        strides[i] = current;
        current *= shape[i];
    }
    return strides;
}

// ----------------- low-level impl -----------------
struct Tensorimpl {
    std::shared_ptr<Storage> data;
    
    // IMPROVEMENT: Grad is now a Tensorimpl, allowing gradients to have views/strides
    std::shared_ptr<Tensorimpl> grad = nullptr; 
    
    size_t offset = 0;
    size_t ndim = 0;
    
    // IMPROVEMENT: Use vector instead of raw pointers for safety
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    
    bool requires_grad = false;
    DType dtype = DType::Float32;
    std::shared_ptr<GradFn> grad_fn;

    // Default constructor
    Tensorimpl(const std::vector<size_t>& shape_, DType dtype_, bool requires_grad_, Device dev_)
        : shape(shape_), strides(calc_strides(shape_)), dtype(dtype_), requires_grad(requires_grad_), ndim(shape_.size()) {
        
        size_t total_el = 1;
        for (auto s : shape) total_el *= s;
        data = Storage::allocate(total_el, dtype, requires_grad, dev_);
    }

    // View constructor
    Tensorimpl(std::shared_ptr<Storage> storage_,
               size_t offset_,
               const std::vector<size_t>& shape_,
               const std::vector<size_t>& strides_,
               DType dtype_,
               bool requires_grad_)
        : data(storage_), offset(offset_), shape(shape_), strides(strides_), 
          dtype(dtype_), requires_grad(requires_grad_), ndim(shape_.size()) {}
    
    // Rule of Zero: No custom destructor needed thanks to std::vector and shared_ptr
};

// ----------------- Tensor (public API) -----------------
struct Tensor {
    std::shared_ptr<Tensorimpl> impl;

    Device device() const;
    Tensor to(Device target_device); 

    Tensor() = default;
    
    // Constructor declarations
    Tensor(const std::vector<size_t>& shape_, DType dtype_ = DType::Float32, bool requires_grad_ = false);
    Tensor(const size_t* shape_ptr, size_t ndim, DType dtype, bool requires_grad)
        : Tensor(std::vector<size_t>(shape_ptr, shape_ptr + ndim), dtype, requires_grad) {}

    // Standard Rule of 5 defaults are fine
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

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
    Tensor detach() ;
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
            return read_scalar_at(impl->data->data.get(), offset, impl->dtype);
        }

        // Helper to recursively fill a sub-tensor
        void recursive_fill(size_t current_depth, size_t current_offset, double val) {
            if (current_depth == impl->ndim) {
                write_scalar_at(impl->data->data.get(), current_offset, impl->dtype, val);
                return;
            }
            
            size_t len = impl->shape[current_depth];
            size_t stride = impl->strides[current_depth];
            for (size_t i = 0; i < len; ++i) {
                recursive_fill(current_depth + 1, current_offset + i * stride, val);
            }
        }

        template <bool W = Writable, typename = std::enable_if_t<W>>
        ProxyBase& operator=(double val) {
            if (!impl) throw std::runtime_error("Invalid tensor");
            
            if (depth == impl->ndim) {
                // Leaf assignment
                write_scalar_at(impl->data->data.get(), offset, impl->dtype, val);
            } else {
                // Slice assignment (Broadcast fill)
                recursive_fill(depth, offset, val);
            }
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

    // Gradient / Autograd API
    void backward(); 
    
    // Helper to get gradient as a Tensor (wraps the internal impl)
    Tensor grad() const {
        if (!impl || !impl->grad) return Tensor();
        Tensor g;
        g.impl = impl->grad;
        return g;
    }

    void zero_grad();
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
    return read_scalar_at(impl->data->data.get(), idx, impl->dtype);
}

inline void Tensor::write_scalar(size_t idx, double val) {
    if (!impl) throw std::runtime_error("Tensor is empty");
    write_scalar_at(impl->data->data.get(), idx, impl->dtype, val);
}

inline bool Tensor::requires_grad() const {
    if (!impl) throw std::runtime_error("Tensor is empty");
    return impl->requires_grad;
}

// Simple implementations for constructors to make it compile with the new impl structure
inline Tensor::Tensor(const std::vector<size_t>& shape_, DType dtype_, bool requires_grad_) {
    impl = std::make_shared<Tensorimpl>(shape_, dtype_, requires_grad_, Device(DeviceType::CPU));
}

inline Tensor::Proxy Tensor::operator[](size_t i) {
    if (!impl) throw std::runtime_error("Invalid tensor");
    if (impl->ndim == 0) throw std::out_of_range("Cannot index scalar");
    if (i >= impl->shape[0]) throw std::out_of_range("Index out of bounds");
    return Proxy(impl, impl->offset + i * impl->strides[0], 1);
}

inline Tensor::ConstProxy Tensor::operator[](size_t i) const {
    if (!impl) throw std::runtime_error("Invalid tensor");
    if (impl->ndim == 0) throw std::out_of_range("Cannot index scalar");
    if (i >= impl->shape[0]) throw std::out_of_range("Index out of bounds");
    return ConstProxy(impl, impl->offset + i * impl->strides[0], 1);
}

inline std::vector<size_t> Tensor::shape() const {
    if(!impl) return {};
    return impl->shape;
}

inline size_t Tensor::numel() const {
    if(!impl) return 0;
    size_t n = 1;
    for(auto s : impl->shape) n *= s;
    return n;
}
inline void print_t(const Tensor& t) {
    size_t n = t.numel_();
    std::cout << "[";
    for (size_t i = 0; i < n; i++) {
        double v = read_scalar_at(t.impl->data.get(), i, t.impl->dtype);
        std::cout << v;
        if (i != n - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
}
inline const char* dtype_to_str(DType dt) {
    switch (dt) {
        case DType::Float32: return "float32";
        case DType::Double64: return "double64";
        case DType::Int32:   return "int32";
        case DType::Int64:   return "int64";
        default:             return "unknown";
    }
}
