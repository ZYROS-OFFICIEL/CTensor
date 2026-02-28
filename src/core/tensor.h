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
#include <numeric> 
#include <atomic>
#include <algorithm>


// DTYPE 

/*
 DType Represents the underlying data type of the Tensor's elements.
  Supports various bit-widths of floats and integers.
 */
enum class DType { 
    Float32, Int32, Double64, UInt8, Int8, Int16, Int64, Bool, Float16  
};

/*
 *  Returns the size in bytes for a given DType.
 */
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
        case DType::Float16: return 2; // Float16 is 2 bytes
    }
    return sizeof(float);
}

/*
   Reads a scalar from a type-erased void* buffer, casting it safely to double.
 data : Pointer to the raw buffer.
 idx : Linear index of the element to read.
 dt : The DType specifying how to interpret the raw bytes.
 */
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

/*
    Writes a double value into a type-erased void* buffer, downcasting to the target type.
 */
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

// Forward declaration for Autograd functionality
struct GradFn; 

// Helper 1: SMALL VECTOR

/*
 A hybrid vector designed to minimize heap allocations.
 Deep learning tensors rarely exceed 5 dimensions. By storing up to N 
 elements directly on the stack, we avoid expensive dynamic memory allocation 
 for shapes and strides during tensor creation. If it exceeds N, it falls back to std::vector.
 */
template <typename T, size_t N>
class SmallVector {
private:
    size_t size_ = 0;
    T stack_data_[N];
    std::vector<T> heap_data_; 

public:
    SmallVector() = default;
    
    SmallVector(const std::vector<T>& vec) : size_(vec.size()) {
        if (size_ > N) {
            heap_data_ = vec;
        } else {
            std::copy(vec.begin(), vec.end(), stack_data_);
        }
    }

    SmallVector(std::initializer_list<T> list) : size_(list.size()) {
        if (size_ > N) {
            heap_data_ = list;
        } else {
            std::copy(list.begin(), list.end(), stack_data_);
        }
    }

    // Accessors transparently handle whether data is on stack or heap
    T& operator[](size_t i) { return (size_ > N) ? heap_data_[i] : stack_data_[i]; }
    const T& operator[](size_t i) const { return (size_ > N) ? heap_data_[i] : stack_data_[i]; }
    const T* data() const { return (size_ > N) ? heap_data_.data() : stack_data_; }
    T* data() { return (size_ > N) ? heap_data_.data() : stack_data_; }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    void push_back(const T& val) {
        if (size_ < N) {
            stack_data_[size_++] = val;
        } else {
            if (size_ == N) { // Transition from stack to heap
                heap_data_.reserve(N + 1);
                heap_data_.assign(stack_data_, stack_data_ + N);
            }
            heap_data_.push_back(val);
            size_++;
        }
    }

    std::vector<T> to_vector() const {
        if (size_ > N) return heap_data_;
        return std::vector<T>(stack_data_, stack_data_ + size_);
    }
    
    T* begin() { return data(); }
    T* end() { return data() + size_; }
    const T* begin() const { return data(); }
    const T* end() const { return data() + size_; }
};

// Helper 2: INTRUSIVE POINTER (REFERENCE COUNTING)

/*
Base class for thread-safe intrusive reference counting.
Unlike std::shared_ptr, the reference count is stored directly inside the object,
saving a separate allocation for the control block.
 */
class RefCounted {
public:
    mutable std::atomic<int32_t> ref_count_{0};

    RefCounted() { ref_count_.store(0, std::memory_order_relaxed); }
    virtual ~RefCounted() = default;

    void retain() const {
        ref_count_.fetch_add(1, std::memory_order_relaxed);
    }

    // Returns true if this was the last reference and object should be destroyed
    bool release() const {
        return ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1;
    }
};


//A smart pointer for objects inheriting from RefCounted.

template <typename T>
class intrusive_ptr {
private:
    T* ptr_ = nullptr;

public:
    intrusive_ptr() : ptr_(nullptr) {}
    intrusive_ptr(std::nullptr_t) : ptr_(nullptr) {}

    explicit intrusive_ptr(T* p, bool add_ref = true) : ptr_(p) {
        if (ptr_ && add_ref) ptr_->retain();
    }

    // Rule of 5 semantics
    intrusive_ptr(const intrusive_ptr& other) : ptr_(other.ptr_) {
        if (ptr_) ptr_->retain();
    }

    intrusive_ptr(intrusive_ptr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    intrusive_ptr& operator=(const intrusive_ptr& other) {
        if (this != &other) {
            if (ptr_ && ptr_->release()) delete ptr_;
            ptr_ = other.ptr_;
            if (ptr_) ptr_->retain();
        }
        return *this;
    }

    intrusive_ptr& operator=(intrusive_ptr&& other) noexcept {
        if (this != &other) {
            if (ptr_ && ptr_->release()) delete ptr_;
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    intrusive_ptr& operator=(std::nullptr_t) {
        if (ptr_ && ptr_->release()) delete ptr_;
        ptr_ = nullptr;
        return *this;
    }

    ~intrusive_ptr() {
        if (ptr_ && ptr_->release()) delete ptr_;
    }

    T* get() const { return ptr_; }
    T* operator->() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    
    void reset() {
        if (ptr_ && ptr_->release()) delete ptr_;
        ptr_ = nullptr;
    }
};


// STORAGE & INTERNAL IMPLEMENTATION

//Represents raw untyped contiguous memory on a specific device (CPU/GPU).

struct Storage {
    std::shared_ptr<void> data;
    size_t size = 0;
    Device device;
    static std::shared_ptr<Storage> allocate(size_t n, DType dt, bool requires_grad = false, Device dev = Device(DeviceType::CPU));
};

/*
 Calculates memory strides required to navigate multi-dimensional shapes.
 Example: Shape [2, 3, 4] -> Strides [12, 4, 1]
 */
inline SmallVector<size_t, 5> calc_strides(const SmallVector<size_t, 5>& shape) {
    if (shape.empty()) return {};
    SmallVector<size_t, 5> strides;
    std::vector<size_t> temp(shape.size());
    size_t current = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        temp[i] = current;
        current *= shape[i];
    }
    return SmallVector<size_t, 5>(temp);
}

/*
 The heavy-weight backend object representing a Tensor.
 Inherits from RefCounted so multiple lightweight `Tensor` frontends can point to it.
 */
struct Tensorimpl : public RefCounted {
    std::shared_ptr<Storage> data;
    
    // Gradient points to another Tensorimpl. Using intrusive_ptr prevents infinite recursion.
    intrusive_ptr<Tensorimpl> grad; 
    
    size_t offset = 0;
    size_t ndim = 0;
    
    SmallVector<size_t, 5> shape;
    SmallVector<size_t, 5> strides;
    
    bool requires_grad = false;
    DType dtype = DType::Float32;
    std::shared_ptr<GradFn> grad_fn; // Node in the computation graph

    // Default physical allocation constructor
    Tensorimpl(const std::vector<size_t>& shape_, DType dtype_, bool requires_grad_, Device dev_)
        : shape(shape_), strides(calc_strides(shape)), dtype(dtype_), requires_grad(requires_grad_), ndim(shape_.size()) {
        
        size_t total_el = 1;
        for (auto s : shape) total_el *= s;
        data = Storage::allocate(total_el, dtype, requires_grad, dev_);
    }

    // Logical view constructor (shares storage with another tensor)
    Tensorimpl(std::shared_ptr<Storage> storage_,
               size_t offset_,
               const SmallVector<size_t, 5>& shape_,
               const SmallVector<size_t, 5>& strides_,
               DType dtype_,
               bool requires_grad_)
        : data(storage_), offset(offset_), shape(shape_), strides(strides_), 
          dtype(dtype_), requires_grad(requires_grad_), ndim(shape_.size()) {}
    
    virtual ~Tensorimpl() = default;
};

// ============================================================================
// PUBLIC TENSOR API
// ============================================================================

/*
    The primary user-facing Tensor class.
  This is a lightweight handle wrapping `intrusive_ptr<Tensorimpl>`.
  Passing this by value is cheap and shares the underlying memory.
 */
struct Tensor {
    intrusive_ptr<Tensorimpl> impl;

    Device device() const;
    Tensor to(Device target_device); 

    Tensor() = default;
    
    Tensor(const std::vector<size_t>& shape_, DType dtype_ = DType::Float32, bool requires_grad_ = false);
    Tensor(const size_t* shape_ptr, size_t ndim, DType dtype, bool requires_grad)
        : Tensor(std::vector<size_t>(shape_ptr, shape_ptr + ndim), dtype, requires_grad) {}

    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    ~Tensor() = default;

    // Core Properties
    size_t numel() const;
    size_t numel_() const { return numel(); }
    std::vector<size_t> shape() const; 
    inline DType _dtype() const;
    inline size_t dtype_bytes() const;
    bool requires_grad() const;

    // Memory layout checks
    bool is_contiguous() const;
    Tensor contiguous() const; 

    // Raw element access
    inline double read_scalar(size_t idx) const;
    inline void write_scalar(size_t idx, double val);

    // Factory methods
    static Tensor ones(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false);
    static Tensor zeros(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false);
    static Tensor full(const std::vector<size_t>& shape_, double value, DType dt = DType::Float32, bool requires_grad_ = false);
    static Tensor rand(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false);
    static Tensor empty(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false);
    static Tensor from_vector(const std::vector<double>& data, const std::vector<size_t>& shape, DType dtype = DType::Float32, bool requires_grad = false);

    // Graph and memory management
    Tensor clone() const;
    Tensor detach() const;
    Tensor detach() ;
    Tensor& requires_grad_(bool b);

    // ------------------------------------------------------------------------
    // MULTI-DIMENSIONAL INDEXING (PROXY PATTERN)
    // ------------------------------------------------------------------------
    /**
     * @brief Enables intuitive indexing like `tensor[1][2] = 5;`.
     * Tracks the current depth and memory offset without copying data.
     */
    template <bool Writable>
    struct ProxyBase {
        intrusive_ptr<Tensorimpl> impl;
        size_t offset;
        size_t depth;

        ProxyBase(intrusive_ptr<Tensorimpl> impl_, size_t off, size_t dp = 0)
            : impl(std::move(impl_)), offset(off), depth(dp) {}

        // Moves one dimension deeper
        ProxyBase operator[](size_t i) const {
            if (!impl) throw std::runtime_error("Invalid tensor");
            if (depth >= impl->ndim) throw std::out_of_range("Too many indices");
            if (i >= impl->shape[depth]) throw std::out_of_range("Index out of bounds");
            
            size_t new_offset = offset + i * impl->strides[depth];
            return ProxyBase(impl, new_offset, depth + 1);
        }

        // Implicit conversion to double for reading
        operator double() const {
            if (!impl) throw std::runtime_error("Invalid tensor");
            if (depth != impl->ndim) throw std::out_of_range("Not at leaf index"); 
            return read_scalar_at(impl->data->data.get(), offset, impl->dtype);
        }

        // Recursive filler for assigning scalars to sub-tensors (e.g. tensor[0] = 5)
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

        // Assignment operator for writing (Only enabled if Writable = true)
        template <bool W = Writable, typename = std::enable_if_t<W>>
        ProxyBase& operator=(double val) {
            if (!impl) throw std::runtime_error("Invalid tensor");
            
            if (depth == impl->ndim) {
                write_scalar_at(impl->data->data.get(), offset, impl->dtype, val);
            } else {
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

    // Operations and Reshaping
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
    Tensor argmax(int dim = -1) const;
    
    // Scalar extraction
    template<typename T>
    T item() const {
        if (!impl) throw std::runtime_error("item() on empty tensor");
        if (numel() != 1) throw std::runtime_error("item() only for 1-element tensors");
        return static_cast<T>(read_scalar(0));
    }
    
    // Autograd
    std::shared_ptr<GradFn> grad_fn;
    void backward(); 
    
    Tensor grad() const {
        if (!impl || !impl->grad) return Tensor();
        Tensor g;
        g.impl = impl->grad; // Copying intrusive_ptr safely increments ref count
        return g;
    }

    void zero_grad();
};

#ifdef USE_CUDA
    #include <cuda_runtime.h>
#endif


// INLINE IMPLEMENTATIONS

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

inline Tensor::Tensor(const std::vector<size_t>& shape_, DType dtype_, bool requires_grad_) {
    Tensorimpl* raw = new Tensorimpl(shape_, dtype_, requires_grad_, Device(DeviceType::CPU));
    impl = intrusive_ptr<Tensorimpl>(raw); 
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
    return impl->shape.to_vector();
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
        double v = read_scalar_at(t.impl->data->data.get(), i, t.impl->dtype);
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