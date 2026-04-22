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
#include <unordered_map>
#include <mutex>

#ifdef USE_CUDA
    #include <cuda_runtime.h>
#endif

enum class DType {
    Float32, Int32, Double64, UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int64, Bool, Float16
};

HOST_DEVICE inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32:  return sizeof(float);
        case DType::Int32:    return sizeof(int32_t);
        case DType::Double64: return sizeof(double);
        case DType::UInt8:    return sizeof(uint8_t);
        case DType::UInt16:   return sizeof(uint16_t);
        case DType::UInt32:   return sizeof(uint32_t);
        case DType::UInt64:   return sizeof(uint64_t);
        case DType::Int8:     return sizeof(int8_t);
        case DType::Int16:    return sizeof(int16_t);
        case DType::Int64:    return sizeof(int64_t);
        case DType::Bool:     return sizeof(bool);
        case DType::Float16:  return 2;
    }
    return sizeof(float);
}

inline const char* dtype_to_str(DType dt) {
    switch (dt) {
        case DType::Float32:  return "float32";
        case DType::Int32:    return "int32";
        case DType::Double64: return "float64";
        case DType::UInt8:    return "uint8";
        case DType::UInt16:   return "uint16";
        case DType::UInt32:   return "uint32";
        case DType::UInt64:   return "uint64";
        case DType::Int8:     return "int8";
        case DType::Int16:    return "int16";
        case DType::Int64:    return "int64";
        case DType::Bool:     return "bool";
        case DType::Float16:  return "float16";
    }
    return "unknown";
}

HOST_DEVICE inline double read_scalar_at(const void* data, size_t idx, DType dt) {
    switch (dt) {
        case DType::Float32:  return static_cast<double>(static_cast<const float*>   (data)[idx]);
        case DType::Int32:    return static_cast<double>(static_cast<const int32_t*> (data)[idx]);
        case DType::Double64: return                     static_cast<const double*>  (data)[idx];
        case DType::UInt8:    return static_cast<double>(static_cast<const uint8_t*> (data)[idx]);
        case DType::UInt16:   return static_cast<double>(static_cast<const uint16_t*>(data)[idx]);
        case DType::UInt32:   return static_cast<double>(static_cast<const uint32_t*>(data)[idx]);
        case DType::UInt64:   return static_cast<double>(static_cast<const uint64_t*>(data)[idx]);
        case DType::Int8:     return static_cast<double>(static_cast<const int8_t*>  (data)[idx]);
        case DType::Int16:    return static_cast<double>(static_cast<const int16_t*> (data)[idx]);
        case DType::Int64:    return static_cast<double>(static_cast<const int64_t*> (data)[idx]);
        case DType::Bool:     return static_cast<double>(static_cast<const bool*>    (data)[idx]);
        default:              return 0.0;
    }
}

HOST_DEVICE inline void write_scalar_at(void* data, size_t idx, DType dt, double val) {
    switch (dt) {
        case DType::Float32:  static_cast<float*>   (data)[idx] = static_cast<float>   (val); return;
        case DType::Int32:    static_cast<int32_t*> (data)[idx] = static_cast<int32_t> (val); return;
        case DType::Double64: static_cast<double*>  (data)[idx] = val;                         return;
        case DType::UInt8:    static_cast<uint8_t*> (data)[idx] = static_cast<uint8_t> (val); return;
        case DType::UInt16:   static_cast<uint16_t*>(data)[idx] = static_cast<uint16_t>(val); return;
        case DType::UInt32:   static_cast<uint32_t*>(data)[idx] = static_cast<uint32_t>(val); return;
        case DType::UInt64:   static_cast<uint64_t*>(data)[idx] = static_cast<uint64_t>(val); return;
        case DType::Int8:     static_cast<int8_t*>  (data)[idx] = static_cast<int8_t>  (val); return;
        case DType::Int16:    static_cast<int16_t*> (data)[idx] = static_cast<int16_t> (val); return;
        case DType::Int64:    static_cast<int64_t*> (data)[idx] = static_cast<int64_t> (val); return;
        case DType::Bool:     static_cast<bool*>    (data)[idx] = (val != 0.0);               return;
        default:                                                                                return;
    }
}

struct GradFn;

template <typename T, size_t N>
class SmallVector {
    static_assert(N > 0, "SmallVector inline capacity must be > 0");

    size_t size_ = 0;
    alignas(T) char                    stack_[sizeof(T) * N];
    alignas(std::vector<T>) char       heap_buf_[sizeof(std::vector<T>)];
    bool heap_active_ = false;

    T*                    sp()       noexcept { return reinterpret_cast<T*>(stack_); }
    const T*              sp() const noexcept { return reinterpret_cast<const T*>(stack_); }
    std::vector<T>&       hv()       noexcept { return *reinterpret_cast<std::vector<T>*>(heap_buf_); }
    const std::vector<T>& hv() const noexcept { return *reinterpret_cast<const std::vector<T>*>(heap_buf_); }

    void init_heap(size_t reserve) {
        new (heap_buf_) std::vector<T>();
        hv().reserve(reserve);
        heap_active_ = true;
    }

    void drop_heap() noexcept {
        if (heap_active_) { hv().~vector<T>(); heap_active_ = false; }
    }

    void spill_to_heap() {
        init_heap(size_ + 1);
        for (size_t i = 0; i < size_; ++i) {
            hv().push_back(std::move(sp()[i]));
            sp()[i].~T();
        }
    }

public:
    SmallVector() noexcept = default;

    SmallVector(const std::vector<T>& v) : size_(v.size()) {
        if (size_ > N) { init_heap(size_); hv().assign(v.begin(), v.end()); }
        else            { for (size_t i = 0; i < size_; ++i) new (sp() + i) T(v[i]); }
    }

    SmallVector(std::initializer_list<T> il) : size_(il.size()) {
        if (size_ > N) { init_heap(size_); hv().assign(il.begin(), il.end()); }
        else            { size_t i = 0; for (const T& x : il) new (sp() + i++) T(x); }
    }

    SmallVector(const SmallVector& o) : size_(o.size_) {
        if (o.heap_active_) { init_heap(o.size_); hv() = o.hv(); }
        else                 { for (size_t i = 0; i < size_; ++i) new (sp() + i) T(o.sp()[i]); }
    }

    SmallVector(SmallVector&& o) noexcept : size_(o.size_), heap_active_(o.heap_active_) {
        if (o.heap_active_) {
            new (heap_buf_) std::vector<T>(std::move(o.hv()));
            o.drop_heap();
        } else {
            for (size_t i = 0; i < size_; ++i) {
                new (sp() + i) T(std::move(o.sp()[i]));
                o.sp()[i].~T();
            }
        }
        o.size_ = 0;
    }

    SmallVector& operator=(const SmallVector& o) {
        if (this == &o) return *this;
        clear();
        size_ = o.size_;
        if (o.heap_active_) { if (!heap_active_) init_heap(o.size_); hv() = o.hv(); }
        else                 { drop_heap(); for (size_t i = 0; i < size_; ++i) new (sp() + i) T(o.sp()[i]); }
        return *this;
    }

    SmallVector& operator=(SmallVector&& o) noexcept {
        if (this == &o) return *this;
        clear();
        size_ = o.size_; heap_active_ = o.heap_active_;
        if (o.heap_active_) {
            new (heap_buf_) std::vector<T>(std::move(o.hv()));
            o.drop_heap();
        } else {
            for (size_t i = 0; i < size_; ++i) {
                new (sp() + i) T(std::move(o.sp()[i]));
                o.sp()[i].~T();
            }
        }
        o.size_ = 0;
        return *this;
    }

    ~SmallVector() { clear(); }

    void clear() noexcept {
        if (heap_active_) { drop_heap(); }
        else               { for (size_t i = 0; i < size_; ++i) sp()[i].~T(); }
        size_ = 0;
    }

    void push_back(const T& val) {
        if (!heap_active_ && size_ < N) { new (sp() + size_++) T(val); return; }
        if (!heap_active_) spill_to_heap();
        hv().push_back(val);
        ++size_;
    }

    T&       operator[](size_t i)       noexcept { return heap_active_ ? hv()[i] : sp()[i]; }
    const T& operator[](size_t i) const noexcept { return heap_active_ ? hv()[i] : sp()[i]; }

    T*       data()       noexcept { return heap_active_ ? hv().data() : sp(); }
    const T* data() const noexcept { return heap_active_ ? hv().data() : sp(); }

    size_t size()  const noexcept { return size_; }
    bool   empty() const noexcept { return size_ == 0; }

    T*       begin()       noexcept { return data(); }
    T*       end()         noexcept { return data() + size_; }
    const T* begin() const noexcept { return data(); }
    const T* end()   const noexcept { return data() + size_; }

    std::vector<T> to_vector() const {
        if (heap_active_) return hv();
        return std::vector<T>(sp(), sp() + size_);
    }
};

class RefCounted {
public:
    mutable std::atomic<int32_t> ref_count_{0};

    RefCounted() noexcept { ref_count_.store(0, std::memory_order_relaxed); }
    virtual ~RefCounted() = default;

    void retain() const noexcept { ref_count_.fetch_add(1, std::memory_order_relaxed); }

    bool release() const noexcept {
        return ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1;
    }

    int32_t use_count() const noexcept { return ref_count_.load(std::memory_order_relaxed); }
};

template <typename T>
class intrusive_ptr {
    static_assert(std::is_base_of<RefCounted, T>::value, "T must inherit RefCounted");
    T* ptr_ = nullptr;

public:
    intrusive_ptr() noexcept = default;
    intrusive_ptr(std::nullptr_t) noexcept : ptr_(nullptr) {}

    explicit intrusive_ptr(T* p, bool add_ref = true) noexcept : ptr_(p) {
        if (ptr_ && add_ref) ptr_->retain();
    }

    intrusive_ptr(const intrusive_ptr& o) noexcept : ptr_(o.ptr_) {
        if (ptr_) ptr_->retain();
    }

    intrusive_ptr(intrusive_ptr&& o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }

    intrusive_ptr& operator=(const intrusive_ptr& o) noexcept {
        if (this != &o) {
            if (ptr_ && ptr_->release()) delete ptr_;
            ptr_ = o.ptr_;
            if (ptr_) ptr_->retain();
        }
        return *this;
    }

    intrusive_ptr& operator=(intrusive_ptr&& o) noexcept {
        if (this != &o) {
            if (ptr_ && ptr_->release()) delete ptr_;
            ptr_ = o.ptr_; o.ptr_ = nullptr;
        }
        return *this;
    }

    intrusive_ptr& operator=(std::nullptr_t) noexcept { reset(); return *this; }

    ~intrusive_ptr() { reset(); }

    void reset() noexcept {
        if (ptr_ && ptr_->release()) delete ptr_;
        ptr_ = nullptr;
    }

    T*       get()        const noexcept { return ptr_; }
    T*       operator->() const noexcept { return ptr_; }
    T&       operator*()  const noexcept { return *ptr_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }
    bool operator==(const intrusive_ptr& o) const noexcept { return ptr_ == o.ptr_; }
    bool operator!=(const intrusive_ptr& o) const noexcept { return ptr_ != o.ptr_; }
};

template <typename T, typename... Args>
intrusive_ptr<T> make_intrusive(Args&&... args) {
    return intrusive_ptr<T>(new T(std::forward<Args>(args)...), true);
}

struct Block {
    void*  ptr;
    size_t size;
    bool   in_use;
};

class CudaMemoryPool {
public:
    static CudaMemoryPool& instance() {
        static CudaMemoryPool pool;
        return pool;
    }

    CudaMemoryPool(const CudaMemoryPool&)            = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

    void* allocate(size_t n);
    void  deallocate(void* ptr, size_t n) noexcept;
    void  release_all() noexcept;

private:
    CudaMemoryPool()  = default;
    ~CudaMemoryPool() { release_all(); }

    static size_t next_pow2(size_t n) noexcept {
        size_t p = 1;
        while (p < n) p <<= 1;
        return p;
    }

    std::unordered_map<size_t, std::vector<Block>> pool_;
    std::mutex mtx_;
};

struct Storage {
    std::shared_ptr<void> data;
    size_t                bytes  = 0;
    Device                device;

    static std::shared_ptr<Storage> allocate(size_t n_elem, DType dt,
                                              Device dev = Device(DeviceType::CPU));
};

inline SmallVector<size_t, 5> calc_strides(const SmallVector<size_t, 5>& shape) {
    const size_t ndim = shape.size();
    if (ndim == 0) return {};
    SmallVector<size_t, 5> st;
    size_t s = 1;
    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
        st.push_back(s);
        s *= shape[i];
    }
    size_t lo = 0, hi = st.size() - 1;
    while (lo < hi) std::swap(st[lo++], st[hi--]);
    return st;
}

struct Tensorimpl : public RefCounted {
    std::shared_ptr<Storage>  data;
    intrusive_ptr<Tensorimpl> grad;
    std::shared_ptr<GradFn>   grad_fn;

    size_t offset = 0;
    size_t ndim   = 0;
    size_t numel_ = 0;

    SmallVector<size_t, 5> shape;
    SmallVector<size_t, 5> strides;

    bool  requires_grad = false;
    DType dtype         = DType::Float32;

    Tensorimpl(const std::vector<size_t>& shape_, DType dtype_,
               bool requires_grad_, Device dev);

    Tensorimpl(std::shared_ptr<Storage>      storage,
               size_t                         offset_,
               const SmallVector<size_t, 5>& shape_,
               const SmallVector<size_t, 5>& strides_,
               DType                          dtype_,
               bool                           requires_grad_);

    virtual ~Tensorimpl() = default;
};

struct Tensor {
    intrusive_ptr<Tensorimpl> impl;

    Tensor()                             = default;
    Tensor(const Tensor&)                = default;
    Tensor(Tensor&&) noexcept            = default;
    Tensor& operator=(const Tensor&)     = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    ~Tensor()                            = default;

    Tensor(const std::vector<size_t>& shape_, DType dtype_ = DType::Float32,
           bool requires_grad_ = false);

    Tensor(const size_t* shape_ptr, size_t ndim_, DType dtype_, bool rg)
        : Tensor(std::vector<size_t>(shape_ptr, shape_ptr + ndim_), dtype_, rg) {}

    Device device()      const noexcept { return impl ? impl->data->device : Device(DeviceType::CPU); }
    size_t numel()       const noexcept { return impl ? impl->numel_ : 0; }
    size_t numel_()      const noexcept { return numel(); }
    DType  _dtype()      const          { require_impl(); return impl->dtype; }
    size_t dtype_bytes() const          { require_impl(); return dtype_size(impl->dtype); }
    bool   requires_grad() const        { require_impl(); return impl->requires_grad; }

    std::vector<size_t> shape() const {
        return impl ? impl->shape.to_vector() : std::vector<size_t>{};
    }

    Tensor& requires_grad_(bool b) { if (impl) impl->requires_grad = b; return *this; }

    bool   is_contiguous() const;
    Tensor contiguous()    const;
    Tensor clone()         const;
    Tensor detach()        const;
    Tensor to(Device target) const;

    double read_scalar (size_t idx) const;
    void   write_scalar(size_t idx, double val);

    static Tensor ones   (const std::vector<size_t>& shape, DType dt = DType::Float32, bool rg = false);
    static Tensor zeros  (const std::vector<size_t>& shape, DType dt = DType::Float32, bool rg = false);
    static Tensor full   (const std::vector<size_t>& shape, double value, DType dt = DType::Float32, bool rg = false);
    static Tensor rand   (const std::vector<size_t>& shape, DType dt = DType::Float32, bool rg = false);
    static Tensor empty  (const std::vector<size_t>& shape, DType dt = DType::Float32, bool rg = false);
    static Tensor arange (double start, double end, double step, DType dtype = DType::Float32);
    static Tensor from_vector(const std::vector<double>& data, const std::vector<size_t>& shape,
                               DType dtype = DType::Float32, bool rg = false);

    Tensor astype(DType new_dtype) const;
    void   to_(DType new_dtype);

    Tensor  reshape (const std::vector<size_t>& new_shape) const;
    Tensor  permute (const std::vector<size_t>& dims)       const;
    Tensor& t_();
    Tensor  squeeze()             const;
    Tensor  unsqueeze(size_t dim) const;
    Tensor  flatten()             const;
    Tensor  select(size_t dim, size_t index) const;
    Tensor  gather(const Tensor& index, size_t dim = 1) const;
    Tensor  argmax(int dim = -1) const;

    static Tensor from_image(const std::string& path, DType dt = DType::Float32);
    void save_image(const std::string& path) const;

    template <typename T>
    T item() const {
        require_impl();
        if (numel() != 1) throw std::runtime_error("item(): tensor must have exactly 1 element");
        return static_cast<T>(read_scalar(impl->offset));
    }

    std::shared_ptr<GradFn> grad_fn() const { return impl ? impl->grad_fn : nullptr; }

    Tensor grad() const {
        if (!impl || !impl->grad) return Tensor();
        Tensor g; g.impl = impl->grad; return g;
    }

    void backward();
    void zero_grad();
    void print_shape() const;

    template <bool Writable>
    struct ProxyBase {
        intrusive_ptr<Tensorimpl> impl;
        size_t offset;
        size_t depth;

        ProxyBase(intrusive_ptr<Tensorimpl> impl_, size_t off, size_t dp)
            : impl(std::move(impl_)), offset(off), depth(dp) {}

        ProxyBase operator[](size_t i) const {
            if (!impl)                   throw std::runtime_error("Invalid tensor");
            if (depth >= impl->ndim)     throw std::out_of_range("Too many indices");
            if (i >= impl->shape[depth]) throw std::out_of_range("Index out of bounds");
            return ProxyBase(impl, offset + i * impl->strides[depth], depth + 1);
        }

        operator double() const {
            if (!impl)               throw std::runtime_error("Invalid tensor");
            if (depth != impl->ndim) throw std::out_of_range("Not at leaf index");
            if (impl->data->device.is_cuda()) {
#ifdef USE_CUDA
                char buf[8] = {};
                const char* src = static_cast<const char*>(impl->data->data.get())
                                  + offset * dtype_size(impl->dtype);
                cudaMemcpy(buf, src, dtype_size(impl->dtype), cudaMemcpyDeviceToHost);
                return read_scalar_at(buf, 0, impl->dtype);
#else
                throw std::runtime_error("Tensor is on CUDA but USE_CUDA is not defined");
#endif
            }
            return read_scalar_at(impl->data->data.get(), offset, impl->dtype);
        }

        template <bool W = Writable, typename = std::enable_if_t<W>>
        ProxyBase& operator=(double val) {
            if (!impl) throw std::runtime_error("Invalid tensor");
            fill_recursive(depth, offset, val);
            return *this;
        }

        template <bool W = Writable, typename U, typename = std::enable_if_t<W>>
        ProxyBase& operator=(U val) { return operator=(static_cast<double>(val)); }

    private:
        void fill_recursive(size_t d, size_t off, double val) {
            if (d == impl->ndim) {
                if (impl->data->device.is_cuda()) {
#ifdef USE_CUDA
                    char buf[8] = {};
                    write_scalar_at(buf, 0, impl->dtype, val);
                    char* dst = static_cast<char*>(impl->data->data.get())
                                + off * dtype_size(impl->dtype);
                    cudaMemcpy(dst, buf, dtype_size(impl->dtype), cudaMemcpyHostToDevice);
#endif
                } else {
                    write_scalar_at(impl->data->data.get(), off, impl->dtype, val);
                }
                return;
            }
            const size_t len    = impl->shape[d];
            const size_t stride = impl->strides[d];
            for (size_t i = 0; i < len; ++i)
                fill_recursive(d + 1, off + i * stride, val);
        }
    };

    using Proxy      = ProxyBase<true>;
    using ConstProxy = ProxyBase<false>;

    Proxy      operator[](size_t i);
    ConstProxy operator[](size_t i) const;

private:
    void require_impl() const {
        if (!impl) throw std::runtime_error("Tensor is empty");
    }
};

inline void print_t(const Tensor& t) {
    Tensor cpu = t.device().is_cuda() ? t.to(Device(DeviceType::CPU)) : t;
    const size_t n = cpu.numel();
    std::cout << "[";
    for (size_t i = 0; i < n; ++i) {
        std::cout << cpu.read_scalar(i);
        if (i + 1 < n) std::cout << ", ";
    }
    std::cout << "]\n";
}