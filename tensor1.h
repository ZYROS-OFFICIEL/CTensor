#pragma once
#include <iostream>
#include <cstddef>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <cassert>
#include <memory>

enum class DType { Float32, Int32, Double64 };

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return sizeof(float);
        case DType::Int32:   return sizeof(int);
        case DType::Double64:return sizeof(double);
    }
    return sizeof(float);
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

struct Storage {
    std::shared_ptr<void> data;
    std::shared_ptr<void> grad;
    size_t size = 0;

    static std::shared_ptr<Storage> allocate(size_t n, DType dt, bool requires_grad = false) {
        auto s = std::make_shared<Storage>();
        s->size = n * dtype_size(dt);

        // allocate data
        void* p = std::malloc(s->size);
        if (!p && s->size) throw std::bad_alloc();
        std::memset(p, 0, s->size);
        s->data = std::shared_ptr<void>(p, std::free);

        // optional grad
        if (requires_grad) {
            void* g = std::malloc(s->size);
            if (!g && s->size) throw std::bad_alloc();
            std::memset(g, 0, s->size);
            s->grad = std::shared_ptr<void>(g, std::free);
        } else {
            s->grad = nullptr;
        }

        return s;
    }
};

struct Tensorimpl {
    std::shared_ptr<Storage> storage;
    size_t offset = 0;
    size_t ndim = 0;
    size_t* shape = nullptr;
    size_t* strides = nullptr;
    bool requires_grad = false;
    DType dtype = DType::Float32;

    Tensorimpl(const std::vector<size_t>& shape_, DType dtype_ = DType::Float32, bool requires_grad_ = false)
        : offset(0), ndim(shape_.size()), requires_grad(requires_grad_), dtype(dtype_)
    {
        // allocate shape & strides
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

    ~Tensorimpl() {
        std::free(shape);
        std::free(strides);
    }
};
struct Tensor{
    std::shared_ptr<Tensorimpl> impl;

    // --- Default constructor ---
    Tensor() = default;

    // --- Primary constructor (create a new impl) ---
    Tensor(const std::vector<size_t>& shape_,
           DType dtype_ = DType::Float32,
           bool requires_grad_ = false)
        : impl(std::make_shared<Tensorimpl>(shape_, dtype_, requires_grad_))
    {}

    // --- Copy constructor ---
    Tensor(const Tensor& other) = default;

    // --- Move constructor ---
    Tensor(Tensor&& other) noexcept = default;

    // --- Copy assignment ---
    Tensor& operator=(const Tensor& other) = default;

    // --- Move assignment ---
    Tensor& operator=(Tensor&& other) noexcept = default;

    // --- Destructor ---
    ~Tensor() = default;
    //helper wrapers
    inline double read_scalar(size_t idx) const {
        return read_scalar_at(impl->storage->data.get(), idx, impl->dtype);
    }

    inline void write_scalar(size_t idx, double val) {
        write_scalar_at(impl->storage->data.get(), idx, impl->dtype, val);
    }
    //Utulities
    size_t numel() const {
        if (!impl) return 0;
        size_t n = 1;
        for (size_t i = 0; i < impl->ndim; ++i) n *= impl->shape[i];
        return n;
    }

    std::vector<size_t> shape() const {
        if (!impl) return {};
        return std::vector<size_t>(impl->shape, impl->shape + impl->ndim);
    }

    void print_shape() const {
        if (!impl) { std::cout << "()\n"; return; }
        std::cout << "(";
        for (size_t i = 0; i < impl->ndim; ++i) {
            std::cout << impl->shape[i];
            if (i < impl->ndim - 1) std::cout << ", ";
        }
        std::cout << ")\n";
    }
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

    // ---------------- Tensor indexing ----------------
    Proxy operator[](size_t i) {
        if (!impl) throw std::runtime_error("Empty tensor");
        if (impl->ndim == 0) throw std::out_of_range("Tensor has no dimensions");
        return Proxy(impl, i * impl->strides[0], 1);
    }

    ConstProxy operator[](size_t i) const {
        if (!impl) throw std::runtime_error("Empty tensor");
        if (impl->ndim == 0) throw std::out_of_range("Tensor has no dimensions");
        return ConstProxy(impl, i * impl->strides[0], 1);
    }

    static Tensor ones(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false){
        Tensor t(shape_, dt, requires_grad_);
        size_t n = t.numel();
        for (size_t i = 0; i < n; ++i) write_scalar_at(t.impl->storage->data.get(), i, dt, 1.0);
        return t;
    }
    static Tensor zeros(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false){
        Tensor t(shape_, dt, requires_grad_);
        size_t n = t.numel();
        for (size_t i = 0; i < n; ++i) write_scalar_at(t.impl->storage->data.get(), i, dt, 0.0);
        return t;
    }
    static Tensor full(const std::vector<size_t>& shape_,float value,DType dt = DType::Float32, bool requires_grad_ = false){
        Tensor t(shape_, dt, requires_grad_);
        size_t n = t.numel();
        for (size_t i = 0; i < n; ++i) write_scalar_at(t.impl->storage->data.get(), i, dt, value);
        return t;
    }
    static Tensor rand(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false) {
        Tensor t(shape_, dt, requires_grad_);
        size_t n = t.numel_();
        // seed only once per program would be better; simple here:
        std::srand((unsigned int)std::time(nullptr));
        for (size_t i = 0; i < n; ++i)
            write_scalar_at(t.impl->storage->data.get(), i, dt, static_cast<double>(std::rand()) / RAND_MAX);
        return t;
    }
    static Tensor empty(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false){
        Tensor t(shape_,dt,requires_grad_);
        return t;
    }
        // ---------- dtype helpers ----------
    DType _dtype() const noexcept { 
        if (!impl) throw std::runtime_error("Tensor is empty");
        return impl->dtype;  
    }
    const char* dtype_name() const noexcept { 
        if (!impl) throw std::runtime_error("Tensor is empty");
        return dtype_to_cstr(impl->dtype); 
    }

    size_t dtype_bytes() const noexcept { 
        if (!impl) throw std::runtime_error("Tensor is empty");
        return dtype_size(impl->dtype); 
    }
        // ---------- conversion: return new tensor ----------
    Tensor astype(DType new_dtype) const {
        if (!impl) throw std::runtime_error("Empty tensor");
        if (new_dtype == impl->dtype) return Tensor(*this); // copy
        Tensor out(shape_(), new_dtype, impl->requires_grad);
        size_t n = numel_();

        // straightforward convert elementwise
        for (size_t i = 0; i < n; ++i) {
            double v = read_scalar_at(impl->storage->data.get(), i, impl->dtype);
            write_scalar_at(out.impl->storage->data.get(), i, out.impl->dtype, v);
        }
        // grad not copied by default; if you want to copy grad, convert similarly:
        if (impl->requires_grad && impl->storage->grad) {
            if (!out.grad && n) throw std::bad_alloc();
            for (size_t i = 0; i < n; ++i) {
                double gv = read_scalar_at(impl->storage->grad.get(), i, impl->dtype);
                write_scalar_at(out.impl->storage->grad.get(), i, out.impl->dtype, gv);
            }
        }
        return out;
    }
    void to_(DType new_dtype) {
        if (!impl) throw std::runtime_error("Empty tensor");
        if (new_dtype == impl->dtype) return;
        
        size_t n = numel_();
        size_t new_tsize = dtype_size(new_dtype);
        
        // allocate new storage buffer (shared_ptr)
        auto new_storage = Storage::allocate(n, new_dtype, impl->requires_grad);
        
        // elementwise convert data
        for (size_t i = 0; i < n; ++i) {
            double v = read_scalar_at(impl->storage->data.get(), i, impl->dtype);
            write_scalar_at(new_storage->data.get(), i, new_dtype, v);
        }
    
        // optional: convert gradient if it exists
        if (impl->requires_grad && impl->storage->grad) {
            for (size_t i = 0; i < n; ++i) {
                double gv = read_scalar_at(impl->storage->grad.get(), i, impl->dtype);
                write_scalar_at(new_storage->grad.get(), i, new_dtype, gv);
            }
        }
    
        // replace old storage & dtype
        impl->storage = new_storage;
        impl->dtype = new_dtype;
    }

};