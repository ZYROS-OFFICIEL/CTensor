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
#include <ctime>

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
    size_t numel_() const { return numel(); }
        inline std::vector<size_t> strides() const {
        if (!impl) return {};
        return std::vector<size_t>(impl->strides, impl->strides + impl->ndim);
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
        const char* dtype_name() const {
        return (_dtype()==DType::Float32) ? "Float32" : (_dtype()==DType::Int32 ? "Int32" : "Double64");
    }

    size_t dtype_bytes() const noexcept { 
        if (!impl) throw std::runtime_error("Tensor is empty");
        return dtype_size(impl->dtype); 
    }
        // ---------- conversion: return new tensor ----------
    Tensor astype(DType new_dtype) const {
        if (!impl) throw std::runtime_error("Empty tensor");
        if (new_dtype == impl->dtype) return Tensor(*this); // copy
        Tensor out(shape(), new_dtype, impl->requires_grad);
        size_t n = numel_();

        // straightforward convert elementwise
        for (size_t i = 0; i < n; ++i) {
            double v = read_scalar_at(impl->storage->data.get(), i, impl->dtype);
            write_scalar_at(out.impl->storage->data.get(), i, out.impl->dtype, v);
        }
        // grad not copied by default; if you want to copy grad, convert similarly:
        if (impl->requires_grad && impl->storage->grad) {
            if (!out.impl->storage->grad && n) throw std::bad_alloc();
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
    //in place transpose 
    Tensor& t_() {
        if (!impl) throw std::runtime_error("Empty tensor");
        if (impl->ndim < 2)
            throw std::invalid_argument("t_: tensor must have at least 2 dimensions");

        std::swap(impl->shape[impl->ndim - 2], impl->shape[impl->ndim - 1]);
        std::swap(impl->strides[impl->ndim - 2], impl->strides[impl->ndim - 1]);
        return *this;
    }
    Tensor permute(const std::vector<size_t>& dims) const {
        if (!impl) throw std::runtime_error("Empty tensor");
        if (dims.size() != impl->ndim)
            throw std::invalid_argument("permute: dims size must match shape size");

        std::vector<bool> seen(impl->ndim, false);
        for (auto d : dims) {
            if (d >= impl->ndim || seen[d])
                throw std::invalid_argument("permute: invalid or duplicate dim");
            seen[d] = true;
        }

        // Create new Tensor sharing same storage
        Tensor out;
        out.impl = std::make_shared<Tensorimpl>(*impl); // shallow copy
        // Free and reallocate shape/strides because we’ll reorder them
        std::free(out.impl->shape);
        std::free(out.impl->strides);

        out.impl->shape = static_cast<size_t*>(std::malloc(impl->ndim * sizeof(size_t)));
        out.impl->strides = static_cast<size_t*>(std::malloc(impl->ndim * sizeof(size_t)));

        for (size_t i = 0; i < impl->ndim; ++i) {
            out.impl->shape[i]   = impl->shape[dims[i]];
            out.impl->strides[i] = impl->strides[dims[i]];
        }

        return out;
    }
        // ------------- arange -------------
    static Tensor arange(double start, double end, double step = 1.0, DType dtype = DType::Float32) {
        if (step == 0.0) throw std::invalid_argument("step must be non-zero");
        std::vector<double> vals;
        if (step > 0) {
            for (double x = start; x < end; x += step) vals.push_back(x);
        } else {
            for (double x = start; x > end; x += step) vals.push_back(x);
        }
        Tensor t({vals.size()}, dtype, false);
        for (size_t i = 0; i < vals.size(); ++i) write_scalar_at(t.impl->storage->data.get(), i, dtype, vals[i]);
        return t;
    }

    // ------------- reshape (returns view sharing storage but with contiguous strides) -------------
    Tensor reshape(const std::vector<size_t>& new_shape) const {
        if (!impl) throw std::runtime_error("Empty tensor");
        size_t old_n = numel();
        size_t new_n = 1;
        for (auto v: new_shape) new_n *= v;
        if (old_n != new_n) throw std::invalid_argument("reshape: number of elements mismatch");
        // compute contiguous strides for new shape (C-contiguous)
        std::vector<size_t> nst(new_shape.size());
        if (!new_shape.empty()) {
            nst.back() = 1;
            for (int i = (int)new_shape.size()-2; i >= 0; --i)
                nst[i] = nst[i+1] * new_shape[i+1];
        }
        // share storage and same offset
        Tensor out;
        out.impl = std::make_shared<Tensorimpl>(impl->storage, impl->offset, new_shape, nst, impl->dtype, impl->requires_grad);
        return out;
    }

    // ------------- select: remove dimension dim by indexing index -------------
    Tensor select(size_t dim, size_t index) const {
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

    // ------------- squeeze / unsqueeze / flatten -------------
    Tensor squeeze() const {
        if (!impl) throw std::runtime_error("Empty tensor");
        std::vector<size_t> nsh;
        std::vector<size_t> nst;
        for (size_t i = 0; i < impl->ndim; ++i) {
            if (impl->shape[i] == 1) continue;
            nsh.push_back(impl->shape[i]);
            nst.push_back(impl->strides[i]);
        }
        if (nsh.empty()) { nsh.push_back(1); nst.push_back(1); } // keep at least 1-d tensor
        Tensor out;
        out.impl = std::make_shared<Tensorimpl>(impl->storage, impl->offset, nsh, nst, impl->dtype, impl->requires_grad);
        return out;
    }

    Tensor unsqueeze(size_t dim) const {
        if (!impl) throw std::runtime_error("Empty tensor");
        if (dim > impl->ndim) throw std::out_of_range("unsqueeze: dim out of range");
        std::vector<size_t> nsh;
        std::vector<size_t> nst;
        // naive approach: create contiguous strides for new shape to be safe
        nsh = shape();
        nsh.insert(nsh.begin() + dim, 1);
        // compute contiguous strides for nsh
        nst.resize(nsh.size());
        nst.back() = 1;
        for (int i = (int)nst.size()-2; i >= 0; --i) nst[i] = nst[i+1] * nsh[i+1];
        Tensor out;
        out.impl = std::make_shared<Tensorimpl>(impl->storage, impl->offset, nsh, nst, impl->dtype, impl->requires_grad);
        return out;
    }

    Tensor flatten() const {
        std::vector<size_t> nsh = { numel() };
        std::vector<size_t> nst = { 1 };
        Tensor out;
        out.impl = std::make_shared<Tensorimpl>(impl->storage, impl->offset, nsh, nst, impl->dtype, impl->requires_grad);
        return out;
    }
};
// ---------- printing utilities ----------

inline void print_flat(const Tensor& t) {
    size_t n = t.numel_();
    std::cout << "[";
    for (size_t i = 0; i < n; ++i) {
        double v = read_scalar_at(t.impl->storage->data.get(), i, t.impl->dtype);
        std::cout << v;
        if (i + 1 != n) std::cout << ", ";
    }
    std::cout << "]\n";
}

// internal helper for recursive printing
static void print_recursive_braces(const Tensor& t, std::vector<size_t>& idx, size_t dim) {
    std::cout << "{";
    size_t dim_size = t.shape()[dim];

    for (size_t i = 0; i < dim_size; ++i) {
        idx[dim] = i;

        if (dim + 1 == t.impl->ndim) {
            // compute flat offset
            size_t offset = 0;
            for (size_t k = 0; k < t.impl->ndim; ++k)
                offset += idx[k] * t.strides()[k];

            double v = read_scalar_at(t.impl->storage->data.get(), offset, t.impl->dtype);

            if (t.impl->dtype == DType::Int32)
                std::cout << static_cast<long long>(std::lrint(v));
            else
                std::cout << v;
        } else {
            print_recursive_braces(t, idx, dim + 1);
        }

        if (i + 1 != dim_size)
            std::cout << ", ";
    }
    std::cout << "}";
}

// main print function — wraps recursive version in []
inline void print_(const Tensor& t) {
    // scalar (0D)
    if (t.impl->ndim == 0) {
        double v = read_scalar_at(t.impl->storage->data.get(), 0, t.impl->dtype);
        if (t.impl->dtype == DType::Int32)
            std::cout << static_cast<long long>(std::lrint(v)) << "\n";
        else
            std::cout << v << "\n";
        return;
    }

    // empty tensor
    for (size_t i = 0; i < t.impl->ndim; ++i)
        if (t.shape()[i] == 0) { std::cout << "[]\n"; return; }

    // general case
    std::vector<size_t> idx(t.impl->ndim, 0);
    std::cout << "[ ";
    print_recursive_braces(t, idx, 0);
    std::cout << " ]\n";
}
// ---------- Helper: compute linear index in original tensor for a given multi-index ----------
// 'orig' has possibly fewer dims than padded_idx.size(); left-pad with 1s.
inline size_t linear_index_from_padded(const Tensor& orig, const std::vector<size_t>& padded_idx) {
    size_t offset = 0;
    size_t pad = padded_idx.size() - orig.impl->ndim;
    for (size_t i = 0; i < orig.impl->ndim; ++i) {
        size_t idx = padded_idx[pad + i];
        size_t dim = orig.shape()[i];
        size_t use_idx = (dim == 1 ? 0 : idx);
        offset += use_idx * orig.strides()[i];
    }
    return offset;
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
