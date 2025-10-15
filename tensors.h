// tensors.h
#pragma once
#include <iostream>
#include <cstddef>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdlib>      // malloc/free
#include <ctime>        // time for rand seed
#include <stdexcept>    // exceptions
#include <cassert>
#include <memory>

enum class DType { Float32, Int32, Double64 };

// ---------- helpers ----------
inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return sizeof(float);
        case DType::Int32:   return sizeof(int);
        case DType::Double64:return sizeof(double);
    }
    return sizeof(float);
}

inline const char* dtype_to_cstr(DType d) {
    switch (d) {
        case DType::Float32:  return "Float32";
        case DType::Int32:    return "Int32";
        case DType::Double64: return "Double64";
    }
    return "Unknown";
}

inline std::ostream& operator<<(std::ostream& os, DType d) {
    os << dtype_to_cstr(d);
    return os;
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

// ---------- Tensor ----------
struct Tensor {
    // storage pointer (type-erased). We allocate with malloc and free, but manage lifetime via shared_ptr.
    std::shared_ptr<void> data;    // points to beginning of allocation (or alias to offset)
    std::shared_ptr<void> grad;    // optional gradient buffer (same semantics)

    // shape & strides (contiguous layout by default)
    size_t ndim;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    bool requires_grad;
    DType dtype;

    // ---------- constructors ----------
    Tensor()
        : data(nullptr), grad(nullptr), ndim(0), shape(), strides(),
          requires_grad(false), dtype(DType::Float32) {}

    Tensor(const std::vector<size_t>& shape_, DType dtype_ = DType::Float32, bool requires_grad_ = false)
        : data(nullptr), grad(nullptr), ndim(shape_.size()), shape(shape_), strides(shape_.size(), 0),
          requires_grad(requires_grad_), dtype(dtype_)
    {
        // compute contiguous strides
        if (ndim > 0) {
            strides[ndim - 1] = 1;
            for (int i = (int)ndim - 2; i >= 0; --i)
                strides[i] = strides[i + 1] * shape[i + 1];
        }

        size_t numel = numel_();
        size_t bytes = numel * dtype_size(dtype);

        if (numel) {
            void* raw = std::malloc(bytes);
            if (!raw) throw std::bad_alloc();
            std::memset(raw, 0, bytes);
            // create shared_ptr owning raw pointer using free as deleter
            data = std::shared_ptr<void>(raw, [](void* p){ std::free(p); });
            if (requires_grad) {
                void* rawg = std::malloc(bytes);
                if (!rawg) throw std::bad_alloc();
                std::memset(rawg, 0, bytes);
                grad = std::shared_ptr<void>(rawg, [](void* p){ std::free(p); });
            } else {
                grad.reset();
            }
        } else {
            data.reset();
            grad.reset();
        }
    }

    // Default copy (shallow copy of storage: views share memory)
    Tensor(const Tensor& other) = default;
    Tensor& operator=(const Tensor& other) = default;

    // Move (default is fine)
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    ~Tensor() = default; // shared_ptr handles freeing

    // ---------- basic utilities ----------
    size_t numel_() const {
        if (ndim == 0) return 0;
        size_t n = 1;
        for (size_t i = 0; i < ndim; ++i) n *= shape[i];
        return n;
    }
    std::vector<size_t> shape_() const { return shape; }
    void print_shape() const {
        std::cout << "(";
        for (size_t i = 0; i < ndim; ++i) {
            std::cout << shape[i];
            if (i < ndim - 1) std::cout << ", ";
        }
        std::cout << ")\n";
    }

    // ---------- Proxy (writeable) and ConstProxy (read-only) ----------
    struct ConstProxy {
        const void* data;
        const std::vector<size_t>* shape;
        const std::vector<size_t>* strides;
        size_t offset; // element offset (in elements)
        size_t depth;
        size_t ndim;
        DType dtype;
        ConstProxy(const void* d, const std::vector<size_t>* s, const std::vector<size_t>* st,
                   size_t off, size_t dp, size_t n, DType dt)
            : data(d), shape(s), strides(st), offset(off), depth(dp), ndim(n), dtype(dt) {}
        ConstProxy operator[](size_t i) const {
            if (depth >= ndim) throw std::out_of_range("Too many indices");
            if (i >= (*shape)[depth]) throw std::out_of_range("Index out of bounds");
            size_t new_offset = offset + i * (*strides)[depth];
            return ConstProxy(data, shape, strides, new_offset, depth + 1, ndim, dtype);
        }
        operator double() const {
            if (depth != ndim) throw std::out_of_range("Not enough indices");
            return read_scalar_at(data, offset, dtype);
        }
    };

    struct Proxy {
        void* data;
        const std::vector<size_t>* shape;
        const std::vector<size_t>* strides;
        size_t offset; // element offset
        size_t depth;
        size_t ndim;
        DType dtype;
        Proxy(void* d, const std::vector<size_t>* s, const std::vector<size_t>* st,
              size_t off, size_t dp, size_t n, DType dt)
            : data(d), shape(s), strides(st), offset(off), depth(dp), ndim(n), dtype(dt) {}
        Proxy operator[](size_t i) const {
            if (depth >= ndim) throw std::out_of_range("Too many indices");
            if (i >= (*shape)[depth]) throw std::out_of_range("Index out of bounds");
            size_t new_offset = offset + i * (*strides)[depth];
            return Proxy(data, shape, strides, new_offset, depth + 1, ndim, dtype);
        }
        operator double() const {
            if (depth != ndim) throw std::out_of_range("Not enough indices");
            return read_scalar_at(data, offset, dtype);
        }
        Proxy& operator=(double val) {
            if (depth != ndim) throw std::out_of_range("Not at leaf index");
            write_scalar_at(data, offset, dtype, val);
            return *this;
        }
    };

    Proxy operator[](size_t i) {
        if (ndim == 0) throw std::out_of_range("Tensor has no dimensions");
        if (i >= shape[0]) throw std::out_of_range("Index out of bounds");
        size_t off = i * strides[0];
        return Proxy(data ? data.get() : nullptr, &shape, &strides, off, 1, ndim, dtype);
    }
    ConstProxy operator[](size_t i) const {
        if (ndim == 0) throw std::out_of_range("Tensor has no dimensions");
        if (i >= shape[0]) throw std::out_of_range("Index out of bounds");
        size_t off = i * strides[0];
        return ConstProxy(data ? data.get() : nullptr, &shape, &strides, off, 1, ndim, dtype);
    }

    // ---------- factories ----------
    static Tensor ones(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false) {
        Tensor t(shape_, dt, requires_grad_);
        size_t n = t.numel_();
        for (size_t i = 0; i < n; ++i) write_scalar_at(t.data.get(), i, t.dtype, 1.0);
        return t;
    }
    static Tensor zeros(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false) {
        Tensor t(shape_, dt, requires_grad_);
        // already zeroed by memset in constructor
        return t;
    }
    static Tensor full(const std::vector<size_t>& shape_, double value, DType dt = DType::Float32, bool requires_grad_ = false) {
        Tensor t(shape_, dt, requires_grad_);
        size_t n = t.numel_();
        for (size_t i = 0; i < n; ++i) write_scalar_at(t.data.get(), i, t.dtype, value);
        return t;
    }
    static Tensor rand(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false) {
        Tensor t(shape_, dt, requires_grad_);
        size_t n = t.numel_();
        std::srand((unsigned int)std::time(nullptr));
        for (size_t i = 0; i < n; ++i)
            write_scalar_at(t.data.get(), i, t.dtype, static_cast<double>(std::rand()) / RAND_MAX);
        return t;
    }
    static Tensor empty(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false) {
        return Tensor(shape_, dt, requires_grad_);
    }

    // ---------- dtype helpers ----------
    DType _dtype() const noexcept { return dtype; }
    const char* dtype_name() const noexcept { return dtype_to_cstr(dtype); }
    size_t dtype_bytes() const noexcept { return dtype_size(dtype); }

    // ---------- conversion: return new tensor ----------
    Tensor astype(DType new_dtype) const {
        if (new_dtype == dtype) return *this; // shallow copy
        Tensor out(shape, new_dtype, requires_grad);
        size_t n = numel_();

        for (size_t i = 0; i < n; ++i) {
            double v = read_scalar_at(data.get(), i, dtype);
            write_scalar_at(out.data.get(), i, out.dtype, v);
        }

        if (requires_grad && grad) {
            out.grad = std::shared_ptr<void>(std::malloc(n * dtype_size(new_dtype)), [](void* p){ std::free(p); });
            if (!out.grad && n) throw std::bad_alloc();
            for (size_t i = 0; i < n; ++i) {
                double gv = read_scalar_at(grad.get(), i, dtype);
                write_scalar_at(out.grad.get(), i, new_dtype, gv);
            }
        }
        return out;
    }

    // ---------- conversion: in-place ----------
    void to_(DType new_dtype) {
        if (new_dtype == dtype) return;
        size_t n = numel_();
        size_t new_tsize = dtype_size(new_dtype);
        size_t bytes = n * new_tsize;
        void* raw = (n ? std::malloc(bytes) : nullptr);
        if (!raw && n) throw std::bad_alloc();

        for (size_t i = 0; i < n; ++i) {
            double v = read_scalar_at(data.get(), i, dtype);
            write_scalar_at(raw, i, new_dtype, v);
        }

        // replace data shared_ptr with new allocation
        data = std::shared_ptr<void>(raw, [](void* p){ std::free(p); });

        if (grad) {
            void* rawg = (n ? std::malloc(bytes) : nullptr);
            if (!rawg && n) throw std::bad_alloc();
            for (size_t i = 0; i < n; ++i) {
                double gv = read_scalar_at(grad.get(), i, dtype);
                write_scalar_at(rawg, i, new_dtype, gv);
            }
            grad = std::shared_ptr<void>(rawg, [](void* p){ std::free(p); });
        }

        dtype = new_dtype;
    }

    // ---------- transpose & permute ----------
    // return a new tensor that is a shallow view with last two dims swapped
    Tensor t() const {
        if (ndim < 2) throw std::invalid_argument("t: tensor must have at least 2 dimensions");
        Tensor out = *this; // shallow copy

        // compute element offset? none needed because stride manipulation suffices for view
        std::vector<size_t> new_shape = shape;
        std::swap(new_shape[ndim - 2], new_shape[ndim - 1]);
        std::vector<size_t> new_strides = strides;
        std::swap(new_strides[ndim - 2], new_strides[ndim - 1]);

        out.shape = std::move(new_shape);
        out.strides = std::move(new_strides);
        out.ndim = out.shape.size();
        return out;
    }

    // in-place transpose of last two dims by swapping shape & strides (view semantics)
    Tensor& t_() {
        if (ndim < 2) throw std::invalid_argument("t_: tensor must have at least 2 dimensions");
        std::swap(shape[ndim - 2], shape[ndim - 1]);
        std::swap(strides[ndim - 2], strides[ndim - 1]);
        return *this;
    }

    Tensor permute(const std::vector<size_t>& dims) const {
        if (dims.size() != ndim) throw std::invalid_argument("permute: dims size must match shape size.");
        std::vector<bool> seen(ndim, false);
        for (auto d : dims) {
            if (d >= ndim || seen[d]) throw std::invalid_argument("permute: invalid or duplicate dim.");
            seen[d] = true;
        }

        Tensor out = *this; // shallow copy
        std::vector<size_t> new_shape(ndim), new_strides(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            new_shape[i] = shape[dims[i]];
            new_strides[i] = strides[dims[i]];
        }
        out.shape = std::move(new_shape);
        out.strides = std::move(new_strides);
        out.ndim = out.shape.size();
        return out;
    }

    // ---------- view / reshape / indexing ----------
    // create 1D range [start, start+step*(n-1)] -> shape {n}
    static Tensor arange(double start, double end, double step = 1.0, DType dtype = DType::Float32) {
        if (step == 0.0) throw std::invalid_argument("arange: step must be non-zero");
        double span = end - start;
        if ((span > 0 && step < 0) || (span < 0 && step > 0)) return Tensor(); // empty
        size_t n = 0;
        if ( (step > 0 && start < end) || (step < 0 && start > end) ) {
            n = static_cast<size_t>(std::ceil(std::abs(span) / std::abs(step)));
        }
        Tensor t({n}, dtype, false);
        for (size_t i = 0; i < n; ++i) {
            double v = start + i * step;
            write_scalar_at(t.data.get(), i, dtype, v);
        }
        return t;
    }

    // reshape (returns a view with new shape but assumes contiguous layout)
    Tensor reshape(const std::vector<size_t>& new_shape) const {
        size_t new_numel = 1;
        for (auto s : new_shape) new_numel *= s;
        if (new_numel != numel_()) throw std::invalid_argument("reshape: total number of elements must remain constant.");

        Tensor out = *this; // shallow copy
        out.shape = new_shape;
        out.ndim = out.shape.size();
        // recompute contiguous strides for new shape
        out.strides.assign(out.ndim, 0);
        if (out.ndim > 0) {
            out.strides[out.ndim - 1] = 1;
            for (int i = (int)out.ndim - 2; i >= 0; --i)
                out.strides[i] = out.strides[i + 1] * out.shape[i + 1];
        }
        return out;
    }

    // select (index along dimension) -> returns view with that dim removed and data pointer alias shifted
    Tensor select(size_t dim, size_t index) const {
        if (dim >= ndim) throw std::out_of_range("select: dim out of range");
        if (index >= shape[dim]) throw std::out_of_range("select: index out of range");

        // compute element offset in elements
        size_t elem_offset = 0;
        for (size_t d = 0; d < dim; ++d) elem_offset += 0 * strides[d]; // no-op kept for clarity
        elem_offset += index * strides[dim];

        // create aliasing shared_ptr that points to base + bytes offset, but shares ownership with original
        size_t byte_offset = elem_offset * dtype_size(dtype);
        std::shared_ptr<void> new_data;
        if (data) {
            new_data = std::shared_ptr<void>(data, static_cast<char*>(data.get()) + byte_offset);
        } else {
            new_data.reset();
        }

        Tensor out = *this; // shallow copy
        // remove the selected dimension
        std::vector<size_t> new_shape;
        std::vector<size_t> new_strides;
        new_shape.reserve(ndim ? ndim - 1 : 0);
        new_strides.reserve(ndim ? ndim - 1 : 0);
        for (size_t d = 0; d < ndim; ++d) {
            if (d == dim) continue;
            new_shape.push_back(shape[d]);
            new_strides.push_back(strides[d]);
        }
        // special case: if result is 0-dim, keep shape empty (ndim=0) or shape {1}? we'll allow ndim=0.
        out.shape = std::move(new_shape);
        out.strides = std::move(new_strides);
        out.ndim = out.shape.size();
        out.data = std::move(new_data);
        // grad: not adjusted (leave null or share original - for now keep no grad)
        out.grad.reset();
        return out;
    }

    // remove all dims == 1
    Tensor squeeze() const {
        std::vector<size_t> new_shape;
        std::vector<size_t> new_strides;
        for (size_t i = 0; i < ndim; ++i) {
            if (shape[i] != 1) {
                new_shape.push_back(shape[i]);
                new_strides.push_back(strides[i]);
            }
        }
        if (new_shape.empty()) {
            // scalar view: shape vector empty, ndim = 0
            Tensor out = *this;
            out.shape.clear();
            out.strides.clear();
            out.ndim = 0;
            return out;
        }
        Tensor out = *this;
        out.shape = std::move(new_shape);
        out.strides = std::move(new_strides);
        out.ndim = out.shape.size();
        return out;
    }

    // insert a dim of size 1 at position dim (0..ndim inclusive)
    Tensor unsqueeze(size_t dim) const {
        if (dim > ndim) throw std::out_of_range("unsqueeze: dim out of range");
        std::vector<size_t> new_shape;
        std::vector<size_t> new_strides;
        new_shape.reserve(ndim + 1);
        new_strides.reserve(ndim + 1);
        for (size_t i = 0; i < dim; ++i) {
            new_shape.push_back(shape[i]);
            new_strides.push_back(strides[i]);
        }
        // inserted dim: size 1. Its stride can be anything; to remain consistent with view semantics, set stride to (dim < ndim ? strides[dim] : 1)
        size_t ins_stride = (dim < ndim ? strides[dim] : (ndim ? strides.back() * shape.back() : 1));
        new_shape.push_back(1);
        new_strides.push_back(ins_stride);
        for (size_t i = dim; i < ndim; ++i) {
            new_shape.push_back(shape[i]);
            new_strides.push_back(strides[i]);
        }
        Tensor out = *this;
        out.shape = std::move(new_shape);
        out.strides = std::move(new_strides);
        out.ndim = out.shape.size();
        return out;
    }

    // flatten -> 1D view
    Tensor flatten() const {
        Tensor out = *this;
        size_t n = numel_();
        out.shape = { n };
        out.strides = { 1 };
        out.ndim = 1;
        return out;
    }

}; // end struct Tensor

// simple flat print (for debugging) — prints as doubles
inline void print_t(const Tensor& t) {
    size_t n = t.numel_();
    std::cout << "[";
    for (size_t i = 0; i < n; i++) {
        double v = read_scalar_at(t.data ? t.data.get() : nullptr, i, t.dtype);
        std::cout << v;
        if (i != n - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

static void print_recursive_braces(const Tensor& t, std::vector<size_t>& idx, size_t dim) {
    std::cout << "{";
    size_t dim_size = (dim < t.ndim ? t.shape[dim] : 0);
    for (size_t i = 0; i < dim_size; ++i) {
        idx[dim] = i;
        if (dim + 1 == t.ndim) {
            // compute flat offset
            size_t offset = 0;
            for (size_t k = 0; k < t.ndim; ++k) offset += idx[k] * t.strides[k];
            double v = read_scalar_at(t.data ? t.data.get() : nullptr, offset, t.dtype);
            if (t.dtype == DType::Int32) {
                long long iv = static_cast<long long>(std::lrint(v));
                std::cout << iv;
            } else {
                std::cout << v;
            }
        } else {
            print_recursive_braces(t, idx, dim + 1);
        }
        if (i + 1 != dim_size) std::cout << ", ";
    }
    std::cout << "}";
}

inline void print_(const Tensor& t) {
    if (t.ndim == 0) {
        // try print scalar — if data exists
        if (t.data && t.numel_() > 0) {
            double v = read_scalar_at(t.data.get(), 0, t.dtype);
            if (t.dtype == DType::Int32) std::cout << static_cast<long long>(std::lrint(v)) << "\n";
            else std::cout << v << "\n";
        } else {
            std::cout << "{}\n";
        }
        return;
    }

    for (size_t i = 0; i < t.ndim; ++i) if (t.shape[i] == 0) { std::cout << "{}\n"; return; }

    std::vector<size_t> idx(t.ndim, 0);
    print_recursive_braces(t, idx, 0);
    std::cout << "\n";
}

// Helper: compute linear index in original tensor for a given multi-index (vec idx)
// 'orig' has possibly fewer dims than idx.size(); left-pad with 1s.
inline size_t linear_index_from_padded(const Tensor& orig, const std::vector<size_t>& padded_idx) {
    size_t offset = 0;
    size_t pad = padded_idx.size() - orig.ndim;
    for (size_t i = 0; i < orig.ndim; ++i) {
        size_t idx = padded_idx[pad + i];
        size_t dim = orig.shape[i];
        size_t use_idx = (dim == 1 ? 0 : idx);
        offset += use_idx * orig.strides[i];
    }
    return offset;
}

// pad_to_ndim: returns a NEW tensor whose shape is padded to target ndim and values broadcasted
inline Tensor pad_to_ndim(const Tensor& t, size_t target_ndim) {
    if (t.ndim == target_ndim) return Tensor(t);
    if (t.ndim > target_ndim) throw std::runtime_error("target_ndim smaller than tensor ndim");

    std::vector<size_t> new_shape(target_ndim, 1);
    for (size_t i = 0; i < t.ndim; ++i)
        new_shape[target_ndim - t.ndim + i] = t.shape[i];

    Tensor result(new_shape, t.dtype);
    size_t N = result.numel_();
    std::vector<size_t> idx(target_ndim, 0);
    for (size_t flat = 0; flat < N; ++flat) {
        size_t rem = flat;
        for (int d = (int)target_ndim - 1; d >= 0; --d) {
            idx[d] = rem % new_shape[d];
            rem /= new_shape[d];
        }
        size_t src_idx = linear_index_from_padded(t, idx);
        double v = read_scalar_at(t.data ? t.data.get() : nullptr, src_idx, t.dtype);
        write_scalar_at(result.data.get(), flat, result.dtype, v);
    }
    return result;
}

// ---------- global helper: broadcast batch shape (place OUTSIDE any other function) ----------
static std::vector<size_t> broadcast_batch_shape_from_vectors(const std::vector<size_t>& a,
                                                              const std::vector<size_t>& b) {
    size_t na = a.size(), nb = b.size();
    size_t n = std::max(na, nb);
    std::vector<size_t> result(n, 1);
    for (size_t i = 0; i < n; ++i) {
        size_t da = (i < n - na) ? 1 : a[i - (n - na)];
        size_t db = (i < n - nb) ? 1 : b[i - (n - nb)];
        if (da != db && da != 1 && db != 1)
            throw std::invalid_argument("Incompatible batch shapes for broadcasting.");
        result[i] = std::max(da, db);
    }
    return result;
}

