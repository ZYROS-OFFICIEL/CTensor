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
    void* data;
    void* grad;
    size_t ndim;
    size_t* shape;
    size_t* strides;
    bool requires_grad;
    DType dtype;

    // Primary constructor
    Tensor(const std::vector<size_t>& shape_, DType dtype_ = DType::Float32, bool requires_grad_ = false)
        : data(nullptr), grad(nullptr), ndim(shape_.size()), shape(nullptr),
          strides(nullptr), requires_grad(requires_grad_), dtype(dtype_)
    {
        // allocate shape & strides
        shape = static_cast<size_t*>(malloc(ndim * sizeof(size_t)));
        strides = static_cast<size_t*>(malloc(ndim * sizeof(size_t)));
        if ((ndim && !shape) || (ndim && !strides)) {
            free(shape); free(strides);
            throw std::bad_alloc();
        }
        for (size_t i = 0; i < ndim; ++i) shape[i] = shape_[i];

        if (ndim > 0) {
            strides[ndim - 1] = 1;
            for (int i = (int)ndim - 2; i >= 0; --i)
                strides[i] = strides[i + 1] * shape[i + 1];
        }

        size_t numel = numel_();
        size_t tsize = dtype_size(dtype);
        data = malloc(numel * tsize);
        if (!data && numel) { free(shape); free(strides); throw std::bad_alloc(); }
        memset(data, 0, numel * tsize);

        if (requires_grad) {
            grad = malloc(numel * tsize);
            if (!grad && numel) { free(data); free(shape); free(strides); throw std::bad_alloc(); }
            memset(grad, 0, numel * tsize);
        } else {
            grad = nullptr;
        }
    }

    // Deep copy constructor
    Tensor(const Tensor& other)
        : data(nullptr), grad(nullptr), ndim(other.ndim), shape(nullptr),
          strides(nullptr), requires_grad(other.requires_grad), dtype(other.dtype)
    {
        shape = static_cast<size_t*>(malloc(ndim * sizeof(size_t)));
        strides = static_cast<size_t*>(malloc(ndim * sizeof(size_t)));
        if ((ndim && !shape) || (ndim && !strides)) { free(shape); free(strides); throw std::bad_alloc(); }
        memcpy(shape, other.shape, ndim * sizeof(size_t));
        memcpy(strides, other.strides, ndim * sizeof(size_t));

        size_t n = numel_();
        size_t tsize = dtype_size(dtype);
        data = malloc(n * tsize);
        if (!data && n) { free(shape); free(strides); throw std::bad_alloc(); }
        memcpy(data, other.data, n * tsize);

        if (requires_grad && other.grad) {
            grad = malloc(n * tsize);
            if (!grad && n) { free(data); free(shape); free(strides); throw std::bad_alloc(); }
            memcpy(grad, other.grad, n * tsize);
        } else {
            grad = nullptr;
        }
    }

    // Move constructor
    Tensor(Tensor&& other) noexcept
        : data(other.data), grad(other.grad), ndim(other.ndim),
          shape(other.shape), strides(other.strides), requires_grad(other.requires_grad),
          dtype(other.dtype)
    {
        other.data = nullptr;
        other.grad = nullptr;
        other.shape = nullptr;
        other.strides = nullptr;
        other.ndim = 0;
        other.requires_grad = false;
    }

    // Delete copy assignment for now
    Tensor& operator=(const Tensor&) = delete;

    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (data) free(data);
            if (grad) free(grad);
            if (shape) free(shape);
            if (strides) free(strides);

            data = other.data;
            grad = other.grad;
            shape = other.shape;
            strides = other.strides;
            ndim = other.ndim;
            requires_grad = other.requires_grad;
            dtype = other.dtype;

            other.data = nullptr;
            other.grad = nullptr;
            other.shape = nullptr;
            other.strides = nullptr;
            other.ndim = 0;
            other.requires_grad = false;
        }
        return *this;
    }

    ~Tensor() {
        if (data) free(data);
        if (grad) free(grad);
        if (shape) free(shape);
        if (strides) free(strides);
    }

    // basic utilities
    size_t numel_() const {
        size_t n = 1;
        for (size_t i = 0; i < ndim; ++i) n *= shape[i];
        return n;
    }
    std::vector<size_t> shape_() const {
        return std::vector<size_t>(this->shape, this->shape + ndim);
    }
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
        size_t* shape;
        size_t* strides;
        size_t offset;
        size_t depth;
        size_t ndim;
        DType dtype;
        ConstProxy(const void* d, size_t* s, size_t* st, size_t off, size_t dp, size_t n, DType dt)
            : data(d), shape(s), strides(st), offset(off), depth(dp), ndim(n), dtype(dt) {}
        ConstProxy operator[](size_t i) const {
            if (depth >= ndim) throw std::out_of_range("Too many indices");
            if (i >= shape[depth]) throw std::out_of_range("Index out of bounds");
            size_t new_offset = offset + i * strides[depth];
            return ConstProxy(data, shape, strides, new_offset, depth + 1, ndim, dtype);
        }
        operator double() const {
            if (depth != ndim) throw std::out_of_range("Not enough indices");
            return read_scalar_at(data, offset, dtype);
        }
    };

    struct Proxy {
        void* data;
        size_t* shape;
        size_t* strides;
        size_t offset;
        size_t depth;
        size_t ndim;
        DType dtype;
        Proxy(void* d, size_t* s, size_t* st, size_t off, size_t dp, size_t n, DType dt)
            : data(d), shape(s), strides(st), offset(off), depth(dp), ndim(n), dtype(dt) {}
        Proxy operator[](size_t i) const {
            if (depth >= ndim) throw std::out_of_range("Too many indices");
            if (i >= shape[depth]) throw std::out_of_range("Index out of bounds");
            size_t new_offset = offset + i * strides[depth];
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
        return Proxy(data, shape, strides, off, 1, ndim, dtype);
    }
    ConstProxy operator[](size_t i) const {
        if (ndim == 0) throw std::out_of_range("Tensor has no dimensions");
        if (i >= shape[0]) throw std::out_of_range("Index out of bounds");
        size_t off = i * strides[0];
        return ConstProxy(data, shape, strides, off, 1, ndim, dtype);
    }

    // ---------- factories ----------
    static Tensor ones(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false) {
        Tensor t(shape_, dt, requires_grad_);
        size_t n = t.numel_();
        for (size_t i = 0; i < n; ++i) write_scalar_at(t.data, i, t.dtype, 1.0);
        return t;
    }
    static Tensor zeros(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false) {
        Tensor t(shape_, dt, requires_grad_);
        // already zeroed by memset but fill explicitly for clarity
        size_t n = t.numel_();
        for (size_t i = 0; i < n; ++i) write_scalar_at(t.data, i, t.dtype, 0.0);
        return t;
    }
    static Tensor full(const std::vector<size_t>& shape_, double value, DType dt = DType::Float32, bool requires_grad_ = false) {
        Tensor t(shape_, dt, requires_grad_);
        size_t n = t.numel_();
        for (size_t i = 0; i < n; ++i) write_scalar_at(t.data, i, t.dtype, value);
        return t;
    }
    static Tensor rand(const std::vector<size_t>& shape_, DType dt = DType::Float32, bool requires_grad_ = false) {
        Tensor t(shape_, dt, requires_grad_);
        size_t n = t.numel_();
        // seed only once per program would be better; simple here:
        std::srand((unsigned int)std::time(nullptr));
        for (size_t i = 0; i < n; ++i)
            write_scalar_at(t.data, i, t.dtype, static_cast<double>(std::rand()) / RAND_MAX);
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
        if (new_dtype == dtype) return Tensor(*this); // copy
        Tensor out(shape_(), new_dtype, requires_grad);
        size_t n = numel_();

        // straightforward convert elementwise
        for (size_t i = 0; i < n; ++i) {
            double v = read_scalar_at(data, i, dtype);
            write_scalar_at(out.data, i, out.dtype, v);
        }
        // grad not copied by default; if you want to copy grad, convert similarly:
        if (requires_grad && grad) {
            out.grad = malloc(n * dtype_size(out.dtype));
            if (!out.grad && n) throw std::bad_alloc();
            for (size_t i = 0; i < n; ++i) {
                double gv = read_scalar_at(grad, i, dtype);
                write_scalar_at(out.grad, i, out.dtype, gv);
            }
        }
        return out;
    }

    // ---------- conversion: in-place ----------
    void to_(DType new_dtype) {
        if (new_dtype == dtype) return;
        size_t n = numel_();
        size_t new_tsize = dtype_size(new_dtype);

        // allocate new buffer
        void* new_data = malloc(n * new_tsize);
        if (!new_data && n) throw std::bad_alloc();

        // convert
        for (size_t i = 0; i < n; ++i) {
            double v = read_scalar_at(data, i, dtype);
            write_scalar_at(new_data, i, new_dtype, v);
        }
        free(data);
        data = new_data;

        if (grad) {
            void* new_grad = malloc(n * new_tsize);
            if (!new_grad && n) throw std::bad_alloc();
            for (size_t i = 0; i < n; ++i) {
                double gv = read_scalar_at(grad, i, dtype);
                write_scalar_at(new_grad, i, new_dtype, gv);
            }
            free(grad);
            grad = new_grad;
        }
        dtype = new_dtype;
    }
    Tensor t() const {
        if (ndim < 2)
            throw std::invalid_argument("t: tensor must have at least 2 dimensions");

        std::vector<size_t> new_shape(shape, shape + ndim);
        std::swap(new_shape[ndim - 2], new_shape[ndim - 1]);

        Tensor out(new_shape, dtype, requires_grad);

        size_t batch_ndim = ndim - 2;
        size_t m = shape[ndim - 2];
        size_t n = shape[ndim - 1];

        size_t batch_size = 1;
        for (size_t i = 0; i < batch_ndim; ++i)
            batch_size *= shape[i];

        for (size_t b = 0; b < batch_size; ++b) {
            size_t batch_offset = 0;
            if (batch_ndim > 0) {
                size_t rem = b;
                for (int d = (int)batch_ndim - 1; d >= 0; --d) {
                    size_t idx = rem % shape[d];
                    rem /= shape[d];
                    batch_offset += idx * strides[d];
                }
            }

            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    size_t src_idx = batch_offset + i * strides[ndim - 2] + j * strides[ndim - 1];
                    size_t dst_idx = batch_offset + j * out.strides[ndim - 2] + i * out.strides[ndim - 1];
                    double val = read_scalar_at(data, src_idx, dtype);
                    write_scalar_at(out.data, dst_idx, out.dtype, val);
                }
            }
        }

        return out;
    }
    Tensor& t_() {
        if (ndim < 2)
            throw std::invalid_argument("t_: tensor must have at least 2 dimensions");

        std::swap(shape[ndim - 2], shape[ndim - 1]);
        std::swap(strides[ndim - 2], strides[ndim - 1]);
        return *this;
    }
    Tensor select(size_t dim, size_t index) const {
        if (dim >= ndim)
            throw std::invalid_argument("select: dim out of range");
        if (index >= shape[dim])
            throw std::invalid_argument("select: index out of range");

        std::vector<size_t> new_shape(shape.begin(), shape.end());
        new_shape.erase(new_shape.begin() + dim);

        Tensor out(new_shape, dtype, false);

        size_t outer = 1, inner = 1;
        for (size_t i = 0; i < dim; ++i)
            outer *= shape[i];
        for (size_t i = dim + 1; i < ndim; ++i)
            inner *= shape[i];

        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                size_t src_idx = o * shape[dim] * inner + index * inner + i;
                size_t dst_idx = o * inner + i;
                double val = read_scalar_at(data, src_idx, dtype);
                write_scalar_at(out.data, dst_idx, dtype, val);
            }
        }

        return out;
    }
    Tensor slice(size_t dim, size_t start, size_t end) const {
        if (dim >= ndim)
            throw std::invalid_argument("slice: dim out of range");
        if (end > shape[dim] || start >= end)
            throw std::invalid_argument("slice: invalid start/end range");

        std::vector<size_t> new_shape(shape.begin(), shape.end());
        new_shape[dim] = end - start;

        Tensor out(new_shape, dtype, false);

        size_t outer = 1, inner = 1;
        for (size_t i = 0; i < dim; ++i)
            outer *= shape[i];
        for (size_t i = dim + 1; i < ndim; ++i)
            inner *= shape[i];

        for (size_t o = 0; o < outer; ++o) {
            for (size_t s = 0; s < end - start; ++s) {
                for (size_t i = 0; i < inner; ++i) {
                    size_t src_idx = o * shape[dim] * inner + (start + s) * inner + i;
                    size_t dst_idx = o * (end - start) * inner + s * inner + i;
                    double val = read_scalar_at(data, src_idx, dtype);
                    write_scalar_at(out.data, dst_idx, dtype, val);
                }
            }
        }

        return out;
    }
    static Tensor arange(double start, double end, double step = 1.0, DType dtype = DType::Float32) {
        if (step == 0) throw std::invalid_argument("arange: step cannot be zero");
        size_t n = static_cast<size_t>((end - start) / step);
        Tensor t({n}, dtype, false);
        for (size_t i = 0; i < n; ++i)
            write_scalar_at(t.data, i, dtype, start + i * step);
        return t;
    }

};

// simple flat print (for debugging) — prints as doubles
inline void print_t(const Tensor& t) {
    size_t n = t.numel_();
    std::cout << "[";
    for (size_t i = 0; i < n; i++) {
        double v = read_scalar_at(t.data, i, t.dtype);
        std::cout << v;
        if (i != n - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}
static void print_recursive_braces(const Tensor& t, std::vector<size_t>& idx, size_t dim) {
    std::cout << "{";
    size_t dim_size = t.shape[dim];
    for (size_t i = 0; i < dim_size; ++i) {
        idx[dim] = i;
        if (dim + 1 == t.ndim) {
            // compute flat offset
            size_t offset = 0;
            for (size_t k = 0; k < t.ndim; ++k) offset += idx[k] * t.strides[k];
            double v = read_scalar_at(t.data, offset, t.dtype);
            // print nicely depending on dtype
            if (t.dtype == DType::Int32) {
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

inline void print_(const Tensor& t) {
    // handle zero-dim (scalar) specially
    if (t.ndim == 0) {
        double v = read_scalar_at(t.data, 0, t.dtype);
        if (t.dtype == DType::Int32) {
            std::cout << static_cast<long long>(std::lrint(v)) << "\n";
        } else {
            std::cout << v << "\n";
        }
        return;
    }
    // empty dims (any dimension is zero) -> print empty braces
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
        double v = read_scalar_at(t.data, src_idx, t.dtype);
        write_scalar_at(result.data, flat, result.dtype, v);
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



