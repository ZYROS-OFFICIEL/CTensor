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

// helper: size of dtype
static size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return sizeof(float);
        case DType::Int32:   return sizeof(int);
        case DType::Double64:return sizeof(double);
    }
    return sizeof(float);
}

// helpers to read/write scalars at flat index (convert via double)
static double read_scalar_at(const void* data, size_t idx, DType dt) {
    switch(dt) {
        case DType::Float32: {
            const float* p = static_cast<const float*>(data);
            return static_cast<double>(p[idx]);
        }
        case DType::Int32: {
            const int* p = static_cast<const int*>(data);
            return static_cast<double>(p[idx]);
        }
        case DType::Double64: {
            const double* p = static_cast<const double*>(data);
            return static_cast<double>(p[idx]);
        }
    }
    return 0.0;
}

static void write_scalar_at(void* data, size_t idx, DType dt, double val) {
    switch(dt) {
        case DType::Float32: {
            float* p = static_cast<float*>(data);
            p[idx] = static_cast<float>(val);
            return;
        }
        case DType::Int32: {
            int* p = static_cast<int*>(data);
            p[idx] = static_cast<int>(std::lrint(val)); // round to nearest int
            return;
        }
        case DType::Double64: {
            double* p = static_cast<double*>(data);
            p[idx] = static_cast<double>(val);
            return;
        }
    }
}

// Tensor struct with runtime dtype
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
        shape = (size_t*) malloc(ndim * sizeof(size_t));
        strides = (size_t*) malloc(ndim * sizeof(size_t));
        for (size_t i = 0; i < ndim; ++i) shape[i] = shape_[i];

        if (ndim > 0) {
            strides[ndim - 1] = 1;
            for (int i = (int)ndim - 2; i >= 0; --i)
                strides[i] = strides[i + 1] * shape[i + 1];
        }

        size_t numel = numel_();
        size_t tsize = dtype_size(dtype);
        data = malloc(numel * tsize);
        if (!data && numel) throw std::bad_alloc();
        memset(data, 0, numel * tsize);

        if (requires_grad) {
            grad = malloc(numel * tsize);
            if (!grad && numel) throw std::bad_alloc();
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
        shape = (size_t*) malloc(ndim * sizeof(size_t));
        strides = (size_t*) malloc(ndim * sizeof(size_t));
        memcpy(shape, other.shape, ndim * sizeof(size_t));
        memcpy(strides, other.strides, ndim * sizeof(size_t));

        size_t n = numel_();
        size_t tsize = dtype_size(dtype);
        data = malloc(n * tsize);
        if (!data && n) throw std::bad_alloc();
        memcpy(data, other.data, n * tsize);

        if (requires_grad && other.grad) {
            grad = malloc(n * tsize);
            if (!grad && n) throw std::bad_alloc();
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

    size_t numel_() const {
        size_t n = 1;
        for (size_t i = 0; i < ndim; ++i) n *= shape[i];
        return n;
    }
    // Returns shape as std::vector
    std::vector<size_t> shape_() const {
        return std::vector<size_t>(this->shape, this->shape + ndim);
    }
    // Print shape for debugging
    void print_shape() const {
        std::cout << "(";
        for (size_t i = 0; i < ndim; ++i) {
            std::cout << shape[i];
            if (i < ndim - 1) std::cout << ", ";
        }
        std::cout << ")\n";
    }

    // --------- ND access proxy ----------
    struct Proxy {
        void* data;
        size_t* shape;
        size_t* strides;
        size_t offset;   // current flat offset
        size_t depth;    // how many indices provided
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

        // read as double
        operator double() const {
            if (depth != ndim) throw std::out_of_range("Not enough indices (or too many)");
            return read_scalar_at(data, offset, dtype);
        }

        // assign from double (works with int/float/double dtypes)
        Proxy& operator=(double val) {
            if (depth != ndim) throw std::out_of_range("Not at leaf index, Dimensions mismatch");
            write_scalar_at(data, offset, dtype, val);
            return *const_cast<Proxy*>(this);
        }
    };

    // Tensor operator[] begins chain: sets first index (depth=1)
    Proxy operator[](size_t i) {
        if (ndim == 0) throw std::out_of_range("Tensor has no dimensions");
        if (i >= shape[0]) throw std::out_of_range("Index out of bounds");
        size_t off = i * strides[0];
        return Proxy(data, shape, strides, off, 1, ndim, dtype);
    }

    // const index operator (returns a Proxy that can be read)
    Proxy operator[](size_t i) const {
        if (ndim == 0) throw std::out_of_range("Tensor has no dimensions");
        if (i >= shape[0]) throw std::out_of_range("Index out of bounds");
        size_t off = i * strides[0];
        return Proxy(data, shape, strides, off, 1, ndim, dtype);
    }

    // Factory functions
    static Tensor ones(const std::vector<size_t>& shape_, DType dtype_ = DType::Float32, bool requires_grad_ = false) {
        Tensor t(shape_, dtype_, requires_grad_);
        size_t n = t.numel_();
        for (size_t i = 0; i < n; ++i) write_scalar_at(t.data, i, t.dtype, 1.0);
        return t;
    }
    static Tensor zeros(const std::vector<size_t>& shape_, DType dtype_ = DType::Float32, bool requires_grad_ = false) {
        Tensor t(shape_, dtype_, requires_grad_);
        // data already zeroed by memset in ctor, but keep explicit for clarity
        size_t n = t.numel_();
        for (size_t i = 0; i < n; ++i) write_scalar_at(t.data, i, t.dtype, 0.0);
        return t;
    }
    static Tensor full(const std::vector<size_t>& shape_, double value, DType dtype_ = DType::Float32, bool requires_grad_ = false) {
        Tensor t(shape_, dtype_, requires_grad_);
        size_t n = t.numel_();
        for (size_t i = 0; i < n; ++i) write_scalar_at(t.data, i, t.dtype, value);
        return t;
    }
    static Tensor rand(const std::vector<size_t>& shape_, DType dtype_ = DType::Float32, bool requires_grad_ = false) {
        Tensor t(shape_, dtype_, requires_grad_);
        size_t n = t.numel_();
        std::srand((unsigned int)std::time(nullptr));
        for (size_t i = 0; i < n; ++i)
            write_scalar_at(t.data, i, t.dtype, static_cast<double>(std::rand()) / RAND_MAX);
        return t;
    }

    // convert to another dtype and return new Tensor
    Tensor astype(DType new_dtype) const {
        Tensor out(shape_(), new_dtype, requires_grad);
        size_t n = numel_();
        for (size_t i = 0; i < n; ++i) {
            double v = read_scalar_at(data, i, dtype);
            write_scalar_at(out.data, i, out.dtype, v);
        }
        return out;
    }

    // convert in-place (destructive)
    void to_(DType new_dtype) {
        if (new_dtype == dtype) return;
        size_t n = numel_();
        size_t new_tsize = dtype_size(new_dtype);
        void* new_data = malloc(n * new_tsize);
        if (!new_data && n) throw std::bad_alloc();
        for (size_t i = 0; i < n; ++i) {
            double v = read_scalar_at(data, i, dtype);
            write_scalar_at(new_data, i, new_dtype, v);
        }
        free(data);
        data = new_data;

        if (grad) {
            void* new_grad = malloc(n * new_tsize);
            if (!new_grad && n) throw std::bad_alloc();
            // optionally convert grad values (here we copy/convert same way)
            for (size_t i = 0; i < n; ++i) {
                double gv = read_scalar_at(grad, i, dtype);
                write_scalar_at(new_grad, i, new_dtype, gv);
            }
            free(grad);
            grad = new_grad;
        }
        dtype = new_dtype;
    }
};

// simple flat print (for debugging) — prints as doubles
static void print_t(const Tensor& t) {
    size_t n = t.numel_();
    std::cout << "[";
    for (size_t i = 0; i < n; i++) {
        double v = read_scalar_at(t.data, i, t.dtype);
        std::cout << v;
        if (i != n - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

// Helper: compute linear index in original tensor for a given multi-index (vec idx)
// 'orig' has possibly fewer dims than idx.size(); left-pad with 1s.
static size_t linear_index_from_padded(const Tensor& orig, const std::vector<size_t>& padded_idx) {
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
static Tensor pad_to_ndim(const Tensor& t, size_t target_ndim) {
    if (t.ndim == target_ndim) {
        return Tensor(t); // copy
    }
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
