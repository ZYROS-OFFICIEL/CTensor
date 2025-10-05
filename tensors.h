// tensors_verified.cpp
#include <iostream>
#include <cstddef>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdlib>      // malloc/free
#include <stdexcept>    // exceptions

struct Tensor {
    float* data;
    float* grad;
    size_t ndim;
    size_t* shape;
    size_t* strides;
    bool requires_grad;

    // Primary constructor
    Tensor(const std::vector<size_t>& shape_, bool requires_grad_ = false)
        : data(nullptr), grad(nullptr), ndim(shape_.size()), shape(nullptr),
          strides(nullptr), requires_grad(requires_grad_) {
        // allocate shape & strides
        shape = (size_t*) malloc(ndim * sizeof(size_t));
        strides = (size_t*) malloc(ndim * sizeof(size_t));
        for (size_t i = 0; i < ndim; ++i) shape[i] = shape_[i];

        if (ndim > 0) {
            strides[ndim - 1] = 1;
            for (int i = (int)ndim - 2; i >= 0; --i)
                strides[i] = strides[i + 1] * shape[i + 1];
        }

        size_t numel = 1;
        for (size_t i = 0; i < ndim; ++i) numel *= shape[i];
        data = (float*) malloc(numel * sizeof(float));
        if (!data && numel) throw std::bad_alloc();
        memset(data, 0, numel * sizeof(float));

        if (requires_grad) {
            grad = (float*) malloc(numel * sizeof(float));
            if (!grad && numel) throw std::bad_alloc();
            memset(grad, 0, numel * sizeof(float));
        } else {
            grad = nullptr;
        }
    }

    // Deep copy constructor (Rule of Three)
    Tensor(const Tensor& other)
        : data(nullptr), grad(nullptr), ndim(other.ndim), shape(nullptr),
          strides(nullptr), requires_grad(other.requires_grad) {
        shape = (size_t*) malloc(ndim * sizeof(size_t));
        strides = (size_t*) malloc(ndim * sizeof(size_t));
        memcpy(shape, other.shape, ndim * sizeof(size_t));
        memcpy(strides, other.strides, ndim * sizeof(size_t));

        size_t n = other.numel();
        data = (float*) malloc(n * sizeof(float));
        memcpy(data, other.data, n * sizeof(float));

        if (requires_grad && other.grad) {
            grad = (float*) malloc(n * sizeof(float));
            memcpy(grad, other.grad, n * sizeof(float));
        } else {
            grad = nullptr;
        }
    }

    // Move constructor
    Tensor(Tensor&& other) noexcept
        : data(other.data), grad(other.grad), ndim(other.ndim),
          shape(other.shape), strides(other.strides), requires_grad(other.requires_grad) {
        other.data = nullptr;
        other.grad = nullptr;
        other.shape = nullptr;
        other.strides = nullptr;
        other.ndim = 0;
        other.requires_grad = false;
    }

    // Delete copy assignment and move assignment for now (implement if needed)
    Tensor& operator=(const Tensor&) = delete;
    Tensor& operator=(Tensor&&) = delete;

    ~Tensor() {
        if (data) free(data);
        if (grad) free(grad);
        if (shape) free(shape);
        if (strides) free(strides);
    }

    size_t numel() const {
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
    // --------- ND access proxy (corrected) ---------
    struct Proxy {
        float* data;
        size_t* shape;
        size_t* strides;
        size_t offset;   // current flat offset
        size_t depth;    // how many indices have been provided
        size_t ndim;

        Proxy(float* d, size_t* s, size_t* st, size_t off, size_t dp, size_t n)
            : data(d), shape(s), strides(st), offset(off), depth(dp), ndim(n) {}

        // chained indexing: returns a new Proxy with one more index set
        Proxy operator[](size_t i) const {
            if (depth >= ndim) throw std::out_of_range("Too many indices");
            if (i >= shape[depth]) throw std::out_of_range("Index out of bounds");
            size_t new_offset = offset + i * strides[depth];
            return Proxy(data, shape, strides, new_offset, depth + 1, ndim);
        }

        // convert to reference (only valid at leaf)
        operator float&() {
            if (depth != ndim) throw std::out_of_range("Not enough indices (or too many)");
            return data[offset];
        }

        // assign at leaf
        Proxy& operator=(float val) {
            if (depth != ndim) throw std::out_of_range("Not at leaf index, Dimensions mismatch:look if you are indexing correctly");
            data[offset] = val;
            return *const_cast<Proxy*>(this);
        }
    };

    // Tensor operator[] begins chain: sets first index (depth=1)
    Proxy operator[](size_t i) {
        if (ndim == 0) throw std::out_of_range("Tensor has no dimensions");
        if (i >= shape[0]) throw std::out_of_range("Index out of bounds");
        size_t off = i * strides[0];
        return Proxy(data, shape, strides, off, 1, ndim);
    }
};

// simple flat print (for debugging)
void print_t(const Tensor& t) {
    size_t n = t.numel();
    std::cout << "[";
    for (size_t i = 0; i < n; i++) {
        std::cout << t.data[i];
        if (i != n - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

// Helper: compute linear index in original tensor for a given multi-index (vec idx)
// 'orig' has possibly fewer dims than idx.size(); left-pad with 1s.
static size_t linear_index_from_padded(const Tensor& orig, const std::vector<size_t>& padded_idx) {
    // padded_idx.size() >= orig.ndim
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
Tensor pad_to_ndim(const Tensor& t, size_t target_ndim) {
    if (t.ndim == target_ndim) {
        return Tensor(t); // deep-copy safe because copy ctor exists
    }
    if (t.ndim > target_ndim) throw std::runtime_error("target_ndim smaller than tensor ndim");

    // build new shape (left-pad with ones)
    std::vector<size_t> new_shape(target_ndim, 1);
    for (size_t i = 0; i < t.ndim; ++i)
        new_shape[target_ndim - t.ndim + i] = t.shape[i];

    Tensor result(new_shape);

    // For each element in result, compute the corresponding index in t (broadcasting dims==1)
    size_t N = result.numel();
    std::vector<size_t> idx(target_ndim, 0);

    for (size_t flat = 0; flat < N; ++flat) {
        // compute multi-index
        size_t rem = flat;
        for (int d = (int)target_ndim - 1; d >= 0; --d) {
            idx[d] = rem % new_shape[d];
            rem /= new_shape[d];
        }
        // map to original tensor linear index
        size_t src_idx = linear_index_from_padded(t, idx);
        result.data[flat] = t.data[src_idx];
    }

    return result;
}
