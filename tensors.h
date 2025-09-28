#include <iostream>
#include <cstddef>
#include <cstring>
#include <vector>
#include <cmath>

struct Tensor {
    float* data;      
    float* grad;     
    size_t ndim;      
    size_t* shape;    
    size_t* strides;  
    bool requires_grad;

    Tensor(const std::vector<size_t>& shape_, bool requires_grad_=false) {
        ndim = shape_.size();
        requires_grad = requires_grad_;

        shape = (size_t*) malloc(ndim * sizeof(size_t));
        strides = (size_t*) malloc(ndim * sizeof(size_t));

        for (size_t i=0; i<ndim; i++) shape[i] = shape_[i];

        if (ndim > 0) {
            strides[ndim-1] = 1;
            for (int i=ndim-2; i>=0; i--)
                strides[i] = strides[i+1] * shape[i+1];
        }

        size_t numel = 1;
        for (size_t i=0; i<ndim; i++) numel *= shape[i];
        data = (float*) malloc(numel * sizeof(float));

        memset(data, 0, numel * sizeof(float));

        grad = requires_grad ? (float*) malloc(numel * sizeof(float)) : nullptr;
        if (grad) memset(grad, 0, numel * sizeof(float));
    }

    ~Tensor() {
        if (data) free(data);
        if (grad) free(grad);
        if (shape) free(shape);
        if (strides) free(strides);
    }

    size_t numel() const {
        size_t n = 1;
        for (size_t i=0; i<ndim; i++) n *= shape[i];
        return n;
    }
    
    // --------- ND access proxy ---------
    struct Proxy {
        float* data;
        size_t* shape;
        size_t* strides;
        size_t offset;
        size_t dim;
        size_t ndim;

        Proxy(float* d, size_t* s, size_t* st, size_t off, size_t dm, size_t n)
            : data(d), shape(s), strides(st), offset(off), dim(dm), ndim(n) {}

        Proxy operator[](size_t i) {
            if (dim + 1 >= ndim) throw std::out_of_range("Too many indices");
            if (i >= shape[dim]) throw std::out_of_range("Index out of bounds");
            return Proxy(data, shape, strides, offset + i * strides[dim], dim + 1, ndim);
        }

        operator float&() { return data[offset]; }      // last dimension returns reference
        Proxy& operator=(float val) { data[offset] = val; return *this; }
    };

    Proxy operator[](size_t i) {
        if (i >= shape[0]) throw std::out_of_range("Index out of bounds");
        if (ndim == 1) return Proxy(data, shape, strides, i, 0, ndim);
        return Proxy(data, shape, strides, i * strides[0], 1, ndim);
    }
};
void print_t(const Tensor& t) {
    size_t n = t.numel();
    std::cout << "[";
    for (size_t i = 0; i < n; i++) {
        std::cout << t.data[i];
        if (i != n-1) std::cout << ", ";
    }
    std::cout << "]\n";
}