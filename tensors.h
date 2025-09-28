#include <iostream>
#include <cstddef>
#include <cstring>
#include <vector>
#include <cmath>

struct Tensor {
    float* data;      // contiguous memory
    float* grad;      // gradient buffer
    size_t ndim;      // number of dimensions
    size_t* shape;    // array of dimension sizes
    size_t* strides;  // array of strides
    bool requires_grad;

    // constructor
    Tensor(const std::vector<size_t>& shape_, bool requires_grad_=false) {
        ndim = shape_.size();
        requires_grad = requires_grad_;

        // allocate shape and strides
        shape = (size_t*) malloc(ndim * sizeof(size_t));
        strides = (size_t*) malloc(ndim * sizeof(size_t));

        // copy shape
        for (size_t i=0; i<ndim; i++) shape[i] = shape_[i];

        // compute strides
        if (ndim > 0) {
            strides[ndim-1] = 1;
            for (int i=ndim-2; i>=0; i--)
                strides[i] = strides[i+1] * shape[i+1];
        }

        // allocate data
        size_t numel = 1;
        for (size_t i=0; i<ndim; i++) numel *= shape[i];
        data = (float*) malloc(numel * sizeof(float));

        // initialize data to zero
        memset(data, 0, numel * sizeof(float));

        // allocate grad if needed
        grad = requires_grad ? (float*) malloc(numel * sizeof(float)) : nullptr;
        if (grad) memset(grad, 0, numel * sizeof(float));
    }

    // destructor
    ~Tensor() {
        if (data) free(data);
        if (grad) free(grad);
        if (shape) free(shape);
        if (strides) free(strides);
    }

    // helper: number of elements
    size_t numel() const {
        size_t n = 1;
        for (size_t i=0; i<ndim; i++) n *= shape[i];
        return n;
    }
};