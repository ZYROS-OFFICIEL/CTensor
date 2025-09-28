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