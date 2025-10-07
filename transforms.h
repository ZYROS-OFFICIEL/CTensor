#pragma once
#include <vector>
#include <functional>
#include <stdexcept>
#include "tensors.h"

struct Transforme {
    // internally store a list of functions that take a tensor and return a tensor
    std::vector<std::function<Tensor(const Tensor&)>> pipeline;

    // add normalize transformation
    void normalize_(const std::vector<float>& mean, const std::vector<float>& stdv) {
        pipeline.push_back([mean, stdv](const Tensor& input) -> Tensor {
            size_t C = input.shape[0];
            if (mean.size() != C && mean.size() != 1)
                throw std::invalid_argument("mean length must match channels or be 1");
            if (stdv.size() != C && stdv.size() != 1)
                throw std::invalid_argument("std length must match channels or be 1");

            Tensor output(input.shape_());
            size_t inner_size = 1;
            for (size_t i = 1; i < input.ndim; ++i)
                inner_size *= input.shape[i];

            for (size_t c = 0; c < C; ++c) {
                float m = mean.size() == 1 ? mean[0] : mean[c];
                float s = stdv.size() == 1 ? stdv[0] : stdv[c];
                size_t offset = c * inner_size;
                for (size_t j = 0; j < inner_size; ++j)
                    output.data[offset + j] = (input.data[offset + j] - m) / s;
            }
            return output;
        });
    }

    // you can add more methods: resize_, flip_, etc.
    void resize_(size_t H, size_t W) {
        pipeline.push_back([H, W](const Tensor& input) -> Tensor {
            // simple placeholder: in real code you resize/interpolate
            Tensor output({input.shape[0], H, W});
            // copy or interpolate data...
            return output;
        });
    }
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

    // apply all transformations
    Tensor operator()(const Tensor& input) const {
        Tensor x = input;
        for (auto& t : pipeline)
            x = t(x);
        return x;
    }


};
