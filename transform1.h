#pragma once
#include <vector>
#include <functional>
#include <stdexcept>
#include "tensor1.h"

struct Transforme {
    std::vector<std::function<Tensor(const Tensor&)>> pipeline;

    // ---------------- N-D normalize ----------------
    void normalize_(const std::vector<float>& mean, const std::vector<float>& stdv) {
        pipeline.push_back([mean, stdv](const Tensor& input) -> Tensor {
            if (!input.impl) throw std::runtime_error("normalize: empty tensor");

            Tensor output(input.shape(), input._dtype(), input.impl->requires_grad);

            size_t C = input.shape()[0];
            if (mean.size() != C && mean.size() != 1)
                throw std::invalid_argument("mean length must match channels or be 1");
            if (stdv.size() != C && stdv.size() != 1)
                throw std::invalid_argument("std length must match channels or be 1");

            // total number of elements
            size_t N = input.numel();

            // multi-index helper
            std::vector<size_t> idx(input.impl->ndim, 0);

            for (size_t flat = 0; flat < N; ++flat) {
                // compute multi-index from flat index
                size_t rem = flat;
                for (int d = static_cast<int>(input.impl->ndim) - 1; d >= 0; --d) {
                    idx[d] = rem % input.impl->shape[d];
                    rem /= input.impl->shape[d];
                }

                // compute real offset in storage
                size_t offset = input.impl->offset;
                for (size_t d = 0; d < input.impl->ndim; ++d)
                    offset += idx[d] * input.impl->strides[d];

                double val = read_scalar_at(input.impl->storage->data.get(), offset, input.impl->dtype);

                // channel index = first dim
                size_t c = idx[0];
                double m = (mean.size() == 1 ? mean[0] : mean[c]);
                double s = (stdv.size() == 1 ? stdv[0] : stdv[c]);

                write_scalar_at(output.impl->storage->data.get(), offset, output.impl->dtype, (val - m)/s);
            }

            return output;
        });
    }

    // ---------------- placeholder resize ----------------
    void resize_(size_t H, size_t W) {
        pipeline.push_back([H, W](const Tensor& input) -> Tensor {
            Tensor output({input.shape()[0], H, W}, input._dtype(), input.impl->requires_grad);
            // real resize/interpolation can be implemented here
            return output;
        });
    }

    // ---------------- apply transformations ----------------
    Tensor operator()(const Tensor& input) const {
        Tensor x = input;
        for (auto& t : pipeline)
            x = t(x);
        return x;
    }
};
