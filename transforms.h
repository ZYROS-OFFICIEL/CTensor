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

    // apply all transformations
    Tensor operator()(const Tensor& input) const {
        Tensor x = input;
        for (auto& t : pipeline)
            x = t(x);
        return x;
    }


};
