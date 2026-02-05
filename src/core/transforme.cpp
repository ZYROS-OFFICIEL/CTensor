#include <vector>
#include <functional>
#include <stdexcept>
#include "transforme.h" // Ensure this points to your header file

// ---------------- N-D normalize implementation ----------------
void Transforme::normalize_(const std::vector<float>& mean, const std::vector<float>& stdv) {
    // Add the lambda to the pipeline
    pipeline.push_back([mean, stdv](const Tensor& input) -> Tensor {
        // We assume 'impl' is accessible here (friend class or public member)
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

            // compute real offset in data
            size_t offset = input.impl->offset;
            for (size_t d = 0; d < input.impl->ndim; ++d)
                offset += idx[d] * input.impl->strides[d];

            // Assuming read_scalar_at is available via tensor.h or similar
            double val = read_scalar_at(input.impl->data->data.get(), offset, input.impl->dtype);

            // channel index = first dim
            size_t c = idx[0];
            double m = (mean.size() == 1 ? mean[0] : mean[c]);
            double s = (stdv.size() == 1 ? stdv[0] : stdv[c]);

            write_scalar_at(output.impl->data->data.get(), offset, output.impl->dtype, (val - m)/s);
        }

        return output;
    });
}

// ---------------- placeholder resize implementation ----------------
void Transforme::resize_(size_t H, size_t W) {
    pipeline.push_back([H, W](const Tensor& input) -> Tensor {
        // Create a new tensor with the target shape (Channels, H, W)
        // Note: Real interpolation logic would go here. Currently just a placeholder.
        Tensor output({input.shape()[0], H, W}, input._dtype(), input.impl->requires_grad);
        return output;
    });
}

// ---------------- apply transformations implementation ----------------
Tensor Transforme::operator()(const Tensor& input) const {
    Tensor x = input;
    for (const auto& t : pipeline) {
        x = t(x);
    }
    return x;
}