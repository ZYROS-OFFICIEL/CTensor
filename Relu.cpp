#include "Relu.h"


Tensor LeakyRelu(const Tensor& a_, double negative_slope) {
    if (!a_.impl)
        throw std::runtime_error("LeakyRelu: null tensor implementation");

    size_t n = a_.numel();
    bool req = a_.requires_grad();

    Tensor result(a_.shape(), a_.dtype(), req);

    if (req)
        result.impl->grad_fn = std::make_shared<GradLeakyRelu>(a_, negative_slope);

    auto* a_data = a_.impl->storage->data.get();
    auto* r_data = result.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double v = read_scalar_at(a_data, i, a_.dtype());
        double out = (v >= 0.0) ? v : v * negative_slope;
        write_scalar_at(r_data, i, result.dtype(), out);
    }

    return result;
}

Tensor PRelu(const Tensor& a_, double init, int num_parameters, DType dtype) {
    if (!a_.impl)
        throw std::runtime_error("PRelu: null tensor implementation");

    size_t in_c = a_.shape()[1];
    size_t n = a_.numel();
    bool req = a_.requires_grad();

    // -------------------------------
    // VALIDATE num_parameters
    // -------------------------------
    if (!(num_parameters == 1 || num_parameters == in_c)) {
        throw std::runtime_error(
            "PRelu: RuntimeError: mismatch in shape between input and weight");
    }

    bool per_channel = (num_parameters == in_c);

    // -------------------------------
    // CREATE α PARAMETER
    // -------------------------------
    std::vector<size_t> wshape;
    if (per_channel)
        wshape = {in_c};
    else
        wshape = {1};

    Tensor weight(wshape, dtype, req);

    // fill weight with "init" value
    auto* w_data = weight.impl->storage->data.get();
    for (size_t i = 0; i < num_parameters; i++)
        write_scalar_at(w_data, i, dtype, init);

    // -------------------------------
    // CREATE OUTPUT
    // -------------------------------
    Tensor result(a_.shape(), a_.dtype(), req);

    if (req)
        result.impl->grad_fn = std::make_shared<GradPRelu>(a_, weight);

    auto* a_data = a_.impl->storage->data.get();
    auto* r_data = result.impl->storage->data.get();

    // -------------------------------
    // APPLY PReLU
    // shape = N, C, ...
    // -------------------------------
    size_t spatial = 1;
    for (int i = 2; i < a_.ndim(); i++)
        spatial *= a_.shape()[i];

    size_t idx = 0;
    for (size_t n_idx = 0; n_idx < a_.shape()[0]; n_idx++) {
        for (size_t c = 0; c < in_c; c++) {

            double alpha = per_channel ?
                read_scalar_at(w_data, c, dtype) :
                read_scalar_at(w_data, 0, dtype);

            for (size_t s = 0; s < spatial; s++, idx++) {
                double v = read_scalar_at(a_data, idx, a_.dtype());
                double out = (v >= 0.0) ? v : v * alpha;
                write_scalar_at(r_data, idx, result.dtype(), out);
            }
        }
    }

    return result;
}
