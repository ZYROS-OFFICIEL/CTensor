#include "loss.h"
#include <cmath>

Tensor Loss::MSE(const Tensor& pred_, const Tensor& target_) {
    if (!pred_.impl || !target_.impl)
        throw std::runtime_error("Loss::MSE: null tensor implementation");

    if (pred_.impl->ndim != target_.impl->ndim)
        throw std::runtime_error("Loss::MSE: dimension mismatch");

    bool req = pred_.requires_grad();
    Tensor result({1}, pred_.impl->dtype, req);

    // Compute (pred - target)^2
    Tensor temp = pow_scalar(pred_ - target_, 2);

    // Sum all elements
    Tensor summed = sum(temp, -1);

    // Divide by number of elements to get MSE
    double mse_value = read_scalar_at(summed.impl->storage->data.get(), 0, summed._dtype()) 
                       / static_cast<double>(pred_.numel_());

    write_scalar_at(result.impl->storage->data.get(), 0, result._dtype(), mse_value);

    // Attach backward function if needed
    if (req)
        result.impl->grad_fn = std::make_shared<GradMSE>(pred_, target_);

    return result;
}

void GradMSE::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradMSE: missing self grad");
    if (!pred.impl || !pred.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = pred.numel_();

    for (size_t i = 0; i < n; ++i) {
        double p = read_scalar_at(pred.impl->storage->data.get(), i, pred._dtype());
        double t = read_scalar_at(target.impl->storage->data.get(), i, target._dtype());
        double res = (2.0 / static_cast<double>(n)) * (p - t);
        write_scalar_at(grad_input.impl->storage->data.get(), i, grad_input._dtype(), res);
    }

    accumulate_grad(pred, grad_input);
}
