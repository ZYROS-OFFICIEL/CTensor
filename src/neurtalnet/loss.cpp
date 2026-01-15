#include <tensor.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include "autograd.h"
#include <immintrin.h>
#include <cstring>
#include "loss.h"
#include "ops_dispatcher.h"

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
    double mse_value = read_scalar_at(summed.impl->data->data.get(), 0, summed._dtype()) 
                       / static_cast<double>(pred_.numel_());

    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), mse_value);

    // Attach backward function if needed
    if (req)
        result.impl->grad_fn = std::make_shared<GradMSE>(pred_, target_);

    return result;
}


Tensor Loss::MAE(const Tensor& pred, const Tensor& target,std::string reduction){
    if (!pred.impl || !target.impl)
        throw std::runtime_error("Loss::MAE: null tensor implementation");

    if (pred.impl->ndim != target.impl->ndim)
        throw std::runtime_error("Loss::MAE: dimension mismatch");

    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    // Compute |pred - target|
    Tensor temp = abs(pred - target);

    Tensor summed = sum_mp(temp, -1);

    double mae_value = read_scalar_at(summed.impl->data->data.get(), 0, summed._dtype());
    if(reduction == "mean") {
        mae_value /= static_cast<double>(pred.numel_());
    }

    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), mae_value);

    // Attach backward function if needed
    if (req) {
        result.impl->grad_fn = std::make_shared<GradMAE>(pred, target, reduction);
    }

    return result;
}