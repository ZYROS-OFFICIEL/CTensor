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


Tensor Loss::HuberLoss(const Tensor& pred, const Tensor& target,std::string reduction,double delta){
    if (!pred.impl || !target.impl)
        throw std::runtime_error("Loss::HuberLoss: null tensor implementation");

    if (pred.impl->ndim != target.impl->ndim)
        throw std::runtime_error("Loss::HuberLoss: dimension mismatch");

    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    
    // Compute Huber Loss
    Tensor diff = pred - target;
    Tensor abs_diff = abs_mp(diff);
    Tensor linear = delta * (abs_diff - 0.5 * delta);
    Tensor quadratic = 0.5 * diff * diff;

    Tensor huber_loss = (abs_diff <= delta) * quadratic + (abs_diff > delta) * linear;
    
    // Sum all elements
    Tensor summed = sum(huber_loss, -1);
    double huber_value = read_scalar_at(summed.impl->data->data.get(), 0, summed._dtype());
    if(reduction == "mean") {
        huber_value /= static_cast<double>(pred.numel_());
    }
    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), huber_value);
    // Attach backward function if needed
    if (req) {
        result.impl->grad_fn = std::make_shared<GradHuberLoss>(pred, target, reduction, delta);
    }

    return result;
}

Tensor Loss::CrossEntropy(const Tensor& pred_, const Tensor& target_, std::string reduction) {
    if (!pred_.impl || !target_.impl) throw std::runtime_error("Loss::CrossEntropy: null tensor");
    
    // Pred: [Batch, Classes], Target: [Batch, 1] (Indices)
    if (pred_.impl->ndim != 2) throw std::runtime_error("CrossEntropy: Pred must be 2D [Batch, Classes]");
    // Target can be [Batch, 1] or [Batch] (if supported), assuming [Batch, 1] from loader
    
    bool req = pred_.requires_grad();
    Tensor result({1}, pred_.impl->dtype, req);

    // 1. LogSoftmax Stability Trick: x - max(x)
    Tensor max_vals = max(pred_, 1).reshape({pred_.shape()[0], 1});
    Tensor shifted = pred_ - max_vals; 

    // 2. Compute Exp and Sum
    Tensor exp_vals = exp(shifted);
    Tensor sum_exp = sum(exp_vals, 1).reshape({pred_.shape()[0], 1});
    Tensor log_sum_exp = ln(sum_exp);

    // 3. LogSoftmax = logits - log_sum_exp
    Tensor log_probs = shifted - log_sum_exp; 

    // 4. Gather correct class probability
    // Target contains indices. gather(dim=1) picks the column.
    Tensor picked = log_probs.gather(target_, 1); // Returns [Batch, 1]
    
    // 5. NLL = -picked
    Tensor nll = picked * -1.0;

    // 6. Reduction
    Tensor reduced;
    if (reduction == "mean") reduced = mean(nll);
    else reduced = sum(nll);
    
    double val = read_scalar_at(reduced.impl->data->data.get(), 0, reduced._dtype());
    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), val);

    if (req) result.impl->grad_fn = std::make_shared<GradCrossEntropy>(pred_, target_, reduction);
    return result;
}