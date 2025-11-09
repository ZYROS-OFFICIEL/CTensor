#include "loss.h"
#include <cmath>
#include <string>
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
Tensor Loss::MAE(const Tensor& pred, const Tensor& target,std::string reduction){
    if (!pred.impl || !target.impl)
        throw std::runtime_error("Loss::MAE: null tensor implementation");

    if (pred.impl->ndim != target.impl->ndim)
        throw std::runtime_error("Loss::MAE: dimension mismatch");

    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    // Compute |pred - target|
    Tensor temp = abs_(pred - target);

    // Sum all elements
    Tensor summed = sum(temp, -1);

    double mae_value = read_scalar_at(summed.impl->storage->data.get(), 0, summed._dtype());

    if(reduction == "mean") {
        mae_value /= static_cast<double>(pred.numel_());
    }

    write_scalar_at(result.impl->storage->data.get(), 0, result._dtype(), mae_value);

    // Attach backward function if needed
    if (req) {
        result.impl->grad_fn = std::make_shared<GradMAE>(pred, target, reduction);
    }

    return result;
}
Tensor Loss::HuberLoss(const Tensor& pred, const Tensor& target,std::string reduction,float delta){
    if (!pred.impl || !target.impl)
        throw std::runtime_error("Loss::HuberLoss: null tensor implementation");

    if (pred.impl->ndim != target.impl->ndim)
        throw std::runtime_error("Loss::HuberLoss: dimension mismatch");

    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    // Compute Huber Loss
    Tensor diff = pred - target;
    Tensor abs_diff = abs_(diff);
    if (abs_diff <= delta)
    {
        Tensor quadratic = 0.5 * pow_scalar(diff, 2);
        Tensor huber_loss = quadratic;
    }else{
        Tensor linear = delta * (abs_diff - 0.5 * delta);

        Tensor huber_loss = linear;

    }
    
    // Sum all elements
    Tensor summed = sum(huber_loss, -1);
    double huber_value = read_scalar_at(summed.impl->storage->data.get(), 0, summed._dtype());
    if(reduction == "mean") {
        huber_value /= static_cast<double>(pred.numel_());
    }
    write_scalar_at(result.impl->storage->data.get(), 0, result._dtype(), huber_value);

    // Attach backward function if needed
    if (req) {
        result.impl->grad_fn = std::make_shared<GradHuberLoss>(pred, target, reduction, delta);
    }

    return result;
}
Tensor Loss::CrossEntropy(const Tensor& pred_, const Tensor& target_) {
    if (!pred_.impl || !target_.impl)
        throw std::runtime_error("Loss::CrossEntropy: null tensor implementation");

    if (pred_.impl->ndim != target_.impl->ndim)
        throw std::runtime_error("Loss::CrossEntropy: dimension mismatch");

    bool req = pred_.requires_grad();
    Tensor result({1}, pred_.impl->dtype, req);

    // --- Softmax manually (numerically stable) ---
    // step 1: subtract max to avoid overflow
    Tensor max_vals = max(pred_, -1);       // keepdims=True
    Tensor shifted = pred_ - max_vals;

    // step 2: exponentiate and normalize
    Tensor exp_shifted = exp_(shifted);
    Tensor sum_exp = sum(exp_shifted, -1);
    Tensor probs = exp_shifted / sum_exp;         // softmax result

    // --- Cross Entropy ---
    Tensor ce = -sum(target_ * ln_(probs), -1);  // element-wise * then sum

    // --- Mean across batch if needed ---
    result = mean(ce);

    // --- Optional backward ---
    if (req)
        result.impl->grad_fn = std::make_shared<GradCrossEntropy>(pred_, target_);

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
void GradMAE::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradMAE: missing self grad");
    if (!pred.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = pred.numel_();

    auto* gdata = grad_input.impl->storage->data.get();
    auto* pdata = pred.impl->storage->data.get();
    auto* tdata = target.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double p = read_scalar_at(pdata, i, pred._dtype());
        double t = read_scalar_at(tdata, i, target._dtype());
        double sign = (p > t) ? 1.0 : ((p < t) ? -1.0 : 0.0);
        double grad_val = sign;
        write_scalar_at(gdata, i, grad_input._dtype(), grad_val);
    }

    // 🔹 If reduction is "mean", scale gradient
    if (reduction == "mean") {
        grad_input = grad_input / static_cast<double>(n);
    }
    // 🔹 If reduction == "sum", leave as-is (no scaling)

    accumulate_grad(pred, grad_input);
}

void GradCrossEntropy::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("CrossEntropy: missing self grad");
    if (!pred.requires_grad()) return;

    // Compute softmax manually
    Tensor max_vals = max(pred, -1);
    Tensor shifted = pred - max_vals;
    Tensor exp_shifted = exp_(shifted);
    Tensor sum_exp = sum(exp_shifted, -1);
    Tensor probs = exp_shifted / sum_exp;  // softmax(pred)

    // Gradient: softmax(pred) - target
    Tensor grad_input = probs - target;

    // Scale by mean if needed (since we averaged the loss)
    grad_input = grad_input / static_cast<double>(pred.shape()[0]);

    accumulate_grad(pred, grad_input);
}


