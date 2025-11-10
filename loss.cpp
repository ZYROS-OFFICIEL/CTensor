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
Tensor Loss::HuberLoss(const Tensor& pred, const Tensor& target,std::string reduction,double delta){
    if (!pred.impl || !target.impl)
        throw std::runtime_error("Loss::HuberLoss: null tensor implementation");

    if (pred.impl->ndim != target.impl->ndim)
        throw std::runtime_error("Loss::HuberLoss: dimension mismatch");

    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    
    // Compute Huber Loss
    Tensor diff = pred - target;
    Tensor abs_diff = abs_(diff);
    Tensor linear = delta * (abs_diff - 0.5 * delta);
    Tensor quadratic = 0.5 * diff * diff;

    Tensor huber_loss = (abs_diff <= delta) * quadratic + (abs_diff > delta) * linear;
    
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

Tensor Loss::LogCosh(const Tensor& pred, const Tensor& target,std::string reduction){
    if (!pred.impl || !target.impl)
        throw std::runtime_error("Loss::LogCosh: null tensor implementation");

    if (pred.impl->ndim != target.impl->ndim)
        throw std::runtime_error("Loss::LogCosh: dimension mismatch");

    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    // Compute Log-Cosh Loss
    Tensor diff = pred - target;
    Tensor log_cosh_loss = ln_(cosh_(diff));

    // Sum all elements
    Tensor summed = sum(log_cosh_loss, -1);
    double log_cosh_value = read_scalar_at(summed.impl->storage->data.get(), 0, summed._dtype());

    if(reduction == "mean") {
        log_cosh_value /= static_cast<double>(pred.numel_());
    }

    write_scalar_at(result.impl->storage->data.get(), 0, result._dtype(), log_cosh_value);

    // Attach backward function if needed
    if (req) {
        // Note: GradLogCosh needs to be implemented similarly to other GradFns
        result.impl->grad_fn = std::make_shared<GradLogCosh>(pred, target, reduction);
    }

    return result;
}

Tensor Loss::BCE(const Tensor& pred, const Tensor& target,std::string reduction){
    if (!pred.impl || !target.impl)
        throw std::runtime_error("Loss::BCE: null tensor implementation");

    if (pred.impl->ndim != target.impl->ndim)
        throw std::runtime_error("Loss::BCE: dimension mismatch");
    if (max(pred) > 1.0 || min(pred) < 0.0)
        throw std::runtime_error("BCE: input must be in [0, 1]");


    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    // Compute BCE Loss
    Tensor log_pred = ln_(pred + 1e-12);               // to avoid log(0)
    Tensor log_one_minus_pred = ln_(1 - pred + 1e-12); // to avoid log(0)

    Tensor bce_loss = - (target * log_pred + (1 - target) * log_one_minus_pred);

    // Sum all elements
    Tensor summed = sum(bce_loss, -1);
    double bce_value = read_scalar_at(summed.impl->storage->data.get(), 0, summed._dtype());

    if(reduction == "mean") {
        bce_value /= static_cast<double>(pred.numel_());
    }

    write_scalar_at(result.impl->storage->data.get(), 0, result._dtype(), bce_value);

    // Attach backward function if needed
    if (req) {
        result.impl->grad_fn = std::make_shared<GradBCE>(pred, target, reduction);
    }

    return result;
}

Tensor Loss::KLDiv(const Tensor& pred, const Tensor& target,std::string reduction){
    if (!pred.impl || !target.impl)
        throw std::runtime_error("Loss::KLDiv: null tensor implementation");

    if (pred.impl->ndim != target.impl->ndim)
        throw std::runtime_error("Loss::KLDiv: dimension mismatch");

    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    // Compute KL Divergence Loss
    Tensor log_pred = ln_(pred + 1e-12); // to avoid log(0)
    Tensor kl_loss = target * (log_pred - ln_(target + 1e-12));

    // Sum all elements
    Tensor summed = sum(kl_loss, -1);
    double kl_value = read_scalar_at(summed.impl->storage->data.get(), 0, summed._dtype());

    if(reduction == "mean") {
        kl_value /= static_cast<double>(pred.numel_());
    }

    write_scalar_at(result.impl->storage->data.get(), 0, result._dtype(), kl_value);

    // Attach backward function if needed
    if (req) {
        result.impl->grad_fn = std::make_shared<GradKLDiv>(pred, target, reduction);
    }

    return result;
}

Tensor Loss::NLLLoss(const Tensor& pred, const Tensor& target,std::string reduction){
    if (!pred.impl || !target.impl)
        throw std::runtime_error("Loss::NLLLoss: null tensor implementation");

    if (pred.impl->ndim != target.impl->ndim)
        throw std::runtime_error("Loss::NLLLoss: dimension mismatch");

    bool req = pred.requires_grad();
    Tensor picked = gather(pred, target, 1);  // selects pred[i, target[i]]

    // 3️⃣ Compute the NLL loss per sample
    Tensor nll_loss = - ln_(picked + 1e-12);  // shape: [batch_size, 1]

    // Sum all elements
    Tensor summed = sum(nll_loss, -1);
    double nll_value = read_scalar_at(summed.impl->storage->data.get(), 0, summed._dtype());

    if(reduction == "mean") {
        nll_value /= static_cast<double>(pred.numel_());
    }

    write_scalar_at(result.impl->storage->data.get(), 0, result._dtype(), nll_value);

    // Attach backward function if needed
    if (req) {
        result.impl->grad_fn = std::make_shared<GradNLLLoss>(pred, target, reduction);
    }

    return result;
}


Tensor HingeLoss(const Tensor& pred, const Tensor& target,std::string reduction){
    if (!pred.impl || !target.impl)
        throw std::runtime_error("Loss::HingeLoss: null tensor implementation");

    if (pred.impl->ndim != target.impl->ndim)
        throw std::runtime_error("Loss::HingeLoss: dimension mismatch");

    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    // Compute Hinge Loss
    Tensor one = Tensor::full(pred.shape(), 1.0, pred._dtype(), false);
    Tensor margin = one - target * pred;
    Tensor hinge_loss = Relu_(margin);

    // Sum all elements
    Tensor summed = sum(hinge_loss);
    double hinge_value = read_scalar_at(summed.impl->storage->data.get(), 0, summed._dtype());

    if(reduction == "mean") {
        hinge_value /= static_cast<double>(pred.numel_());
    }

    write_scalar_at(result.impl->storage->data.get(), 0, result._dtype(), hinge_value);

    // Attach backward function if needed
    if (req) {
        result.impl->grad_fn = std::make_shared<GradHingeLoss>(pred, target, reduction);
    }

    return result;
}


Tensor MarginRankingLoss(const Tensor& input1, const Tensor& input2, const Tensor& target, double margin, std::string reduction) {
    if (!input1.impl || !input2.impl || !target.impl)
        throw std::runtime_error("Loss::MarginRankingLoss: null tensor implementation");

    if (input1.impl->ndim != input2.impl->ndim || input1.impl->ndim != target.impl->ndim)
        throw std::runtime_error("Loss::MarginRankingLoss: dimension mismatch");

    bool req = input1.requires_grad();
    Tensor result({1}, input1.impl->dtype, req);

    // Compute Margin Ranking Loss
    Tensor diff = input1 - input2;
    Tensor margin_tensor = margin - target * diff;
    Tensor loss = Relu_(margin_tensor);

    // Sum all elements
    Tensor summed = sum(loss);
    double loss_value = read_scalar_at(summed.impl->storage->data.get(), 0, summed._dtype());

    if(reduction == "mean") {
        loss_value /= static_cast<double>(input1.numel_());
    }

    write_scalar_at(result.impl->storage->data.get(), 0, result._dtype(), loss_value);

    // Attach backward function if needed
    if (req) {
        result.impl->grad_fn = std::make_shared<GradMarginRankingLoss>(input1, input2, target, margin, reduction);
    }

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

void GradHuberLoss::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradHuberLoss: missing self grad");
    if (!pred.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = pred.numel_();

    auto* gdata = grad_input.impl->storage->data.get();
    auto* pdata = pred.impl->storage->data.get();
    auto* tdata = target.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double p = read_scalar_at(pdata, i, pred._dtype());
        double t = read_scalar_at(tdata, i, target._dtype());
        double diff = p - t;
        double abs_diff = std::abs(diff);
        double grad_val = 0.0;

        if (abs_diff <= delta) {
            grad_val = diff; // derivative of 0.5 * diff^2
        } else {
            grad_val = delta * ((diff > 0) ? 1.0 : -1.0); // derivative of delta * (|diff| - 0.5 * delta)
        }

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

    if (reduction == "mean") {
    grad_input = grad_input / static_cast<double>(pred.shape()[0]);
    }

    accumulate_grad(pred, grad_input);
}

void GradBCE::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradBCE: missing self grad");
    if (!pred.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = pred.numel_();

    auto* gdata = grad_input.impl->storage->data.get();
    auto* pdata = pred.impl->storage->data.get();
    auto* tdata = target.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double p = read_scalar_at(pdata, i, pred._dtype());
        double t = read_scalar_at(tdata, i, target._dtype());
        double grad_val = (p - t) / ( (p * (1 - p)) + 1e-12 ); // derivative of BCE
        write_scalar_at(gdata, i, grad_input._dtype(), grad_val);
    }

    // 🔹 If reduction is "mean", scale gradient
    if (reduction == "mean") {
        grad_input = grad_input / static_cast<double>(n);
    }
    // 🔹 If reduction == "sum", leave as-is (no scaling)

    accumulate_grad(pred, grad_input);
}

void GradLogCosh::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradLogCosh: missing self grad");
    if (!pred.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = pred.numel_();

    auto* gdata = grad_input.impl->storage->data.get();
    auto* pdata = pred.impl->storage->data.get();
    auto* tdata = target.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double p = read_scalar_at(pdata, i, pred._dtype());
        double t = read_scalar_at(tdata, i, target._dtype());
        double diff = p - t;
        double grad_val = std::tanh(diff); // derivative of Log-Cosh
        write_scalar_at(gdata, i, grad_input._dtype(), grad_val);
    }

    // 🔹 If reduction is "mean", scale gradient
    if (reduction == "mean") {
        grad_input = grad_input / static_cast<double>(n);
    }
    // 🔹 If reduction == "sum", leave as-is (no scaling)

    accumulate_grad(pred, grad_input);
}

void GradKLDiv::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradKLDiv: missing self grad");
    if (!pred.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = pred.numel_();

    auto* gdata = grad_input.impl->storage->data.get();
    auto* pdata = pred.impl->storage->data.get();
    auto* tdata = target.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double p = read_scalar_at(pdata, i, pred._dtype());
        double t = read_scalar_at(tdata, i, target._dtype());
        double grad_val = ( (p > 1e-12) ? (t / (p + 1e-12)) : 0.0 ); // derivative of KLDiv
        write_scalar_at(gdata, i, grad_input._dtype(), grad_val);
    }

    // 🔹 If reduction is "mean", scale gradient
    if (reduction == "mean") {
        grad_input = grad_input / static_cast<double>(n);
    }
    // 🔹 If reduction == "sum", leave as-is (no scaling)

    accumulate_grad(pred, grad_input);
}

void GradNLLLoss::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradNLLLoss: missing self grad");
    if (!pred.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = pred.numel_();

    auto* gdata = grad_input.impl->storage->data.get();
    auto* pdata = pred.impl->storage->data.get();
    auto* tdata = target.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double p = read_scalar_at(pdata, i, pred._dtype());
        double t = read_scalar_at(tdata, i, target._dtype());
        double grad_val = 0.0;
        if (static_cast<size_t>(t) == i % pred.shape()[1]) { // assuming target contains class indices
            grad_val = -1.0 / (p + 1e-12); // derivative of NLLLoss
        }
        write_scalar_at(gdata, i, grad_input._dtype(), grad_val);
    }

    // 🔹 If reduction is "mean", scale gradient
    if (reduction == "mean") {
        grad_input = grad_input / static_cast<double>(n);
    }
    // 🔹 If reduction == "sum", leave as-is (no scaling)

    accumulate_grad(pred, grad_input);
}
void GradHingeLoss::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradHingeLoss: missing self grad");
    if (!pred.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = pred.numel_();

    auto* gdata = grad_input.impl->storage->data.get();
    auto* pdata = pred.impl->storage->data.get();
    auto* tdata = target.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double p = read_scalar_at(pdata, i, pred._dtype());
        double t = read_scalar_at(tdata, i, target._dtype());
        double margin = 1.0 - t * p;
        double grad_val = (margin > 0) ? -t : 0.0; // derivative of Hinge Loss
        write_scalar_at(gdata, i, grad_input._dtype(), grad_val);
    }

    // 🔹 If reduction is "mean", scale gradient
    if (reduction == "mean") {
        grad_input = grad_input / static_cast<double>(n);
    }
    // 🔹 If reduction == "sum", leave as-is (no scaling)

    accumulate_grad(pred, grad_input);
}
void GradMarginRankingLoss::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradMarginRankingLoss: missing self grad");
    if (!input1.requires_grad() && !input2.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = input1.numel_();

    auto* gdata = grad_input.impl->storage->data.get();
    auto* in1data = input1.impl->storage->data.get();
    auto* in2data = input2.impl->storage->data.get();
    auto* tdata = target.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double in1 = read_scalar_at(in1data, i, input1._dtype());
        double in2 = read_scalar_at(in2data, i, input2._dtype());
        double t = read_scalar_at(tdata, i, target._dtype());
        double margin_val = margin - t * (in1 - in2);
        double grad_val = (margin_val > 0) ? -t : 0.0; // derivative of Margin Ranking Loss

        if (input1.requires_grad()) {
            write_scalar_at(gdata, i, grad_input._dtype(), grad_val);
            accumulate_grad(input1, grad_input);
        }
        if (input2.requires_grad()) {
            write_scalar_at(gdata, i, grad_input._dtype(), -grad_val);
            accumulate_grad(input2, grad_input);
        }
    }

    // 🔹 If reduction is "mean", scale gradients
    if (reduction == "mean") {
        if (input1.requires_grad()) {
            Tensor scaled_grad1 = tensor_from_grad(self) / static_cast<double>(n);
            accumulate_grad(input1, scaled_grad1);
        }
        if (input2.requires_grad()) {
            Tensor scaled_grad2 = tensor_from_grad(self) / static_cast<double>(n);
            accumulate_grad(input2, scaled_grad2);
        }
    }
    // 🔹 If reduction == "sum", leave as-is (no scaling)
}
