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

    Tensor summed = sum(temp, -1);

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
    Tensor abs_diff = abs(diff);
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

Tensor Loss::LogCosh(const Tensor& pred, const Tensor& target,std::string reduction){
    if (!pred.impl || !target.impl)
        throw std::runtime_error("Loss::LogCosh: null tensor implementation");

    if (pred.impl->ndim != target.impl->ndim)
        throw std::runtime_error("Loss::LogCosh: dimension mismatch");

    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    // Compute Log-Cosh Loss
    Tensor diff = pred - target;
    Tensor log_cosh_loss = ln(cosh(diff));

    // Sum all elements
    Tensor summed = sum(log_cosh_loss, -1);
    double log_cosh_value = read_scalar_at(summed.impl->data->data.get(), 0, summed._dtype());

    if(reduction == "mean") {
        log_cosh_value /= static_cast<double>(pred.numel_());
    }

    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), log_cosh_value);

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
    Tensor t_max = max(pred); // returns shape [1]
    Tensor t_min = min(pred); // returns shape [1]
    
    // read scalar value from the underlying data
    double max_val = read_scalar_at(t_max.impl->data->data.get(), 0, t_max._dtype());
    double min_val = read_scalar_at(t_min.impl->data->data.get(), 0, t_min._dtype());
    if (max_val > 1.0 || min_val < 0.0)
        throw std::runtime_error("BCE: input must be in [0, 1]");


    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    // Compute BCE Loss
    Tensor log_pred = ln(pred + 1e-12);               // to avoid log(0)
    Tensor log_one_minus_pred = ln(1 - pred + 1e-12); // to avoid log(0)

    Tensor bce_loss = - (target * log_pred + (1 - target) * log_one_minus_pred);

    // Sum all elements
    Tensor summed = sum(bce_loss, -1);
    double bce_value = read_scalar_at(summed.impl->data->data.get(), 0, summed._dtype());

    if(reduction == "mean") {
        bce_value /= static_cast<double>(pred.numel_());
    }

    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), bce_value);

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
    Tensor log_pred = ln(pred + 1e-12); // to avoid log(0)
    Tensor kl_loss = target * (log_pred - ln(target + 1e-12));

    // Sum all elements
    Tensor summed = sum(kl_loss, -1);
    double kl_value = read_scalar_at(summed.impl->data->data.get(), 0, summed._dtype());

    if(reduction == "mean") {
        kl_value /= static_cast<double>(pred.numel_());
    }

    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), kl_value);

    // Attach backward function if needed
    if (req) {
        result.impl->grad_fn = std::make_shared<GradKLDiv>(pred, target, reduction);
    }

    return result;
}

Tensor Loss::NLLLoss(const Tensor& pred, const Tensor& target, std::string reduction){
    if (!pred.impl || !target.impl) throw std::runtime_error("Loss::NLLLoss: null tensor");
    
    // Assumes pred is ALREADY Log-Probabilities [Batch, Classes]
    // Target is Indices [Batch, 1]

    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    // Gather specific indices
    Tensor picked = pred.gather(target, 1);  
    Tensor loss = picked * -1.0; // NLL is negative log likelihood

    Tensor reduced;
    if(reduction == "mean") reduced = mean(loss);
    else reduced = sum(loss);

    double val = read_scalar_at(reduced.impl->data->data.get(), 0, reduced._dtype());
    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), val);
    if (req) result.impl->grad_fn = std::make_shared<GradNLLLoss>(pred, target, reduction);
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
    Tensor hinge_loss = Relu(margin);

    // Sum all elements
    Tensor summed = sum(hinge_loss);
    double hinge_value = read_scalar_at(summed.impl->data->data.get(), 0, summed._dtype());
    if(reduction == "mean") {
        hinge_value /= static_cast<double>(pred.numel_());
    }

    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), hinge_value);

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
    Tensor loss = Relu(margin_tensor);

    // Sum all elements
    Tensor summed = sum(loss);
    double loss_value = read_scalar_at(summed.impl->data->data.get(), 0, summed._dtype());

    if(reduction == "mean") {
        loss_value /= static_cast<double>(input1.numel_());
    }

    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), loss_value);

    // Attach backward function if needed
    if (req) {
        result.impl->grad_fn = std::make_shared<GradMarginRankingLoss>(input1, input2, target, margin, reduction);
    }

    return result;
}

void GradMSE::backward(const Tensor& self) {
    if (!self.impl || !self.impl->data || !self.impl->data->grad)
        throw std::runtime_error("GradMSE: missing self grad");
    if (!pred.impl || !pred.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = pred.numel_();

    for (size_t i = 0; i < n; ++i) {
        double p = read_scalar_at(pred.impl->data->data.get(), i, pred._dtype());
        double t = read_scalar_at(target.impl->data->data.get(), i, target._dtype());
        double res = (2.0 / static_cast<double>(n)) * (p - t);
        write_scalar_at(grad_input.impl->data->data.get(), i, grad_input._dtype(), res);
    }

    accumulate_grad(pred, grad_input);
}

void GradMAE::backward(const Tensor& self) {
    if (!self.impl || !self.impl->data || !self.impl->data->grad)
        throw std::runtime_error("GradMAE: missing self grad");
    if (!pred.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = pred.numel_();

    auto* gdata = grad_input.impl->data->data.get();
    auto* pdata = pred.impl->data->data.get();
    auto* tdata = target.impl->data->data.get();

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
    if (!self.impl || !self.impl->data || !self.impl->data->grad)
        throw std::runtime_error("GradHuberLoss: missing self grad");
    if (!pred.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = pred.numel_();

    auto* gdata = grad_input.impl->data->data.get();
    auto* pdata = pred.impl->data->data.get();
    auto* tdata = target.impl->data->data.get();

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


// --- CROSS ENTROPY BACKWARD (UPDATED FOR INDICES) ---
void GradCrossEntropy::backward(const Tensor& self) {
    if (!self.impl->data->grad) throw std::runtime_error("GradCrossEntropy: missing self grad");
    if (!pred.requires_grad()) return;

    // 1. Re-compute Softmax (can be optimized if cached, but safer to recompute)
    Tensor max_vals = max_mp(pred, 1).reshape({pred.shape()[0], 1});
    Tensor shifted = pred - max_vals;
    Tensor sum_exp = sum_mp(exp_mp(shifted), 1).reshape({pred.shape()[0], 1});
    Tensor probs = exp_mp(shifted) / sum_exp; 

    // 2. Initialize Gradient with 'probs'
    // Deriv is (probs - 1) at target index, (probs - 0) elsewhere.
    Tensor grad_input = probs; 

    // 3. Subtract 1.0 at target indices
    size_t batch_size = pred.shape()[0];
    size_t num_classes = pred.shape()[1];
    
    // We assume target is Int32 [Batch, 1]
    int32_t* t_ptr = (int32_t*)target.impl->data->data.get();
    float* g_ptr = (float*)grad_input.impl->data->data.get(); // assuming float gradient

    #pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b) {
        int label = t_ptr[b];
        if (label >= 0 && label < (int)num_classes) {
            size_t idx = b * num_classes + label;
            g_ptr[idx] -= 1.0f; 
        }
    }

    // 4. Handle Reduction and Incoming Gradient
    // Get scalar gradient from loss (usually 1.0)
    double grad_out = read_scalar_at(self.impl->data->grad.get(), 0, self._dtype());

    if (reduction == "mean") {
        grad_out /= static_cast<double>(batch_size);
    }

    // Apply scaling
    if (grad_out != 1.0) {
        size_t n = grad_input.numel();
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) g_ptr[i] *= (float)grad_out;
    }

    accumulate_grad(pred, grad_input);
}


void GradBCE::backward(const Tensor& self) {
    if (!self.impl || !self.impl->data || !self.impl->data->grad)
        throw std::runtime_error("GradBCE: missing self grad");
    if (!pred.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = pred.numel_();

    auto* gdata = grad_input.impl->data->data.get();
    auto* pdata = pred.impl->data->data.get();
    auto* tdata = target.impl->data->data.get();

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
    if (!self.impl || !self.impl->data || !self.impl->data->grad)
        throw std::runtime_error("GradLogCosh: missing self grad");
    if (!pred.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = pred.numel_();

    auto* gdata = grad_input.impl->data->data.get();
    auto* pdata = pred.impl->data->data.get();
    auto* tdata = target.impl->data->data.get();

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