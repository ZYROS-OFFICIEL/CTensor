#include <tensor.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include "autograd.h"
#include <immintrin.h>
#include <cstring>
#include "loss.h"
#include "ops_dispatch.h"

Tensor Loss::MSE(const Tensor& pred_, const Tensor& target_) {
    if (!pred_.impl || !target_.impl)
        throw std::runtime_error("Loss::MSE: null tensor implementation");

    if (pred_.impl->ndim != target_.impl->ndim)
        throw std::runtime_error("Loss::MSE: dimension mismatch");

    bool req = pred_.requires_grad();
    Tensor result({1}, pred_.impl->dtype, req);

    Tensor temp = pow_scalar(pred_ - target_, 2);
    Tensor summed = sum(temp, -1);

    double mse_value = read_scalar_at(summed.impl->data->data.get(), 0, summed._dtype()) 
                       / static_cast<double>(pred_.numel_());

    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), mse_value);

    if (req) result.impl->grad_fn = std::make_shared<GradMSE>(pred_, target_);
    return result;
}

Tensor Loss::MAE(const Tensor& pred, const Tensor& target,std::string reduction){
    if (!pred.impl || !target.impl)
        throw std::runtime_error("Loss::MAE: null tensor implementation");

    if (pred.impl->ndim != target.impl->ndim)
        throw std::runtime_error("Loss::MAE: dimension mismatch");

    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    Tensor temp = abs(pred - target);
    Tensor summed = sum(temp, -1);

    double mae_value = read_scalar_at(summed.impl->data->data.get(), 0, summed._dtype());
    if(reduction == "mean") {
        mae_value /= static_cast<double>(pred.numel_());
    }

    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), mae_value);

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

    Tensor diff = pred - target;
    Tensor abs_diff = abs(diff);
    Tensor linear = delta * (abs_diff - 0.5 * delta);
    Tensor quadratic = 0.5 * diff * diff;

    Tensor huber_loss = (abs_diff <= delta) * quadratic + (abs_diff > delta) * linear;
    
    Tensor summed = sum(huber_loss, -1);
    double huber_value = read_scalar_at(summed.impl->data->data.get(), 0, summed._dtype());
    if(reduction == "mean") {
        huber_value /= static_cast<double>(pred.numel_());
    }
    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), huber_value);
    
    if (req) {
        result.impl->grad_fn = std::make_shared<GradHuberLoss>(pred, target, reduction, delta);
    }
    return result;
}

Tensor Loss::CrossEntropy(const Tensor& pred_, const Tensor& target_, std::string reduction) {
    if (!pred_.impl || !target_.impl) throw std::runtime_error("Loss::CrossEntropy: null tensor");
    if (pred_.impl->ndim != 2) throw std::runtime_error("CrossEntropy: Pred must be 2D [Batch, Classes]");
    
    bool req = pred_.requires_grad();
    Tensor result({1}, pred_.impl->dtype, req);

    Tensor max_vals = max(pred_, 1).reshape({pred_.shape()[0], 1});
    Tensor shifted = pred_ - max_vals; 

    Tensor exp_vals = exp(shifted);
    Tensor sum_exp = sum(exp_vals, 1).reshape({pred_.shape()[0], 1});
    Tensor log_sum_exp = ln(sum_exp);

    Tensor log_probs = shifted - log_sum_exp; 
    Tensor picked = log_probs.gather(target_, 1); 
    
    Tensor nll = picked * -1.0;

    Tensor reduced;
    if (reduction == "mean") reduced = mean(nll);
    else reduced = sum(nll);
    
    double val = read_scalar_at(reduced.impl->data->data.get(), 0, reduced._dtype());
    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), val);

    if (req) result.impl->grad_fn = std::make_shared<GradCrossEntropy>(pred_, target_, reduction);
    return result;
}

Tensor Loss::LogCosh(const Tensor& pred, const Tensor& target,std::string reduction){
    if (!pred.impl || !target.impl) throw std::runtime_error("Loss::LogCosh: null tensor implementation");
    if (pred.impl->ndim != target.impl->ndim) throw std::runtime_error("Loss::LogCosh: dimension mismatch");

    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    Tensor diff = pred - target;
    Tensor log_cosh_loss = ln(cosh(diff));

    Tensor summed = sum(log_cosh_loss, -1);
    double log_cosh_value = read_scalar_at(summed.impl->data->data.get(), 0, summed._dtype());

    if(reduction == "mean") log_cosh_value /= static_cast<double>(pred.numel_());

    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), log_cosh_value);

    if (req) result.impl->grad_fn = std::make_shared<GradLogCosh>(pred, target, reduction);
    return result;
}

Tensor Loss::BCE(const Tensor& pred, const Tensor& target,std::string reduction){
    if (!pred.impl || !target.impl) throw std::runtime_error("Loss::BCE: null tensor implementation");
    if (pred.impl->ndim != target.impl->ndim) throw std::runtime_error("Loss::BCE: dimension mismatch");
    
    Tensor t_max = max(pred); 
    Tensor t_min = min(pred); 
    
    double max_val = read_scalar_at(t_max.impl->data->data.get(), 0, t_max._dtype());
    double min_val = read_scalar_at(t_min.impl->data->data.get(), 0, t_min._dtype());
    if (max_val > 1.0 || min_val < 0.0) throw std::runtime_error("BCE: input must be in [0, 1]");

    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    Tensor log_pred = ln(pred + 1e-12);               
    Tensor log_one_minus_pred = ln(1 - pred + 1e-12); 

    Tensor bce_loss = - (target * log_pred + (1 - target) * log_one_minus_pred);

    Tensor summed = sum(bce_loss, -1);
    double bce_value = read_scalar_at(summed.impl->data->data.get(), 0, summed._dtype());

    if(reduction == "mean") bce_value /= static_cast<double>(pred.numel_());

    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), bce_value);

    if (req) result.impl->grad_fn = std::make_shared<GradBCE>(pred, target, reduction);
    return result;
}

Tensor Loss::KLDiv(const Tensor& pred, const Tensor& target,std::string reduction){
    if (!pred.impl || !target.impl) throw std::runtime_error("Loss::KLDiv: null tensor implementation");
    if (pred.impl->ndim != target.impl->ndim) throw std::runtime_error("Loss::KLDiv: dimension mismatch");

    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    Tensor log_pred = ln(pred + 1e-12); 
    Tensor kl_loss = target * (log_pred - ln(target + 1e-12));

    Tensor summed = sum(kl_loss, -1);
    double kl_value = read_scalar_at(summed.impl->data->data.get(), 0, summed._dtype());

    if(reduction == "mean") kl_value /= static_cast<double>(pred.numel_());

    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), kl_value);

    if (req) result.impl->grad_fn = std::make_shared<GradKLDiv>(pred, target, reduction);
    return result;
}

Tensor Loss::NLLLoss(const Tensor& pred, const Tensor& target, std::string reduction){
    if (!pred.impl || !target.impl) throw std::runtime_error("Loss::NLLLoss: null tensor");

    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    Tensor picked = pred.gather(target, 1);  
    Tensor loss = picked * -1.0; 

    Tensor reduced;
    if(reduction == "mean") reduced = mean(loss);
    else reduced = sum(loss);

    double val = read_scalar_at(reduced.impl->data->data.get(), 0, reduced._dtype());
    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), val);
    if (req) result.impl->grad_fn = std::make_shared<GradNLLLoss>(pred, target, reduction);
    return result;
}

Tensor HingeLoss(const Tensor& pred, const Tensor& target,std::string reduction){
    if (!pred.impl || !target.impl) throw std::runtime_error("Loss::HingeLoss: null tensor implementation");
    if (pred.impl->ndim != target.impl->ndim) throw std::runtime_error("Loss::HingeLoss: dimension mismatch");

    bool req = pred.requires_grad();
    Tensor result({1}, pred.impl->dtype, req);

    Tensor one = Tensor::full(pred.shape(), 1.0, pred._dtype(), false);
    Tensor margin = one - target * pred;
    Tensor hinge_loss = relu(margin);

    Tensor summed = sum(hinge_loss);
    double hinge_value = read_scalar_at(summed.impl->data->data.get(), 0, summed._dtype());
    if(reduction == "mean") hinge_value /= static_cast<double>(pred.numel_());

    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), hinge_value);

    if (req) result.impl->grad_fn = std::make_shared<GradHingeLoss>(pred, target, reduction);
    return result;
}

Tensor MarginRankingLoss(const Tensor& input1, const Tensor& input2, const Tensor& target, double margin, std::string reduction) {
    if (!input1.impl || !input2.impl || !target.impl) throw std::runtime_error("Loss::MarginRankingLoss: null tensor implementation");
    if (input1.impl->ndim != input2.impl->ndim || input1.impl->ndim != target.impl->ndim) throw std::runtime_error("Loss::MarginRankingLoss: dimension mismatch");

    bool req = input1.requires_grad();
    Tensor result({1}, input1.impl->dtype, req);

    Tensor diff = input1 - input2;
    Tensor margin_tensor = margin - target * diff;
    Tensor loss = relu(margin_tensor);

    Tensor summed = sum(loss);
    double loss_value = read_scalar_at(summed.impl->data->data.get(), 0, summed._dtype());

    if(reduction == "mean") loss_value /= static_cast<double>(input1.numel_());

    write_scalar_at(result.impl->data->data.get(), 0, result._dtype(), loss_value);

    if (req) result.impl->grad_fn = std::make_shared<GradMarginRankingLoss>(input1, input2, target, margin, reduction);
    return result;
}

void GradMSE::backward(const Tensor& self) {
    if (!self.impl || !self.impl->data || !self.impl->grad) throw std::runtime_error("GradMSE: missing self grad");
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
    if (!self.impl || !self.impl->data || !self.impl->grad) throw std::runtime_error("GradMAE: missing self grad");
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
        write_scalar_at(gdata, i, grad_input._dtype(), sign);
    }

    if (reduction == "mean") grad_input = grad_input / static_cast<double>(n);
    accumulate_grad(pred, grad_input);
}

void GradHuberLoss::backward(const Tensor& self) {
    if (!self.impl || !self.impl->data || !self.impl->grad) throw std::runtime_error("GradHuberLoss: missing self grad");
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
        double grad_val = (abs_diff <= delta) ? diff : delta * ((diff > 0) ? 1.0 : -1.0);
        write_scalar_at(gdata, i, grad_input._dtype(), grad_val);
    }

    if (reduction == "mean") grad_input = grad_input / static_cast<double>(n);
    accumulate_grad(pred, grad_input);
}


// --- CROSS ENTROPY BACKWARD (MANUAL NUMERIC IMPLEMENTATION) ---
void GradCrossEntropy::backward(const Tensor& self) {
    if (!self.impl->grad) throw std::runtime_error("GradCrossEntropy: missing self grad");
    if (!pred.requires_grad()) return;

    Tensor grad_input = Tensor::zeros(pred.shape(), pred._dtype(), false);
    
    size_t batch_size = pred.shape()[0];
    size_t num_classes = pred.shape()[1];
    
    const float* p_ptr = (const float*)pred.impl->data->data.get();
    const int32_t* t_ptr = (const int32_t*)target.impl->data->data.get();
    float* g_ptr = (float*)grad_input.impl->data->data.get();
    
    // ==========================================
    // CRITICAL BUG FIX 1: Read from DATA, not from Tensorimpl V-Table!
    // ==========================================
    double grad_out = read_scalar_at(self.impl->grad->data->data.get(), 0, self._dtype());

    if (reduction == "mean") {
        grad_out /= static_cast<double>(batch_size);
    }
    float scale = (float)grad_out;

    #pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b) {
        const float* row_pred = p_ptr + b * num_classes;
        float* row_grad = g_ptr + b * num_classes;
        
        float max_val = -1e9f; 
        for(size_t c=0; c<num_classes; ++c) {
            if(row_pred[c] > max_val) max_val = row_pred[c];
        }
        
        float sum_exp = 0.0f;
        for(size_t c=0; c<num_classes; ++c) {
            float val = std::exp(row_pred[c] - max_val);
            row_grad[c] = val; 
            sum_exp += val;
        }
        
        int label = t_ptr[b];
        float inv_sum = (sum_exp > 1e-12f) ? (1.0f / sum_exp) : 0.0f;

        for(size_t c=0; c<num_classes; ++c) {
            float prob = row_grad[c] * inv_sum; 
            if (static_cast<int>(c) == label) prob -= 1.0f;
            row_grad[c] = prob * scale;
        }
    }

    accumulate_grad(pred, grad_input);
}


void GradBCE::backward(const Tensor& self) {
    if (!self.impl || !self.impl->data || !self.impl->grad) throw std::runtime_error("GradBCE: missing self grad");
    if (!pred.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = pred.numel_();

    auto* gdata = grad_input.impl->data->data.get();
    auto* pdata = pred.impl->data->data.get();
    auto* tdata = target.impl->data->data.get();

    for (size_t i = 0; i < n; ++i) {
        double p = read_scalar_at(pdata, i, pred._dtype());
        double t = read_scalar_at(tdata, i, target._dtype());
        double grad_val = (p - t) / ( (p * (1 - p)) + 1e-12 ); 
        write_scalar_at(gdata, i, grad_input._dtype(), grad_val);
    }

    if (reduction == "mean") grad_input = grad_input / static_cast<double>(n);
    accumulate_grad(pred, grad_input);
}

void GradLogCosh::backward(const Tensor& self) {
    if (!self.impl || !self.impl->data || !self.impl->grad) throw std::runtime_error("GradLogCosh: missing self grad");
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
        double grad_val = std::tanh(diff); 
        write_scalar_at(gdata, i, grad_input._dtype(), grad_val);
    }

    if (reduction == "mean") grad_input = grad_input / static_cast<double>(n);
    accumulate_grad(pred, grad_input);
}


void GradKLDiv::backward(const Tensor& self) {
    if (!self.impl || !self.impl->data || !self.impl->grad) throw std::runtime_error("GradKLDiv: missing self grad");
    if (!pred.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = pred.numel_();

    auto* gdata = grad_input.impl->data->data.get();
    auto* pdata = pred.impl->data->data.get();
    auto* tdata = target.impl->data->data.get();

    for (size_t i = 0; i < n; ++i) {
        double p = read_scalar_at(pdata, i, pred._dtype());
        double t = read_scalar_at(tdata, i, target._dtype());
        double grad_val = ( (p > 1e-12) ? (t / (p + 1e-12)) : 0.0 ); 
        write_scalar_at(gdata, i, grad_input._dtype(), grad_val);
    }

    if (reduction == "mean") grad_input = grad_input / static_cast<double>(n);
    accumulate_grad(pred, grad_input);
}

void GradNLLLoss::backward(const Tensor& self) {
    if (!self.impl->grad) throw std::runtime_error("GradNLLLoss: missing self grad");
    if (!pred.requires_grad()) return;

    Tensor grad_input = Tensor::zeros(pred.shape(), pred._dtype(), false);
    
    size_t batch_size = pred.shape()[0];
    size_t num_classes = pred.shape()[1];
    int32_t* t_ptr = (int32_t*)target.impl->data->data.get();
    float* g_ptr = (float*)grad_input.impl->data->data.get();

    #pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b) {
        int label = t_ptr[b];
        if (label >= 0 && label < (int)num_classes) {
            size_t idx = b * num_classes + label;
            g_ptr[idx] = -1.0f;
        }
    }

    // ==========================================
    // CRITICAL BUG FIX 1: Read from DATA, not from Tensorimpl V-Table!
    // ==========================================
    double grad_out = read_scalar_at(self.impl->grad->data->data.get(), 0, self._dtype());
    
    if (reduction == "mean") grad_out /= static_cast<double>(batch_size);

    if (grad_out != 1.0) {
        size_t n = grad_input.numel();
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) g_ptr[i] *= (float)grad_out;
    }

    accumulate_grad(pred, grad_input);
}


void GradHingeLoss::backward(const Tensor& self) {
    if (!self.impl || !self.impl->data || !self.impl->grad) throw std::runtime_error("GradHingeLoss: missing self grad");
    if (!pred.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = pred.numel_();

    auto* gdata = grad_input.impl->data->data.get();
    auto* pdata = pred.impl->data->data.get();
    auto* tdata = target.impl->data->data.get();

    for (size_t i = 0; i < n; ++i) {
        double p = read_scalar_at(pdata, i, pred._dtype());
        double t = read_scalar_at(tdata, i, target._dtype());
        double margin = 1.0 - t * p;
        double grad_val = (margin > 0) ? -t : 0.0; 
        write_scalar_at(gdata, i, grad_input._dtype(), grad_val);
    }

    if (reduction == "mean") grad_input = grad_input / static_cast<double>(n);
    accumulate_grad(pred, grad_input);
}

void GradMarginRankingLoss::backward(const Tensor& self) {
    if (!self.impl || !self.impl->data || !self.impl->grad) throw std::runtime_error("GradMarginRankingLoss: missing self grad");
    if (!input1.requires_grad() && !input2.requires_grad()) return;

    Tensor grad_input = tensor_from_grad(self);
    size_t n = input1.numel_();

    auto* gdata = grad_input.impl->data->data.get();
    auto* in1data = input1.impl->data->data.get();
    auto* in2data = input2.impl->data->data.get();
    auto* tdata = target.impl->data->data.get();

    for (size_t i = 0; i < n; ++i) {
        double in1 = read_scalar_at(in1data, i, input1._dtype());
        double in2 = read_scalar_at(in2data, i, input2._dtype());
        double t = read_scalar_at(tdata, i, target._dtype());
        double margin_val = margin - t * (in1 - in2);
        double grad_val = (margin_val > 0) ? -t : 0.0; 

        if (input1.requires_grad()) {
            write_scalar_at(gdata, i, grad_input._dtype(), grad_val);
            accumulate_grad(input1, grad_input);
        }
        if (input2.requires_grad()) {
            write_scalar_at(gdata, i, grad_input._dtype(), -grad_val);
            accumulate_grad(input2, grad_input);
        }
    }

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
}