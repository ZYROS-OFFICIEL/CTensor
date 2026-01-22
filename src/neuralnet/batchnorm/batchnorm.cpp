#include "batchnorm.h"
#include "ops_dispatch.h"
#include <stdexcept>
#include <cmath>
#include <numeric> 

// Constructor
BatchNorm::BatchNorm(int num_features, double eps, double momentum)
    : num_features(num_features), eps(eps), momentum(momentum), training(true)
{
    // Gamma initialized to 1, Beta to 0
    gamma = Tensor::full({(size_t)num_features}, 1.0, DType::Float32, true);
    beta = Tensor::zeros({(size_t)num_features}, DType::Float32, true);

    // Running stats initialized to 0 and 1 (mean=0, var=1)
    running_mean = Tensor::zeros({(size_t)num_features}, DType::Float32, false);
    running_var = Tensor::full({(size_t)num_features}, 1.0, DType::Float32, false);
}

Tensor BatchNorm::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("BatchNorm: null input");
    
    // Input must be [N, C] or [N, C, H, W] etc.
    if (input.impl->shape[1] != (size_t)num_features) {
        throw std::runtime_error("BatchNorm: input channels != num_features");
    }

    int ndim = (int)input.impl->ndim;
    
    // We need to reshape mean/var/gamma/beta to broadcast correctly.
    // They are [C]. We need them to be [1, C, 1, 1, ...] matching input ndim.
    std::vector<size_t> broadcast_shape(ndim, 1);
    broadcast_shape[1] = num_features;

    Tensor mean_val, var_val;

    if (training) {
        // --- TRAINING MODE ---
        
        // 1. Permute to put Channel at dim 0: [C, N, H, W...]
        std::vector<size_t> perm(ndim);
        perm[0] = 1; // C
        perm[1] = 0; // N
        for (int i = 2; i < ndim; ++i) perm[i] = i; // H, W...
        
        Tensor input_perm = input.permute(perm);
        
        // 2. Flatten to [C, Rest]
        size_t C = num_features;
        size_t rest = input.numel() / C;
        Tensor input_flat = input_perm.reshape({C, rest});
        
        // 3. Compute Mean and Var over dim 1 (the flattened spatial/batch dims)
        mean_val = mean(input_flat, 1); // Shape [C]
        
        // Variance: mean((x - mean)^2)
        // Broadcast mean manually for the subtraction
        Tensor mean_reshaped = mean_val.reshape({C, 1});
        Tensor diff = input_flat - mean_reshaped;
        Tensor sq_diff = diff * diff;
        var_val = mean(sq_diff, 1); // Shape [C] (biased variance)
        
        // 4. Update running stats (momentum update)
        // running_mean = (1-m)*running_mean + m*mean
        // running_var  = (1-m)*running_var  + m*unbiased_var
        
        // We use a manual loop for now as we lack some tensor ops
        auto* rm_data = running_mean.impl->storage->data.get();
        auto* rv_data = running_var.impl->storage->data.get();
        
        // Calculate unbiased variance factor: N / (N - 1)
        double n_count = (double)rest;
        double unbiased_factor = (n_count > 1.0) ? (n_count / (n_count - 1.0)) : 1.0;

        for(size_t i=0; i<C; ++i) {
            double m_curr = read_scalar_at(mean_val.impl->storage->data.get(), i, DType::Float32);
            double v_curr = read_scalar_at(var_val.impl->storage->data.get(), i, DType::Float32);
            
            double rm_old = read_scalar_at(rm_data, i, DType::Float32);
            double rv_old = read_scalar_at(rv_data, i, DType::Float32);
            
            double unbiased_v = v_curr * unbiased_factor;
            
            write_scalar_at(rm_data, i, DType::Float32, (1.0 - momentum) * rm_old + momentum * m_curr);
            write_scalar_at(rv_data, i, DType::Float32, (1.0 - momentum) * rv_old + momentum * unbiased_v);
        }

    } else {
        // --- INFERENCE MODE ---
        mean_val = running_mean;
        var_val = running_var;
    }

    // --- Normalization ---
    // y = (x - mean) / sqrt(var + eps) * gamma + beta
    
    // Reshape statistics for broadcasting
    Tensor mean_bc = mean_val.reshape(broadcast_shape);
    Tensor var_bc = var_val.reshape(broadcast_shape);
    Tensor gamma_bc = gamma.reshape(broadcast_shape);
    Tensor beta_bc = beta.reshape(broadcast_shape);

    // x_centered = input - mean
    Tensor x_centered = input - mean_bc;
    
    // inv_std = 1 / sqrt(var + eps)
    Tensor inv_std = pow_scalar(var_bc + eps, -0.5);
    
    // normalized = x_centered * inv_std
    Tensor normalized = x_centered * inv_std;
    
    // output = normalized * gamma + beta
    Tensor output = normalized * gamma_bc + beta_bc;

    if (training && (input.requires_grad() || gamma.requires_grad() || beta.requires_grad())) {
        output.impl->grad_fn = std::make_shared<GradBatchNorm>(input, gamma, beta, x_centered, inv_std);
    }

    return output;
}

void GradBatchNorm::backward(const Tensor& self) {
    if (!self.impl->grad->data) throw std::runtime_error("GradBatchNorm: missing self grad");

    Tensor grad_output = tensor_from_grad(self);
    
    int ndim = (int)input.impl->ndim;
    size_t C = gamma.numel();
    
    // Create broadcast shape: [1, C, 1, 1...]
    std::vector<size_t> broadcast_shape(ndim, 1);
    broadcast_shape[1] = C;
    
    Tensor gamma_bc = gamma.reshape(broadcast_shape);

    // normalized = x_centered * inv_std
    Tensor normalized = x_centered * inv_std;
    
    // Helper to sum over all dims EXCEPT dim 1 (Channel)
    auto sum_exclude_dim1 = [&](const Tensor& t) -> Tensor {
        // Permute to [C, Rest] logic
        std::vector<size_t> perm(t.impl->ndim);
        perm[0] = 1; 
        perm[1] = 0;
        for (int i = 2; i < (int)t.impl->ndim; ++i) perm[i] = i;
        
        Tensor t_perm = t.permute(perm);
        Tensor t_flat = t_perm.reshape({C, t_perm.numel() / C});
        
        // Sum over dim 1 -> [C]
        return sum(t_flat, 1);
    };

    // 1. Gradients for Gamma and Beta
    if (gamma.requires_grad()) {
        Tensor grad_gamma_full = grad_output * normalized;
        Tensor dgamma = sum_exclude_dim1(grad_gamma_full);
        // Reshape to [C] just in case, though sum returns [C]
        accumulate_grad(gamma, dgamma);
    }

    if (beta.requires_grad()) {
        Tensor dbeta = sum_exclude_dim1(grad_output);
        accumulate_grad(beta, dbeta);
    }

    // 2. Gradient for Input
    if (input.requires_grad()) {
        double M = (double)(input.numel() / C);
        
        // dx_hat = grad_output * gamma
        Tensor dx_hat = grad_output * gamma_bc;
        
        // Sums need to be broadcast back to [1, C, 1...] for elementwise math
        Tensor sum_dx_hat = sum_exclude_dim1(dx_hat).reshape(broadcast_shape);
        Tensor sum_dx_hat_norm = sum_exclude_dim1(dx_hat * normalized).reshape(broadcast_shape);
        
        // dL/dx formula for Batch Norm
        // = (inv_std / M) * (M * dx_hat - sum_dx_hat - normalized * sum_dx_hat_norm)
        
        Tensor term1 = dx_hat * M;
        Tensor term2 = sum_dx_hat;
        Tensor term3 = normalized * sum_dx_hat_norm;
        
        Tensor grad_input = (term1 - term2 - term3) * (inv_std / M);
        
        accumulate_grad(input, grad_input);
    }
}