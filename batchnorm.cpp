#include "batchnorm.h"
#include "ops1.h" // sum, mean, etc.
#include <stdexcept>
#include <cmath>

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
    
    // Input must be [N, C] or [N, C, H, W] (or [N, C, D, H, W])
    if (input.impl->shape[1] != (size_t)num_features) {
        throw std::runtime_error("BatchNorm: input channels != num_features");
    }

    int ndim = (int)input.impl->ndim;
    // Identify axes to reduce over. 
    // For [N, C, H, W], we reduce over N(0), H(2), W(3). We keep C(1).
    std::vector<int> reduce_axes;
    reduce_axes.push_back(0); // Batch dim
    for (int i = 2; i < ndim; ++i) reduce_axes.push_back(i);

    Tensor mean_val, var_val;

    if (training) {
        // 1. Calculate Mean and Variance of current batch
        // We don't have a single "mean over axes" function exposed cleanly in ops1.h yet 
        // (only `mean` over one dim).
        // But we can implement it using sum and dividing by count.
        
        // Calculate sum over reduction axes
        Tensor sum_val = input;
        // To avoid complex reduce logic here, let's loop over axes?
        // Or better: implement a helper or use the `mean` repeatedly?
        // `ops1.h` has `mean(t, dim)`.
        // Let's reduce repeatedly.
        // Note: reduces rank each time. Indices shift.
        // E.g. [N, C, H, W] -> mean(0) -> [C, H, W] -> mean(1) -> [C, W] ...
        // Wait, reducing dim 0 shifts C to dim 0. 
        // Correct strategy: Always reduce the *last* dimension involved, or handle permutations.
        // 
        // Alternative: Permute input to [C, N, H, W], flatten to [C, Rest], then mean/var over dim 1.
        // This is robust.
        
        // Permute: put Channel at dim 0.
        std::vector<size_t> perm(ndim);
        perm[0] = 1; // C
        perm[1] = 0; // N
        for (int i = 2; i < ndim; ++i) perm[i] = i; // H, W...
        
        Tensor input_perm = input.permute(perm);
        
        // Reshape to [C, -1]
        size_t C = num_features;
        size_t rest = input.numel() / C;
        Tensor input_flat = input_perm.reshape({C, rest});
        
        // Mean over dim 1
        mean_val = mean(input_flat, 1); // Shape [C]
        
        // Variance: mean((x - mean)^2)
        // We need to broadcast mean to [C, rest].
        // Manual broadcast for now since our ops broadcasting logic handles tensors.
        // input_flat - mean_val.reshape({C, 1})
        Tensor mean_reshaped = mean_val.reshape({C, 1});
        Tensor diff = input_flat - mean_reshaped;
        Tensor sq_diff = diff * diff;
        var_val = mean(sq_diff, 1); // Shape [C] (biased variance)
        
        // Unbiased variance correction for running stats? Standard is usually biased for BN forward
        // but unbiased for running stats update.
        // Let's use biased for the normalization (standard PyTorch behavior).

        // Update running stats (detached)
        // running_mean = (1-m)*running_mean + m*mean
        // running_var  = (1-m)*running_var  + m*var * (n/(n-1))
        // We'll stick to simple update for now.
        Tensor m_tensor = Tensor::full(running_mean.shape(), momentum, DType::Float32, false);
        
        // We need proper operator support or loops for these updates.
        // Let's do a manual loop for the update to be safe and easy.
        // Note: `running_mean` is a 1D tensor of size C.
        auto* rm_data = running_mean.impl->storage->data.get();
        auto* rv_data = running_var.impl->storage->data.get();
        for(size_t i=0; i<C; ++i) {
            double m_curr = read_scalar_at(mean_val.impl->storage->data.get(), i, DType::Float32);
            double v_curr = read_scalar_at(var_val.impl->storage->data.get(), i, DType::Float32);
            double rm_old = read_scalar_at(rm_data, i, DType::Float32);
            double rv_old = read_scalar_at(rv_data, i, DType::Float32);
            
            // Update
            double n_count = (double)rest;
            double unbiased_v = v_curr * (n_count / (n_count - 1.0));
            
            write_scalar_at(rm_data, i, DType::Float32, (1.0 - momentum) * rm_old + momentum * m_curr);
            write_scalar_at(rv_data, i, DType::Float32, (1.0 - momentum) * rv_old + momentum * unbiased_v);
        }

    } else {
        // Evaluation mode: use running stats
        mean_val = running_mean;
        var_val = running_var;
    }

    // --- Normalization ---
    // y = (x - mean) / sqrt(var + eps) * gamma + beta
    
    // We need to reshape mean/var/gamma/beta to broadcast correctly against [N, C, H, W].
    // They are all [C]. We need them to look like [1, C, 1, 1].
    std::vector<size_t> broadcast_shape(ndim, 1);
    broadcast_shape[1] = num_features;
    
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
        // Save state for backward
        // Note: we save x_centered and inv_std as they are reused in backward formulas
        // They are already broadcast-ready 4D tensors (or whatever input ndim is).
        output.impl->grad_fn = std::make_shared<GradBatchNorm>(input, gamma, beta, x_centered, inv_std);
    }

    return output;
}

void GradBatchNorm::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradBatchNorm: missing self grad");

    // Inputs:
    // self.grad (dL/dy): [N, C, H, W]
    // x_centered: [N, C, H, W]
    // inv_std: [1, C, 1, 1] (broadcastable)
    // gamma: [C]
    
    Tensor grad_output = tensor_from_grad(self);
    
    // Broadcast dimensions logic
    int ndim = (int)input.impl->ndim;
    std::vector<size_t> broadcast_shape(ndim, 1);
    broadcast_shape[1] = gamma.numel();
    Tensor gamma_bc = gamma.reshape(broadcast_shape);

    // 1. dL/dGamma = sum(grad_output * normalized)
    // normalized = x_centered * inv_std
    Tensor normalized = x_centered * inv_std;
    
    // dBeta and dGamma are summed over (N, H, W)
    // We need to sum over all dims EXCEPT dim 1.
    // Helper to sum over N, H, W...
    // We can repurpose the "permute -> reshape -> sum" trick.
    
    auto sum_exclude_dim1 = [&](const Tensor& t) -> Tensor {
        // Permute to [C, N, H, W]
        std::vector<size_t> perm(t.impl->ndim);
        perm[0] = 1; 
        perm[1] = 0;
        for (int i = 2; i < (int)t.impl->ndim; ++i) perm[i] = i;
        Tensor t_perm = t.permute(perm);
        
        // Flatten to [C, Rest]
        size_t C = t_perm.impl->shape[0];
        size_t Rest = t_perm.numel() / C;
        Tensor t_flat = t_perm.reshape({C, Rest});
        
        // Sum over dim 1 -> [C]
        return sum(t_flat, 1);
    };

    if (gamma.requires_grad()) {
        Tensor grad_gamma_full = grad_output * normalized;
        Tensor dgamma = sum_exclude_dim1(grad_gamma_full);
        accumulate_grad(gamma, dgamma);
    }

    if (beta.requires_grad()) {
        Tensor dbeta = sum_exclude_dim1(grad_output);
        accumulate_grad(beta, dbeta);
    }

    if (input.requires_grad()) {
        // dL/dx calculation for Batch Norm is specific.
        // Refer to standard derivation (e.g., from the original paper or reliable source).
        // dL/dx = (1 / (N * std)) * ( N * dL/dx_hat - sum(dL/dx_hat) - x_hat * sum(dL/dx_hat * x_hat) )
        // where x_hat is normalized input.
        // Wait, simpler form involves dGamma.
        
        // Let M = N * H * W (number of elements per channel)
        size_t C = gamma.numel();
        double M = (double)(input.numel() / C);
        
        // dx_hat = grad_output * gamma
        Tensor dx_hat = grad_output * gamma_bc;
        
        // term1 = sum(dx_hat) over axis (N,H,W) -> Shape [C] -> broadcast to [1, C, 1, 1]
        Tensor sum_dx_hat = sum_exclude_dim1(dx_hat).reshape(broadcast_shape);
        
        // term2 = sum(dx_hat * normalized) -> Shape [C] -> broadcast
        Tensor sum_dx_hat_norm = sum_exclude_dim1(dx_hat * normalized).reshape(broadcast_shape);
        
        // dL/dx = inv_std * (dx_hat - sum_dx_hat/M - normalized * sum_dx_hat_norm/M)
        //       = (inv_std / M) * (M * dx_hat - sum_dx_hat - normalized * sum_dx_hat_norm)
        
        Tensor term_brackets = (dx_hat * M) - sum_dx_hat - (normalized * sum_dx_hat_norm);
        Tensor grad_input = term_brackets * (inv_std / M);
        
        accumulate_grad(input, grad_input);
    }
}