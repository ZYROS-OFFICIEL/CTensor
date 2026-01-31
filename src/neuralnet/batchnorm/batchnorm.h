#pragma once
#include "core/tensor.h"
#include "core/autograd.h"

// --- BatchNorm1d / BatchNorm2d ---
// Applies Batch Normalization over a 2D or 4D input.
// 2D Input: [N, C] -> Normalizes over N for each C.
// 4D Input: [N, C, H, W] -> Normalizes over (N, H, W) for each C.

class BatchNorm {
public:
    int num_features;
    double eps;
    double momentum;
    bool training; // Flag to switch between training and inference modes

    // Learnable parameters
    Tensor gamma; // Scale
    Tensor beta;  // Shift

    // Running statistics (not learned via gradient descent)
    Tensor running_mean;
    Tensor running_var;

    BatchNorm(int num_features, double eps = 1e-5, double momentum = 0.1);

    // Switch modes
    void train() { training = true; }
    void eval() { training = false; }

    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }
};
// --- Autograd Node ---
struct GradBatchNorm : GradFn {
    Tensor input;
    Tensor gamma; // We need gamma for input gradient
    Tensor mean;  // Saved from forward
    Tensor var;   // Saved from forward (actually 1/sqrt(var+eps) usually saved as inv_std)
    double eps;
    
    // We save centered input (x - mean) and inv_std for efficiency
    Tensor x_centered; 
    Tensor inv_std;

    GradBatchNorm(const Tensor& input_, const Tensor& gamma_, const Tensor& x_centered_, const Tensor& inv_std_)
        : input(input_), gamma(gamma_), x_centered(x_centered_), inv_std(inv_std_) {
        parents = {input, gamma_}; 
        // But our GradFn structure usually stores references.
        // Beta is a separate leaf. We need to store it if we compute its grad here?
        // Actually, standard BN backward computes dgamma and dbeta and dx all at once.
        // So we should list them as parents if they require grad.
    }
    
    // We'll handle beta manually in backward or add it to parents.
    // To be safe/clean, let's store references to everything we need to update.
    // Re-defining constructor below.
    
    Tensor beta; // Stored just to have the reference for accumulation
    
    GradBatchNorm(const Tensor& input_, const Tensor& gamma_, const Tensor& beta_, 
                  const Tensor& x_centered_, const Tensor& inv_std_)
        : input(input_), gamma(gamma_), beta(beta_), x_centered(x_centered_), inv_std(inv_std_) {
        parents = {input, gamma, beta};
    }

    void backward(const Tensor& self) override;
};