#include "conv.h"
#include "ops1.h"   // for any ops helpers you have (abs, pow, etc.)
#include <stdexcept>

// Constructor
Conv1d::Conv1d(int in_c, int out_c, int k, int s, int p)
    : in_channels(in_c), out_channels(out_c),
      kernel_size(k), stride(s), padding(p)
{
    // weights: [out_c, in_c, kernel_size]
    weight = Tensor::rand({(size_t)out_c, (size_t)in_c, (size_t)k}, DType::Float32, true);
    bias   = Tensor::zeros({(size_t)out_c}, DType::Float32, true);
}

Conv2d::Conv2d(int in_c, int out_c, int kh, int kw, int sh, int sw, int ph, int pw)
    : in_channels(in_c), out_channels(out_c),
      kernel_size_h(kh), kernel_size_w(kw),
      stride_h(sh), stride_w(sw),
      padding_h(ph), padding_w(pw)
{
    // weights: [out_c, in_c, kernel_size_h, kernel_size_w]
    weight = Tensor::rand({(size_t)out_c, (size_t)in_c, (size_t)kh, (size_t)kw}, DType::Float32, true);
    bias   = Tensor::zeros({(size_t)out_c}, DType::Float32, true);
}

// Forward (Conv1d remains naive for simplicity)
Tensor Conv1d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("Conv1d::forward: null input");

    // input shape assumed [batch, in_channels, width]
    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t width = input.impl->shape[2];

    int out_w = (int)(( (int)width + 2 * padding - kernel_size) / stride + 1);
    if (out_w <= 0) throw std::runtime_error("Conv1d::forward: invalid output width");

    std::vector<size_t> out_shape = { batch, (size_t)out_channels, (size_t)out_w };
    bool req = input.requires_grad() || weight.requires_grad() || bias.requires_grad();

    Tensor output(out_shape, input._dtype(), req);

    // compute convolution (naive)
    for (size_t b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ow = 0; ow < out_w; ++ow) {
                // start with bias
                double acc = bias[(size_t)oc]; // proxy -> double
                for (size_t ic = 0; ic < in_c; ++ic) {
                    for (int k = 0; k < kernel_size; ++k) {
                        int iw = ow * stride + k - padding;
                        if (iw >= 0 && iw < (int)width) {
                            double in_val = input[b][ic][(size_t)iw];
                            double w_val  = weight[(size_t)oc][ic][(size_t)k];
                            acc += in_val * w_val;
                        }
                    }
                }
                output[b][(size_t)oc][(size_t)ow] = acc;
            }
        }
    }

    // attach grad fn if needed
    if (req) {
        output.impl->grad_fn = std::make_shared<GradConv1d>(input, weight, bias, stride, padding);
    }

    return output;
}

// --- HIGH-PERFORMANCE Conv2d::forward (im2col + MatMul) ---
Tensor Conv2d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("Conv2d::forward: null input");

    // input shape: [batch, in_c, height, width]
    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t height = input.impl->shape[2];
    size_t width  = input.impl->shape[3];

    // Calculate output dimensions
    int out_h = (int)(( (int)height + 2 * padding_h - kernel_size_h) / stride_h + 1);
    int out_w = (int)(( (int)width  + 2 * padding_w - kernel_size_w) / stride_w + 1);
    if (out_h <= 0 || out_w <= 0) throw std::runtime_error("Conv2d::forward: invalid output dimensions");

    // --- Step 1: Flatten the Kernels (Weight Matrix) ---
    // Reshape from [out_c, in_c, k_h, k_w] to [out_c, (in_c * k_h * k_w)]
    size_t kernel_patch_size = in_c * kernel_size_h * kernel_size_w;
    std::vector<size_t> w_flat_shape = {(size_t)out_channels, kernel_patch_size};
    Tensor w_flat = weight.reshape(w_flat_shape);

    // --- Step 2: Create the im2col "Patch Matrix" ---
    // Output shape will be [kernel_patch_size, (batch * out_h * out_w)]
    size_t num_patches = batch * out_h * out_w;
    Tensor input_patches = Tensor::zeros({kernel_patch_size, num_patches}, input._dtype(), false);

    // This loop is the im2col transformation
    size_t patch_col_idx = 0; // Current column in input_patches
    for (size_t b = 0; b < batch; ++b) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                // For this output pixel (oh, ow), extract its corresponding patch
                size_t patch_row_idx = 0; // Current row in this column
                for (size_t ic = 0; ic < in_c; ++ic) {
                    for (int kh = 0; kh < kernel_size_h; ++kh) {
                        for (int kw = 0; kw < kernel_size_w; ++kw) {
                            
                            int ih = oh * stride_h + kh - padding_h;
                            int iw = ow * stride_w + kw - padding_w;

                            // Handle padding: if (ih, iw) is outside, write 0.0
                            if (ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width) {
                                // Use proxy access for readability
                                input_patches[patch_row_idx][patch_col_idx] = input[b][ic][(size_t)ih][(size_t)iw];
                            }
                            // else: it's already 0 from Tensor::zeros
                            
                            patch_row_idx++;
                        }
                    }
                }
                patch_col_idx++;
            }
        }
    }

    // --- Step 3: The MatMul ---
    // C = W_flat @ Input_patches
    // Shapes: [out_c, kernel_patch_size] @ [kernel_patch_size, num_patches]
    // Result shape: [out_c, num_patches] or [out_c, (batch * out_h * out_w)]
    Tensor output_flat = matmul_(w_flat, input_patches);

    // --- Step 4: Add Bias ---
    // Reshape bias to [out_c, 1]
    Tensor bias_col = bias.reshape({(size_t)out_channels, 1});
    
    // Use operator+ broadcasting (assumes your add_ op handles broadcasting)
    // output_flat = [out_c, num_patches] + [out_c, 1]
    output_flat = output_flat + bias_col;

    // --- Step 5: Reshape Output ---
    // Reshape from [out_c, (batch * out_h * out_w)] to [out_c, batch, out_h, out_w]
    Tensor output_reshaped = output_flat.reshape({(size_t)out_channels, batch, (size_t)out_h, (size_t)out_w});
    
    // Permute to [batch, out_c, out_h, out_w]
    Tensor output = output_reshaped.permute({1, 0, 2, 3});

    // --- Attach Grad Fn (if needed) ---
    bool req = input.requires_grad() || weight.requires_grad() || bias.requires_grad();
    if (req) {
        // !! --- CRITICAL CAVEAT --- !!
        // We have changed the forward pass logic. The *original*
        // GradConv2d::backward function is now INCORRECT, as it
        // assumes the naive loop structure.
        // A complete implementation would require a new GradConv2dMatMul
        // that performs the backward pass using matrix multiplications
        // (i.e., grad_weight = grad_output @ patches.T)
        // (i.e., grad_patches = weights.T @ grad_output)
        //
        // For now, we link the old one, but it will NOT produce
        // correct gradients for this forward pass.
        // !! ------------------------- !!
        output.impl->grad_fn = std::make_shared<GradConv2d>(input, weight, bias, stride_h, stride_w, padding_h, padding_w);
    }
    
    return output;
}




// Backward
void GradConv1d::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradConv1d: missing self grad");

    // get grad of output (same shape as forward output)
    Tensor grad_output = tensor_from_grad(self);

    // prepare zero tensors for grads (same shapes as originals)
    Tensor grad_input  = Tensor::zeros(input.impl->shape, input._dtype(), false);
    Tensor grad_weight = Tensor::zeros(weight.impl->shape, weight._dtype(), false);
    Tensor grad_bias   = Tensor::zeros(bias.impl->shape, bias._dtype(), false);

    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t width = input.impl->shape[2];
    size_t out_c = weight.impl->shape[0];
    size_t k_len = weight.impl->shape[2];
    size_t out_w = grad_output.impl->shape[2];

    // accumulate gradients
    for (size_t b = 0; b < batch; ++b) {
        for (size_t oc = 0; oc < out_c; ++oc) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                double go = grad_output[b][oc][ow]; // grad at this output element

                // grad bias
                double cur_b = grad_bias[oc];
                grad_bias[oc] = cur_b + go;

                for (size_t ic = 0; ic < in_c; ++ic) {
                    for (size_t kk = 0; kk < k_len; ++kk) {
                        int iw = (int)ow * stride + (int)kk - padding;
                        if (iw >= 0 && iw < (int)width) {
                            // grad_input[b, ic, iw] += go * weight[oc, ic, kk]
                            double cur_in = grad_input[b][ic][(size_t)iw];
                            double w_val  = weight[oc][ic][kk];
                            grad_input[b][ic][(size_t)iw] = cur_in + go * w_val;

                            // grad_weight[oc, ic, kk] += go * input[b, ic, iw]
                            double cur_w = grad_weight[oc][ic][kk];
                            double in_val = input[b][ic][(size_t)iw];
                            grad_weight[oc][ic][kk] = cur_w + go * in_val;
                        }
                    }
                }
            }
        }
    }

    // accumulate into parents' grad buffers using your helper
    accumulate_grad(input, grad_input);
    accumulate_grad(weight, grad_weight);
    accumulate_grad(bias, grad_bias);
}

// --- HIGH-PERFORMANCE GradConv2d::backward (col2im + MatMul) ---
void GradConv2d::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradConv2d: missing self grad");

    // --- Get shapes and parameters from saved tensors ---
    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t height = input.impl->shape[2];
    size_t width  = input.impl->shape[3];
    size_t out_c = weight.impl->shape[0];
    size_t k_h   = weight.impl->shape[2];
    size_t k_w   = weight.impl->shape[3];
    size_t out_h = self.impl->shape[2]; // 'self' is the output tensor
    size_t out_w = self.impl->shape[3];

    size_t kernel_patch_size = in_c * k_h * k_w;
    size_t num_patches = batch * out_h * out_w;

    // --- Step 1: Get dL/dOutput (grad_output) ---
    // This is the incoming gradient from the next layer
    Tensor grad_output = tensor_from_grad(self); // Shape: [batch, out_c, out_h, out_w]

    // --- Step 2: "Un-permute" and "Un-reshape" grad_output ---
    // Forward was: permute({1, 0, 2, 3}) -> reshape
    // Backward is: permute({1, 0, 2, 3}) -> reshape
    Tensor grad_output_reshaped = grad_output.permute({1, 0, 2, 3});
    Tensor grad_output_flat = grad_output_reshaped.reshape({out_c, num_patches});
    // grad_output_flat is dL/dOutput_flat, Shape: [out_c, num_patches]

    // --- Step 3: Gradient w.r.t Bias ---
    // Forward: output_flat = ... + bias_col
    // Backward: dL/dBias = sum(dL/dOutput_flat) along the broadcasted dimension (dim 1)
    if (bias.requires_grad()) {
        Tensor grad_bias = sum(grad_output_flat, 1); // Sums along dim 1
        // grad_bias shape is [out_c, 1] or [out_c]. Ensure it matches bias shape [out_c]
        accumulate_grad(bias, grad_bias.reshape(bias.shape()));
    }

    // --- Step 4: Re-create w_flat and input_patches (needed for other grads) ---
    // Re-flatten weights
    Tensor w_flat = weight.reshape({out_c, kernel_patch_size});

    // Re-create input_patches (im2col)
    // This is expensive, but necessary as we didn't store it in the GradConv2d struct
    Tensor input_patches = Tensor::zeros({kernel_patch_size, num_patches}, input._dtype(), false);
    size_t patch_col_idx = 0;
    for (size_t b = 0; b < batch; ++b) {
        for (size_t oh = 0; oh < out_h; ++oh) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                size_t patch_row_idx = 0;
                for (size_t ic = 0; ic < in_c; ++ic) {
                    for (size_t kh = 0; kh < k_h; ++kh) {
                        for (size_t kw = 0; kw < k_w; ++kw) {
                            int ih = (int)oh * stride_h + (int)kh - padding_h;
                            int iw = (int)ow * stride_w + (int)kw - padding_w;
                            if (ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width) {
                                input_patches[patch_row_idx][patch_col_idx] = input[b][ic][(size_t)ih][(size_t)iw];
                            }
                            patch_row_idx++;
                        }
                    }
                }
                patch_col_idx++;
            }
        }
    }
    
    // --- Step 5: Gradient w.r.t Weight ---
    // Forward: output_flat = w_flat @ input_patches
    // Backward: dL/dW_flat = dL/dOutput_flat @ input_patches.T
    if (weight.requires_grad()) {
        Tensor input_patches_T = input_patches.t_(); // Shape: [num_patches, k_patch_size]
        Tensor grad_w_flat = matmul_(grad_output_flat, input_patches_T); // Shape: [out_c, k_patch_size]
        
        // Reshape grad_w_flat back to original weight shape and accumulate
        Tensor grad_weight = grad_w_flat.reshape(weight.shape());
        accumulate_grad(weight, grad_weight);
    }

    // --- Step 6: Gradient w.r.t Input ---
    // Forward: output_flat = w_flat @ input_patches
    // Backward: dL/dInput_patches = w_flat.T @ dL/dOutput_flat
    if (input.requires_grad()) {
        Tensor w_flat_T = w_flat.t_(); // Shape: [k_patch_size, out_c]
        Tensor grad_input_patches = matmul_(w_flat_T, grad_output_flat); // Shape: [k_patch_size, num_patches]

        // Now, perform "col2im" to map grad_input_patches back to grad_input
        Tensor grad_input = Tensor::zeros(input.shape(), input._dtype(), false);
        
        patch_col_idx = 0; // Reset column index
        for (size_t b = 0; b < batch; ++b) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    size_t patch_row_idx = 0; // Reset row index
                    for (size_t ic = 0; ic < in_c; ++ic) {
                        for (size_t kh = 0; kh < k_h; ++kh) {
                            for (size_t kw = 0; kw < k_w; ++kw) {
                                int ih = (int)oh * stride_h + (int)kh - padding_h;
                                int iw = (int)ow * stride_w + (int)kw - padding_w;
                                
                                // Check if this was a valid (non-padded) input
                                if (ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width) {
                                    // Read from grad_input_patches
                                    double grad_val = grad_input_patches[patch_row_idx][patch_col_idx];
                                    
                                    // Add (accumulate) to grad_input
                                    // Note: This needs to be an atomic add if parallelized
                                    double cur_val = grad_input[b][ic][(size_t)ih][(size_t)iw];
                                    grad_input[b][ic][(size_t)ih][(size_t)iw] = cur_val + grad_val;
                                }
                                patch_row_idx++;
                            }
                        }
                    }
                    patch_col_idx++;
                }
            }
        }
        
        // Finally, accumulate the computed gradient into the original input
        accumulate_grad(input, grad_input);
    }
}