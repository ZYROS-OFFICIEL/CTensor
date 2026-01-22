#include "conv.h"
#include "ops.h" // Use optimized ops
#include <stdexcept>

// --- Conv1d ---
Conv1d::Conv1d(int in_c, int out_c, int k, int s, int p, DType dt)
    : in_channels(in_c), out_channels(out_c),
      kernel_size(k), stride(s), padding(p)
{
    weight = Tensor::rand({(size_t)out_c, (size_t)in_c, (size_t)k}, dt, true);
    bias   = Tensor::zeros({(size_t)out_c}, dt, true);
}

Tensor Conv1d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("Conv1d: null input");
    size_t batch = input.impl->shape[0];
    size_t width = input.impl->shape[2];
    int out_w = (int)(( (int)width + 2 * padding - kernel_size) / stride + 1);
    if (out_w <= 0) throw std::runtime_error("Conv1d: invalid output width");

    std::vector<size_t> out_shape = { batch, (size_t)out_channels, (size_t)out_w };
    bool req = input.requires_grad() || weight.requires_grad() || bias.requires_grad();
    Tensor output(out_shape, input._dtype(), req);

    // Naive implementation for Conv1d (can be optimized later)
    for (size_t b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ow = 0; ow < out_w; ++ow) {
                double acc = bias[(size_t)oc]; 
                for (size_t ic = 0; ic < (size_t)in_channels; ++ic) {
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

    if (req) {
        output.impl->grad_fn = std::make_shared<GradConv1d>(input, weight, bias, stride, padding);
    }
    return output;
}


// --- Conv2d (Optimized im2col) ---
Conv2d::Conv2d(int in_c, int out_c, int kh, int kw, int sh, int sw, int ph, int pw, DType dt)
    : in_channels(in_c), out_channels(out_c),
      kernel_size_h(kh), kernel_size_w(kw),
      stride_h(sh), stride_w(sw),
      padding_h(ph), padding_w(pw)
{
    weight = Tensor::rand({(size_t)out_c, (size_t)in_c, (size_t)kh, (size_t)kw}, dt, true);
    bias   = Tensor::zeros({(size_t)out_c}, dt, true);
}

Tensor Conv2d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("Conv2d: null input");
    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t height = input.impl->shape[2];
    size_t width  = input.impl->shape[3];

    int out_h = (int)(( (int)height + 2 * padding_h - kernel_size_h) / stride_h + 1);
    int out_w = (int)(( (int)width  + 2 * padding_w - kernel_size_w) / stride_w + 1);
    if (out_h <= 0 || out_w <= 0) throw std::runtime_error("Conv2d: invalid output dimensions");

    // 1. Flatten Weights
    size_t kernel_patch_size = in_c * kernel_size_h * kernel_size_w;
    Tensor w_flat = weight.reshape({(size_t)out_channels, kernel_patch_size});

    // 2. Im2Col
    size_t num_patches = batch * out_h * out_w;
    Tensor input_patches = Tensor::zeros({kernel_patch_size, num_patches}, input._dtype(), false);
    // --- RAW POINTER OPTIMIZATION START ---
    const float* in_ptr = (const float*)input.impl->data->data.get();
    float* patch_ptr = (float*)input_patches.impl->data->data.get();
    
    size_t s0 = input.impl->strides[0]; 
    size_t s1 = input.impl->strides[1]; 
    size_t s2 = input.impl->strides[2]; 
    size_t s3 = input.impl->strides[3]; 
    size_t patch_stride = num_patches; // Stride for row-major matrix

    #pragma omp parallel for
    for (size_t i = 0; i < num_patches; ++i) {
        size_t temp = i;
        size_t ow = temp % out_w; temp /= out_w;
        size_t oh = temp % out_h; temp /= out_h;
        size_t b  = temp;

        size_t patch_row = 0;
        for (size_t ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < kernel_size_h; ++kh) {
                for (int kw = 0; kw < kernel_size_w; ++kw) {
                    int ih = (int)oh * stride_h + kh - padding_h;
                    int iw = (int)ow * stride_w + kw - padding_w;
                    
                    float val = 0.0f;
                    if (ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width) {
                        size_t offset = b * s0 + ic * s1 + ih * s2 + iw * s3;
                        val = in_ptr[offset];
                    }
                    
                    // Direct Write: Bypasses operator[] checks/allocations
                    patch_ptr[patch_row * patch_stride + i] = val;
                    patch_row++;
                }
            }
        }
    }
    // 3. MatMul
    Tensor output_flat = matmul(w_flat, input_patches);

    // 4. Bias
    Tensor bias_col = bias.reshape({(size_t)out_channels, 1});
    output_flat = output_flat + bias_col;

    // 5. Reshape
    Tensor output_reshaped = output_flat.reshape({(size_t)out_channels, batch, (size_t)out_h, (size_t)out_w});
    Tensor output = output_reshaped.permute({1, 0, 2, 3});

    if (input.requires_grad() || weight.requires_grad() || bias.requires_grad()) {
        // Link to GradConv2d (which contains the optimized backward logic)
        output.impl->grad_fn = std::make_shared<GradConv2d>(input, weight, bias, stride_h, stride_w, padding_h, padding_w);
    }
    return output;
}

// --- Conv3d (Optimized im2col) ---
Conv3d::Conv3d(int in_c, int out_c, int kd, int kh, int kw, int sd, int sh, int sw, int pd, int ph, int pw, DType dt)
    : in_channels(in_c), out_channels(out_c),
      kernel_size_d(kd), kernel_size_h(kh), kernel_size_w(kw),
      stride_d(sd), stride_h(sh), stride_w(sw),
      padding_d(pd), padding_h(ph), padding_w(pw)
{
    weight = Tensor::rand({(size_t)out_c, (size_t)in_c, (size_t)kd, (size_t)kh, (size_t)kw}, dt, true);
    bias   = Tensor::zeros({(size_t)out_c}, dt, true);
}

Tensor Conv3d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("Conv3d: null input");
    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t depth = input.impl->shape[2];
    size_t height = input.impl->shape[3];
    size_t width  = input.impl->shape[4];

    int out_d = (int)(( (int)depth + 2 * padding_d - kernel_size_d) / stride_d + 1);
    int out_h = (int)(( (int)height + 2 * padding_h - kernel_size_h) / stride_h + 1);
    int out_w = (int)(( (int)width  + 2 * padding_w - kernel_size_w) / stride_w + 1);
    if (out_d <= 0 || out_h <= 0 || out_w <= 0) throw std::runtime_error("Conv3d: invalid output dimensions");

    size_t kernel_patch_size = in_c * kernel_size_d * kernel_size_h * kernel_size_w;
    Tensor w_flat = weight.reshape({(size_t)out_channels, kernel_patch_size});

    size_t num_patches = batch * out_d * out_h * out_w;
    Tensor input_patches = Tensor::zeros({kernel_patch_size, num_patches}, input._dtype(), false);

    #pragma omp parallel for
    for (size_t i = 0; i < num_patches; ++i) {
        size_t temp = i;
        size_t ow = temp % out_w; temp /= out_w;
        size_t oh = temp % out_h; temp /= out_h;
        size_t od = temp % out_d; temp /= out_d;
        size_t b  = temp;

        size_t patch_row = 0;
        for (size_t ic = 0; ic < in_c; ++ic) {
            for (int kd = 0; kd < kernel_size_d; ++kd) {
                for (int kh = 0; kh < kernel_size_h; ++kh) {
                    for (int kw = 0; kw < kernel_size_w; ++kw) {
                        int id = (int)od * stride_d + kd - padding_d;
                        int ih = (int)oh * stride_h + kh - padding_h;
                        int iw = (int)ow * stride_w + kw - padding_w;
                        
                        if (id >= 0 && id < (int)depth && ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width) {
                            input_patches[patch_row][i] = input[b][ic][(size_t)id][(size_t)ih][(size_t)iw];
                        }
                        patch_row++;
                    }
                }
            }
        }
    }

    Tensor output_flat = matmul(w_flat, input_patches);
    Tensor bias_col = bias.reshape({(size_t)out_channels, 1});
    output_flat = output_flat + bias_col;

    Tensor output_reshaped = output_flat.reshape({(size_t)out_channels, batch, (size_t)out_d, (size_t)out_h, (size_t)out_w});
    Tensor output = output_reshaped.permute({1, 0, 2, 3, 4});

    if (input.requires_grad() || weight.requires_grad() || bias.requires_grad()) {
        output.impl->grad_fn = std::make_shared<GradConv3d>(input, weight, bias, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w);
    }
    return output;
}


// --- Backward Implementations ---

void GradConv1d::backward(const Tensor& self) {
    if (!self.impl->grad->data) throw std::runtime_error("GradConv1d: missing self grad");
    Tensor grad_output = tensor_from_grad(self);
    Tensor grad_input  = Tensor::zeros(input.shape(), input._dtype(), false);
    Tensor grad_weight = Tensor::zeros(weight.shape(), weight._dtype(), false);
    Tensor grad_bias   = Tensor::zeros(bias.shape(), bias._dtype(), false);

    // Naive backward (consistent with naive forward)
    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t width = input.impl->shape[2];
    size_t out_c = weight.impl->shape[0];
    size_t k_len = weight.impl->shape[2];
    size_t out_w = grad_output.impl->shape[2];

    for (size_t b = 0; b < batch; ++b) {
        for (size_t oc = 0; oc < out_c; ++oc) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                double go = grad_output[b][oc][ow]; 
                // Bias
                double cur_b = grad_bias[oc];
                grad_bias[oc] = cur_b + go;

                for (size_t ic = 0; ic < in_c; ++ic) {
                    for (size_t kk = 0; kk < k_len; ++kk) {
                        int iw = (int)ow * stride + (int)kk - padding;
                        if (iw >= 0 && iw < (int)width) {
                            double cur_in = grad_input[b][ic][(size_t)iw];
                            double w_val  = weight[oc][ic][kk];
                            grad_input[b][ic][(size_t)iw] = cur_in + go * w_val;

                            double cur_w = grad_weight[oc][ic][kk];
                            double in_val = input[b][ic][(size_t)iw];
                            grad_weight[oc][ic][kk] = cur_w + go * in_val;
                        }
                    }
                }
            }
        }
    }
    accumulate_grad(input, grad_input);
    accumulate_grad(weight, grad_weight);
    accumulate_grad(bias, grad_bias);
}


// Optimized Backward for Conv2d
void GradConv2d::backward(const Tensor& self) {
    if (!self.impl->grad->data) throw std::runtime_error("GradConv2d: missing self grad");

    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t out_c = weight.impl->shape[0];
    size_t k_h   = weight.impl->shape[2];
    size_t k_w   = weight.impl->shape[3];
    size_t out_h = self.impl->shape[2];
    size_t out_w = self.impl->shape[3];
    size_t height = input.impl->shape[2];
    size_t width = input.impl->shape[3];

    size_t kernel_patch_size = in_c * k_h * k_w;
    size_t num_patches = batch * out_h * out_w;

    Tensor grad_output = tensor_from_grad(self);
    
    // FIX 1: Ensure Contiguous Memory before Reshape
    Tensor grad_output_permuted = grad_output.permute({1, 0, 2, 3});
    Tensor grad_output_contig = add_scalar(grad_output_permuted, 0.0); 
    Tensor grad_output_flat = grad_output_contig.reshape({out_c, num_patches});

    // 2. Bias Grad
    if (bias.requires_grad()) {
        Tensor grad_bias = sum(grad_output_flat, 1); 
        accumulate_grad(bias, grad_bias.reshape(bias.shape()));
    }

    // 3. Weight and Input Gradients
    if (weight.requires_grad() || input.requires_grad()) {
        Tensor input_patches = Tensor::zeros({kernel_patch_size, num_patches}, input._dtype(), false);
        
        // FIX 2: Use Raw Pointers (No operator[] inside OMP)
        const float* in_ptr = (const float*)input.impl->data->data.get();
        float* patch_ptr = (float*)input_patches.impl->data->data.get();
        
        size_t s0 = input.impl->strides[0];
        size_t s1 = input.impl->strides[1];
        size_t s2 = input.impl->strides[2];
        size_t s3 = input.impl->strides[3];
        size_t patch_stride = num_patches;

        #pragma omp parallel for
        for (size_t i = 0; i < num_patches; ++i) {
            size_t temp = i;
            size_t ow = temp % out_w; temp /= out_w;
            size_t oh = temp % out_h; temp /= out_h;
            size_t b  = temp;
            
            size_t patch_row = 0;
            for (size_t ic = 0; ic < in_c; ++ic) {
                for (int kh = 0; kh < k_h; ++kh) {
                    for (int kw = 0; kw < k_w; ++kw) {
                        int ih = (int)oh * stride_h + kh - padding_h;
                        int iw = (int)ow * stride_w + kw - padding_w;
                        
                        float val = 0.0f;
                        if (ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width) {
                            size_t offset = b * s0 + ic * s1 + ih * s2 + iw * s3;
                            val = in_ptr[offset];
                        }
                        patch_ptr[patch_row * patch_stride + i] = val;
                        patch_row++;
                    }
                }
            }
        }

        if (weight.requires_grad()) {
            Tensor patches_T = input_patches.t_();
            Tensor grad_w_flat = matmul(grad_output_flat, patches_T);
            accumulate_grad(weight, grad_w_flat.reshape(weight.shape()));
        }

        if (input.requires_grad()) {
            Tensor w_flat = weight.reshape({out_c, kernel_patch_size});
            Tensor w_flat_T = w_flat.t_();
            Tensor grad_input_patches = matmul(w_flat_T, grad_output_flat);
            
            Tensor grad_input = Tensor::zeros(input.shape(), input._dtype(), false);
            auto* gi_data = grad_input.impl->data->data.get();
            float* g_patches_ptr = (float*)grad_input_patches.impl->data->data.get();
            size_t gp_stride = num_patches; // Row-major stride

            #pragma omp parallel for
            for (size_t i = 0; i < num_patches; ++i) {
                size_t temp = i;
                size_t ow = temp % out_w; temp /= out_w;
                size_t oh = temp % out_h; temp /= out_h;
                size_t b  = temp;
                
                size_t patch_row = 0;
                for (size_t ic = 0; ic < in_c; ++ic) {
                    for (int kh = 0; kh < k_h; ++kh) {
                        for (int kw = 0; kw < k_w; ++kw) {
                            int ih = (int)oh * stride_h + kh - padding_h;
                            int iw = (int)ow * stride_w + kw - padding_w;
                            
                            if (ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width) {
                                float val = g_patches_ptr[patch_row * gp_stride + i];
                                
                                size_t idx = grad_input.impl->offset + 
                                    b*grad_input.impl->strides[0] + 
                                    ic*grad_input.impl->strides[1] + 
                                    ih*grad_input.impl->strides[2] + 
                                    iw*grad_input.impl->strides[3];
                                
                                float* ptr = (float*)gi_data + idx;
                                #pragma omp atomic
                                *ptr += val;
                            }
                            patch_row++;
                        }
                    }
                }
            }
            accumulate_grad(input, grad_input);
        }
    }
}


// Optimized Backward for Conv3d
void GradConv3d::backward(const Tensor& self) {
    if (!self.impl->grad->data) throw std::runtime_error("GradConv3d: missing self grad");

    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t depth  = input.impl->shape[2];
    size_t height = input.impl->shape[3];
    size_t width  = input.impl->shape[4];
    size_t out_c = weight.impl->shape[0];
    size_t k_d   = weight.impl->shape[2];
    size_t k_h   = weight.impl->shape[3];
    size_t k_w   = weight.impl->shape[4];
    
    size_t out_d = self.impl->shape[2];
    size_t out_h = self.impl->shape[3];
    size_t out_w = self.impl->shape[4];

    size_t kernel_patch_size = in_c * k_d * k_h * k_w;
    size_t num_patches = batch * out_d * out_h * out_w;

    Tensor grad_output = tensor_from_grad(self);
    
    Tensor grad_output_reshaped = grad_output.permute({1, 0, 2, 3, 4});
    Tensor grad_output_reshaped_con = add_scalar(grad_output_reshaped,0.0); // ensure contiguous
    Tensor grad_output_flat = grad_output_reshaped.reshape({out_c, num_patches});

    if (bias.requires_grad()) {
        Tensor grad_bias = sum(grad_output_flat, 1);
        accumulate_grad(bias, grad_bias.reshape(bias.shape()));
    }

    if (weight.requires_grad() || input.requires_grad()) {
        Tensor input_patches = Tensor::zeros({kernel_patch_size, num_patches}, input._dtype(), false);
        
        #pragma omp parallel for
        for (size_t i = 0; i < num_patches; ++i) {
            size_t temp = i;
            size_t ow = temp % out_w; temp /= out_w;
            size_t oh = temp % out_h; temp /= out_h;
            size_t od = temp % out_d; temp /= out_d;
            size_t b  = temp;
            size_t patch_row = 0;
            
            for (size_t ic = 0; ic < in_c; ++ic) {
                for (int kd = 0; kd < k_d; ++kd) {
                    for (int kh = 0; kh < k_h; ++kh) {
                        for (int kw = 0; kw < k_w; ++kw) {
                            int id = (int)od * stride_d + kd - padding_d;
                            int ih = (int)oh * stride_h + kh - padding_h;
                            int iw = (int)ow * stride_w + kw - padding_w;
                            
                            if (id >= 0 && id < (int)depth && ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width) {
                                input_patches[patch_row][i] = input[b][ic][(size_t)id][(size_t)ih][(size_t)iw];
                            }
                            patch_row++;
                        }
                    }
                }
            }
        }

        if (weight.requires_grad()) {
            Tensor patches_T = input_patches.t_();
            Tensor grad_w_flat = matmul(grad_output_flat, patches_T);
            accumulate_grad(weight, grad_w_flat.reshape(weight.shape()));
        }

        if (input.requires_grad()) {
            Tensor w_flat = weight.reshape({out_c, kernel_patch_size});
            Tensor w_flat_T = w_flat.t_();
            Tensor grad_input_patches = matmul(w_flat_T, grad_output_flat);
            
            Tensor grad_input = Tensor::zeros(input.shape(), input._dtype(), false);
            auto* gi_data = grad_input.impl->data->data.get();

            #pragma omp parallel for
            for (size_t i = 0; i < num_patches; ++i) {
                size_t temp = i;
                size_t ow = temp % out_w; temp /= out_w;
                size_t oh = temp % out_h; temp /= out_h;
                size_t od = temp % out_d; temp /= out_d;
                size_t b  = temp;
                size_t patch_row = 0;
                
                for (size_t ic = 0; ic < in_c; ++ic) {
                    for (int kd = 0; kd < k_d; ++kd) {
                        for (int kh = 0; kh < k_h; ++kh) {
                            for (int kw = 0; kw < k_w; ++kw) {
                                int id = (int)od * stride_d + kd - padding_d;
                                int ih = (int)oh * stride_h + kh - padding_h;
                                int iw = (int)ow * stride_w + kw - padding_w;
                                
                                if (id >= 0 && id < (int)depth && ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width) {
                                    double val = grad_input_patches[patch_row][i];
                                    size_t idx = grad_input.impl->offset + 
                                        b*grad_input.impl->strides[0] + 
                                        ic*grad_input.impl->strides[1] + 
                                        id*grad_input.impl->strides[2] + 
                                        ih*grad_input.impl->strides[3] + 
                                        iw*grad_input.impl->strides[4];
                                    
                                    float* ptr = (float*)gi_data + idx;
                                    #pragma omp atomic
                                    *ptr += (float)val;
                                }
                                patch_row++;
                            }
                        }
                    }
                }
            }
            accumulate_grad(input, grad_input);
        }
    }
}