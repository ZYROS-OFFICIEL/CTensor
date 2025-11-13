#include "conv.h"
#include "ops1.h"   // for any ops helpers you have (abs, pow, etc.)
#include <stdexcept>


//------------Helpers---------------------------------------------
Tensor im2col_2d(const Tensor& input,
                 int kernel_h, int kernel_w,
                 int stride_h, int stride_w,
                 int pad_h, int pad_w)
{
    // input [batch, in_c, H, W]
    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t H     = input.impl->shape[2];
    size_t W     = input.impl->shape[3];

    int out_h = (int)(( (int)H + 2 * pad_h - kernel_h) / stride_h + 1);
    int out_w = (int)(( (int)W + 2 * pad_w - kernel_w) / stride_w + 1);
    if (out_h <= 0 || out_w <= 0) throw std::runtime_error("im2col_2d: invalid out dims");

    size_t patch_h = in_c * kernel_h * kernel_w;
    size_t num_patches = batch * out_h * out_w;
    Tensor patches = Tensor::zeros({patch_h, num_patches}, input._dtype(), false);

    size_t col = 0;
    for (size_t b = 0; b < batch; ++b) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                size_t row = 0;
                for (size_t ic = 0; ic < in_c; ++ic) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = oh * stride_h + kh - pad_h;
                            int iw = ow * stride_w + kw - pad_w;
                            double v = 0.0;
                            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                                v = input[b][ic][(size_t)ih][(size_t)iw];
                            }
                            patches[row][col] = v;
                            ++row;
                        }
                    }
                }
                ++col;
            }
        }
    }
    return patches;
}

// col2im for 2D: inverse of im2col; accumulates contributions.
// grad_patches shape: [patch_h, num_patches] ; we map back into grad_input (batch,in_c,H,W)
void col2im_2d(const Tensor& grad_patches,
               Tensor& grad_input,
               int kernel_h, int kernel_w,
               int stride_h, int stride_w,
               int pad_h, int pad_w)
{
    // grad_input must be zeros-initialized on entry (we will accumulate)
    size_t batch = grad_input.impl->shape[0];
    size_t in_c  = grad_input.impl->shape[1];
    size_t H     = grad_input.impl->shape[2];
    size_t W     = grad_input.impl->shape[3];

    int out_h = (int)(( (int)H + 2 * pad_h - kernel_h) / stride_h + 1);
    int out_w = (int)(( (int)W + 2 * pad_w - kernel_w) / stride_w + 1);

    size_t col = 0;
    for (size_t b = 0; b < batch; ++b) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                size_t row = 0;
                for (size_t ic = 0; ic < in_c; ++ic) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = oh * stride_h + kh - pad_h;
                            int iw = ow * stride_w + kw - pad_w;
                            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                                double addv = grad_patches[row][col];
                                double cur = grad_input[b][ic][(size_t)ih][(size_t)iw];
                                grad_input[b][ic][(size_t)ih][(size_t)iw] = cur + addv;
                            }
                            ++row;
                        }
                    }
                }
                ++col;
            }
        }
    }
}

// im2col for 3D (depth, height, width)
Tensor im2col_3d(const Tensor& input,
                 int kernel_d, int kernel_h, int kernel_w,
                 int stride_d, int stride_h, int stride_w,
                 int pad_d, int pad_h, int pad_w)
{
    // input [batch, in_c, D, H, W]
    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t D     = input.impl->shape[2];
    size_t H     = input.impl->shape[3];
    size_t W     = input.impl->shape[4];

    int out_d = (int)(( (int)D + 2 * pad_d - kernel_d) / stride_d + 1);
    int out_h = (int)(( (int)H + 2 * pad_h - kernel_h) / stride_h + 1);
    int out_w = (int)(( (int)W + 2 * pad_w - kernel_w) / stride_w + 1);

    if (out_d <= 0 || out_h <= 0 || out_w <= 0) throw std::runtime_error("im2col_3d: invalid out dims");

    size_t patch_h = in_c * kernel_d * kernel_h * kernel_w;
    size_t num_patches = batch * out_d * out_h * out_w;
    Tensor patches = Tensor::zeros({patch_h, num_patches}, input._dtype(), false);

    size_t col = 0;
    for (size_t b = 0; b < batch; ++b) {
        for (int od = 0; od < out_d; ++od) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    size_t row = 0;
                    for (size_t ic = 0; ic < in_c; ++ic) {
                        for (int kd = 0; kd < kernel_d; ++kd) {
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    int id = od * stride_d + kd - pad_d;
                                    int ih = oh * stride_h + kh - pad_h;
                                    int iw = ow * stride_w + kw - pad_w;
                                    double v = 0.0;
                                    if (id >= 0 && id < (int)D && ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                                        v = input[b][ic][(size_t)id][(size_t)ih][(size_t)iw];
                                    }
                                    patches[row][col] = v;
                                    ++row;
                                }
                            }
                        }
                    }
                    ++col;
                }
            }
        }
    }
    return patches;
}

void col2im_3d(const Tensor& grad_patches,
               Tensor& grad_input,
               int kernel_d, int kernel_h, int kernel_w,
               int stride_d, int stride_h, int stride_w,
               int pad_d, int pad_h, int pad_w)
{
    size_t batch = grad_input.impl->shape[0];
    size_t in_c  = grad_input.impl->shape[1];
    size_t D     = grad_input.impl->shape[2];
    size_t H     = grad_input.impl->shape[3];
    size_t W     = grad_input.impl->shape[4];

    int out_d = (int)(( (int)D + 2 * pad_d - kernel_d) / stride_d + 1);
    int out_h = (int)(( (int)H + 2 * pad_h - kernel_h) / stride_h + 1);
    int out_w = (int)(( (int)W + 2 * pad_w - kernel_w) / stride_w + 1);

    size_t col = 0;
    for (size_t b = 0; b < batch; ++b) {
        for (int od = 0; od < out_d; ++od) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    size_t row = 0;
                    for (size_t ic = 0; ic < in_c; ++ic) {
                        for (int kd = 0; kd < kernel_d; ++kd) {
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    int id = od * stride_d + kd - pad_d;
                                    int ih = oh * stride_h + kh - pad_h;
                                    int iw = ow * stride_w + kw - pad_w;
                                    if (id >= 0 && id < (int)D && ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                                        double addv = grad_patches[row][col];
                                        double cur = grad_input[b][ic][(size_t)id][(size_t)ih][(size_t)iw];
                                        grad_input[b][ic][(size_t)id][(size_t)ih][(size_t)iw] = cur + addv;
                                    }
                                    ++row;
                                }
                            }
                        }
                    }
                    ++col;
                }
            }
        }
    }
}



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
Conv3d::Conv3d(int in_c, int out_c,int kd ,int kh, int kw,int sd, int sh, int sw,int pd, int ph, int pw)
    : in_channels(in_c), out_channels(out_c),
      kernel_size_d(kd),kernel_size_h(kh), kernel_size_w(kw),
      stride_d(sd),stride_h(sh), stride_w(sw),
      padding_d(pd),padding_h(ph), padding_w(pw)
{
    // weights: [out_c, in_c,kernel_size_d, kernel_size_h, kernel_size_w]
    weight = Tensor::rand({(size_t)out_c, (size_t)in_c,(size_t)kd, (size_t)kh, (size_t)kw}, DType::Float32, true);
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
    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t H     = input.impl->shape[2];
    size_t W     = input.impl->shape[3];

    int out_h = (int)(( (int)H + 2 * padding_h - kernel_size_h) / stride_h + 1);
    int out_w = (int)(( (int)W + 2 * padding_w - kernel_size_w) / stride_w + 1);

    size_t kernel_patch_size = in_c * kernel_size_h * kernel_size_w;
    Tensor w_flat = weight.reshape({(size_t)out_channels, kernel_patch_size});

    // create patches and store them
    Tensor input_patches = im2col_2d(input, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w);

    // size_t num_patches = batch * out_h * out_w; // Unused
    // matmul
    Tensor output_flat = matmul_(w_flat, input_patches);

    // add bias and reshape back
    Tensor bias_col = bias.reshape({(size_t)out_channels, 1});
    output_flat = output_flat + bias_col;

    Tensor output_reshaped = output_flat.reshape({(size_t)out_channels, batch, (size_t)out_h, (size_t)out_w});
    Tensor output = output_reshaped.permute({1,0,2,3});

    bool req = input.requires_grad() || weight.requires_grad() || bias.requires_grad();
    if (req) {
        // attach grad node storing patches (exactly those used in forward)
        output.impl->grad_fn = std::make_shared<GradConv2dMatmul>(input, weight, bias, input_patches,
                                                                 stride_h, stride_w, padding_h, padding_w,
                                                                 kernel_size_h, kernel_size_w);
    }
    return output;
}

// --- Conv3d forward using im2col + matmul (stores input_patches) ---
Tensor Conv3d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("Conv3d::forward: null input");
    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t D     = input.impl->shape[2];
    size_t H     = input.impl->shape[3];
    size_t W     = input.impl->shape[4];

    int out_d = (int)(( (int)D + 2 * padding_d - kernel_size_d) / stride_d + 1);
    int out_h = (int)(( (int)H + 2 * padding_h - kernel_size_h) / stride_h + 1);
    int out_w = (int)(( (int)W + 2 * padding_w - kernel_size_w) / stride_w + 1);

    size_t kernel_patch_size = in_c * kernel_size_d * kernel_size_h * kernel_size_w;
    Tensor w_flat = weight.reshape({(size_t)out_channels, kernel_patch_size});

    Tensor input_patches = im2col_3d(input, kernel_size_d, kernel_size_h, kernel_size_w,
                                     stride_d, stride_h, stride_w,
                                     padding_d, padding_h, padding_w);

    // size_t num_patches = batch * out_d * out_h * out_w; // Unused
    Tensor output_flat = matmul_(w_flat, input_patches);

    Tensor bias_col = bias.reshape({(size_t)out_channels, 1});
    output_flat = output_flat + bias_col;

    Tensor output_reshaped = output_flat.reshape({(size_t)out_channels, batch, (size_t)out_d, (size_t)out_h, (size_t)out_w});
    Tensor output = output_reshaped.permute({1,0,2,3,4});

    bool req = input.requires_grad() || weight.requires_grad() || bias.requires_grad();
    if (req) {
        output.impl->grad_fn = std::make_shared<GradConv3dMatmul>(input, weight, bias, input_patches,
                                                                 stride_d, stride_h, stride_w,
                                                                 padding_d, padding_h, padding_w,
                                                                 kernel_size_d, kernel_size_h, kernel_size_w);
    }
    return output;
}


// Backward
void GradConv1d::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradConv1d: missing self grad");

    // Detach parents while we compute intermediate grad tensors to avoid
    // accidentally building new grad-ops that reference the original graph.
    bool old_req_input  = input.impl ? input.impl->requires_grad : false;
    bool old_req_weight = weight.impl ? weight.impl->requires_grad : false;
    bool old_req_bias   = bias.impl ? bias.impl->requires_grad : false;

    if (input.impl)  input.impl->requires_grad  = false;
    if (weight.impl) weight.impl->requires_grad = false;
    if (bias.impl)   bias.impl->requires_grad   = false;

    // get grad of output (same shape as forward output) — contiguous DATA tensor
    Tensor grad_output = tensor_from_grad(self);

    // prepare zero tensors for grads (same shapes as originals), not requiring grad
    Tensor grad_input  = Tensor::zeros(input.shape(), input._dtype(), false);
    Tensor grad_weight = Tensor::zeros(weight.shape(), weight._dtype(), false);
    Tensor grad_bias   = Tensor::zeros(bias.shape(), bias._dtype(), false);

    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t width = input.impl->shape[2];
    size_t out_c = weight.impl->shape[0];
    size_t k_len = weight.impl->shape[2];
    size_t out_w = grad_output.impl->shape[2];

    // accumulate gradients (pure scalar math; no grad ops are created because parents are detached)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t oc = 0; oc < out_c; ++oc) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                // read grad at this output element (should be a small/copy-access)
                double go = grad_output[b][oc][ow];

                // grad bias: sum over batch & width
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

    // restore original requires_grad flags
    if (input.impl)  input.impl->requires_grad  = old_req_input;
    if (weight.impl) weight.impl->requires_grad = old_req_weight;
    if (bias.impl)   bias.impl->requires_grad   = old_req_bias;

    // accumulate into parents' grad buffers using your helper (only if they requested grads)
    if (input.requires_grad())  accumulate_grad(input,  grad_input);
    if (weight.requires_grad()) accumulate_grad(weight, grad_weight);
    if (bias.requires_grad())   accumulate_grad(bias,   grad_bias);
}

void GradConv2dMatmul::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradConv2dMatmul: missing self grad");

    // incoming grad (batch, out_c, out_h, out_w)
    Tensor grad_output = tensor_from_grad(self);

    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t H     = input.impl->shape[2];
    size_t W     = input.impl->shape[3];
    size_t out_c = weight.impl->shape[0];

    int out_h = (int)(( (int)H + 2 * pad_h - kernel_h) / stride_h + 1);
    int out_w = (int)(( (int)W + 2 * pad_w - kernel_w) / stride_w + 1);
    size_t num_patches = batch * out_h * out_w;
    size_t kernel_patch_size = in_c * kernel_h * kernel_w;

    // reshape grad_output -> [out_c, num_patches]
    Tensor grad_output_reshaped = grad_output.permute({1,0,2,3});
    Tensor grad_output_flat = grad_output_reshaped.reshape({out_c, num_patches});

    // --- BUG FIX: Detach parents ---
    bool old_grad_input = input.impl->requires_grad;
    bool old_grad_weight = weight.impl->requires_grad;
    bool old_grad_bias = bias.impl->requires_grad;
    input.impl->requires_grad = false;
    weight.impl->requires_grad = false;
    bias.impl->requires_grad = false;


    // bias grad: sum over columns
    if (old_grad_bias) {
        Tensor grad_bias = sum(grad_output_flat, 1); // [out_c] or [out_c,1]
        accumulate_grad(bias, grad_bias.reshape(bias.shape()));
    }

    // weight grad: grad_output_flat @ input_patches.T
    if (old_grad_weight) {
        Tensor input_patches_T = input_patches.t_(); // [num_patches, kernel_patch_size]
        Tensor grad_w_flat = matmul_(grad_output_flat, input_patches_T); // [out_c, kernel_patch_size]
        Tensor grad_weight = grad_w_flat.reshape(weight.shape());
        accumulate_grad(weight, grad_weight);
    }

    // input grad: w_flat.T @ grad_output_flat -> grad_input_patches -> col2im
    if (old_grad_input) {
        Tensor w_flat = weight.reshape({out_c, kernel_patch_size});
        Tensor w_flat_T = w_flat.t_(); // [kernel_patch_size, out_c]
        Tensor grad_input_patches = matmul_(w_flat_T, grad_output_flat); // [kernel_patch_size, num_patches]

        Tensor grad_input = Tensor::zeros(input.shape(), input._dtype(), false);
        col2im_2d(grad_input_patches, grad_input, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
        accumulate_grad(input, grad_input);
    }
    
    // --- Restore flags ---
    input.impl->requires_grad = old_grad_input;
    weight.impl->requires_grad = old_grad_weight;
    bias.impl->requires_grad = old_grad_bias;
}

void GradConv3dMatmul::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradConv3dMatmul: missing self grad");

    Tensor grad_output = tensor_from_grad(self);

    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t D     = input.impl->shape[2];
    size_t H     = input.impl->shape[3];
    size_t W     = input.impl->shape[4];
    size_t out_c = weight.impl->shape[0];

    int out_d = (int)(( (int)D + 2 * pad_d - kernel_d) / stride_d + 1);
    int out_h = (int)(( (int)H + 2 * pad_h - kernel_h) / stride_h + 1);
    int out_w = (int)(( (int)W + 2 * pad_w - kernel_w) / stride_w + 1);
    size_t num_patches = batch * out_d * out_h * out_w;
    size_t kernel_patch_size = in_c * kernel_d * kernel_h * kernel_w;

    Tensor grad_output_reshaped = grad_output.permute({1,0,2,3,4});
    Tensor grad_output_flat = grad_output_reshaped.reshape({out_c, num_patches});

    // --- BUG FIX: Detach parents ---
    bool old_grad_input = input.impl->requires_grad;
    bool old_grad_weight = weight.impl->requires_grad;
    bool old_grad_bias = bias.impl->requires_grad;
    input.impl->requires_grad = false;
    weight.impl->requires_grad = false;
    bias.impl->requires_grad = false;

    if (old_grad_bias) {
        Tensor grad_bias = sum(grad_output_flat, 1);
        accumulate_grad(bias, grad_bias.reshape(bias.shape()));
    }

    if (old_grad_weight) {
        Tensor input_patches_T = input_patches.t_();
        Tensor grad_w_flat = matmul_(grad_output_flat, input_patches_T);
        Tensor grad_weight = grad_w_flat.reshape(weight.shape());
        accumulate_grad(weight, grad_weight);
    }

    if (old_grad_input) {
        Tensor w_flat = weight.reshape({out_c, kernel_patch_size});
        Tensor w_flat_T = w_flat.t_();
        Tensor grad_input_patches = matmul_(w_flat_T, grad_output_flat);
        Tensor grad_input = Tensor::zeros(input.shape(), input._dtype(), false);
        col2im_3d(grad_input_patches, grad_input,
                  kernel_d, kernel_h, kernel_w,
                  stride_d, stride_h, stride_w,
                  pad_d, pad_h, pad_w);
        accumulate_grad(input, grad_input);
    }
    
    // --- Restore flags ---
    input.impl->requires_grad = old_grad_input;
    weight.impl->requires_grad = old_grad_weight;
    bias.impl->requires_grad = old_grad_bias;
}