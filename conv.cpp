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
Conv1d::Conv1d(int in_c, int out_c, int k, int s, int p, DType dt )
    : in_channels(in_c), out_channels(out_c),
      kernel_size(k), stride(s), padding(p)
{
    // weights: [out_c, in_c, kernel_size]
    weight = Tensor::rand({(size_t)out_c, (size_t)in_c, (size_t)k}, dt, true);
    bias   = Tensor::zeros({(size_t)out_c}, dt, true);
}

Conv2d::Conv2d(int in_c, int out_c, int kh, int kw, int sh, int sw, int ph, int pw,DType dt)
    : in_channels(in_c), out_channels(out_c),
      kernel_size_h(kh), kernel_size_w(kw),
      stride_h(sh), stride_w(sw),
      padding_h(ph), padding_w(pw)
{
    // weights: [out_c, in_c, kernel_size_h, kernel_size_w]
    weight = Tensor::rand({(size_t)out_c, (size_t)in_c, (size_t)kh, (size_t)kw}, dt, true);
    bias   = Tensor::zeros({(size_t)out_c}, dt, true);
}
Conv3d::Conv3d(int in_c, int out_c,int kd ,int kh, int kw,int sd, int sh, int sw,int pd, int ph, int pw,DType dt )
    : in_channels(in_c), out_channels(out_c),
      kernel_size_d(kd),kernel_size_h(kh), kernel_size_w(kw),
      stride_d(sd),stride_h(sh), stride_w(sw),
      padding_d(pd),padding_h(ph), padding_w(pw)
{
    // weights: [out_c, in_c,kernel_size_d, kernel_size_h, kernel_size_w]
    weight = Tensor::rand({(size_t)out_c, (size_t)in_c,(size_t)kd, (size_t)kh, (size_t)kw}, dt, true);
    bias   = Tensor::zeros({(size_t)out_c}, dt, true);
}


// Forward (Conv1d )
Tensor Conv1d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("Conv1d::forward: null input");
    if (input.impl->ndim != 3)
    throw std::runtime_error("Conv1d forward: input must be [batch, channels, width]");

    // input shape assumed [batch, in_channels, width]
    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t width = input.impl->shape[2];

    int out_w = (int)(( (int)width + 2 * padding - kernel_size) / stride + 1);
    if (out_w <= 0) throw std::runtime_error("Conv1d::forward: invalid output width");

    std::vector<size_t> out_shape = { batch, (size_t)out_channels, (size_t)out_w };
    bool req = input.requires_grad() || weight.requires_grad() || bias.requires_grad();

    Tensor output(out_shape, input._dtype(), req);

    for (size_t b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ow = 0; ow < out_w; ++ow) {
                // start with bias
                // --- FIX START ---
                // The bias tensor is 1D [out_c]. Its stride is 1 (or bias.impl->strides[0])
                // We must read it using the correct DType, not hardcode float*
                size_t bias_idx = bias.impl->offset + (size_t)oc * bias.impl->strides[0];
                double acc = read_scalar_at(bias.impl->storage->data.get(), bias_idx, bias._dtype());
                // --- FIX END ---
                
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

    size_t num_patches = batch * out_h * out_w;
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

    size_t num_patches = batch * out_d * out_h * out_w;
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
                                                                 kernel_size_d, kernel_size_h, kernel_size_w); // <-- FIX: The middle argument was kernel_size_d
    }
    return output;
}

void GradConv1d::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradConv1d: missing self grad");

    // --- detach parents while computing numeric intermediates ---
    bool old_req_input  = input.impl ? input.impl->requires_grad : false;
    bool old_req_weight = weight.impl ? weight.impl->requires_grad : false;
    bool old_req_bias   = bias.impl ? bias.impl->requires_grad : false;
    if (input.impl)  input.impl->requires_grad  = false;
    if (weight.impl) weight.impl->requires_grad = false;
    if (bias.impl)   bias.impl->requires_grad   = false;

    // incoming gradient as contiguous DATA tensor
    Tensor grad_output = tensor_from_grad(self); // contiguous DATA

    // accumulators (no grad tracking)
    Tensor grad_input  = Tensor::zeros(input.shape(),  input._dtype(),  false);
    Tensor grad_weight = Tensor::zeros(weight.shape(), weight._dtype(), false);
    Tensor grad_bias   = Tensor::zeros(bias.shape(),   bias._dtype(),   false);

    // shapes
    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t width = input.impl->shape[2];
    size_t out_c = weight.impl->shape[0];
    size_t k_len = weight.impl->shape[2];
    size_t out_w = grad_output.impl->shape[2];

    // raw pointers + offsets + strides (consistent naming)
    void* in_data  = input.impl->storage->data.get();
    void* w_data   = weight.impl->storage->data.get();
    void* b_data   = bias.impl->storage->data.get();
    void* go_data  = grad_output.impl->storage->data.get();

    void* gin_data = grad_input.impl->storage->data.get();
    void* gw_data  = grad_weight.impl->storage->data.get();
    void* gb_data  = grad_bias.impl->storage->data.get();

    size_t in_off  = input.impl->offset;
    size_t w_off   = weight.impl->offset;
    size_t b_off   = bias.impl->offset;
    size_t go_off  = grad_output.impl->offset;
    size_t gin_off = grad_input.impl->offset;
    size_t gw_off  = grad_weight.impl->offset;
    size_t gb_off  = grad_bias.impl->offset;

    size_t in_s0 = input.impl->strides[0], in_s1 = input.impl->strides[1], in_s2 = input.impl->strides[2];
    size_t w_s0  = weight.impl->strides[0], w_s1 = weight.impl->strides[1], w_s2 = weight.impl->strides[2];
    size_t go_s0 = grad_output.impl->strides[0], go_s1 = grad_output.impl->strides[1], go_s2 = grad_output.impl->strides[2];
    size_t gin_s0 = grad_input.impl->strides[0], gin_s1 = grad_input.impl->strides[1], gin_s2 = grad_input.impl->strides[2];
    size_t gw_s0 = grad_weight.impl->strides[0], gw_s1 = grad_weight.impl->strides[1], gw_s2 = grad_weight.impl->strides[2];
    size_t gb_s0 = (bias.impl->ndim > 0) ? bias.impl->strides[0] : 1;

    // main accumulation
    for (size_t b = 0; b < batch; ++b) {
        for (size_t oc = 0; oc < out_c; ++oc) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                // read incoming grad (go)
                size_t go_idx = go_off + b*go_s0 + oc*go_s1 + ow*go_s2;
                double go = read_scalar_at(go_data, go_idx, grad_output._dtype());

                // bias accumulation
                size_t gb_idx = gb_off + oc * gb_s0;
                double curb = read_scalar_at(gb_data, gb_idx, grad_bias._dtype());
                write_scalar_at(gb_data, gb_idx, grad_bias._dtype(), curb + go);

                for (size_t ic = 0; ic < in_c; ++ic) {
                    for (size_t k = 0; k < k_len; ++k) {
                        int iw = (int)ow * stride + (int)k - padding;
                        if (iw < 0 || iw >= (int)width) continue;

                        // grad_input[b, ic, iw] += go * weight[oc, ic, k]
                        size_t w_idx = w_off + oc*w_s0 + ic*w_s1 + k*w_s2;
                        double wval = read_scalar_at(w_data, w_idx, weight._dtype());
                        size_t gin_idx = gin_off + b*gin_s0 + ic*gin_s1 + (size_t)iw * gin_s2;
                        double cur_in = read_scalar_at(gin_data, gin_idx, grad_input._dtype());
                        write_scalar_at(gin_data, gin_idx, grad_input._dtype(), cur_in + go * wval);


                        // grad_weight[oc, ic, k] += go * input[b, ic, iw]
                        size_t in_idx = in_off + b*in_s0 + ic*in_s1 + (size_t)iw * in_s2;
                        double inval = read_scalar_at(in_data, in_idx, input._dtype());
                        size_t gw_idx = gw_off + oc*gw_s0 + ic*gw_s1 + k*gw_s2;
                        double cur_w = read_scalar_at(gw_data, gw_idx, grad_weight._dtype());
                        // inside the k loop, right before updating grad_weight:
                        std::cerr << "B" << b << " OC" << oc << " IC" << ic << " K" << k << " IW" << iw
                        << " go="<<go<<" inval="<<inval<<" idx_gw="<<gw_idx<<"\n";
                        write_scalar_at(gw_data, gw_idx, grad_weight._dtype(), cur_w + go * inval);
                    }
                }
            }
        }
    }

    // restore requires_grad flags
    if (input.impl)  input.impl->requires_grad  = old_req_input;
    if (weight.impl) weight.impl->requires_grad = old_req_weight;
    if (bias.impl)   bias.impl->requires_grad   = old_req_bias;
    std::cerr << "DEBUG conv shapes: batch=" << batch << " in_c=" << in_c << " width=" << width
          << " out_c=" << out_c << " k_len=" << k_len << " out_w=" << out_w << "\n";
    std::cerr << "strides: in=("<<in_s0<<","<<in_s1<<","<<in_s2<<") weight=("<<w_s0<<","<<w_s1<<","<<w_s2<<") grad_weight=("<<gw_s0<<","<<gw_s1<<","<<gw_s2<<")\n";

    // accumulate into parents' grad buffers only if they originally requested grad
    if (old_req_input)  accumulate_grad(input,  grad_input);
    // --- debug dump: show grad_weight local accumulator (values & raw storage) ---
    std::cerr << "DEBUG: local grad_weight (shape): ";
    grad_weight.print_shape();
    std::cerr << " local grad_weight values (flat): ";
    for (size_t i=0; i<grad_weight.numel(); ++i) {
        std::cerr << grad_weight.read_scalar(i) << (i+1<grad_weight.numel() ? " " : "\n");
    }

    // storage size / offset / strides info for weight
    std::cerr << "DEBUG weight storage: storage_size=" << (weight.impl->storage ? weight.impl->storage->size : 0)
              << " offset=" << weight.impl->offset << " strides=[";
    for (size_t i=0;i<weight.impl->ndim;++i) std::cerr << weight.impl->strides[i] << (i+1<weight.impl->ndim? ",":"");
    std::cerr << "] shape=[";
    for (size_t i=0;i<weight.impl->ndim;++i) std::cerr << weight.impl->shape[i] << (i+1<weight.impl->ndim? ",":"");
    std::cerr << "]\n";

    // Show what accumulate_grad will write: print gw_data raw BEFORE
    void* gw_storage_data = grad_weight.impl->storage->data.get();
    std::cerr << "DEBUG grad_weight.storage->data (flat): ";
    for (size_t i=0;i<grad_weight.numel(); ++i) {
        std::cerr << read_scalar_at(gw_storage_data, i + grad_weight.impl->offset, grad_weight._dtype())
                  << (i+1<grad_weight.numel() ? " " : "\n");
    }

    // Now show weight's grad buffer BEFORE accumulate (if any)
    if (weight.impl->storage->grad) {
        std::cerr << "DEBUG weight.grad (before accumulate): ";
        void* wgrad_raw = weight.impl->storage->grad.get();
        for (size_t i=0;i<weight.numel_(); ++i) {
            std::cerr << read_scalar_at(wgrad_raw, i + weight.impl->offset, weight._dtype())
                      << (i+1<weight.numel_() ? " " : "\n");
        }
    } else {
        std::cerr << "DEBUG weight.grad == null (before accumulate)\n";
    }

    if (old_req_weight) accumulate_grad(weight, grad_weight);
    // print weight.grad after accumulate
    if (weight.impl->storage->grad) {
        std::cerr << "DEBUG weight.grad (after accumulate): ";
        void* wgrad_raw = weight.impl->storage->grad.get();
        for (size_t i=0;i<weight.numel_(); ++i) {
            std::cerr << read_scalar_at(wgrad_raw, i + weight.impl->offset, weight._dtype())
                      << (i+1<weight.numel_() ? " " : "\n");
        }
    } else {
        std::cerr << "DEBUG weight.grad still null after accumulate\n";
    }
    // debug: show whether bias requested grad originally
    std::cerr << "DEBUG: old_req_bias=" << old_req_bias << "\n";

    // debug: local grad_bias (values)
    std::cerr << "DEBUG local grad_bias (flat): ";
    for (size_t i = 0; i < grad_bias.numel(); ++i) {
        std::cerr << grad_bias.read_scalar(i) << (i+1<grad_bias.numel()? " ":"\n");
    }

    // debug: bias.grad before accumulate (if any)
    if (bias.impl->storage->grad) {
        std::cerr << "DEBUG bias.grad (before): ";
        void* bgrad_raw = bias.impl->storage->grad.get();
        for (size_t i = 0; i < bias.numel_(); ++i)
            std::cerr << read_scalar_at(bgrad_raw, i + bias.impl->offset, bias._dtype())
                      << (i+1<bias.numel_()? " ":"\n");
    } else {
        std::cerr << "DEBUG bias.grad == null (before)\n";
    }

    if (old_req_bias) accumulate_grad(bias,  grad_bias);
    if (bias.impl->storage->grad) {
    void* g = bias.impl->storage->grad.get();
        std::cerr << "bias.grad raw[0] = " << read_scalar_at(g, bias.impl->offset, bias._dtype()) << "\n";
    }

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

    // bias grad: sum over columns
    if (bias.requires_grad()) {
        Tensor grad_bias = sum(grad_output_flat, 1); // [out_c] or [out_c,1]
        accumulate_grad(bias, grad_bias.reshape(bias.shape()));
    }

    // weight grad: grad_output_flat @ input_patches.T
    if (weight.requires_grad()) {
        Tensor input_patches_T = input_patches.t_(); // [num_patches, kernel_patch_size]
        Tensor grad_w_flat = matmul_(grad_output_flat, input_patches_T); // [out_c, kernel_patch_size]
        Tensor grad_weight = grad_w_flat.reshape(weight.shape());
        accumulate_grad(weight, grad_weight);
    }

    // input grad: w_flat.T @ grad_output_flat -> grad_input_patches -> col2im
    if (input.requires_grad()) {
        Tensor w_flat = weight.reshape({out_c, kernel_patch_size});
        Tensor w_flat_T = w_flat.t_(); // [kernel_patch_size, out_c]
        Tensor grad_input_patches = matmul_(w_flat_T, grad_output_flat); // [kernel_patch_size, num_patches]

        Tensor grad_input = Tensor::zeros(input.shape(), input._dtype(), false);
        col2im_2d(grad_input_patches, grad_input, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
        accumulate_grad(input, grad_input);
    }
        /*
    std::cerr   << "DEBUG GradConv2d: input.storage->size=" << input.impl->storage->size
                << " weight.storage->size=" << weight.impl->storage->size
                << " output.storage->size=" << self.impl->storage->size << "\n";

    std::cerr << "shapes: input=";
    input.print_shape();
    std::cerr << " weight=";
    weight.print_shape();
    std::cerr << " grad_output=";
    grad_output.print_shape();
    */
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

    if (bias.requires_grad()) {
        Tensor grad_bias = sum(grad_output_flat, 1);
        accumulate_grad(bias, grad_bias.reshape(bias.shape()));
    }

    if (weight.requires_grad()) {
        Tensor input_patches_T = input_patches.t_();
        Tensor grad_w_flat = matmul_(grad_output_flat, input_patches_T);
        Tensor grad_weight = grad_w_flat.reshape(weight.shape());
        accumulate_grad(weight, grad_weight);
    }

    if (input.requires_grad()) {
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
}