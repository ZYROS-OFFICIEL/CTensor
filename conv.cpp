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

// Forward
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

Tensor Conv2d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("Conv2d::forward: null input");

    // input shape assumed [batch, in_channels, height, width]
    size_t batch = input.impl->shape[0];
    size_t in_c  = input.impl->shape[1];
    size_t height = input.impl->shape[2];
    size_t width  = input.impl->shape[3];

    int out_h = (int)(( (int)height + 2 * padding_h - kernel_size_h) / stride_h + 1);
    int out_w = (int)(( (int)width  + 2 * padding_w - kernel_size_w) / stride_w + 1);
    if (out_h <= 0 || out_w <= 0) throw std::runtime_error("Conv2d::forward: invalid output dimensions");

    std::vector<size_t> out_shape = { batch, (size_t)out_channels, (size_t)out_h, (size_t)out_w };
    bool req = input.requires_grad() || weight.requires_grad() || bias.requires_grad();

    Tensor output(out_shape, input._dtype(), req);

    // compute convolution (naive)
    for (size_t b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    // start with bias
                    double acc = bias[(size_t)oc]; // proxy -> double
                    for (size_t ic = 0; ic < in_c; ++ic) {
                        for (int kh = 0; kh < kernel_size_h; ++kh) {
                            for (int kw = 0; kw < kernel_size_w; ++kw) {
                                int ih = oh * stride_h + kh - padding_h;
                                int iw = ow * stride_w + kw - padding_w;
                                if (ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width) {
                                    double in_val = input[b][ic][(size_t)ih][(size_t)iw];
                                    double w_val  = weight[(size_t)oc][ic][(size_t)kh][(size_t)kw];
                                    acc += in_val * w_val;
                                }
                            }
                        }
                    }
                    output[b][(size_t)oc][(size_t)oh][(size_t)ow] = acc;
                }
            }
        }
    }
    // attach grad fn if needed
    if (req) {
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
