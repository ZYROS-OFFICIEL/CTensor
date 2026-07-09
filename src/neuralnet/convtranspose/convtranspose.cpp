#include "neuralnet/convtranspose/convtranspose.h"
#include "core.h"
#include <stdexcept>

ConvTranspose1d::ConvTranspose1d(int in_c, int out_c, int k, int s, int p, int op, DType dt)
    : in_channels(in_c), out_channels(out_c),
      kernel_size(k), stride(s), padding(p), output_padding(op)
{
    weight = Tensor::rand({(size_t)in_c, (size_t)out_c, (size_t)k}, dt, true);
    bias   = Tensor::zeros({(size_t)out_c}, dt, true);
}

Tensor ConvTranspose1d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("");
    size_t batch = input.impl->shape[0];
    size_t width = input.impl->shape[2];
    int out_w = (int)(((int)width - 1) * stride - 2 * padding + kernel_size + output_padding);
    if (out_w <= 0) throw std::runtime_error("");

    std::vector<size_t> out_shape = { batch, (size_t)out_channels, (size_t)out_w };
    bool req = input.requires_grad() || weight.requires_grad() || bias.requires_grad();
    Tensor output(out_shape, input._dtype(), req);

    const float* in_ptr = (const float*)input.impl->data->data.get() + input.impl->offset;
    const float* w_ptr = (const float*)weight.impl->data->data.get() + weight.impl->offset;
    const float* b_ptr = (const float*)bias.impl->data->data.get() + bias.impl->offset;
    float* out_ptr = (float*)output.impl->data->data.get() + output.impl->offset;

    size_t i_s0 = input.impl->strides[0], i_s1 = input.impl->strides[1], i_s2 = input.impl->strides[2];
    size_t w_s0 = weight.impl->strides[0], w_s1 = weight.impl->strides[1], w_s2 = weight.impl->strides[2];
    size_t o_s0 = output.impl->strides[0], o_s1 = output.impl->strides[1], o_s2 = output.impl->strides[2];

    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ow = 0; ow < out_w; ++ow) {
                double sum = b_ptr[oc];
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int k = 0; k < kernel_size; ++k) {
                        int iw_s = ow + padding - k;
                        if (iw_s % stride == 0) {
                            int iw = iw_s / stride;
                            if (iw >= 0 && iw < (int)width) {
                                sum += in_ptr[b*i_s0 + ic*i_s1 + iw*i_s2] * w_ptr[ic*w_s0 + oc*w_s1 + k*w_s2];
                            }
                        }
                    }
                }
                out_ptr[b*o_s0 + oc*o_s1 + ow*o_s2] = sum;
            }
        }
    }

    if (req) output.impl->grad_fn = std::make_shared<GradConvTranspose1d>(input, weight, bias, stride, padding);
    return output;
}

ConvTranspose2d::ConvTranspose2d(int in_c, int out_c, int kh, int kw, int sh, int sw, int ph, int pw, int oph, int opw, DType dt)
    : in_channels(in_c), out_channels(out_c),
      kernel_size_h(kh), kernel_size_w(kw == -1 ? kh : kw),
      stride_h(sh), stride_w(sw), padding_h(ph), padding_w(pw),
      output_padding_h(oph), output_padding_w(opw)
{
    weight = Tensor::rand({(size_t)in_c, (size_t)out_c, (size_t)kh, (size_t)kernel_size_w}, dt, true);
    bias   = Tensor::zeros({(size_t)out_c}, dt, true);
}

Tensor ConvTranspose2d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("");
    size_t batch = input.impl->shape[0];
    size_t height = input.impl->shape[2];
    size_t width  = input.impl->shape[3];

    int out_h = (int)(((int)height - 1) * stride_h - 2 * padding_h + kernel_size_h + output_padding_h);
    int out_w = (int)(((int)width - 1) * stride_w - 2 * padding_w + kernel_size_w + output_padding_w);
    if (out_h <= 0 || out_w <= 0) throw std::runtime_error("");

    std::vector<size_t> out_shape = { batch, (size_t)out_channels, (size_t)out_h, (size_t)out_w };
    bool req = input.requires_grad() || weight.requires_grad() || bias.requires_grad();
    Tensor output(out_shape, input._dtype(), req);

    const float* in_ptr = (const float*)input.impl->data->data.get() + input.impl->offset;
    const float* w_ptr = (const float*)weight.impl->data->data.get() + weight.impl->offset;
    const float* b_ptr = (const float*)bias.impl->data->data.get() + bias.impl->offset;
    float* out_ptr = (float*)output.impl->data->data.get() + output.impl->offset;

    size_t i_s0 = input.impl->strides[0], i_s1 = input.impl->strides[1], i_s2 = input.impl->strides[2], i_s3 = input.impl->strides[3];
    size_t w_s0 = weight.impl->strides[0], w_s1 = weight.impl->strides[1], w_s2 = weight.impl->strides[2], w_s3 = weight.impl->strides[3];
    size_t o_s0 = output.impl->strides[0], o_s1 = output.impl->strides[1], o_s2 = output.impl->strides[2], o_s3 = output.impl->strides[3];

    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    double sum = b_ptr[oc];
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size_h; ++kh) {
                            for (int kw = 0; kw < kernel_size_w; ++kw) {
                                int ih_s = oh + padding_h - kh;
                                int iw_s = ow + padding_w - kw;
                                if (ih_s % stride_h == 0 && iw_s % stride_w == 0) {
                                    int ih = ih_s / stride_h;
                                    int iw = iw_s / stride_w;
                                    if (ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width) {
                                        sum += in_ptr[b*i_s0 + ic*i_s1 + ih*i_s2 + iw*i_s3] * w_ptr[ic*w_s0 + oc*w_s1 + kh*w_s2 + kw*w_s3];
                                    }
                                }
                            }
                        }
                    }
                    out_ptr[b*o_s0 + oc*o_s1 + oh*o_s2 + ow*o_s3] = sum;
                }
            }
        }
    }

    if (req) output.impl->grad_fn = std::make_shared<GradConvTranspose2d>(input, weight, bias, stride_h, stride_w, padding_h, padding_w);
    return output;
}

ConvTranspose3d::ConvTranspose3d(int in_c, int out_c, int kd, int kh, int kw, int sd, int sh, int sw, int pd, int ph, int pw, int opd, int oph, int opw, DType dt)
    : in_channels(in_c), out_channels(out_c),
      kernel_size_d(kd), kernel_size_h(kh), kernel_size_w(kw),
      stride_d(sd), stride_h(sh), stride_w(sw),
      padding_d(pd), padding_h(ph), padding_w(pw),
      output_padding_d(opd), output_padding_h(oph), output_padding_w(opw)
{
    weight = Tensor::rand({(size_t)in_c, (size_t)out_c, (size_t)kd, (size_t)kh, (size_t)kw}, dt, true);
    bias   = Tensor::zeros({(size_t)out_c}, dt, true);
}

Tensor ConvTranspose3d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("");
    size_t batch = input.impl->shape[0];
    size_t depth = input.impl->shape[2];
    size_t height = input.impl->shape[3];
    size_t width  = input.impl->shape[4];

    int out_d = (int)(((int)depth - 1) * stride_d - 2 * padding_d + kernel_size_d + output_padding_d);
    int out_h = (int)(((int)height - 1) * stride_h - 2 * padding_h + kernel_size_h + output_padding_h);
    int out_w = (int)(((int)width - 1) * stride_w - 2 * padding_w + kernel_size_w + output_padding_w);
    if (out_d <= 0 || out_h <= 0 || out_w <= 0) throw std::runtime_error("");

    std::vector<size_t> out_shape = { batch, (size_t)out_channels, (size_t)out_d, (size_t)out_h, (size_t)out_w };
    bool req = input.requires_grad() || weight.requires_grad() || bias.requires_grad();
    Tensor output(out_shape, input._dtype(), req);

    const float* in_ptr = (const float*)input.impl->data->data.get() + input.impl->offset;
    const float* w_ptr = (const float*)weight.impl->data->data.get() + weight.impl->offset;
    const float* b_ptr = (const float*)bias.impl->data->data.get() + bias.impl->offset;
    float* out_ptr = (float*)output.impl->data->data.get() + output.impl->offset;

    size_t i_s0 = input.impl->strides[0], i_s1 = input.impl->strides[1], i_s2 = input.impl->strides[2], i_s3 = input.impl->strides[3], i_s4 = input.impl->strides[4];
    size_t w_s0 = weight.impl->strides[0], w_s1 = weight.impl->strides[1], w_s2 = weight.impl->strides[2], w_s3 = weight.impl->strides[3], w_s4 = weight.impl->strides[4];
    size_t o_s0 = output.impl->strides[0], o_s1 = output.impl->strides[1], o_s2 = output.impl->strides[2], o_s3 = output.impl->strides[3], o_s4 = output.impl->strides[4];

    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int od = 0; od < out_d; ++od) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        double sum = b_ptr[oc];
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kd = 0; kd < kernel_size_d; ++kd) {
                                for (int kh = 0; kh < kernel_size_h; ++kh) {
                                    for (int kw = 0; kw < kernel_size_w; ++kw) {
                                        int id_s = od + padding_d - kd;
                                        int ih_s = oh + padding_h - kh;
                                        int iw_s = ow + padding_w - kw;
                                        if (id_s % stride_d == 0 && ih_s % stride_h == 0 && iw_s % stride_w == 0) {
                                            int id = id_s / stride_d;
                                            int ih = ih_s / stride_h;
                                            int iw = iw_s / stride_w;
                                            if (id >= 0 && id < (int)depth && ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width) {
                                                sum += in_ptr[b*i_s0 + ic*i_s1 + id*i_s2 + ih*i_s3 + iw*i_s4] * w_ptr[ic*w_s0 + oc*w_s1 + kd*w_s2 + kh*w_s3 + kw*w_s4];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        out_ptr[b*o_s0 + oc*o_s1 + od*o_s2 + oh*o_s3 + ow*o_s4] = sum;
                    }
                }
            }
        }
    }

    if (req) output.impl->grad_fn = std::make_shared<GradConvTranspose3d>(input, weight, bias, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w);
    return output;
}

void GradConvTranspose1d::backward(const Tensor& self) {
    if (!self.impl->grad->data) throw std::runtime_error("");
    Tensor grad_output = tensor_from_grad(self);
    Tensor grad_input  = Tensor::zeros(input.shape(), input._dtype(), false);
    Tensor grad_weight = Tensor::zeros(weight.shape(), weight._dtype(), false);
    Tensor grad_bias   = Tensor::zeros(bias.shape(), bias._dtype(), false);

    size_t batch = input.impl->shape[0], in_c = input.impl->shape[1], width = input.impl->shape[2];
    size_t out_c = weight.impl->shape[1], k_len = weight.impl->shape[2], out_w = grad_output.impl->shape[2];

    const float* go_ptr = (const float*)grad_output.impl->data->data.get() + grad_output.impl->offset;
    const float* in_ptr = (const float*)input.impl->data->data.get() + input.impl->offset;
    const float* w_ptr = (const float*)weight.impl->data->data.get() + weight.impl->offset;
    float* gi_ptr = (float*)grad_input.impl->data->data.get() + grad_input.impl->offset;
    float* gw_ptr = (float*)grad_weight.impl->data->data.get() + grad_weight.impl->offset;
    float* gb_ptr = (float*)grad_bias.impl->data->data.get() + grad_bias.impl->offset;

    size_t go_s0 = grad_output.impl->strides[0], go_s1 = grad_output.impl->strides[1], go_s2 = grad_output.impl->strides[2];
    size_t i_s0 = input.impl->strides[0], i_s1 = input.impl->strides[1], i_s2 = input.impl->strides[2];
    size_t w_s0 = weight.impl->strides[0], w_s1 = weight.impl->strides[1], w_s2 = weight.impl->strides[2];
    size_t gi_s0 = grad_input.impl->strides[0], gi_s1 = grad_input.impl->strides[1], gi_s2 = grad_input.impl->strides[2];
    size_t gw_s0 = grad_weight.impl->strides[0], gw_s1 = grad_weight.impl->strides[1], gw_s2 = grad_weight.impl->strides[2];

    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch; ++b) {
        for (int ic = 0; ic < in_c; ++ic) {
            for (int iw = 0; iw < width; ++iw) {
                double sum_in = 0.0;
                for (int oc = 0; oc < out_c; ++oc) {
                    for (int k = 0; k < k_len; ++k) {
                        int ow = iw * stride - padding + k;
                        if (ow >= 0 && ow < out_w) {
                            double go = go_ptr[b*go_s0 + oc*go_s1 + ow*go_s2];
                            double w_val = w_ptr[ic*w_s0 + oc*w_s1 + k*w_s2];
                            sum_in += go * w_val;
                            double in_val = in_ptr[b*i_s0 + ic*i_s1 + iw*i_s2];
                            float* gw = &gw_ptr[ic*gw_s0 + oc*gw_s1 + k*gw_s2];
                            #pragma omp atomic
                            *gw += (float)(go * in_val);
                        }
                    }
                }
                gi_ptr[b*gi_s0 + ic*gi_s1 + iw*gi_s2] += sum_in;
            }
        }
    }

    if (bias.requires_grad()) {
        for (int oc = 0; oc < out_c; ++oc) {
            double sum_b = 0.0;
            for (size_t b = 0; b < batch; ++b) {
                for (int ow = 0; ow < out_w; ++ow) {
                    sum_b += go_ptr[b*go_s0 + oc*go_s1 + ow*go_s2];
                }
            }
            gb_ptr[oc] += sum_b;
        }
    }

    if (input.requires_grad()) accumulate_grad(input, grad_input);
    if (weight.requires_grad()) accumulate_grad(weight, grad_weight);
    if (bias.requires_grad()) accumulate_grad(bias, grad_bias);
}

void GradConvTranspose2d::backward(const Tensor& self) {
    if (!self.impl->grad->data) throw std::runtime_error("");
    Tensor grad_output = tensor_from_grad(self);
    Tensor grad_input  = Tensor::zeros(input.shape(), input._dtype(), false);
    Tensor grad_weight = Tensor::zeros(weight.shape(), weight._dtype(), false);
    Tensor grad_bias   = Tensor::zeros(bias.shape(), bias._dtype(), false);

    size_t batch = input.impl->shape[0], in_c = input.impl->shape[1], height = input.impl->shape[2], width = input.impl->shape[3];
    size_t out_c = weight.impl->shape[1], k_h = weight.impl->shape[2], k_w = weight.impl->shape[3];
    size_t out_h = grad_output.impl->shape[2], out_w = grad_output.impl->shape[3];

    const float* go_ptr = (const float*)grad_output.impl->data->data.get() + grad_output.impl->offset;
    const float* in_ptr = (const float*)input.impl->data->data.get() + input.impl->offset;
    const float* w_ptr = (const float*)weight.impl->data->data.get() + weight.impl->offset;
    float* gi_ptr = (float*)grad_input.impl->data->data.get() + grad_input.impl->offset;
    float* gw_ptr = (float*)grad_weight.impl->data->data.get() + grad_weight.impl->offset;
    float* gb_ptr = (float*)grad_bias.impl->data->data.get() + grad_bias.impl->offset;

    size_t go_s0 = grad_output.impl->strides[0], go_s1 = grad_output.impl->strides[1], go_s2 = grad_output.impl->strides[2], go_s3 = grad_output.impl->strides[3];
    size_t i_s0 = input.impl->strides[0], i_s1 = input.impl->strides[1], i_s2 = input.impl->strides[2], i_s3 = input.impl->strides[3];
    size_t w_s0 = weight.impl->strides[0], w_s1 = weight.impl->strides[1], w_s2 = weight.impl->strides[2], w_s3 = weight.impl->strides[3];
    size_t gi_s0 = grad_input.impl->strides[0], gi_s1 = grad_input.impl->strides[1], gi_s2 = grad_input.impl->strides[2], gi_s3 = grad_input.impl->strides[3];
    size_t gw_s0 = grad_weight.impl->strides[0], gw_s1 = grad_weight.impl->strides[1], gw_s2 = grad_weight.impl->strides[2], gw_s3 = grad_weight.impl->strides[3];

    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch; ++b) {
        for (int ic = 0; ic < in_c; ++ic) {
            for (int ih = 0; ih < height; ++ih) {
                for (int iw = 0; iw < width; ++iw) {
                    double sum_in = 0.0;
                    for (int oc = 0; oc < out_c; ++oc) {
                        for (int kh = 0; kh < k_h; ++kh) {
                            for (int kw = 0; kw < k_w; ++kw) {
                                int oh = ih * stride_h - padding_h + kh;
                                int ow = iw * stride_w - padding_w + kw;
                                if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                                    double go = go_ptr[b*go_s0 + oc*go_s1 + oh*go_s2 + ow*go_s3];
                                    double w_val = w_ptr[ic*w_s0 + oc*w_s1 + kh*w_s2 + kw*w_s3];
                                    sum_in += go * w_val;
                                    double in_val = in_ptr[b*i_s0 + ic*i_s1 + ih*i_s2 + iw*i_s3];
                                    float* gw = &gw_ptr[ic*gw_s0 + oc*gw_s1 + kh*gw_s2 + kw*gw_s3];
                                    #pragma omp atomic
                                    *gw += (float)(go * in_val);
                                }
                            }
                        }
                    }
                    gi_ptr[b*gi_s0 + ic*gi_s1 + ih*gi_s2 + iw*gi_s3] += sum_in;
                }
            }
        }
    }

    if (bias.requires_grad()) {
        for (int oc = 0; oc < out_c; ++oc) {
            double sum_b = 0.0;
            for (size_t b = 0; b < batch; ++b) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        sum_b += go_ptr[b*go_s0 + oc*go_s1 + oh*go_s2 + ow*go_s3];
                    }
                }
            }
            gb_ptr[oc] += sum_b;
        }
    }

    if (input.requires_grad()) accumulate_grad(input, grad_input);
    if (weight.requires_grad()) accumulate_grad(weight, grad_weight);
    if (bias.requires_grad()) accumulate_grad(bias, grad_bias);
}

void GradConvTranspose3d::backward(const Tensor& self) {
    if (!self.impl->grad->data) throw std::runtime_error("");
    Tensor grad_output = tensor_from_grad(self);
    Tensor grad_input  = Tensor::zeros(input.shape(), input._dtype(), false);
    Tensor grad_weight = Tensor::zeros(weight.shape(), weight._dtype(), false);
    Tensor grad_bias   = Tensor::zeros(bias.shape(), bias._dtype(), false);

    size_t batch = input.impl->shape[0], in_c = input.impl->shape[1], depth = input.impl->shape[2], height = input.impl->shape[3], width = input.impl->shape[4];
    size_t out_c = weight.impl->shape[1], k_d = weight.impl->shape[2], k_h = weight.impl->shape[3], k_w = weight.impl->shape[4];
    size_t out_d = grad_output.impl->shape[2], out_h = grad_output.impl->shape[3], out_w = grad_output.impl->shape[4];

    const float* go_ptr = (const float*)grad_output.impl->data->data.get() + grad_output.impl->offset;
    const float* in_ptr = (const float*)input.impl->data->data.get() + input.impl->offset;
    const float* w_ptr = (const float*)weight.impl->data->data.get() + weight.impl->offset;
    float* gi_ptr = (float*)grad_input.impl->data->data.get() + grad_input.impl->offset;
    float* gw_ptr = (float*)grad_weight.impl->data->data.get() + grad_weight.impl->offset;
    float* gb_ptr = (float*)grad_bias.impl->data->data.get() + grad_bias.impl->offset;

    size_t go_s0 = grad_output.impl->strides[0], go_s1 = grad_output.impl->strides[1], go_s2 = grad_output.impl->strides[2], go_s3 = grad_output.impl->strides[3], go_s4 = grad_output.impl->strides[4];
    size_t i_s0 = input.impl->strides[0], i_s1 = input.impl->strides[1], i_s2 = input.impl->strides[2], i_s3 = input.impl->strides[3], i_s4 = input.impl->strides[4];
    size_t w_s0 = weight.impl->strides[0], w_s1 = weight.impl->strides[1], w_s2 = weight.impl->strides[2], w_s3 = weight.impl->strides[3], w_s4 = weight.impl->strides[4];
    size_t gi_s0 = grad_input.impl->strides[0], gi_s1 = grad_input.impl->strides[1], gi_s2 = grad_input.impl->strides[2], gi_s3 = grad_input.impl->strides[3], gi_s4 = grad_input.impl->strides[4];
    size_t gw_s0 = grad_weight.impl->strides[0], gw_s1 = grad_weight.impl->strides[1], gw_s2 = grad_weight.impl->strides[2], gw_s3 = grad_weight.impl->strides[3], gw_s4 = grad_weight.impl->strides[4];

    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch; ++b) {
        for (int ic = 0; ic < in_c; ++ic) {
            for (int id = 0; id < depth; ++id) {
                for (int ih = 0; ih < height; ++ih) {
                    for (int iw = 0; iw < width; ++iw) {
                        double sum_in = 0.0;
                        for (int oc = 0; oc < out_c; ++oc) {
                            for (int kd = 0; kd < k_d; ++kd) {
                                for (int kh = 0; kh < k_h; ++kh) {
                                    for (int kw = 0; kw < k_w; ++kw) {
                                        int od = id * stride_d - padding_d + kd;
                                        int oh = ih * stride_h - padding_h + kh;
                                        int ow = iw * stride_w - padding_w + kw;
                                        if (od >= 0 && od < out_d && oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                                            double go = go_ptr[b*go_s0 + oc*go_s1 + od*go_s2 + oh*go_s3 + ow*go_s4];
                                            double w_val = w_ptr[ic*w_s0 + oc*w_s1 + kd*w_s2 + kh*w_s3 + kw*w_s4];
                                            sum_in += go * w_val;
                                            double in_val = in_ptr[b*i_s0 + ic*i_s1 + id*i_s2 + ih*i_s3 + iw*i_s4];
                                            float* gw = &gw_ptr[ic*gw_s0 + oc*gw_s1 + kd*gw_s2 + kh*gw_s3 + kw*gw_s4];
                                            #pragma omp atomic
                                            *gw += (float)(go * in_val);
                                        }
                                    }
                                }
                            }
                        }
                        gi_ptr[b*gi_s0 + ic*gi_s1 + id*gi_s2 + ih*gi_s3 + iw*gi_s4] += sum_in;
                    }
                }
            }
        }
    }

    if (bias.requires_grad()) {
        for (int oc = 0; oc < out_c; ++oc) {
            double sum_b = 0.0;
            for (size_t b = 0; b < batch; ++b) {
                for (int od = 0; od < out_d; ++od) {
                    for (int oh = 0; oh < out_h; ++oh) {
                        for (int ow = 0; ow < out_w; ++ow) {
                            sum_b += go_ptr[b*go_s0 + oc*go_s1 + od*go_s2 + oh*go_s3 + ow*go_s4];
                        }
                    }
                }
            }
            gb_ptr[oc] += sum_b;
        }
    }

    if (input.requires_grad()) accumulate_grad(input, grad_input);
    if (weight.requires_grad()) accumulate_grad(weight, grad_weight);
    if (bias.requires_grad()) accumulate_grad(bias, grad_bias);
}