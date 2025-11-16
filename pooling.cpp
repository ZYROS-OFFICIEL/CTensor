#include "tensor1.h"
#include <vector>
#include <stdexcept>
#include <cmath>
#include "autograd.h"
#include <immintrin.h>
#include <cstring>
#include "ops1.h"
#include <string>
#pragma once

//----------------Helpers---------------------------------------------
Tensor im2col_2d_pool(const Tensor& input,
                      int kernel_h, int kernel_w,
                      int stride_h, int stride_w,
                      int pad_h, int pad_w){
    // Calculate output dimensions
    int batch_size = input.shape()[0];
    int in_channels = input.shape()[1];
    int in_h = input.shape()[2];
    int in_w = input.shape()[3];
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    Tensor cols = Tensor::zeros({batch_size, in_channels, kernel_h, kernel_w, out_h, out_w}, input._dtype());
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_channels; ++c) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    for (int oh = 0; oh < out_h; ++oh) {
                        for (int ow = 0; ow < out_w; ++ow) {
                            int ih = oh * stride_h - pad_h + kh;
                            int iw = ow * stride_w - pad_w + kw;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                cols.write_scalar((((b * in_channels + c) * kernel_h + kh) * kernel_w + kw) * out_h * out_w + oh * out_w + ow,
                                                  input.read_scalar(((b * in_channels + c) * in_h + ih) * in_w + iw));
                            }
                        }
                    }
                }
            }
        }
    }
    return cols;
}
void col2im_2d_pool(const Tensor& grad_patches,
                    Tensor& grad_input,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w){

    int batch_size = grad_input.shape()[0];
    int in_channels = grad_input.shape()[1];
    int in_h = grad_input.shape()[2];
    int in_w = grad_input.shape()[3];
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_channels; ++c) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    for (int oh = 0; oh < out_h; ++oh) {
                        for (int ow = 0; ow < out_w; ++ow) {
                            int ih = oh * stride_h - pad_h + kh;
                            int iw = ow * stride_w - pad_w + kw;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                double val = grad_patches.read_scalar((((b * in_channels + c) * kernel_h + kh) * kernel_w + kw) * out_h * out_w + oh * out_w + ow);
                                double existing = grad_input.read_scalar(((b * in_channels + c) * in_h + ih) * in_w + iw);
                                grad_input.write_scalar(((b * in_channels + c) * in_h + ih) * in_w + iw, existing + val);
                            }

                        }

                    }

                }

            }

        }

    }

}



Tensor im2col_3d_pool(const Tensor& input,
                      int kernel_d, int kernel_h, int kernel_w,
                      int stride_d, int stride_h, int stride_w,
                      int pad_d, int pad_h, int pad_w){
    // Calculate output dimensions
    int batch_size = input.shape()[0];
    int in_channels = input.shape()[1];
    int in_d = input.shape()[2];
    int in_h = input.shape()[3];
    int in_w = input.shape()[4];
    int out_d = (in_d + 2 * pad_d - kernel_d) / stride_d + 1;
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    Tensor cols = Tensor::zeros({batch_size, in_channels, kernel_d, kernel_h, kernel_w, out_d, out_h, out_w}, input._dtype());
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_channels; ++c) {
            for (int kd = 0; kd < kernel_d; ++kd) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        for (int od = 0; od < out_d; ++od) {
                            for (int oh = 0; oh < out_h; ++oh) {
                                for (int ow = 0; ow < out_w; ++ow) {
                                    int id = od * stride_d - pad_d + kd;
                                    int ih = oh * stride_h - pad_h + kh;
                                    int iw = ow * stride_w - pad_w + kw;
                                    if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                        cols.write_scalar((((((b * in_channels + c) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw) * out_d * out_h * out_w) + (od * out_h * out_w) + (oh * out_w) + ow,
                                                          input.read_scalar((((b * in_channels + c) * in_d + id) * in_h + ih) * in_w + iw));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    return cols;}
}
void col2im_3d_pool(const Tensor& grad_patches,
                    Tensor& grad_input,
                    int kernel_d, int kernel_h, int kernel_w,
                    int stride_d, int stride_h, int stride_w,
                    int pad_d, int pad_h, int pad_w){
    int batch_size = grad_input.shape()[0];
    int in_channels = grad_input.shape()[1];
    int in_d = grad_input.shape()[2];
    int in_h = grad_input.shape()[3];
    int in_w = grad_input.shape()[4];
    int out_d = (in_d + 2 * pad_d - kernel_d) / stride_d + 1;
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_channels; ++c) {
            for (int kd = 0; kd < kernel_d; ++kd) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        for (int od = 0; od < out_d; ++od) {
                            for (int oh = 0; oh < out_h; ++oh) {
                                for (int ow = 0; ow < out_w; ++ow) {
                                    int id = od * stride_d - pad_d + kd;
                                    int ih = oh * stride_h - pad_h + kh;
                                    int iw = ow * stride_w - pad_w + kw;
                                    if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                        double val = grad_patches.read_scalar((((((b * in_channels + c) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw) * out_d * out_h * out_w) + (od * out_h * out_w) + (oh * out_w) + ow);
                                        double existing = grad_input.read_scalar((((b * in_channels + c) * in_d + id) * in_h + ih) * in_w + iw);
                                        grad_input.write_scalar((((b * in_channels + c) * in_d + id) * in_h + ih) * in_w + iw, existing + val);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

//----------------Pooling Classes---------------------------------------------
MaxPool1d::MaxPool1d(int k, int s, int p)
    : kernel_size(k), stride(s), padding(p) {}
// forward (MaxPool1d)
Tensor MaxPool1d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("MaxPool1d::forward: null input");
    if (input.impl->ndim != 3)
        throw std::runtime_error("MaxPool1d forward: input must be [batch, channels, width]");

    // input shape assumed [batch, channels, width]
    size_t batch = input.impl->shape[0];
    size_t channels  = input.impl->shape[1];
    size_t width = input.impl->shape[2];

    int out_w = (int)(( (int)width + 2 * padding - kernel_size) / stride + 1);
    if (out_w <= 0) throw std::runtime_error("MaxPool1d::forward: invalid output width");

    std::vector<size_t> out_shape = { batch, channels, (size_t)out_w };
    Tensor output(out_shape, input._dtype(), input.requires_grad());

    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (int ow = 0; ow < out_w; ++ow) {
                double max_val = -std::numeric_limits<double>::infinity();
                for (int k = 0; k < kernel_size; ++k) {
                    int iw = ow * stride + k - padding;
                    if (iw >= 0 && iw < (int)width) {
                        double in_val = input[b][c][(size_t)iw];
                        if (in_val > max_val) {
                            max_val = in_val;
                        }
                    }
                }
                output[b][c][(size_t)ow] = max_val;
            }
        }
    }
    return output;
}
// forward (MaxPool2d)
Tensor MaxPool2d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("MaxPool2d::forward: null input");
    if (input.impl->ndim != 4)
        throw std::runtime_error("MaxPool2d forward: input must be [batch, channels, height, width]");

    // input shape assumed [batch, channels, height, width]
    size_t batch = input.impl->shape[0];
    size_t channels  = input.impl->shape[1];
    size_t height = input.impl->shape[2];
    size_t width = input.impl->shape[3];

    int out_h = (int)(( (int)height + 2 * padding_h - kernel_size_h) / stride_h + 1);
    int out_w = (int)(( (int)width + 2 * padding_w - kernel_size_w) / stride_w + 1);
    if (out_h <= 0 || out_w <= 0) throw std::runtime_error("MaxPool2d::forward: invalid output dimensions");

    std::vector<size_t> out_shape = { batch, channels, (size_t)out_h, (size_t)out_w };
    Tensor output(out_shape, input._dtype(), input.requires_grad());

    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    double max_val = -std::numeric_limits<double>::infinity();
                    for (int kh = 0; kh < kernel_size_h; ++kh) {
                        for (int kw = 0; kw < kernel_size_w; ++kw) {
                            int ih = oh * stride_h + kh - padding_h;
                            int iw = ow * stride_w + kw - padding_w;
                            if (ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width) {
                                double in_val = input[b][c][(size_t)ih][(size_t)iw];
                                if (in_val > max_val) {
                                    max_val = in_val;
                                }
                            }
                        }
                    }
                    output[b][c][(size_t)oh][(size_t)ow] = max_val;
                }
            }
        }
    return output;
    }
}
// forward (MaxPool3d)
Tensor MaxPool3d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("MaxPool3d::forward: null input");
    if (input.impl->ndim != 5)
        throw std::runtime_error("MaxPool3d forward: input must be [batch, channels, depth, height, width]");

    // input shape assumed [batch, channels, depth, height, width]
    size_t batch = input.impl->shape[0];
    size_t channels  = input.impl->shape[1];
    size_t depth = input.impl->shape[2];
    size_t height = input.impl->shape[3];
    size_t width = input.impl->shape[4];

    int out_d = (int)(( (int)depth + 2 * padding_d - kernel_size_d) / stride_d + 1);
    int out_h = (int)(( (int)height + 2 * padding_h - kernel_size_h) / stride_h + 1);
    int out_w = (int)(( (int)width + 2 * padding_w - kernel_size_w) / stride_w + 1);
    if (out_d <= 0 || out_h <= 0 || out_w <= 0) throw std::runtime_error("MaxPool3d::forward: invalid output dimensions");

    std::vector<size_t> out_shape = { batch, channels, (size_t)out_d, (size_t)out_h, (size_t)out_w };
    Tensor output(out_shape, input._dtype(), input.requires_grad());

    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (int od = 0; od < out_d; ++od) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        double max_val = -std::numeric_limits<double>::infinity();
                        for (int kd = 0; kd < kernel_size_d; ++kd) {
                            for (int kh = 0; kh < kernel_size_h; ++kh) {
                                for (int kw = 0; kw < kernel_size_w; ++kw) {
                                    int id = od * stride_d + kd - padding_d;
                                    int ih = oh * stride_h + kh - padding_h;
                                    int iw = ow * stride_w + kw - padding_w;
                                    if (id >= 0 && id < (int)depth && ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width) {
                                        double in_val = input[b][c][(size_t)id][(size_t)ih][(size_t)iw];
                                        if (in_val > max_val) {
                                            max_val = in_val;
                                        }
                                    }
                                }
                            }
                        output[b][c][(size_t)od][(size_t)oh][(size_t)ow] = max_val;
                        }
                    }
                }
            }
        }
    return output;
    }
}
AvgPool1d::AvgPool1d(int k, int s, int p)
    : kernel_size(k), stride(s), padding(p) {}
// forward (AvgPool1d)
Tensor AvgPool1d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("AvgPool1d::forward: null input");
    if (input.impl->ndim != 3)
        throw std::runtime_error("AvgPool1d forward: input must be [batch, channels, width]");

    // input shape assumed [batch, channels, width]
    size_t batch = input.impl->shape[0];
    size_t channels  = input.impl->shape[1];
    size_t width = input.impl->shape[2];

    int out_w = (int)(( (int)width + 2 * padding - kernel_size) / stride + 1);
    if (out_w <= 0) throw std::runtime_error("AvgPool1d::forward: invalid output width");

    std::vector<size_t> out_shape = { batch, channels, (size_t)out_w };
    Tensor output(out_shape, input._dtype(), input.requires_grad());

    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (int ow = 0; ow < out_w; ++ow) {
                double sum_val = 0.0;
                int count = 0;
                for (int k = 0; k < kernel_size; ++k) {
                    int iw = ow * stride + k - padding;
                    if (iw >= 0 && iw < (int)width) {
                        double in_val = input[b][c][(size_t)iw];
                        sum_val += in_val;
                        count += 1;
                    }
                }
                output[b][c][(size_t)ow] = sum_val / count;
            }
        }
    }
    return output;
}
// forward (AvgPool2d)
Tensor AvgPool2d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("AvgPool2d::forward: null input");
    if (input.impl->ndim != 4)
        throw std::runtime_error("AvgPool2d forward: input must be [batch, channels, height, width]");

    // input shape assumed [batch, channels, height, width]
    size_t batch = input.impl->shape[0];
    size_t channels  = input.impl->shape[1];
    size_t height = input.impl->shape[2];
    size_t width = input.impl->shape[3];

    int out_h = (int)(( (int)height + 2 * padding_h - kernel_size_h) / stride_h + 1);
    int out_w = (int)(( (int)width + 2 * padding_w - kernel_size_w) / stride_w + 1);
    if (out_h <= 0 || out_w <= 0) throw std::runtime_error("AvgPool2d::forward: invalid output dimensions");

    std::vector<size_t> out_shape = { batch, channels, (size_t)out_h, (size_t)out_w };
    Tensor output(out_shape, input._dtype(), input.requires_grad());

    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    double sum_val = 0.0;
                    int count = 0;
                    for (int kh = 0; kh < kernel_size_h; ++kh) {
                        for (int kw = 0; kw < kernel_size_w; ++kw) {
                            int ih = oh * stride_h + kh - padding_h;
                            int iw = ow * stride_w + kw - padding_w;
                            if (ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width) {
                                double in_val = input[b][c][(size_t)ih][(size_t)iw];
                                sum_val += in_val;
                                count += 1;
                            }
                        }
                    }
                    output[b][c][(size_t)oh][(size_t)ow] = sum_val / count;
                }
            }
        }
    return output;
    }
}
// forward (AvgPool3d)
Tensor AvgPool3d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("AvgPool3d::forward: null input");
    if (input.impl->ndim != 5)
        throw std::runtime_error("AvgPool3d forward: input must be [batch, channels, depth, height, width]");

    // input shape assumed [batch, channels, depth, height, width]
    size_t batch = input.impl->shape[0];
    size_t channels  = input.impl->shape[1];
    size_t depth = input.impl->shape[2];
    size_t height = input.impl->shape[3];
    size_t width = input.impl->shape[4];

    int out_d = (int)(( (int)depth + 2 * padding_d - kernel_size_d) / stride_d + 1);
    int out_h = (int)(( (int)height + 2 * padding_h - kernel_size_h) / stride_h + 1);
    int out_w = (int)(( (int)width + 2 * padding_w - kernel_size_w) / stride_w + 1);
    if (out_d <= 0 || out_h <= 0 || out_w <= 0) throw std::runtime_error("AvgPool3d::forward: invalid output dimensions");

    std::vector<size_t> out_shape = { batch, channels, (size_t)out_d, (size_t)out_h, (size_t)out_w };
    Tensor output(out_shape, input._dtype(), input.requires_grad());

    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (int od = 0; od < out_d; ++od) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        double sum_val = 0.0;
                        int count = 0;
                        for (int kd = 0; kd < kernel_size_d; ++kd) {
                            for (int kh = 0; kh < kernel_size_h; ++kh) {
                                for (int kw = 0; kw < kernel_size_w; ++kw) {
                                    int id = od * stride_d + kd - padding_d;
                                    int ih = oh * stride_h + kh - padding_h;
                                    int iw = ow * stride_w + kw - padding_w;
                                    if (id >= 0 && id < (int)depth && ih >= 0
                                        && ih < (int)height && iw >= 0 && iw < (int)width) {
                                        double in_val = input[b][c][(size_t)id][(size_t)ih][(size_t)iw];
                                        sum_val += in_val;
                                        count += 1;
                                    }
                                }
                            }
                        output[b][c][(size_t)od][(size_t)oh][(size_t)ow] = sum_val / count;
                        }
                    }
                }
            }
        }
    return output;
    }
}
void GradMaxPool1d::backward(const Tensor& self) {
    Tensor input = parents[0];
    if (!input.impl || !self.impl) throw std::runtime_error("GradMaxPool1d::backward: null tensor");
    if (input.impl->ndim != 3 || self.impl->ndim != 3)
        throw std::runtime_error("GradMaxPool1d backward: input and self must be [batch, channels, width]");

    // input shape assumed [batch, channels, width]
    size_t batch = input.impl->shape[0];
    size_t channels  = input.impl->shape[1];
    size_t width = input.impl->shape[2];

    int out_w = (int)(( (int)width + 2 * padding - kernel_size) / stride + 1);
    if (out_w <= 0) throw std::runtime_error("GradMaxPool1d::backward: invalid output width");

    Tensor grad_input = Tensor::zeros(input.shape(), input._dtype(), false);
    Tensor grad_output = self.impl->storage->grad;

    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (int ow = 0; ow < out_w; ++ow) {
                double max_val = -std::numeric_limits<double>::infinity();
                int max_idx = -1;
                for (int k = 0; k < kernel_size; ++k) {
                    int iw = ow * stride + k - padding;
                    if (iw >= 0 && iw < (int)width) {
                        double in_val = input[b][c][(size_t)iw];
                        if (in_val > max_val) {
                            max_val = in_val;
                            max_idx = iw;
                        }
                    }
                }
                if (max_idx != -1) {
                    double grad_out_val = grad_output[b][c][(size_t)ow];
                    double existing_grad = grad_input[b][c][(size_t)max_idx];
                    grad_input.write_scalar(((b * channels + c) * width + max_idx),
                                            existing_grad + grad_out_val);
                }
            }
        }
    }
    input.impl->storage->grad = grad_input.impl->storage->grad;
}
void GradMaxPool2d::backward(const Tensor& self) {
    Tensor input = parents[0];
    if (!input.impl || !self.impl) throw std::runtime_error("GradMaxPool2d::backward: null tensor");
    if (input.impl->ndim != 4 || self.impl->ndim != 4)
        throw std::runtime_error("GradMaxPool2d backward: input and self must be [batch, channels, height, width]");

    // input shape assumed [batch, channels, height, width]
    size_t batch = input.impl->shape[0];
    size_t channels  = input.impl->shape[1];
    size_t height = input.impl->shape[2];
    size_t width = input.impl->shape[3];

    int out_h = (int)(( (int)height + 2 * padding_h - kernel_size_h) / stride_h + 1);
    int out_w = (int)(( (int)width + 2 * padding_w - kernel_size_w) / stride_w + 1);
    if (out_h <= 0 || out_w <= 0) throw std::runtime_error("GradMaxPool2d::backward: invalid output dimensions");

    Tensor grad_input = Tensor::zeros(input.shape(), input._dtype(), false);
    Tensor grad_output = self.impl->storage->grad;

    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    double max_val = -std::numeric_limits<double>::infinity();
                    int max_ih = -1;
                    int max_iw = -1;
                    for (int kh = 0; kh < kernel_size_h; ++kh) {
                        for (int kw = 0; kw < kernel_size_w; ++kw) {
                            int ih = oh * stride_h + kh - padding_h;
                            int iw = ow * stride_w + kw - padding_w;
                            if (ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width) {
                                double in_val = input[b][c][(size_t)ih][(size_t)iw];
                                if (in_val > max_val) {
                                    max_val = in_val;
                                    max_ih = ih;
                                    max_iw = iw;
                                }
                            }
                        }
                    if (max_ih != -1 && max_iw != -1) {
                        double grad_out_val = grad_output[b][c][(size_t)oh][(size_t)ow];
                        double existing_grad = grad_input[b][c][(size_t)max_ih][(size_t)max_iw];
                        grad_input.write_scalar((( (b * channels + c) * height + max_ih) * width + max_iw),
                                                existing_grad + grad_out_val);
                    }
                }
            }
        }
    }
    input.impl->storage->grad = grad_input.impl->storage->grad;
}
void GradMaxPool3d::backward(const Tensor& self) {
    Tensor input = parents[0];
    if (!input.impl || !self.impl) throw std::runtime_error("GradMaxPool3d::backward: null tensor");
    if (input.impl->ndim != 5 || self.impl->ndim != 5)
        throw std::runtime_error("GradMaxPool3d backward: input and self must be [batch, channels, depth, height, width]");

    // input shape assumed [batch, channels, depth, height, width]
    size_t batch = input.impl->shape[0];
    size_t channels  = input.impl->shape[1];
    size_t depth = input.impl->shape[2];
    size_t height = input.impl->shape[3];
    size_t width = input.impl->shape[4];

    int out_d = (int)(( (int)depth + 2 * padding_d - kernel_size_d) / stride_d + 1);
    int out_h = (int)(( (int)height + 2 * padding_h - kernel_size_h) / stride_h + 1);
    int out_w = (int)(( (int)width + 2 * padding_w - kernel_size_w) / stride_w + 1);
    if (out_d <= 0 || out_h <= 0 || out_w <= 0) throw std::runtime_error("GradMaxPool3d::backward: invalid output dimensions");

    Tensor grad_input = Tensor::zeros(input.shape(), input._dtype(), false);
    Tensor grad_output = self.impl->storage->grad;

    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (int od = 0; od < out_d; ++od) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        double max_val = -std::numeric_limits<double>::infinity();
                        int max_id = -1;
                        int max_ih = -1;
                        int max_iw = -1;
                        for (int kd = 0; kd < kernel_size_d; ++kd) {
                            for (int kh = 0; kh < kernel_size_h; ++kh) {
                                for (int kw = 0; kw < kernel_size_w; ++kw) {
                                    int id = od * stride_d + kd - padding_d;
                                    int ih = oh * stride_h + kh - padding_h;
                                    int iw = ow * stride_w + kw - padding_w;
                                    if (id >= 0 && id < (int)depth && ih >= 0 && ih < (int)height && iw >= 0 && iw < (int)width) {
                                        double in_val = input[b][c][(size_t)id][(size_t)ih][(size_t)iw];
                                        if (in_val > max_val) {
                                            max_val = in_val;
                                            max_id = id;
                                            max_ih = ih;
                                            max_iw = iw;
                                        }
                                    }
                                }
                            if (max_id != -1 && max_ih != -1 && max_iw != -1) {
                                double grad_out_val = grad_output[b][c][(size_t)od][(size_t)oh][(size_t)ow];
                                double existing_grad = grad_input[b][c][(size_t)max_id][(size_t)max_ih][(size_t)max_iw];
                                grad_input.write_scalar((((b * channels + c) * depth + max_id) * height + max_ih) * width + max_iw,
                                                        existing_grad + grad_out_val);
                            }
                        }
                    }
                }
            }
        }
    input.impl->storage->grad = grad_input.impl->storage->grad;
}
