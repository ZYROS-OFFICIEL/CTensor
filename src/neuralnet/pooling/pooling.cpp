#include "pooling.h"
#include "ops_dispatch.h"
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <omp.h>

// ======================================================================================
//                                      HELPERS
// ======================================================================================
Tensor im2col_2d_pool(const Tensor& input, int kh, int kw, int sh, int sw, int ph, int pw) { return Tensor(); }
void col2im_2d_pool(const Tensor& gp, Tensor& gi, int kh, int kw, int sh, int sw, int ph, int pw) {}
Tensor im2col_3d_pool(const Tensor& i, int kd, int kh, int kw, int sd, int sh, int sw, int pd, int ph, int pw) { return Tensor(); }
void col2im_3d_pool(const Tensor& gp, Tensor& gi, int kd, int kh, int kw, int sd, int sh, int sw, int pd, int ph, int pw) {}


// ======================================================================================
//                                   MAX POOLING
// ======================================================================================

// --- MaxPool1d ---
MaxPool1d::MaxPool1d(int k, int s, int p) : kernel_size(k), stride(s), padding(p) {
    if (stride == -1) stride = kernel_size;
}

Tensor MaxPool1d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("MaxPool1d: null input");
    size_t N = input.impl->shape[0];
    size_t C = input.impl->shape[1];
    size_t L = input.impl->shape[2];

    int out_l = (int)((L + 2 * padding - kernel_size) / stride + 1);
    if (out_l <= 0) throw std::runtime_error("MaxPool1d: invalid output dims");

    Tensor output({N, C, (size_t)out_l}, input._dtype(), input.requires_grad());

    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (int i = 0; i < out_l; ++i) {
                int start = i * stride - padding;
                int end = std::min((int)L, start + kernel_size);
                start = std::max(0, start);

                double max_val = -std::numeric_limits<double>::infinity();
                
                for (int k = start; k < end; ++k) {
                    size_t offset = input.impl->offset + 
                                    n * input.impl->strides[0] + 
                                    c * input.impl->strides[1] + 
                                    k * input.impl->strides[2];
                    double val = read_scalar_at(input.impl->storage->data.get(), offset, input._dtype());
                    if (val > max_val) max_val = val;
                }
                
                size_t out_offset = output.impl->offset + 
                                    n * output.impl->strides[0] + 
                                    c * output.impl->strides[1] + 
                                    i * output.impl->strides[2];
                write_scalar_at(output.impl->storage->data.get(), out_offset, output._dtype(), max_val);
            }
        }
    }

    if (input.requires_grad()) {
        output.impl->grad_fn = std::make_shared<GradMaxPool1d>(input, kernel_size, stride, padding);
    }
    return output;
}

void GradMaxPool1d::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradMaxPool1d: missing self grad");
    Tensor grad_output = tensor_from_grad(self);
    Tensor grad_input = Tensor::zeros(input.shape(), input._dtype(), false);

    size_t N = input.impl->shape[0];
    size_t C = input.impl->shape[1];
    size_t L = input.impl->shape[2];
    int out_l = grad_output.impl->shape[2];

    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (int i = 0; i < out_l; ++i) {
                size_t go_idx = grad_output.impl->offset + n * grad_output.impl->strides[0] + c * grad_output.impl->strides[1] + i * grad_output.impl->strides[2];
                double g = read_scalar_at(grad_output.impl->storage->data.get(), go_idx, grad_output._dtype());

                int start = i * stride - padding;
                int end = std::min((int)L, start + kernel_size);
                start = std::max(0, start);

                double max_val = -std::numeric_limits<double>::infinity();
                int max_idx = -1;

                for (int k = start; k < end; ++k) {
                    size_t in_idx = input.impl->offset + n * input.impl->strides[0] + c * input.impl->strides[1] + k * input.impl->strides[2];
                    double val = read_scalar_at(input.impl->storage->data.get(), in_idx, input._dtype());
                    if (val > max_val) {
                        max_val = val;
                        max_idx = k;
                    }
                }

                if (max_idx != -1) {
                    size_t in_idx = input.impl->offset + n * input.impl->strides[0] + c * input.impl->strides[1] + max_idx * input.impl->strides[2];
                    double cur = read_scalar_at(grad_input.impl->storage->data.get(), in_idx, grad_input._dtype());
                    write_scalar_at(grad_input.impl->storage->data.get(), in_idx, grad_input._dtype(), cur + g);
                }
            }
        }
    }
    accumulate_grad(input, grad_input);
}

// --- MaxPool2d ---
MaxPool2d::MaxPool2d(int kh, int kw, int sh, int sw, int ph, int pw) 
    : kernel_size_h(kh), kernel_size_w(kw), stride_h(sh), stride_w(sw), padding_h(ph), padding_w(pw) {
    if (stride_h == -1) stride_h = kernel_size_h;
    if (stride_w == -1) stride_w = kernel_size_w;
}

Tensor MaxPool2d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("MaxPool2d: null input");
    size_t N = input.impl->shape[0];
    size_t C = input.impl->shape[1];
    size_t H = input.impl->shape[2];
    size_t W = input.impl->shape[3];

    int out_h = (int)((H + 2 * padding_h - kernel_size_h) / stride_h + 1);
    int out_w = (int)((W + 2 * padding_w - kernel_size_w) / stride_w + 1);
    if (out_h <= 0 || out_w <= 0) throw std::runtime_error("MaxPool2d: invalid output dims");

    Tensor output({N, C, (size_t)out_h, (size_t)out_w}, input._dtype(), input.requires_grad());

    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int h_start = std::max(0, oh * stride_h - padding_h);
                    int w_start = std::max(0, ow * stride_w - padding_w);
                    int h_end = std::min((int)H, oh * stride_h - padding_h + kernel_size_h);
                    int w_end = std::min((int)W, ow * stride_w - padding_w + kernel_size_w);

                    double max_val = -std::numeric_limits<double>::infinity();

                    for (int h = h_start; h < h_end; ++h) {
                        for (int w = w_start; w < w_end; ++w) {
                            size_t off = input.impl->offset + 
                                         n * input.impl->strides[0] + c * input.impl->strides[1] + 
                                         h * input.impl->strides[2] + w * input.impl->strides[3];
                            double v = read_scalar_at(input.impl->storage->data.get(), off, input._dtype());
                            if (v > max_val) max_val = v;
                        }
                    }
                    size_t out_off = output.impl->offset + 
                                     n * output.impl->strides[0] + c * output.impl->strides[1] + 
                                     oh * output.impl->strides[2] + ow * output.impl->strides[3];
                    write_scalar_at(output.impl->storage->data.get(), out_off, output._dtype(), max_val);
                }
            }
        }
    }

    if (input.requires_grad()) {
        output.impl->grad_fn = std::make_shared<GradMaxPool2d>(input, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w);
    }
    return output;
}

void GradMaxPool2d::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradMaxPool2d: missing self grad");
    Tensor grad_output = tensor_from_grad(self);
    Tensor grad_input = Tensor::zeros(input.shape(), input._dtype(), false);

    size_t N = input.impl->shape[0];
    size_t C = input.impl->shape[1];
    size_t H = input.impl->shape[2];
    size_t W = input.impl->shape[3];
    int out_h = grad_output.impl->shape[2];
    int out_w = grad_output.impl->shape[3];

    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    size_t g_off = grad_output.impl->offset + n * grad_output.impl->strides[0] + c * grad_output.impl->strides[1] + oh * grad_output.impl->strides[2] + ow * grad_output.impl->strides[3];
                    double g = read_scalar_at(grad_output.impl->storage->data.get(), g_off, grad_output._dtype());

                    int h_start = std::max(0, oh * stride_h - padding_h);
                    int w_start = std::max(0, ow * stride_w - padding_w);
                    int h_end = std::min((int)H, oh * stride_h - padding_h + kernel_size_h);
                    int w_end = std::min((int)W, ow * stride_w - padding_w + kernel_size_w);

                    double max_val = -std::numeric_limits<double>::infinity();
                    int max_h = -1, max_w = -1;

                    for (int h = h_start; h < h_end; ++h) {
                        for (int w = w_start; w < w_end; ++w) {
                            size_t off = input.impl->offset + n * input.impl->strides[0] + c * input.impl->strides[1] + h * input.impl->strides[2] + w * input.impl->strides[3];
                            double v = read_scalar_at(input.impl->storage->data.get(), off, input._dtype());
                            if (v > max_val) {
                                max_val = v;
                                max_h = h;
                                max_w = w;
                            }
                        }
                    }

                    if (max_h != -1) {
                        size_t in_off = input.impl->offset + n * input.impl->strides[0] + c * input.impl->strides[1] + max_h * input.impl->strides[2] + max_w * input.impl->strides[3];
                        double cur = read_scalar_at(grad_input.impl->storage->data.get(), in_off, grad_input._dtype());
                        write_scalar_at(grad_input.impl->storage->data.get(), in_off, grad_input._dtype(), cur + g);
                    }
                }
            }
        }
    }
    accumulate_grad(input, grad_input);
}


// --- MaxPool3d ---
MaxPool3d::MaxPool3d(int kd, int kh, int kw, int sd, int sh, int sw, int pd, int ph, int pw)
    : kernel_size_d(kd), kernel_size_h(kh), kernel_size_w(kw),
      stride_d(sd), stride_h(sh), stride_w(sw),
      padding_d(pd), padding_h(ph), padding_w(pw) {
    if (stride_d == -1) stride_d = kernel_size_d;
    if (stride_h == -1) stride_h = kernel_size_h;
    if (stride_w == -1) stride_w = kernel_size_w;
}

Tensor MaxPool3d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("MaxPool3d: null input");
    size_t N = input.impl->shape[0];
    size_t C = input.impl->shape[1];
    size_t D = input.impl->shape[2];
    size_t H = input.impl->shape[3];
    size_t W = input.impl->shape[4];

    int out_d = (int)((D + 2 * padding_d - kernel_size_d) / stride_d + 1);
    int out_h = (int)((H + 2 * padding_h - kernel_size_h) / stride_h + 1);
    int out_w = (int)((W + 2 * padding_w - kernel_size_w) / stride_w + 1);
    
    if (out_d <= 0 || out_h <= 0 || out_w <= 0) throw std::runtime_error("MaxPool3d: invalid output dims");

    Tensor output({N, C, (size_t)out_d, (size_t)out_h, (size_t)out_w}, input._dtype(), input.requires_grad());

    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (int od = 0; od < out_d; ++od) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        int d_start = std::max(0, od * stride_d - padding_d);
                        int h_start = std::max(0, oh * stride_h - padding_h);
                        int w_start = std::max(0, ow * stride_w - padding_w);
                        int d_end = std::min((int)D, od * stride_d - padding_d + kernel_size_d);
                        int h_end = std::min((int)H, oh * stride_h - padding_h + kernel_size_h);
                        int w_end = std::min((int)W, ow * stride_w - padding_w + kernel_size_w);

                        double max_val = -std::numeric_limits<double>::infinity();

                        for (int d_idx = d_start; d_idx < d_end; ++d_idx) {
                            for (int h = h_start; h < h_end; ++h) {
                                for (int w = w_start; w < w_end; ++w) {
                                    size_t off = input.impl->offset + 
                                        n*input.impl->strides[0] + c*input.impl->strides[1] + 
                                        d_idx*input.impl->strides[2] + h*input.impl->strides[3] + w*input.impl->strides[4];
                                    double v = read_scalar_at(input.impl->storage->data.get(), off, input._dtype());
                                    if (v > max_val) max_val = v;
                                }
                            }
                        }
                        size_t out_off = output.impl->offset + 
                            n*output.impl->strides[0] + c*output.impl->strides[1] + 
                            od*output.impl->strides[2] + oh*output.impl->strides[3] + ow*output.impl->strides[4];
                        write_scalar_at(output.impl->storage->data.get(), out_off, output._dtype(), max_val);
                    }
                }
            }
        }
    }

    if (input.requires_grad()) {
        output.impl->grad_fn = std::make_shared<GradMaxPool3d>(input, kernel_size_d, kernel_size_h, kernel_size_w, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w);
    }
    return output;
}

void GradMaxPool3d::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradMaxPool3d: missing self grad");
    Tensor grad_output = tensor_from_grad(self);
    Tensor grad_input = Tensor::zeros(input.shape(), input._dtype(), false);

    size_t N = input.impl->shape[0];
    size_t C = input.impl->shape[1];
    size_t D = input.impl->shape[2];
    size_t H = input.impl->shape[3];
    size_t W = input.impl->shape[4];
    
    int out_d = grad_output.impl->shape[2];
    int out_h = grad_output.impl->shape[3];
    int out_w = grad_output.impl->shape[4];

    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (int od = 0; od < out_d; ++od) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        size_t g_off = grad_output.impl->offset + n*grad_output.impl->strides[0] + c*grad_output.impl->strides[1] + od*grad_output.impl->strides[2] + oh*grad_output.impl->strides[3] + ow*grad_output.impl->strides[4];
                        double g = read_scalar_at(grad_output.impl->storage->data.get(), g_off, grad_output._dtype());

                        int d_start = std::max(0, od * stride_d - padding_d);
                        int h_start = std::max(0, oh * stride_h - padding_h);
                        int w_start = std::max(0, ow * stride_w - padding_w);
                        int d_end = std::min((int)D, od * stride_d - padding_d + kernel_size_d);
                        int h_end = std::min((int)H, oh * stride_h - padding_h + kernel_size_h);
                        int w_end = std::min((int)W, ow * stride_w - padding_w + kernel_size_w);

                        double max_val = -std::numeric_limits<double>::infinity();
                        int max_d = -1, max_h = -1, max_w = -1;

                        for (int d_idx = d_start; d_idx < d_end; ++d_idx) {
                            for (int h = h_start; h < h_end; ++h) {
                                for (int w = w_start; w < w_end; ++w) {
                                    size_t off = input.impl->offset + n*input.impl->strides[0] + c*input.impl->strides[1] + d_idx*input.impl->strides[2] + h*input.impl->strides[3] + w*input.impl->strides[4];
                                    double v = read_scalar_at(input.impl->storage->data.get(), off, input._dtype());
                                    if (v > max_val) {
                                        max_val = v;
                                        max_d = d_idx;
                                        max_h = h;
                                        max_w = w;
                                    }
                                }
                            }
                        }

                        if (max_d != -1) {
                            size_t in_off = input.impl->offset + n*input.impl->strides[0] + c*input.impl->strides[1] + max_d*input.impl->strides[2] + max_h*input.impl->strides[3] + max_w*input.impl->strides[4];
                            double cur = read_scalar_at(grad_input.impl->storage->data.get(), in_off, grad_input._dtype());
                            write_scalar_at(grad_input.impl->storage->data.get(), in_off, grad_input._dtype(), cur + g);
                        }
                    }
                }
            }
        }
    }
    accumulate_grad(input, grad_input);
}

// ======================================================================================
//                                   AVERAGE POOLING
// ======================================================================================

// --- AvgPool1d ---
AvgPool1d::AvgPool1d(int k, int s, int p) : kernel_size(k), stride(s), padding(p) {
    if (stride == -1) stride = kernel_size;
}

Tensor AvgPool1d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("AvgPool1d: null input");
    size_t N = input.impl->shape[0];
    size_t C = input.impl->shape[1];
    size_t L = input.impl->shape[2];

    int out_l = (int)((L + 2 * padding - kernel_size) / stride + 1);
    if (out_l <= 0) throw std::runtime_error("AvgPool1d: invalid output dims");

    Tensor output({N, C, (size_t)out_l}, input._dtype(), input.requires_grad());

    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (int i = 0; i < out_l; ++i) {
                int start = i * stride - padding;
                int end = std::min((int)L, start + kernel_size);
                start = std::max(0, start);
                
                double sum_val = 0.0;
                int count = 0;
                for (int k = start; k < end; ++k) {
                    size_t offset = input.impl->offset + n * input.impl->strides[0] + c * input.impl->strides[1] + k * input.impl->strides[2];
                    sum_val += read_scalar_at(input.impl->storage->data.get(), offset, input._dtype());
                    count++;
                }
                
                double avg = (count > 0) ? sum_val / kernel_size : 0.0; // Div by kernel_size (standard) or count (if count_include_pad=False)
                // Standard AvgPool usually divides by kernel_size, but PyTorch has count_include_pad.
                // Let's divide by actual window size (count) for correctness at edges
                if (count > 0) avg = sum_val / count;

                size_t out_offset = output.impl->offset + n * output.impl->strides[0] + c * output.impl->strides[1] + i * output.impl->strides[2];
                write_scalar_at(output.impl->storage->data.get(), out_offset, output._dtype(), avg);
            }
        }
    }

    if (input.requires_grad()) {
        output.impl->grad_fn = std::make_shared<GradAvgPool1d>(input, kernel_size, stride, padding);
    }
    return output;
}

void GradAvgPool1d::backward(const Tensor& self) {
    if (!self.impl->storage->grad) throw std::runtime_error("GradAvgPool1d: missing self grad");
    Tensor grad_output = tensor_from_grad(self);
    Tensor grad_input = Tensor::zeros(input.shape(), input._dtype(), false);

    size_t N = input.impl->shape[0];
    size_t C = input.impl->shape[1];
    size_t L = input.impl->shape[2];
    int out_l = grad_output.impl->shape[2];

    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (int i = 0; i < out_l; ++i) {
                size_t go_idx = grad_output.impl->offset + n * grad_output.impl->strides[0] + c * grad_output.impl->strides[1] + i * grad_output.impl->strides[2];
                double g = read_scalar_at(grad_output.impl->storage->data.get(), go_idx, grad_output._dtype());

                int start = i * stride - padding;
                int end = std::min((int)L, start + kernel_size);
                start = std::max(0, start);
                
                int count = end - start; // Using actual overlap count
                if (count > 0) {
                    double grad_val = g / count;
                    for (int k = start; k < end; ++k) {
                        size_t in_idx = input.impl->offset + n * input.impl->strides[0] + c * input.impl->strides[1] + k * input.impl->strides[2];
                        double cur = read_scalar_at(grad_input.impl->storage->data.get(), in_idx, grad_input._dtype());
                        write_scalar_at(grad_input.impl->storage->data.get(), in_idx, grad_input._dtype(), cur + grad_val);
                    }
                }
            }
        }
    }
    accumulate_grad(input, grad_input);
}

// --- AvgPool2d ---
AvgPool2d::AvgPool2d(int kh, int kw, int sh, int sw, int ph, int pw) 
    : kernel_size_h(kh), kernel_size_w(kw), stride_h(sh), stride_w(sw), padding_h(ph), padding_w(pw) {
    if (stride_h == -1) stride_h = kernel_size_h;
    if (stride_w == -1) stride_w = kernel_size_w;
}

Tensor AvgPool2d::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("AvgPool2d: null input");
    size_t N = input.impl->shape[0];
    size_t C = input.impl->shape[1];
    size_t H = input.impl->shape[2];
    size_t W = input.impl->shape[3];

    int out_h = (int)((H + 2 * padding_h - kernel_size_h) / stride_h + 1);
    int out_w = (int)((W + 2 * padding_w - kernel_size_w) / stride_w + 1);
    if (out_h <= 0 || out_w <= 0) throw std::runtime_error("AvgPool2d: invalid output dims");

    Tensor output({N, C, (size_t)out_h, (size_t)out_w}, input._dtype(), input.requires_grad());

    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int h_start = std::max(0, oh * stride_h - padding_h);
                    int w_start = std::max(0, ow * stride_w - padding_w);
                    int h_end = std::min((int)H, oh * stride_h - padding_h + kernel_size_h);
                    int w_end = std::min((int)W, ow * stride_w - padding_w + kernel_size_w);

                    double sum_val = 0.0;
                    int count = 0;
                    for (int h = h_start; h < h_end; ++h) {
                        for (int w = w_start; w < w_end; ++w) {
                            size_t off = input.impl->offset + n*input.impl->strides[0] + c*input.impl->strides[1] + h*input.impl->strides[2] + w*input.impl->strides[3];
                            sum_val += read_scalar_at(input.impl->storage->data.get(), off, input._dtype());
                            count++;
                        }
                    }
                    
                    double avg = (count > 0) ? sum_val / count : 0.0;
                    size_t out_off = output.impl->offset + n*output.impl->strides[0] + c*output.impl->strides[1] + oh*output.impl->strides[2] + ow*output.impl->strides[3];
                    write_scalar_at(output.impl->storage->data.get(), out_off, output._dtype(), avg);
                }
            }
        }
    }

    if (input.requires_grad()) {
        output.impl->grad_fn = std::make_shared<GradAvgPool2d>(input, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w);
    }
    return output;
}