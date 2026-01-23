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