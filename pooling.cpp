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
                    int pad_h, int pad_w);  

Tensor im2col_3d_pool(const Tensor& input,
                      int kernel_d, int kernel_h, int kernel_w,
                      int stride_d, int stride_h, int stride_w,
                      int pad_d, int pad_h, int pad_w);
void col2im_3d_pool(const Tensor& grad_patches,
                    Tensor& grad_input,
                    int kernel_d, int kernel_h, int kernel_w,
                    int stride_d, int stride_h, int stride_w,
                    int pad_d, int pad_h, int pad_w);
