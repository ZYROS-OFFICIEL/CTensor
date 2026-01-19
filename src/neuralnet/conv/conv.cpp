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
