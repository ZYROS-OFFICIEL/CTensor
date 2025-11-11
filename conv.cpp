#include "conv.h"
#include "ops1.h"


Conv1d::Conv1d(int in_c, int out_c, int k, int s, int p)
    : in_channels(in_c), out_channels(out_c),
      kernel_size(k), stride(s), padding(p)
{
    weight = Tensor::randn({out_c, in_c, k});
    bias   = Tensor::zeros({out_c});
}

Tensor Conv1d::Conv1d(const Tensor& input) {
    int batch = input.shape[0];
    int in_c  = input.shape[1];
    int width = input.shape[2];

    int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    Tensor output({batch, out_channels, out_w});

    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ow = 0; ow < out_w; ++ow) {
                float sum = bias[oc].item<float>();
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int k = 0; k < kernel_size; ++k) {
                        int iw = ow * stride + k - padding;
                        if (iw >= 0 && iw < width)
                            sum += input(b, ic, iw).item<float>() * weight(oc, ic, k).item<float>();
                    }
                }
                output(b, oc, ow) = sum;
            }
        }
    }
    if (req)
    output.impl->grad_fn = std::make_shared<GradConv1d>(input, weight, bias, stride, padding);

    return output;
}
void GradConv1d::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradConv1d: missing self grad");

    Tensor grad_output = tensor_from_grad(self);
    Tensor grad_input  = input;
    Tensor grad_weight = weight;
    Tensor grad_bias   = bias;

    int batch = input.shape[0];
    int in_c  = input.shape[1];
    int width = input.shape[2];
    int out_c = weight.shape[0];
    int out_w = grad_output.shape[2];

    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_c; ++oc) {
            for (int ow = 0; ow < out_w; ++ow) {
                double go = grad_output(b, oc, ow).item<double>();
                grad_bias(oc) += go;

                for (int ic = 0; ic < in_c; ++ic) {
                    for (int k = 0; k < weight.shape[2]; ++k) {
                        int iw = ow * stride + k - padding;
                        if (iw >= 0 && iw < width) {
                            grad_input(b, ic, iw) += go * weight(oc, ic, k);
                            grad_weight(oc, ic, k) += go * input(b, ic, iw);
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

