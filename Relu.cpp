#include "Relu.h"
#include "ops1.h" 


// LeakyRelu (fixed stride handling)
Tensor LeakyRelu(const Tensor& a_, double negative_slope) {
    if (!a_.impl) throw std::runtime_error("LeakyRelu: null tensor implementation");

    size_t n = a_.numel_();
    bool req = a_.requires_grad();

    Tensor result(a_.shape(), a_._dtype(), req);

    if (req) result.impl->grad_fn = std::make_shared<GradLeakyRelu>(a_, negative_slope);

    size_t ndim = a_.impl->ndim;
    auto shape     = a_.impl->shape;
    auto stridesA  = a_.impl->strides;
    auto stridesR  = result.impl->strides;
    size_t offsetA = a_.impl->offset;
    size_t offsetR = result.impl->offset;

    #pragma omp parallel 
    {
        std::vector<size_t> idx_vec(ndim, 0);
        #pragma omp for
        for (size_t flat = 0; flat < n; ++flat) {

            // decode multi-index for this logical position
            size_t rem = flat;
            for (int d = (int)ndim - 1; d >= 0; --d) {
                idx_vec[d] = rem % shape[d];
                rem /= shape[d];
            }

            // source strided index (a_)
            size_t src_idx = offsetA;
            for (size_t d = 0; d < ndim; ++d)
                src_idx += idx_vec[d] * stridesA[d];

            // dest strided index (result) — must compute with result strides/offset
            size_t dst_idx = offsetR;
            for (size_t d = 0; d < result.impl->ndim; ++d)
                dst_idx += idx_vec[d] * stridesR[d];

            double v = read_scalar_at(a_.impl->storage->data.get(), src_idx, a_._dtype());
            double out = (v >= 0.0) ? v : v * negative_slope;
            write_scalar_at(result.impl->storage->data.get(), dst_idx, result._dtype(), out);
        }
    }
    return result;
}

// GradLeakyRelu::backward (fixed stride handling & grad_output reading)
void GradLeakyRelu::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradLeakyRelu: missing self grad");
    if (!a.impl) return;
    if (!a.requires_grad()) return;

    // grad_output is contiguous copy (tensor_from_grad)
    Tensor grad_output = tensor_from_grad(self);

    // grad_input as zeros with same shape as a (no grad tracking)
    Tensor grad_input = Tensor::zeros(a.shape(), a._dtype(), false);

    size_t n = a.numel_();


    size_t ndim = a.impl->ndim;
    auto shape = a.impl->shape;
    auto stridesA = a.impl->strides;
    auto stridesG = grad_input.impl->strides;
    size_t offsetA = a.impl->offset;
    size_t offsetG = grad_input.impl->offset;
    #pragma omp parallel
    {   
        std::vector<size_t> idx_vec(ndim, 0);
        #pragma omp for
        for (size_t flat = 0; flat < n; ++flat) {
            // multi-index
            size_t rem = flat;
            for (int d = (int)ndim - 1; d >= 0; --d) {
                idx_vec[d] = rem % shape[d];
                rem /= shape[d];
            }

            // compute source/dest strided indices for 'a' and 'grad_input'
            size_t src_idx = offsetA;
            size_t dst_idx = offsetG;
            for (size_t d = 0; d < ndim; ++d) {
                src_idx += idx_vec[d] * stridesA[d];
                dst_idx += idx_vec[d] * stridesG[d];
            }

            double v = read_scalar_at(a.impl->storage->data.get(), src_idx, a_._dtype());
            // read grad_output by flat index because tensor_from_grad returned contiguous copy
            double go = read_scalar_at(grad_output.impl->storage->data.get(), flat, grad_output._dtype());

            double gin = (v >= 0.0) ? go : go * negative_slope;
            write_scalar_at(grad_input.impl->storage->data.get(), dst_idx, grad_input._dtype(), gin);
        }
    }
    // accumulate into parent's grad buffer
    accumulate_grad(a, grad_input);
}

// PReLU constructor (your version with Tensor::full)
PRelu::PRelu(int num_params, double init, DType dtype)
    : num_parameters(num_params)
{
    if (num_parameters == 1) {
        weight = Tensor::full({1}, init, dtype, true);
    } else {
        weight = Tensor::full({(size_t)num_parameters}, init, dtype, true);
    }
}

// PRelu forward (fixed stride handling & channel indexing)
Tensor PRelu::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("PRelu: null input tensor");
    if (num_parameters > 1 && num_parameters != (int)input.impl->shape[1])
        throw std::runtime_error("PRelu: num_parameters must be 1 or equal to input channels");

    size_t n = input.numel_();
    bool req = input.requires_grad() || weight.requires_grad();

    Tensor result(input.shape(), input._dtype(), req);
    if (req) result.impl->grad_fn = std::make_shared<GradPRelu>(input, weight);

    bool per_channel = (num_parameters > 1);
    auto ndim = input.impl->ndim;
    auto shape = input.impl->shape;
    // input src index and result dst index
    auto stridesI = input.impl->strides;
    auto stridesR = result.impl->strides;
    #pragma omp parallel 
    {   
        std::vector<size_t> idx_vec(ndim, 0);
        #pragma omp for
        for (size_t flat = 0; flat < n; ++flat) {
            size_t rem = flat;
            for (int d = (int)ndim - 1; d >= 0; --d) {
                idx_vec[d] = rem % shape[d];
                rem /= shape[d];
            }
        size_t src_idx = input.impl->offset;
        size_t dst_idx = result.impl->offset;

            for (size_t d = 0; d < ndim; ++d) {
                src_idx += idx_vec[d] * stridesI[d];
                dst_idx += idx_vec[d] * stridesR[d];
            }

            size_t channel_idx = per_channel ? idx_vec[1] : 0;
            double alpha = read_scalar_at(weight.impl->storage->data.get(), channel_idx, weight._dtype());
            double v = read_scalar_at(input.impl->storage->data.get(), src_idx, input._dtype());
            double out = (v >= 0.0) ? v : v * alpha;
            write_scalar_at(result.impl->storage->data.get(), dst_idx, result._dtype(), out);
        }
    }
    return result;
}

// GradPRelu::backward (fixed stride handling for grad_output reading and accumulations)
void GradPRelu::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradPRelu: missing self grad");
    if (!input.impl || !weight.impl) throw std::runtime_error("GradPRelu: missing parents");

    Tensor grad_output = tensor_from_grad(self); // contiguous copy
    Tensor grad_input  = Tensor::zeros(input.shape(), input._dtype(), false);
    Tensor grad_weight = Tensor::zeros(weight.shape(), weight._dtype(), false);

    bool per_channel = (weight.numel_() > 1);
    size_t n = input.numel_();
    auto ndim = input.impl->ndim;
    auto shape =input.impl->shape;
    size_t strideI = input.impl->strides;
    size_t strideG = grad_input.impl->strides;

    #pragma omp parallel
    {

        std::vector<size_t> idx_vec(ndim, 0);
        #pragma omp for
        for (size_t flat = 0; flat < n; ++flat) {
            size_t rem = flat;
            for (int d = (int)ndim - 1; d >= 0; --d) {
                idx_vec[d] = rem % shape[d];
                rem /= shape[d];
            }

            // compute strided indices for input & grad_input
            size_t in_idx = input.impl->offset;
            size_t gin_idx = grad_input.impl->offset;
            for (size_t d = 0; d < ndim; ++d) {
                in_idx += idx_vec[d] * strideI[d];
                gin_idx += idx_vec[d] * strideG[d];
            }

            // read input value and grad_output value (grad_output is contiguous -> use flat)
            double v = read_scalar_at(input.impl->storage->data.get(), in_idx, input._dtype());
            double go = read_scalar_at(grad_output.impl->storage->data.get(), flat, grad_output._dtype());

            // compute alpha
            size_t channel_idx = per_channel ? idx_vec[1] : 0;
            double alpha = read_scalar_at(weight.impl->storage->data.get(), channel_idx, weight._dtype());

            // grad input
            double gin = (v >= 0.0) ? go : go * alpha;
            write_scalar_at(grad_input.impl->storage->data.get(), gin_idx, grad_input._dtype(), gin);

            // grad alpha contribution (only when v < 0)
            if (weight.requires_grad() && v < 0.0) {
                double contrib = go * v;
                // accumulate in local grad_weight raw storage (using grad_weight's strided index)
                size_t gw_idx = grad_weight.impl->offset + channel_idx * grad_weight.impl->strides[0];
                double cur = read_scalar_at(grad_weight.impl->storage->data.get(), gw_idx, grad_weight._dtype());
                write_scalar_at(grad_weight.impl->storage->data.get(), gw_idx, grad_weight._dtype(), cur + contrib);
            }
        }
    }
    if (input.requires_grad())  accumulate_grad(input, grad_input);
    if (weight.requires_grad()) accumulate_grad(weight, grad_weight);
}
