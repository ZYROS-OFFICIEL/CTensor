#include "Relu.h"
#include "ops1.h" 


Tensor LeakyRelu(const Tensor& a_, double negative_slope) {
    if (!a_.impl)
        throw std::runtime_error("LeakyRelu: null tensor implementation");

    size_t n = a_.numel();
    bool req = a_.requires_grad();

    Tensor result(a_.shape(), a_._dtype(), req);

    if (req)
        result.impl->grad_fn = std::make_shared<GradLeakyRelu>(a_, negative_slope);

    std::vector<size_t> idx_vec(a_.impl->ndim, 0);
    for (size_t flat = 0; flat < n; ++flat) {
        // 1. Convert flat index to multi-dim
        size_t rem = flat;
        for (int d = (int)a_.impl->ndim - 1; d >= 0; --d) {
            idx_vec[d] = rem % a_.impl->shape[d];
            rem /= a_.impl->shape[d];
        }
        // 2. Convert multi-dim to strided index
        size_t strided_idx = a_.impl->offset;
        for (size_t d = 0; d < a_.impl->ndim; ++d) {
            strided_idx += idx_vec[d] * a_.impl->strides[d];
        }

        // 3. Read from strided src, write to strided dest
        double v = read_scalar_at(a_.impl->storage->data.get(), strided_idx, a_._dtype());
        double out = (v >= 0.0) ? v : v * negative_slope;
        write_scalar_at(result.impl->storage->data.get(), strided_idx, result._dtype(), out);
    }

    return result;
}

void GradLeakyRelu::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradLeakyRelu: missing self grad");
    if (!a.impl || !a.requires_grad()) return;

    Tensor grad_output = tensor_from_grad(self); 
    Tensor grad_input = Tensor::zeros(a.shape(), a._dtype(), false);
    
    size_t n = a.numel();
    std::vector<size_t> idx_vec(a.impl->ndim, 0);
    for (size_t flat = 0; flat < n; ++flat) {
        // 1. Convert flat index to multi-dim
        size_t rem = flat;
        for (int d = (int)a.impl->ndim - 1; d >= 0; --d) {
            idx_vec[d] = rem % a_.impl->shape[d];
            rem /= a_.impl->shape[d];
        }
        // 2. Convert multi-dim to strided index
        size_t strided_idx = a.impl->offset;
        for (size_t d = 0; d < a.impl->ndim; ++d) {
            strided_idx += idx_vec[d] * a_.impl->strides[d];
        }

        // 3. Read/Write using strided index
        double v = read_scalar_at(a.impl->storage->data.get(), strided_idx, a._dtype());
        double go = read_scalar_at(grad_output.impl->storage->data.get(), strided_idx, grad_output._dtype());
        double gin = (v >= 0.0) ? go : go * negative_slope;
        write_scalar_at(grad_input.impl->storage->data.get(), strided_idx, grad_input._dtype(), gin);
    }

    accumulate_grad(a, grad_input);
}

PRelu::PRelu(int num_params, double init, DType dtype) 
    : num_parameters(num_params) 
{
    if (num_parameters == 1) {
        // Shared parameter
        weight = Tensor::full({1}, init, dtype, true);
    } else {
        // Per-channel parameter
        weight = Tensor::full({(size_t)num_parameters}, init, dtype, true);
    }
}

Tensor PRelu::forward(const Tensor& input) {
    if (!input.impl)
        throw std::runtime_error("PRelu: null input tensor");
    if (num_parameters > 1 && num_parameters != (int)input.impl->shape[1])
        throw std::runtime_error("PRelu: num_parameters must be 1 or equal to input channels");

    size_t n = input.numel();
    bool req = input.requires_grad() || weight.requires_grad();

    Tensor result(input.shape(), input._dtype(), req);

    if (req)
        result.impl->grad_fn = std::make_shared<GradPRelu>(input, weight);
    
    bool per_channel = (num_parameters > 1);
    size_t C = per_channel ? input.impl->shape[1] : 1;
    size_t spatial_dims = n / (input.impl->shape[0] * C);

    std::vector<size_t> idx_vec(input.impl->ndim, 0);
    for (size_t flat = 0; flat < n; ++flat) {
        // 1. Convert flat index to multi-dim
        size_t rem = flat;
        for (int d = (int)input.impl->ndim - 1; d >= 0; --d) {
            idx_vec[d] = rem % input.impl->shape[d];
            rem /= input.impl->shape[d];
        }
        // 2. Convert multi-dim to strided index
        size_t strided_idx = input.impl->offset;
        for (size_t d = 0; d < input.impl->ndim; ++d) {
            strided_idx += idx_vec[d] * input.impl->strides[d];
        }

        // 3. Get alpha (weight)
        size_t channel_idx = per_channel ? idx_vec[1] : 0;
        double alpha = read_scalar_at(weight.impl->storage->data.get(), channel_idx, weight._dtype());

        // 4. Read/Write using strided index
        double v = read_scalar_at(input.impl->storage->data.get(), strided_idx, input._dtype());
        double out = (v >= 0.0) ? v : v * alpha;
        write_scalar_at(result.impl->storage->data.get(), strided_idx, result._dtype(), out);
    }

    return result;
}

void GradPRelu::backward(const Tensor& self) {
    if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
        throw std::runtime_error("GradPRelu: missing self grad");
    if (!input.impl)
        throw std::runtime_error("GradPRelu: missing input tensor");
    
    Tensor grad_output = tensor_from_grad(self); // Stride-aware
    Tensor grad_input = Tensor::zeros(input.shape(), input._dtype(), false);
    Tensor grad_weight = Tensor::zeros(weight.shape(), weight._dtype(), false);

    bool per_channel = (weight.numel() > 1);
    size_t n = input.numel();
    
    std::vector<size_t> idx_vec(input.impl->ndim, 0);
    for (size_t flat = 0; flat < n; ++flat) {
        // 1. Convert flat index to multi-dim
        size_t rem = flat;
        for (int d = (int)input.impl->ndim - 1; d >= 0; --d) {
            idx_vec[d] = rem % input.impl->shape[d];
            rem /= input.impl->shape[d];
        }
        // 2. Convert multi-dim to strided index
        size_t strided_idx = input.impl->offset;
        for (size_t d = 0; d < input.impl->ndim; ++d) {
            strided_idx += idx_vec[d] * input.impl->strides[d];
        }

        // 3. Get alpha (weight) and channel index
        size_t channel_idx = per_channel ? idx_vec[1] : 0;
        double alpha = read_scalar_at(weight.impl->storage->data.get(), channel_idx, weight._dtype());

        // 4. Read/Write using strided index
        double v = read_scalar_at(input.impl->storage->data.get(), strided_idx, input._dtype());
        double go = read_scalar_at(grad_output.impl->storage->data.get(), strided_idx, grad_output._dtype());
        
        // Grad Input
        double gin = (v >= 0.0) ? go : go * alpha;
        write_scalar_at(grad_input.impl->storage->data.get(), strided_idx, grad_input._dtype(), gin);

        // Grad Weight (Alpha)
        if (weight.requires_grad() && v < 0.0) {
            double g_alpha = go * v;
            // Accumulate gradient for this alpha
            double cur_w_grad = read_scalar_at(grad_weight.impl->storage->data.get(), channel_idx, grad_weight._dtype());
            write_scalar_at(grad_weight.impl->storage->data.get(), channel_idx, grad_weight._dtype(), cur_w_grad + g_alpha);
        }
    }

    // Accumulate final gradients
    if (input.requires_grad()) {
        accumulate_grad(input, grad_input);
    }
    if (weight.requires_grad()) {
        accumulate_grad(weight, grad_weight);
    }
}