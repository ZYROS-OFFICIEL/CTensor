#include "Relu.h"
#include "ops_dispatch.h" 
#include <vector>
#include <omp.h>
#include <stdexcept>

// ===========================================================================
//                               STANDARD ReLU (Class)
// ===========================================================================

Tensor Relu::forward(const Tensor& input) {
    return relu(input);
}

// ===========================================================================
//                               LEAKY RELU (Op)
// ===========================================================================

Tensor LeakyRelu(const Tensor& input, double negative_slope) {
    if (!input.impl) throw std::runtime_error("LeakyRelu: null input");

    size_t n = input.numel();
    bool req = input.requires_grad();

    Tensor result(input.shape(), input._dtype(), req);

    // Attach GradFn
    if (req) {
        result.impl->grad_fn = std::make_shared<GradLeakyRelu>(input, negative_slope);
    }

    // Pointers for fast access
    size_t ndim = input.impl->ndim;
    const size_t* shape = input.impl->shape.data();
    const size_t* strides_in = input.impl->strides.data();
    const size_t* strides_out = result.impl->strides.data(); // Contiguous
    
    auto* in_data = input.impl->data->data.get();
    auto* out_data = result.impl->data->data.get();
    DType dt = input._dtype();
    size_t off_in = input.impl->offset;

    // Parallel Loop
    #pragma omp parallel for
    for (size_t flat = 0; flat < n; ++flat) {
        // Decode index
        size_t rem = flat;
        size_t idx_in = off_in;
        
        for (int d = (int)ndim - 1; d >= 0; --d) {
            size_t coord = rem % shape[d];
            rem /= shape[d];
            idx_in += coord * strides_in[d];
        }

        double v = read_scalar_at(in_data, idx_in, dt);
        double val = (v >= 0.0) ? v : v * negative_slope;
        write_scalar_at(out_data, flat, dt, val); // out is contiguous
    }

    return result;
}

void GradLeakyRelu::backward(const Tensor& self) {
    if (!self.impl->grad->data) throw std::runtime_error("GradLeakyRelu: missing self grad");
    
    Tensor grad_output = tensor_from_grad(self); // Stride-aware copy
    Tensor grad_input = Tensor::zeros(input.shape(), input._dtype(), false);
    
    size_t n = input.numel();
    size_t ndim = input.impl->ndim;
    const size_t* shape = input.impl->shape.data();
    const size_t* strides_in = input.impl->strides.data();
    const size_t* strides_gi = grad_input.impl->strides.data();
    // grad_output is contiguous from tensor_from_grad
    
    auto* in_data = input.impl->data->data.get();
    auto* go_data = grad_output.impl->data->data.get();
    auto* gi_data = grad_input.impl->data->data.get();
    DType dt = input._dtype();
    size_t off_in = input.impl->offset;
    size_t off_gi = grad_input.impl->offset;

    #pragma omp parallel for
    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat;
        size_t idx_in = off_in;
        size_t idx_gi = off_gi;

        for (int d = (int)ndim - 1; d >= 0; --d) {
            size_t coord = rem % shape[d];
            rem /= shape[d];
            idx_in += coord * strides_in[d];
            idx_gi += coord * strides_gi[d];
        }

        double v = read_scalar_at(in_data, idx_in, dt);
        double go = read_scalar_at(go_data, flat, dt); // grad_output is flat/contiguous
        
        double gin = (v >= 0.0) ? go : go * negative_slope;
        write_scalar_at(gi_data, idx_gi, dt, gin);
    }

    accumulate_grad(input, grad_input);
}


// ===========================================================================
//                               PReLU (Class)
// ===========================================================================

PRelu::PRelu(int num_params, double init) : num_parameters(num_params) {
    // Weight shape: [1] or [C]
    if (num_parameters == 1) {
        weight = Tensor::full({1}, init, DType::Float32, true);
    } else {
        weight = Tensor::full({(size_t)num_parameters}, init, DType::Float32, true);
    }
}

Tensor PRelu::forward(const Tensor& input) {
    if (!input.impl) throw std::runtime_error("PRelu: null input");
    
    // Input validation for per-channel
    if (num_parameters > 1) {
        if (input.impl->ndim < 2 || (int)input.impl->shape[1] != num_parameters) {
            throw std::runtime_error("PRelu: input channels dim 1 must match num_parameters");
        }
    }

    size_t n = input.numel();
    bool req = input.requires_grad() || weight.requires_grad();
    Tensor result(input.shape(), input._dtype(), req);

    if (req) {
        result.impl->grad_fn = std::make_shared<GradPRelu>(input, weight);
    }

    // Pointers
    size_t ndim = input.impl->ndim;
    const size_t* shape = input.impl->shape.data();
    const size_t* strides_in = input.impl->strides.data();
    const size_t* strides_w = weight.impl->strides.data();

    auto* in_data = input.impl->data->data.get();
    auto* w_data = weight.impl->data->data.get();
    auto* out_data = result.impl->data->data.get();
    
    DType dt = input._dtype();
    size_t off_in = input.impl->offset;
    size_t off_w = weight.impl->offset;
    bool per_channel = (num_parameters > 1);

    #pragma omp parallel for
    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat;
        size_t idx_in = off_in;
        size_t channel_idx = 0;

        for (int d = (int)ndim - 1; d >= 0; --d) {
            size_t coord = rem % shape[d];
            rem /= shape[d];
            idx_in += coord * strides_in[d];
            if (per_channel && d == 1) channel_idx = coord;
        }

        // Get weight (alpha)
        size_t idx_w = off_w; 
        if (per_channel) idx_w += channel_idx * strides_w[0];
        
        double alpha = read_scalar_at(w_data, idx_w, dt);
        double v = read_scalar_at(in_data, idx_in, dt);
        
        double val = (v >= 0.0) ? v : v * alpha;
        write_scalar_at(out_data, flat, dt, val);
    }

    return result;
}

void GradPRelu::backward(const Tensor& self) {
    if (!self.impl->grad->data) throw std::runtime_error("GradPRelu: missing self grad");
    
    Tensor grad_output = tensor_from_grad(self); 
    Tensor grad_input = Tensor::zeros(input.shape(), input._dtype(), false);
    Tensor grad_weight = Tensor::zeros(weight.shape(), weight._dtype(), false);

    size_t n = input.numel();
    size_t ndim = input.impl->ndim;
    const size_t* shape = input.impl->shape.data();
    const size_t* strides_in = input.impl->strides.data();
    const size_t* strides_gi = grad_input.impl->strides.data();
    const size_t* strides_w = weight.impl->strides.data();
    const size_t* strides_gw = grad_weight.impl->strides.data();
    auto* in_data = input.impl->data->data.get();
    auto* w_data = weight.impl->data->data.get();
    auto* go_data = grad_output.impl->data->data.get();
    auto* gi_data = grad_input.impl->data->data.get();
    auto* gw_data = grad_weight.impl->data->data.get(); // Assuming contiguous accumulation buffer usually
    
    DType dt = input._dtype();
    size_t off_in = input.impl->offset;
    size_t off_gi = grad_input.impl->offset;
    size_t off_w = weight.impl->offset;
    size_t off_gw = grad_weight.impl->offset;
    bool per_channel = (weight.numel() > 1);

    #pragma omp parallel for
    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat;
        size_t idx_in = off_in;
        size_t idx_gi = off_gi;
        size_t channel = 0;

        for (int d = (int)ndim - 1; d >= 0; --d) {
            size_t coord = rem % shape[d];
            rem /= shape[d];
            idx_in += coord * strides_in[d];
            idx_gi += coord * strides_gi[d];
            if (per_channel && d == 1) channel = coord;
        }

        // Get alpha
        size_t idx_w = off_w;
        if (per_channel) idx_w += channel * strides_w[0];
        double alpha = read_scalar_at(w_data, idx_w, dt);

        double v = read_scalar_at(in_data, idx_in, dt);
        double go = read_scalar_at(go_data, flat, dt);

        // 1. Grad Input
        double gin = (v >= 0.0) ? go : go * alpha;
        write_scalar_at(gi_data, idx_gi, dt, gin);

        // 2. Grad Weight (Accumulate using ATOMIC)
        if (weight.requires_grad() && v < 0.0) {
            double g_alpha = go * v;
            
            size_t idx_gw = off_gw;
            if (per_channel) idx_gw += channel * strides_gw[0];

            // Atomic update based on type
            if (dt == DType::Float32) {
                float* ptr = (float*)gw_data + idx_gw;
                #pragma omp atomic
                *ptr += (float)g_alpha;
            } else if (dt == DType::Double64) {
                double* ptr = (double*)gw_data + idx_gw;
                #pragma omp atomic
                *ptr += g_alpha;
            }
        }
    }

    if (input.requires_grad()) accumulate_grad(input, grad_input);
    if (weight.requires_grad()) accumulate_grad(weight, grad_weight);
}