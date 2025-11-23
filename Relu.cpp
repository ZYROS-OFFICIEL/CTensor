#include "Relu.h"
#include "ops1.h" 
#include <omp.h>


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
    if (!input.impl)
        throw std::runtime_error("GradPRelu: missing input tensor");
    
    Tensor grad_output = tensor_from_grad(self); 
    Tensor grad_input = Tensor::zeros(input.shape(), input._dtype(), false);
    Tensor grad_weight = Tensor::zeros(weight.shape(), weight._dtype(), false);

    bool per_channel = (weight.numel() > 1);
    size_t n = input.numel();
    size_t ndim = input.impl->ndim;
    
    const size_t* shape = input.impl->shape;
    const size_t* strides_in = input.impl->strides;
    const size_t* strides_go = grad_output.impl->strides;
    const size_t* strides_gi = grad_input.impl->strides;
    const size_t* strides_w = weight.impl->strides;
    const size_t* strides_gw = grad_weight.impl->strides;

    auto* in_data = input.impl->storage->data.get();
    auto* w_data = weight.impl->storage->data.get();
    auto* go_data = grad_output.impl->storage->data.get();
    auto* gi_data = grad_input.impl->storage->data.get();
    auto* gw_data = grad_weight.impl->storage->data.get();
    DType dt = input._dtype();

    // Parallel loop with atomic updates for weights
    #pragma omp parallel for
    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat;
        size_t idx_in = input.impl->offset;
        size_t idx_go = grad_output.impl->offset;
        size_t idx_gi = grad_input.impl->offset;
        size_t channel = 0;

        for (int d = (int)ndim - 1; d >= 0; --d) {
            size_t coord = rem % shape[d];
            rem /= shape[d];
            idx_in += coord * strides_in[d];
            idx_go += coord * strides_go[d];
            idx_gi += coord * strides_gi[d];
            if (per_channel && d == 1) channel = coord;
        }

        // Get alpha (read-only, safe)
        size_t idx_w = weight.impl->offset + (per_channel ? channel * strides_w[0] : 0);
        double alpha = read_scalar_at(w_data, idx_w, dt);

        double v = read_scalar_at(in_data, idx_in, dt);
        double go = read_scalar_at(go_data, idx_go, dt);
        
        // --- Calculate Gradients ---
        
        // Grad Input (Write only to unique location idx_gi per thread -> Safe)
        double gin = (v >= 0.0) ? go : go * alpha;
        write_scalar_at(gi_data, idx_gi, dt, gin);

        // Grad Weight (Alpha)
        // Multiple threads will try to update the SAME weight index (idx_gw)
        // MUST USE ATOMIC
        if (weight.requires_grad() && v < 0.0) {
            double g_alpha = go * v;
            size_t idx_gw = grad_weight.impl->offset + (per_channel ? channel * strides_gw[0] : 0);
            
            // Atomic update for float/double
            // Note: OpenMP atomic supports float/double accumulation
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

    // Accumulate final gradients
    if (input.requires_grad()) {
        accumulate_grad(input, grad_input);
    }
    if (weight.requires_grad()) {
        accumulate_grad(weight, grad_weight);
    }
}