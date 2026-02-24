#include "core/tensor.h"
#include "core/ops_dispatch.h"
#include "neuralnet/weights/weights_init.h"
#include <iostream>
#include <fstream> 
#include <numeric>
#include <cmath>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <omp.h>


//Helpers : 

// Thread-local Mersenne Twister for fast, safe parallel initialization
static thread_local std::mt19937 rng(std::random_device{}());


// Helper to calculate Fan-In and Fan-Out for Xavier/Kaiming
void calculate_fan_in_and_fan_out(const Tensor& tensor, size_t& fan_in, size_t& fan_out) {
    auto shape = tensor.shape();
    if (shape.size() < 2) {
        fan_in = fan_out = shape.empty() ? 1 : shape[0];
        return;
    }
    
    fan_in = shape[1];  // in_features / in_channels
    fan_out = shape[0]; // out_features / out_channels
    
    // If it's a Conv layer (e.g. 4D: [out, in, H, W]), multiply by receptive field
    size_t receptive_field_size = 1;
    for (size_t i = 2; i < shape.size(); ++i) {
        receptive_field_size *= shape[i];
    }
    
    fan_in *= receptive_field_size;
    fan_out *= receptive_field_size;
}

void uniform_(Tensor& tensor, double a, double b) {
    if (!tensor.impl) return;
    std::uniform_real_distribution<float> dist((float)a, (float)b);
    size_t n = tensor.numel();
    
    if (tensor._dtype() == DType::Float32) {
        float* ptr = (float*)tensor.impl->data->data.get();
        for (size_t i = 0; i < n; ++i) ptr[i] = dist(rng);
    } else {
        for (size_t i = 0; i < n; ++i) tensor.write_scalar(i, dist(rng));
    }
}

void normal_(Tensor& tensor, double mean, double std) {
    if (!tensor.impl) return;
    std::normal_distribution<float> dist((float)mean, (float)std);
    size_t n = tensor.numel();
    
    if (tensor._dtype() == DType::Float32) {
        float* ptr = (float*)tensor.impl->data->data.get();
        for (size_t i = 0; i < n; ++i) ptr[i] = dist(rng);
    } else {
        for (size_t i = 0; i < n; ++i) tensor.write_scalar(i, dist(rng));
    }
}


void zeros_(Tensor& tensor) { constant_(tensor, 0.0); }
void ones_(Tensor& tensor) { constant_(tensor, 1.0); }

void constant_(Tensor& tensor, double val) {
    if (!tensor.impl) return;
    size_t n = tensor.numel();
    if (tensor._dtype() == DType::Float32) {
        float* ptr = (float*)tensor.impl->data->data.get();
        for (size_t i = 0; i < n; ++i) ptr[i] = (float)val;
    } else {
        for (size_t i = 0; i < n; ++i) tensor.write_scalar(i, val);
    }
}

void xavier_uniform_(Tensor& tensor, double gain) {
    size_t fan_in, fan_out;
    calculate_fan_in_and_fan_out(tensor, fan_in, fan_out);
    double std = gain * std::sqrt(2.0 / (fan_in + fan_out));
    double a = std::sqrt(3.0) * std; // Calculate uniform bounds from standard deviation
    uniform_(tensor, -a, a);
}

void xavier_normal_(Tensor& tensor, double gain) {
    size_t fan_in, fan_out;
    calculate_fan_in_and_fan_out(tensor, fan_in, fan_out);
    double std = gain * std::sqrt(2.0 / (fan_in + fan_out));
    normal_(tensor, 0.0, std);
}

void kaiming_uniform_(Tensor& tensor, double a) {
    size_t fan_in, fan_out;
    calculate_fan_in_and_fan_out(tensor, fan_in, fan_out);
    // a is negative slope of rectifier. std = sqrt(2 / ((1 + a^2) * fan_in))
    double std = std::sqrt(2.0 / ((1.0 + a * a) * fan_in));
    double bound = std::sqrt(3.0) * std;
    uniform_(tensor, -bound, bound);
}
void kaiming_normal_(Tensor& tensor, double a) {
    size_t fan_in, fan_out;
    calculate_fan_in_and_fan_out(tensor, fan_in, fan_out);
    double std = std::sqrt(2.0 / ((1.0 + a * a) * fan_in));
    normal_(tensor, 0.0, std);
}

template <typename InitFunc>
void apply_bulk(std::vector<Tensor*>& params, InitFunc func) {
    for (auto* p : params) {
        if (!p || !p->impl) continue;
        if (p->impl->ndim >= 2) {
            func(*p);
        } else {
            zeros_(*p); 
        }
    }
}

void kaiming_uniform_(std::vector<Tensor*>& params) { apply_bulk(params, [](Tensor& t){ kaiming_uniform_(t, 0.0); }); }
void kaiming_normal_(std::vector<Tensor*>& params)  { apply_bulk(params, [](Tensor& t){ kaiming_normal_(t, 0.0); }); }
void xavier_uniform_(std::vector<Tensor*>& params)  { apply_bulk(params, [](Tensor& t){ xavier_uniform_(t, 1.0); }); }
void xavier_normal_(std::vector<Tensor*>& params)   { apply_bulk(params, [](Tensor& t){ xavier_normal_(t, 1.0); }); }

void zeros_(std::vector<Tensor*>& params) { for(auto* p : params) if(p) zeros_(*p); }
void ones_(std::vector<Tensor*>& params)  { for(auto* p : params) if(p) ones_(*p); }

void kaiming_init(std::vector<Tensor*>& params) {
    std::cout << "Initializing weights (Kaiming Uniform)..." << std::endl;
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    
    for (auto* p : params) {
        if (!p->impl) continue;
        
        size_t fan_in = 1;
        if (p->impl->ndim == 2) {
            fan_in = p->shape()[1];
        } else if (p->impl->ndim == 4) {
             fan_in = p->shape()[1] * p->shape()[2] * p->shape()[3];
        }

        float bound = std::sqrt(6.0f / (float)fan_in);
        bool is_bias = (p->impl->ndim == 1);

        size_t n = p->numel();
        float* ptr = (float*)p->impl->data->data.get();
        
        for (size_t i = 0; i < n; ++i) {
            if (is_bias) {
                ptr[i] = 0.0f;
            } else {
                float r = static_cast<float>(std::rand()) / RAND_MAX; 
                ptr[i] = (r * 2.0f - 1.0f) * bound;
            }
        }
    }
}