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