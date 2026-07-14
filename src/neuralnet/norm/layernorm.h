#pragma once
#include "core.h"
#include "neuralnet.h"
#include <vector>

class LayerNorm : public Module {
public:
    Tensor weight, bias;
    double eps;

    LayerNorm(int normalized_shape, double eps = 1e-5);

    Tensor forward(const Tensor& x);
    Tensor operator()(const Tensor& x) { return forward(x); }

    std::vector<Tensor*> parameters() override { return {&weight, &bias}; }
};
