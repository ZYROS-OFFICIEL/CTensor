#pragma once
#include "core.h"
#include "neuralnet.h"
#include "neuralnet/functional/functional.h"
#include <vector>
#include <stdexcept>

class Upsample : public Module {
public:
    std::vector<int> size;
    std::vector<float> scale_factor;
    InterpolateMode mode;
    bool align_corners;

    // Constructor
    Upsample(
        std::vector<int> size = {},
        std::vector<float> scale_factor = {},
        InterpolateMode mode = InterpolateMode::Nearest,
        bool align_corners = false
    ) : size(size), 
        scale_factor(scale_factor), 
        mode(mode), 
        align_corners(align_corners) 
    {}

    // Forward pass
    // Input shape is typically [Batch, Channels, Height, Width] for Bilinear/Nearest 2D
    Tensor forward(const Tensor& x) {
        return functional::interpolate(x, size, scale_factor, mode, align_corners);
    }

    // Overload for elegant module(x) syntax
    Tensor operator()(const Tensor& x) {
        return forward(x);
    }

    // Upsampling is a purely mathematical routing operation (like ReLU or Pooling),
    // meaning it contains no learnable weights or biases.
    std::vector<Tensor*> parameters() override {
        return {};
    }
};
