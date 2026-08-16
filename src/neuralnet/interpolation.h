#pragma once
#include "core.h"
#include "neuralnet.h"
#include <vector>
#include <stdexcept>


// Define supported interpolation modes
enum class InterpolateMode {
    Nearest,
    Linear,
    Bilinear,
    Bicubic,
    Trilinear
};




class Upsample : public Module {
public:
    std::vector<int> size;
    std::vector<float> scale_factor;
    InterpolateMode mode;
    bool align_corners;

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

    Tensor forward(const Tensor& x) {
        return functional::interpolate(x, size, scale_factor, mode, align_corners);
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }

    std::vector<Tensor*> parameters() override {
        return {};
    }
};

