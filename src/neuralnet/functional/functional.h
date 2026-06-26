#pragma once
#include "core.h"
#include "neuralnet.h"
#include <vector>
#include <stdexcept>

enum class InterpolateMode {
    Nearest,
    Linear,
    Bilinear,
    Bicubic,
    Trilinear
};
namespace functional {

inline Tensor embedding(
    const Tensor& weight, 
    const Tensor& indices, 
    int padding_idx = -1
) {
    return gather(weight, indices, padding_idx); 
}


inline Tensor interpolate(
    const Tensor& input,
    const std::vector<int>& size = {},
    const std::vector<float>& scale_factor = {},
    InterpolateMode mode = InterpolateMode::Nearest,
    bool align_corners = false
) {
    if (size.empty() && scale_factor.empty()) {
        throw std::invalid_argument("interpolate() requires either 'size' or 'scale_factor' to be specified.");
    }
    if (!size.empty() && !scale_factor.empty()) {
        throw std::invalid_argument("interpolate() cannot take both 'size' and 'scale_factor' simultaneously.");
    }

    return ops::interpolate(input, size, scale_factor, static_cast<int>(mode), align_corners);
}


} 