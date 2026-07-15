#pragma once
#include "core.h"
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
    (void)padding_idx;
    return embedding_lookup(weight, indices);
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
    if (!scale_factor.empty()) {
        throw std::invalid_argument("interpolate() with 'scale_factor' is not supported; pass 'size' instead.");
    }

    std::vector<size_t> out_size(size.begin(), size.end());
    std::string mode_str = (mode == InterpolateMode::Nearest) ? "nearest" : "bilinear";
    return ::interpolate(input, out_size, mode_str, align_corners);
}

inline Tensor softmax(const Tensor& x, int dim = -1) {
    int nd = (int)x.shape().size();
    size_t dim_pos = (size_t)(dim < 0 ? dim + nd : dim);

    Tensor m = max(x, dim);
    Tensor shifted = x - m.unsqueeze(dim_pos);
    Tensor e = exp(shifted);
    Tensor s = sum(e, dim);
    return e / s.unsqueeze(dim_pos);
}

}
