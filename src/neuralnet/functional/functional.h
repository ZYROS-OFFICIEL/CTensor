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

} 