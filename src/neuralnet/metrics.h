#pragma once
#include "core.h"
#include "neuralnet.h"

namespace torch{

namespace metrics {
    inline size_t accuracy(const Tensor& pred_logits, const Tensor& target) {
        Tensor p = pred_logits.argmax(1); // [B]
        Tensor t = target;
        // Ensure target is flat for accuracy to avoid broadcasting bugs
        if (t.shape().size() > 1 && t.shape().back() == 1) t = t.flatten();
        
        Tensor match = (p == t).astype(DType::Float32);
        return (size_t)::sum(match).item<float>();
    }
}

}