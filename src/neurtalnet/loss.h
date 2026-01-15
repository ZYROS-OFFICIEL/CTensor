#pragma once
#include "tensor.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include "autograd.h"
#include <immintrin.h>
#include <cstring>
#include "ops_dispatcher.h"

// Base class containing all available loss functions
class Loss {
public:
    static Tensor MSE(const Tensor& pred, const Tensor& target);
    static Tensor MAE(const Tensor& pred, const Tensor& target, std::string reduction = "mean");
    static Tensor HuberLoss(const Tensor& pred, const Tensor& target, std::string reduction = "mean", double delta=1.0);
    
    static Tensor SmoothL1Loss(const Tensor& pred, const Tensor& target, std::string reduction = "mean") {
        return HuberLoss(pred, target, reduction, 1.0);
    }

    // --- UPDATED: Expects Integer Index Targets [Batch, 1] ---
    static Tensor CrossEntropy(const Tensor& pred, const Tensor& target, std::string reduction = "mean");
    static Tensor NLLLoss(const Tensor& pred, const Tensor& target, std::string reduction = "mean");
    // ---------------------------------------------------------

    static Tensor LogCosh(const Tensor& pred, const Tensor& target, std::string reduction = "mean");
    
    // Classification (Element-wise / One-Hot still required for BCE/KL)
    static Tensor BCE(const Tensor& pred, const Tensor& target, std::string reduction = "mean");
    static Tensor KLDiv(const Tensor& pred, const Tensor& target, std::string reduction = "mean");
    
    static Tensor HingeLoss(const Tensor& pred, const Tensor& target, std::string reduction = "mean");

    // Ranking losses:
    static Tensor MarginRankingLoss(const Tensor& input1, const Tensor& input2, const Tensor& target, double margin = 0.0, std::string reduction = "mean");
};

