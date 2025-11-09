#include "tensor1.h"
#include <vector>
#include <stdexcept>
#include <cmath>
#include "autograd.h"
#include <immintrin.h>
#include <cstring>
#include "ops1.h"
#include <string>
#pragma once
#include "tensor1.h"

// Base class containing all available loss functions
class Loss {
public:
    static Tensor MSE(const Tensor& pred, const Tensor& target);
    static Tensor MAE(const Tensor& pred, const Tensor& target,std::string reduction = "mean");
    static Tensor HuberLoss(const Tensor& pred, const Tensor& target,,std::string reduction = "mean",float delta=1.0);
    // Later you can add:
    static Tensor CrossEntropy(const Tensor& pred, const Tensor& target);
};

// Gradient function for MSE
struct GradMSE : GradFn {
    Tensor pred, target;
    GradMSE(const Tensor& pred_, const Tensor& target_) : pred(pred_), target(target_) {
        parents = {pred};
    }
    void backward(const Tensor& self) override;
};
struct GradMAE : GradFn {
    Tensor pred, target;
    std::string reduction;

    GradMAE(const Tensor& pred_, const Tensor& target_, const std::string& reduction_)
        : pred(pred_), target(target_), reduction(reduction_) {
        parents = {pred};
    }

    void backward(const Tensor& self) override;
};

struct GradHuberLoss : GradFn {
    Tensor pred, target;
    std::string reduction;
    float delta;

    GradHuberLoss(const Tensor& pred_, const Tensor& target_, const std::string& reduction_, float delta_)
        : pred(pred_), target(target_), reduction(reduction_), delta(delta_) {
        parents = {pred};
    }

    void backward(const Tensor& self) override;
};

struct GradCrossEntropy : GradFn {
    Tensor pred, target;
    GradCrossEntropy(const Tensor& pred_, const Tensor& target_) : pred(pred_), target(target_) {
        parents = {pred};
    }
    void backward(const Tensor& self) override;
};