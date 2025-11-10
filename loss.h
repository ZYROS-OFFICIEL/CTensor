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
    static Tensor HuberLoss(const Tensor& pred, const Tensor& target,std::string reduction = "mean",double delta=1.0);
    static Tensor SmoothL1Loss(const Tensor& pred, const Tensor& target,std::string reduction = "mean") {
        return HuberLoss(pred, target, reduction, 1.0);
    }
    static Tensor CrossEntropy(const Tensor& pred, const Tensor& target);
    static Tensor LogCosh(const Tensor& pred, const Tensor& target,std::string reduction = "mean");
    //Classification losses:
    static Tensor BCE(const Tensor& pred, const Tensor& target,std::string reduction = "mean");
    static Tensor KLDiv(const Tensor& pred, const Tensor& target,std::string reduction = "mean");
    static Tensor NLLLoss(const Tensor& pred, const Tensor& target,std::string reduction = "mean");
    static Tensor HingeLoss(const Tensor& pred, const Tensor& target,std::string reduction = "mean");
    //Ranking losses:
    static Tensor MarginRankingLoss(const Tensor& input1, const Tensor& input2, const Tensor& target, double margin = 0.0, std::string reduction = "mean");

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
    double delta;

    GradHuberLoss(const Tensor& pred_, const Tensor& target_, const std::string& reduction_, double delta_)
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
struct GradBCE : GradFn {
    Tensor pred, target;
    std::string reduction;

    GradBCE(const Tensor& pred_, const Tensor& target_, const std::string& reduction_)
        : pred(pred_), target(target_), reduction(reduction_) {
        parents = {pred};
    }

    void backward(const Tensor& self) override;
};

struct GradKLDiv : GradFn {
    Tensor pred, target;
    std::string reduction;

    GradKLDiv(const Tensor& pred_, const Tensor& target_, const std::string& reduction_)
        : pred(pred_), target(target_), reduction(reduction_) {
        parents = {pred};
    }

    void backward(const Tensor& self) override;
};

struct GradNLLLoss : GradFn {
    Tensor pred, target;
    std::string reduction;

    GradNLLLoss(const Tensor& pred_, const Tensor& target_, const std::string& reduction_)
        : pred(pred_), target(target_), reduction(reduction_) {
        parents = {pred};
    }

    void backward(const Tensor& self) override;
};

struct GradHingeLoss : GradFn {
    Tensor pred, target;
    std::string reduction;

    GradHingeLoss(const Tensor& pred_, const Tensor& target_, const std::string& reduction_)
        : pred(pred_), target(target_), reduction(reduction_) {
        parents = {pred};
    }

    void backward(const Tensor& self) override;
};
struct GradMarginRankingLoss : GradFn {
    Tensor input1, input2, target;
    double margin;
    std::string reduction;

    GradMarginRankingLoss(const Tensor& input1_, const Tensor& input2_, const Tensor& target_, double margin_, const std::string& reduction_)
        : input1(input1_), input2(input2_), target(target_), margin(margin_), reduction(reduction_) {
        parents = {input1, input2};
    }

    void backward(const Tensor& self) override;
};
struct GradLogCosh : GradFn {
    Tensor pred, target;
    std::string reduction;

    GradLogCosh(const Tensor& pred_, const Tensor& target_, const std::string& reduction_)
        : pred(pred_), target(target_), reduction(reduction_) {
        parents = {pred};
    }

    void backward(const Tensor& self) override;
};