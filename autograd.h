// ------------------- autograd core -------------------
#include <unordered_map>
#include <functional>
#include <stack>
#include <set>
#include "tensor1.h"
#include "ops1.h"


inline void accumulate_grad(Tensor& target, const Tensor& grad) {
    if (!target.impl->grad) {
        // allocate grad if missing
        target.impl->storage->grad = std::shared_ptr<void>(
            std::malloc(target.numel() * target.dtype_bytes()), std::free);
        std::memset(target.impl->storage->grad.get(), 0, target.numel() * target.dtype_bytes());
    }

    // elementwise addition
    size_t n = target.numel();
    for (size_t i = 0; i < n; ++i) {
        double current = read_scalar_at(target.impl->storage->grad.get(), i, target._dtype());
        double g = read_scalar_at(grad.impl->storage->data.get(), i, grad._dtype());
        write_scalar_at(target.impl->storage->grad.get(), i, target._dtype(), current + g);
    }
}

struct GradFn {
    virtual void backward(const Tensor& grad_output) = 0;
    virtual ~GradFn() = default;
};

struct GradAdd : GradFn {
    Tensor a, b;
    GradAdd(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) {}

    void backward(Tensor& grad_output) override {
        if (a.requires_grad()) accumulate_grad(a, grad_output);
        if (b.requires_grad()) accumulate_grad(b, grad_output);
    }
};

struct GradSub : GradFn {
    Tensor a, b;
    GradSub(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) {}

    void backward(Tensor& grad_output) override {
        if (a.requires_grad()) accumulate_grad(a, grad_output);
        if (b.requires_grad()) {
            Tensor neg = grad_output.clone();
            size_t n = neg.numel();
            for (size_t i = 0; i < n; ++i) neg[i] = -neg[i];
            accumulate_grad(b, neg);
        }
    }
};


// ------------------- Tensor::backward (add to your Tensor) -------------------
void backward(Tensor& loss) {
    if (!loss.impl->grad_fn)
        throw std::runtime_error("loss has no grad_fn");

    Tensor grad_out(loss.impl->shape, loss.impl->dtype, false);
    double* g = grad_out.data();
    for (size_t i = 0; i < grad_out.numel(); ++i)
        g[i] = 1.0;

    loss.impl->grad_fn->backward(grad_out);
}