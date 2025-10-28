#pragma once
#include "tensor1.h"

Tensor add_backward(const Tensor& grad_output) {
    return grad_output; // d(a+b)/da = 1
}

Tensor mul_backward(const Tensor& grad_output, const Tensor& a, const Tensor& b) {
    return grad_output * b; // d(ab)/da = b
}