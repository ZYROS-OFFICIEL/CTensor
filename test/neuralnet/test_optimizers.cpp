#include <iostream>
#include <cassert>
#include <cmath>
#include "neuralnet.h"

void test_sgd() {
    Tensor w = Tensor::full({2}, 1.0, DType::Float32, true);
    w.impl->grad = intrusive_ptr<Tensorimpl>(new Tensorimpl(w.shape(), w._dtype(), false, w.device()));
    w.grad().write_scalar(0, 0.1);
    w.grad().write_scalar(1, 0.1);
    
    std::vector<Tensor*> params = {&w};
    SGD optim(params, 0.1);
    optim.step();
    
    assert(std::abs(w.read_scalar(0) - 0.99) < 1e-5);
}

void test_adam() {
    Tensor w = Tensor::full({2}, 1.0, DType::Float32, true);
    w.impl->grad = intrusive_ptr<Tensorimpl>(new Tensorimpl(w.shape(), w._dtype(), false, w.device()));
    w.grad().write_scalar(0, 0.1);
    w.grad().write_scalar(1, 0.1);
    
    std::vector<Tensor*> params = {&w};
    Adam optim(params, 0.1);
    optim.step();
}

int main() {
    test_sgd();
    test_adam();
    std::cout << "test_optim passed\n";
    return 0;
}