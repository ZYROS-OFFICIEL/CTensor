#include <iostream>
#include <cassert>
#include <cmath>
#include "neuralnet.h"

void test_mse() {
    Tensor p = Tensor::full({2, 2}, 2.0, DType::Float32, true);
    Tensor t = Tensor::full({2, 2}, 3.0, DType::Float32, false);
    Tensor l = Loss::MSE(p, t);
    
    assert(l.read_scalar(0) == 1.0);
    l.backward();
    assert(p.grad().read_scalar(0) == -0.5);
}

void test_ce() {
    Tensor p = Tensor::zeros({2, 3}, DType::Float32, true);
    Tensor t = Tensor::zeros({2, 1}, DType::Int32, false);
    Tensor l = Loss::CrossEntropy(p, t, "mean");
    
    l.backward();
    assert(p.grad().shape()[0] == 2);
    assert(p.grad().shape()[1] == 3);
}

int main() {
    test_mse();
    test_ce();
    std::cout << "test_loss passed\n";
    return 0;
}