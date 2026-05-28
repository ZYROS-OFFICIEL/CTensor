#include <iostream>
#include <cassert>
#include <cmath>
#include "core/tensor.h"
#include "neuralnet/Relu.h"

void test_standard_relu() {
    Relu relu_layer;
    Tensor x = Tensor::full({4}, -2.0, DType::Float32, false);
    x.write_scalar(0, 3.0);
    x.write_scalar(2, 5.0);
    
    Tensor y = relu_layer(x);
    assert(y.read_scalar(0) == 3.0);
    assert(y.read_scalar(1) == 0.0);
    assert(y.read_scalar(2) == 5.0);
    assert(y.read_scalar(3) == 0.0);
}

void test_leaky_relu() {
    Tensor x = Tensor::full({2}, -2.0, DType::Float32, true);
    x.write_scalar(0, 2.0);
    
    Tensor y = LeakyRelu(x, 0.1);
    assert(std::abs(y.read_scalar(0) - 2.0) < 1e-5);
    assert(std::abs(y.read_scalar(1) - (-0.2)) < 1e-5);
    
    Tensor loss = y * y;
    loss.backward();
    
    assert(x.grad().read_scalar(0) != 0.0);
    assert(std::abs(x.grad().read_scalar(1) - (-0.04)) < 1e-5);
}

void test_prelu() {
    PRelu prelu_layer(1, 0.25);
    Tensor x = Tensor::full({2}, -4.0, DType::Float32, true);
    x.write_scalar(0, 4.0);
    
    Tensor y = prelu_layer(x);
    assert(std::abs(y.read_scalar(0) - 4.0) < 1e-5);
    assert(std::abs(y.read_scalar(1) - (-1.0)) < 1e-5);
    
    Tensor loss = y * y;
    loss.backward();
    
    assert(prelu_layer.weight.grad().read_scalar(0) != 0.0);
}

int main() {
    test_standard_relu();
    test_leaky_relu();
    test_prelu();
    std::cout << "test_relu passed\n";
    return 0;
}