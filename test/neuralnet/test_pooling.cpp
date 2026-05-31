#include <iostream>
#include <cassert>
#include "core/tensor.h"
#include "core/ops_dispatch.h"
#include "neuralnet/pooling/pooling.h"

void test_maxpool2d() {
    Tensor x = Tensor::arange(0, 16, 1.0, DType::Float32).reshape({1, 1, 4, 4});
    x.requires_grad_(true);
    MaxPool2d pool(2, 2, 2, 2);
    Tensor y = pool(x);
    
    assert(y.shape().size() == 4);
    assert(y.shape()[2] == 2);
    assert(y.shape()[3] == 2);
    
    Tensor loss = sum(y, -1);
    loss.backward();
    assert(x.grad().shape().size() == 4);
}

void test_avgpool2d() {
    Tensor x = Tensor::ones({1, 1, 4, 4});
    AvgPool2d pool(2, 2, 2, 2);
    Tensor y = pool(x);
    
    assert(y.shape().size() == 4);
    assert(y.read_scalar(0) == 1.0);
}

int main() {
    test_maxpool2d();
    test_avgpool2d();
    std::cout << "test_pooling passed\n";
    return 0;
}