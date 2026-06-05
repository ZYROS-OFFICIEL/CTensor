#include <iostream>
#include <cassert>
#include "core/tensor.h"
#include "core/ops_dispatch.h"
#include "neuralnet.h"

void test_conv1d() {
    Tensor x = Tensor::ones({2, 3, 10});
    x.requires_grad_(true);
    
    Conv1d conv(3, 4, 3, 1, 1);
    Tensor y = conv(x);
    
    assert(y.shape().size() == 3);
    assert(y.shape()[0] == 2);
    assert(y.shape()[1] == 4);
    assert(y.shape()[2] == 10);
    
    Tensor loss = sum(y, -1);
    loss.backward();
    
    assert(x.grad().shape()[0] == 2);
    assert(conv.weight.grad().shape()[0] == 4);
}

void test_conv2d() {
    Tensor x = Tensor::ones({2, 3, 10, 10});
    x.requires_grad_(true);
    
    Conv2d conv(3, 4, 3, 3, 1, 1, 1, 1);
    Tensor y = conv(x);
    
    assert(y.shape().size() == 4);
    assert(y.shape()[0] == 2);
    assert(y.shape()[1] == 4);
    assert(y.shape()[2] == 10);
    assert(y.shape()[3] == 10);
    
    Tensor loss = sum(y, -1);
    loss.backward();
    
    assert(x.grad().shape()[0] == 2);
    assert(conv.weight.grad().shape()[0] == 4);
}

void test_conv3d() {
    Tensor x = Tensor::ones({2, 3, 5, 5, 5});
    x.requires_grad_(true);
    
    Conv3d conv(3, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1);
    Tensor y = conv(x);
    
    assert(y.shape().size() == 5);
    assert(y.shape()[0] == 2);
    assert(y.shape()[1] == 4);
    assert(y.shape()[2] == 5);
    assert(y.shape()[3] == 5);
    assert(y.shape()[4] == 5);
    
    Tensor loss = sum(y, -1);
    loss.backward();
    
    assert(x.grad().shape()[0] == 2);
    assert(conv.weight.grad().shape()[0] == 4);
}

int main() {
    test_conv1d();
    test_conv2d();
    test_conv3d();
    std::cout << "test_conv passed\n";
    return 0;
}