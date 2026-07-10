#include <iostream>
#include <cassert>
#include "core.h"
#include "neuralnet.h"

void test_conv_transpose1d() {
    Tensor x = Tensor::ones({2, 3, 10});
    x.requires_grad_(true);
    ConvTranspose1d conv(3, 4, 3, 2, 1, 1);
    Tensor y = conv(x);
    assert(y.shape().size() == 3);
    assert(y.shape()[0] == 2);
    assert(y.shape()[1] == 4);
    assert(y.shape()[2] == 20); 
    Tensor loss = sum(y, -1);
    loss.backward();
    assert(x.grad().shape()[0] == 2);
    assert(conv.weight.grad().shape()[0] == 3);
}

void test_conv_transpose2d() {
    Tensor x = Tensor::ones({2, 3, 10, 10});
    x.requires_grad_(true);
    ConvTranspose2d conv(3, 4, 3, 3, 2, 2, 1, 1, 1, 1);
    Tensor y = conv(x);
    assert(y.shape().size() == 4);
    assert(y.shape()[0] == 2);
    assert(y.shape()[1] == 4);
    assert(y.shape()[2] == 20);
    assert(y.shape()[3] == 20);
    Tensor loss = sum(y, -1);
    loss.backward();
    assert(x.grad().shape()[0] == 2);
    assert(conv.weight.grad().shape()[0] == 3);
}

void test_conv_transpose3d() {
    Tensor x = Tensor::ones({2, 3, 5, 5, 5});
    x.requires_grad_(true);
    ConvTranspose3d conv(3, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1);
    Tensor y = conv(x);
    assert(y.shape().size() == 5);
    assert(y.shape()[0] == 2);
    assert(y.shape()[1] == 4);
    assert(y.shape()[2] == 10);
    assert(y.shape()[3] == 10);
    assert(y.shape()[4] == 10);
    Tensor loss = sum(y, -1);
    loss.backward();
    assert(x.grad().shape()[0] == 2);
    assert(conv.weight.grad().shape()[0] == 3);
}

int main() {
    test_conv_transpose1d();
    test_conv_transpose2d();
    test_conv_transpose3d();
    std::cout << "test_conv_transpose passed\n";
    return 0;
}

