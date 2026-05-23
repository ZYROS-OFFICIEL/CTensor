#include <iostream>
#include <cassert>
#include <memory>
#include "neuralnet.h"

void test_linear() {
    Linear fc(10, 5);
    Tensor x = Tensor::ones({2, 10});
    Tensor y = fc(x);
    assert(y.shape()[0] == 2);
    assert(y.shape()[1] == 5);
    auto params = fc.parameters();
    assert(params.size() == 2);
}

void test_flatten() {
    Flatten flat;
    Tensor x = Tensor::ones({2, 3, 4, 5});
    Tensor y = flat(x);
    assert(y.shape().size() == 2);
    assert(y.shape()[0] == 2);
    assert(y.shape()[1] == 60);
}

void test_sequential() {
    Sequential seq;
    seq.add(std::make_shared<Flatten>());
    seq.add(std::make_shared<Linear>(60, 10));
    
    auto params = seq.parameters();
    assert(params.size() == 2);
}

int main() {
    test_linear();
    test_flatten();
    test_sequential();
    std::cout << "test_layer passed\n";
    return 0;
}