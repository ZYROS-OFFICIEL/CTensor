#include <iostream>
#include <cassert>
#include "core/tensor.h"
#include "core/ops_dispatch.h"
#include "neuralnet/dropout/dropout.h"

void test_dropout() {
    Tensor x = Tensor::ones({10, 10});
    x.requires_grad_(true);
    
    Dropout drop_all(1.0);
    Tensor y1 = drop_all(x);
    assert(y1.read_scalar(0) == 0.0);
    
    Dropout drop_none(0.0);
    Tensor y2 = drop_none(x);
    assert(y2.read_scalar(0) == 1.0);
    
    Tensor loss = sum(y2, -1);
    loss.backward();
    assert(x.grad().read_scalar(0) == 1.0);
}

int main() {
    test_dropout();
    std::cout << "test_dropout passed\n";
    return 0;
}