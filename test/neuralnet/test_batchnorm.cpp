#include <iostream>
#include <cassert>
#include <cmath>
#include "core/tensor.h"
#include "core/ops_dispatch.h"
#include "neuralnet/batchnorm/batchnorm.h"

void test_batchnorm() {
    Tensor x = Tensor::ones({4, 3});
    x.write_scalar(0, 2.0);
    x.requires_grad_(true);
    
    BatchNorm bn(3);
    Tensor y = bn(x);
    
    assert(y.shape()[0] == 4);
    assert(y.shape()[1] == 3);
    
    bn.eval();
    Tensor y_eval = bn(x);
    
    Tensor loss = sum(y, -1);
    loss.backward();
    
    assert(bn.gamma.grad().shape()[0] == 3);
    assert(bn.beta.grad().shape()[0] == 3);
}

int main() {
    test_batchnorm();
    std::cout << "test_batchnorm passed\n";
    return 0;
}