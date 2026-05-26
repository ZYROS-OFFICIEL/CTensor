#include <iostream>
#include <cassert>
#include "neuralnet.h"

void test_nn_linear() {
    Tensor x = Tensor::ones({2, 10});
    torch::nn::Linear fc(10, 5);
    Tensor y = fc(x);
    assert(y.shape()[0] == 2);
    assert(y.shape()[1] == 5);
}

int main() {
    test_nn_linear();
    std::cout << "test_nn passed\n";
    return 0;
}