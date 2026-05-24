#include <iostream>
#include <cassert>
#include "neuralnet.h"

void test_accuracy() {
    Tensor p = Tensor::arange(0, 6, 1).reshape({2, 3});
    Tensor t = Tensor::full({2}, 2.0, DType::Float32);
    size_t acc = torch::metrics::accuracy(p, t);
    std::cout << "Accuracy count: " << acc << std::endl;
    assert(acc == 2);
}

void test_binary_metrics() {
    Tensor p = Tensor::from_vector({0.1, 0.9, 0.8, 0.2}, {4});
    Tensor t = Tensor::from_vector({0.0, 1.0, 0.0, 1.0}, {4});
    float acc = torch::metrics::binary_accuracy(p, t);
    assert(acc == 0.5f);
}

int main() {
    test_accuracy();
    test_binary_metrics();
    std::cout << "test_metrics passed\n";
    return 0;
}