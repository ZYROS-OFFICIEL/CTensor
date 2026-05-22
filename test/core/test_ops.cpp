#include <iostream>
#include <cassert>
#include <cmath>
#include "core.h"

void test_basic_ops() {
    Tensor a = Tensor::full({2, 2}, 2.0, DType::Float32, false);
    Tensor b = Tensor::full({2, 2}, 3.0, DType::Float32, false);
    Tensor c = a + b;
    assert(c.read_scalar(0) == 5.0);
    Tensor d = a * b;
    assert(d.read_scalar(0) == 6.0);
}

void test_matmul() {
    Tensor a = Tensor::arange(1, 5, 1, DType::Float32).reshape({2, 2});
    Tensor b = Tensor::arange(1, 5, 1, DType::Float32).reshape({2, 2});
    Tensor c = matmul(a, b);
    assert(c.read_scalar(0) == 7.0); 
    assert(c.read_scalar(1) == 10.0);
    assert(c.read_scalar(2) == 15.0);
    assert(c.read_scalar(3) == 22.0);
}

int main() {
    test_basic_ops();
    test_matmul();
    std::cout << "test_ops passed\n";
    return 0;
}