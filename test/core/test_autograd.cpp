#include <iostream>
#include <cassert>
#include <cmath>
#include "core.h"

void test_autograd_basic() {
    Tensor x = Tensor::full({2}, 3.0, DType::Float32, true);
    Tensor y = x * x;
    Tensor z = sum(y, -1);
    z.backward();
    Tensor grad = x.grad();
    assert(grad.read_scalar(0) == 6.0);
    assert(grad.read_scalar(1) == 6.0);
}

void test_autograd_matmul() {
    Tensor a = Tensor::full({2, 2}, 2.0, DType::Float32, true);
    Tensor b = Tensor::full({2, 2}, 3.0, DType::Float32, true);
    Tensor c = matmul(a, b);
    Tensor d = sum(c, -1);
    d.backward();
    Tensor grad_a = a.grad();
    assert(grad_a.read_scalar(0) == 6.0);
}

int main() {
    test_autograd_basic();
    test_autograd_matmul();
    std::cout << "test_autograd passed\n";
    return 0;
}