#include <iostream>
#include <cassert>
#include "core.h"
#include "core/tensor.h"

void test_tensor_creation() {
    Tensor t({2, 3}, DType::Float32, false);
    assert(t.shape()[0] == 2);
    assert(t.shape()[1] == 3);
    assert(t.numel() == 6);
}

void test_tensor_manipulation() {
    Tensor t = Tensor::arange(0, 6, 1, DType::Float32);
    Tensor reshaped = t.reshape({2, 3});
    assert(reshaped.shape()[0] == 2);
    assert(reshaped.shape()[1] == 3);

    Tensor permuted = reshaped.permute({1, 0});
    assert(permuted.shape()[0] == 3);
    assert(permuted.shape()[1] == 2);
}

int main() {
    test_tensor_creation();
    test_tensor_manipulation();
    std::cout << "test_tensor passed\n";
    return 0;
}