#include <iostream>
#include <cassert>
#include <vector>
#include "core.h"

void test_from_flat_vector() {
    std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
    Tensor t = from_flat_vector(v, {2, 2}, DType::Float32, false);
    assert(t.shape()[0] == 2);
    assert(t.read_scalar(1) == 2.0);
}

void test_from_2d_vector() {
    std::vector<std::vector<double>> v = {{1.0, 2.0}, {3.0, 4.0}};
    Tensor t = from_2d_vector(v, DType::Float32, false);
    assert(t.shape()[0] == 2);
    assert(t.shape()[1] == 2);
    assert(t.read_scalar(3) == 4.0);
}

int main() {
    test_from_flat_vector();
    test_from_2d_vector();
    std::cout << "test_data passed\n";
    return 0;
}