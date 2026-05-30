#include <iostream>
#include <cassert>
#include <cmath>
#include "core.h"
#include "neuralnet.h"

void test_norms() {
    Tensor t = Tensor::arange(-2.0, 3.0, 1.0);
    Tensor l1 = norm(t, 1.0);
    assert(std::abs(l1.read_scalar(0) - 6.0) < 1e-5);
    
    Tensor l2 = norm(t, 2.0);
    assert(std::abs(l2.read_scalar(0) - std::sqrt(10.0)) < 1e-5);
    
    Tensor inf = infinity_norm(t);
    assert(std::abs(inf.read_scalar(0) - 2.0) < 1e-5);
    
    Tensor z = zero_norm(t);
    assert(std::abs(z.read_scalar(0) - 4.0) < 1e-5);
}

void test_lp_normalize() {
    Tensor t = Tensor::full({2}, 3.0);
    t.write_scalar(1, 4.0);
    
    Tensor norm_t = Lp_Norm(t, 2.0);
    assert(std::abs(norm_t.read_scalar(0) - 0.6) < 1e-5);
    assert(std::abs(norm_t.read_scalar(1) - 0.8) < 1e-5);
}

int main() {
    test_norms();
    test_lp_normalize();
    std::cout << "test_norm passed\n";
    return 0;
}