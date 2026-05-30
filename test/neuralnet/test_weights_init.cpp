#include <iostream>
#include <cassert>
#include <vector>
#include "core/tensor.h"
#include "neuralnet/weights/weights_init.h"

void test_constants() {
    Tensor t = Tensor::empty({5, 5});
    
    constant_(t, 7.0);
    for(size_t i = 0; i < 25; ++i) {
        assert(t.read_scalar(i) == 7.0);
    }
    
    zeros_(t);
    for(size_t i = 0; i < 25; ++i) {
        assert(t.read_scalar(i) == 0.0);
    }
    
    ones_(t);
    for(size_t i = 0; i < 25; ++i) {
        assert(t.read_scalar(i) == 1.0);
    }
}

void test_randoms() {
    Tensor t1 = Tensor::empty({10, 10});
    Tensor t2 = Tensor::empty({5});
    std::vector<Tensor*> params = {&t1, &t2};
    
    uniform_(t1, -1.0, 1.0);
    for(size_t i = 0; i < 100; ++i) {
        double v = t1.read_scalar(i);
        assert(v >= -1.0 && v <= 1.0);
    }
    
    normal_(t2, 0.0, 1.0);
    
    xavier_uniform_(params);
    xavier_normal_(params);
    kaiming_uniform_(params);
    kaiming_normal_(params);
}

void test_bulk_init() {
    Tensor t1 = Tensor::empty({10, 10});
    Tensor t2 = Tensor::empty({5});
    std::vector<Tensor*> params = {&t1, &t2};
    
    zeros_(params);
    assert(t1.read_scalar(0) == 0.0);
    assert(t2.read_scalar(0) == 0.0);
    
    kaiming_init(params);
}

int main() {
    test_constants();
    test_randoms();
    test_bulk_init();
    std::cout << "test_weights_init passed\n";
    return 0;
}