#include <iostream>
#include <cassert>
#include <vector>
#include <cstdio>
#include "core.h"
#include "neuralnet.h"

void test_save_load() {
    Tensor t1 = Tensor::full({2, 2}, 3.14, DType::Float32, true);
    Tensor t2 = Tensor::full({3}, -1.5, DType::Float32, true);
    
    t1.write_scalar(1, 2.71);
    
    std::vector<Tensor*> save_params = {&t1, &t2};
    std::string filename = "test_checkpoint_tmp.bin";
    
    checkpoints::save_weights(save_params, filename);
    
    Tensor l1 = Tensor::zeros({2, 2}, DType::Float32, true);
    Tensor l2 = Tensor::zeros({3}, DType::Float32, true);
    
    std::vector<Tensor*> load_params = {&l1, &l2};
    checkpoints::load_weights(load_params, filename);
    
    assert(std::abs(l1.read_scalar(0) - 3.14) < 1e-5);
    assert(std::abs(l1.read_scalar(1) - 2.71) < 1e-5);
    assert(std::abs(l1.read_scalar(2) - 3.14) < 1e-5);
    assert(std::abs(l2.read_scalar(0) - (-1.5)) < 1e-5);
    
    std::remove(filename.c_str());
}

int main() {
    test_save_load();
    std::cout << "test_check passed\n";
    return 0;
}