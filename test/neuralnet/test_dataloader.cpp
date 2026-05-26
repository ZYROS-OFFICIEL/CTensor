#include <iostream>
#include <cassert>
#include "neuralnet.h"

void test_dataloader() {
    Tensor x = Tensor::ones({10, 3, 4, 4});
    Tensor y = Tensor::zeros({10, 1}, DType::Int32);
    
    TensorDataset ds(x, y);
    SimpleDataLoader loader(ds, 4, false);
    
    assert(loader.size() == 10);
    assert(loader.has_next() == true);
    
    auto batch1 = loader.next();
    assert(batch1.first.shape()[0] == 4);
    assert(batch1.first.shape()[1] == 3);
    
    auto batch2 = loader.next();
    assert(batch2.first.shape()[0] == 4);
    
    auto batch3 = loader.next();
    assert(batch3.first.shape()[0] == 2);
    
    assert(loader.has_next() == false);
}

int main() {
    test_dataloader();
    std::cout << "test_dataloader passed\n";
    return 0;
}