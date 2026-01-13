#include <iostream>
#include <chrono>
#include "tensor.h"


main(){

    auto start1 = std::chrono::high_resolution_clock::now();
    Tensor a = Tensor::ones({1000, 1000}, DType::Float64);
    Tensor b = Tensor::ones({1000, 1000}, DType::Float64);
    for (int i = 0; i < 100; ++i) {
        Tensor c = addmp(a, b);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end1 - start1;
    std::cout << "Time taken for 100 additions of 1000x1000 tensors: " << elapsed1.count() << " seconds\n";

}
