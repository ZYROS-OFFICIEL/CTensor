#include <iostream>
#include <chrono>
#include "tensor.h"
#include "opsmp.h"
#include "ops_dispatch.h"
main(){
    
    Tensor a = Tensor::rand({1000, 1000}, DType::Float32);
    Tensor b = Tensor::rand({1000, 1000}, DType::Float32);
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        Tensor c = add_mp(a, b);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end1 - start1;
    std::cout << "Time taken for 100 Scalar additions of 1000x1000 tensors: " << elapsed1.count() << " seconds\n";
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        Tensor c = add(a, b);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end2 - start2;
    std::cout << "Time taken for 100 Avx additions of 1000x1000 tensors: " << elapsed2.count() << " seconds\n";

    std::cout << "RAPIDITY : " << elapsed1.count() / elapsed2.count() << " \n";

}
