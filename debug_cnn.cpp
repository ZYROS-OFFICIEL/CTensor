#include <iostream>
#include <omp.h>
#include "tensor1.h"
#include "conv.h"
#include "opsmp.h"
#include "autograd.h"

int main() {
    try {
        std::cout << "=== TEST: Backward Stress Test (Batch 64) ===\n";
        omp_set_num_threads(8);

        // 1. Setup Input [64, 1, 28, 28]
        Tensor x = Tensor::rand({64, 1, 28, 28}, DType::Float32, true);
        Conv2d layer(1, 6, 5, 5, 1, 1, 2, 2, DType::Float32);

        // 2. Forward
        std::cout << "Running Forward...\n";
        Tensor y = layer.forward(x);
        
        // 3. Backward (The Crash Site)
        std::cout << "Running Backward...\n";
        Tensor loss = sum_mp(y);
        backward(loss); // <--- Expect Silent Crash Here

        std::cout << "[PASS] Backward finished successfully.\n";
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Exception: " << e.what() << "\n";
    }
    return 0;
}