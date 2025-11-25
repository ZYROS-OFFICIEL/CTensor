#include <iostream>
#include "opsmp.h"
#include "ops1.h"
#include <chrono>
#include <vector>

int main() {
    try {
        std::cout << "BENCHMARK: add_ (Sequential) vs add_mp (OpenMP)\n";

        // 1. Use LARGER tensors to see parallel benefits
        // 1024x1024 = ~1 million elements. This is large enough.
        size_t N_elements = 128 * 128; 
        Tensor a = Tensor::ones({128, 128});
        Tensor b = Tensor::full({128, 128}, 2.0); 
        
        // Warmup (to spin up threads if needed)
        Tensor warm = add_mp(a, b);

        const int ITERATIONS = 10;

        // --- Benchmark Sequential (add_) ---
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i) {
            Tensor c = matmul_(a, b); 
            // Prevent optimization by using result
            if (c.impl->storage->data.get() == nullptr) std::cerr << "null";
        }
        auto end = std::chrono::high_resolution_clock::now();
        double time_seq = std::chrono::duration<double, std::milli>(end - start).count() / ITERATIONS;
        
        std::cout << "Sequential Time: " << time_seq << " ms\n";

        // --- Benchmark Parallel (add_mp) ---
        auto start_mp = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i) {
            Tensor d = matmul_mp(a, b);
             // Prevent optimization
            if (d.impl->storage->data.get() == nullptr) std::cerr << "null";
        }
        auto end_mp = std::chrono::high_resolution_clock::now();
        double time_mp = std::chrono::duration<double, std::milli>(end_mp - start_mp).count() / ITERATIONS;

        std::cout << "OpenMP Time:     " << time_mp << " ms\n";
        std::cout << "Speedup:         " << time_seq / time_mp << "x\n";

        // Verify correctness
        Tensor diff = diff_mp(warm, add_(a, b)); // Should be near zero
        // (Add check logic here if desired)

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}