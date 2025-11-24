#include <iostream>
#include "opsmp.h"
#include <chrono>

int main() {
    try {
        std::cout << "TEST add_ with broadcasting BEGIN\n";

        // Input tensors
        Tensor a = Tensor::ones({3, 3});
        Tensor b = Tensor::full({1, 3}, 2.0);

        std::cout << "Tensor a:\n"; print_tensor(a);
        std::cout << "Tensor b:\n"; print_tensor(b);

        // Output tensor reused for timing
        Tensor c(a.shape(), a._dtype());

        // Number of repetitions
        const int N = 1000000;

        // Start timer
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            c = add_(a, b);  // reuse c, avoids repeated allocation
        }
        auto end = std::chrono::high_resolution_clock::now();

        // Compute average time per addition in microseconds
        double avg = std::chrono::duration<double, std::micro>(end - start).count() / N;

        std::cout << "Average add_ time: " << avg << " microseconds\n";
        std::cout << "Tensor c:\n"; print_tensor(c);
        std::cout << "add_ test OK\n";
    } catch (const std::exception& e) {
        std::cerr << "add_ test FAILED: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
