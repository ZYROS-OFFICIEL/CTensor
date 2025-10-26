#include <iostream>
#include "ops1.h"
#include <ctime>
#include <chrono>

int main() {
    
    try {
        
        std::cout << "TEST add_ with broadcasting BEGIN\n";

        Tensor a = Tensor::ones({3, 3});
        Tensor b = Tensor::full({1, 3}, 2.0); ;

        std::cout << "Tensor a:\n"; print_(a);
        std::cout << "Tensor b:\n"; print_(b);
        int N = 1000000;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            Tensor c = add(a, b);
        }
        auto end = std::chrono::high_resolution_clock::now();
        Tensor c = add_simd(a, b);
        double avg = std::chrono::duration<double, std::micro>(end - start).count() / N;
        std::cout << "Average add_ time: " << avg << " microseconds\n";
        std::cout << "Tensor c:\n"; print_(c);
        std::cout << "add_ test OK\n";
    } catch (const std::exception& e) {
        std::cerr << "add_ test FAILED: " << e.what() << "\n";
        return 1;
    }

}
