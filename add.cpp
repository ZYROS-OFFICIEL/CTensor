#include <iostream>
#include "ops1.h"
#include <ctime>

int main() {
    std::clock_t start = std::clock();
    try {
        
        std::cout << "TEST add_ with broadcasting BEGIN\n";

        Tensor a = Tensor::ones({3, 3});
        Tensor b = Tensor::full({1, 3}, 2.0); ;

        std::cout << "Tensor a:\n"; print_(a);
        std::cout << "Tensor b:\n"; print_(b);

        Tensor c = add_(a, b);
        std::cout << "Tensor c = a + b:\n"; print_(c);

        std::cout << "add_ test OK\n";
    } catch (const std::exception& e) {
        std::cerr << "add_ test FAILED: " << e.what() << "\n";
        return 1;
    }
    std::clock_t end = std::clock();
    double elapsed = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time: " << elapsed << " seconds\n";
}
