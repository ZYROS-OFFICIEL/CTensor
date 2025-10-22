#include "transform1.h"

int main() {
    try {
        std::cout << "TEST Transforme BEGIN\n";

        // Create fake tensor with shape (3, 2, 2)
        Tensor x = Tensor::ones({3, 2, 2});
        std::cout << "Input x: "; print_(x);

        // Create transform pipeline
        Transforme T;
        T.normalize_({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5});
        T.resize_(4, 4);

        // Apply transformations
        Tensor out = T(x);
        std::cout << "Output after normalize + resize: "; print_(out);

        std::cout << "Transforme test passed.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Transforme test failed: " << e.what() << "\n";
        return 1;
    }
}
