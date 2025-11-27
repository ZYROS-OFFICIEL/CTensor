#include <iostream>
#include <vector>
#include "tensor1.h"
#include "opsmp.h"
#include "conv.h"
#include "layer.h"
#include "pooling.h"
#include "Relu.h"

// Helper to print shapes clearly
void check(const std::string& tag, const Tensor& t) {
    std::cout << "[" << tag << "] Shape: (";
    for (size_t i = 0; i < t.impl->ndim; ++i) {
        std::cout << t.impl->shape[i] << (i < t.impl->ndim - 1 ? ", " : "");
    }
    std::cout << ")" << std::endl;
}

int main() {
    std::cout << "\n=== DEBUGGING MNIST SHAPES ===\n";

    try {
        // 1. Simulate a Batch of Images [64, 1, 28, 28]
        int B = 64;
        Tensor x = Tensor::rand({(size_t)B, 1, 28, 28});
        check("Input Batch", x);

        // 2. Conv1 Setup (1 -> 6, 5x5, pad 2)
        Conv2d conv1(1, 6, 5, 5, 1, 1, 2, 2);
        check("Conv1 Weight", conv1.weight); // Should be [6, 1, 5, 5]

        // 3. Conv1 Forward
        std::cout << "Running Conv1...\n";
        x = conv1(x);
        check("Conv1 Output", x); // Expected: [64, 6, 28, 28]

        // 4. Pool1 (2x2)
        MaxPool2d pool1(2, 2);
        std::cout << "Running Pool1...\n";
        x = pool1(x);
        check("Pool1 Output", x); // Expected: [64, 6, 14, 14]

        // 5. Conv2 Setup (6 -> 16, 5x5, pad 0)
        Conv2d conv2(6, 16, 5, 5);
        check("Conv2 Weight", conv2.weight); // Should be [16, 6, 5, 5]

        // 6. Conv2 Forward
        std::cout << "Running Conv2...\n";
        x = conv2(x);
        check("Conv2 Output", x); // Expected: [64, 16, 10, 10]

        // 7. Pool2 (2x2)
        MaxPool2d pool2(2, 2);
        std::cout << "Running Pool2...\n";
        x = pool2(x);
        check("Pool2 Output", x); // Expected: [64, 16, 5, 5]

        // 8. Flatten
        Flatten flat;
        std::cout << "Running Flatten...\n";
        x = flat(x);
        check("Flatten Output", x); // CRITICAL: Expected [64, 400] (16*5*5=400)

        // 9. Safety Check (reproducing logic from train_mnist)
        if (x.impl->ndim != 2) {
            std::cout << "!!! Flatten produced non-2D tensor. Attempting Reshape fix...\n";
            size_t batch_size = x.impl->shape[0];
            size_t features = x.numel() / batch_size;
            x = x.reshape({batch_size, features});
            check("Reshaped Output", x);
        }

        // 10. Linear 1 (400 -> 120)
        Linear fc1(400, 120);
        check("FC1 Weight", fc1.weight); // [120, 400]
        
        std::cout << "Running FC1...\n";
        
        // --- DEBUGGING INSIDE LINEAR ---
        // We manually run the steps of Linear::forward here to see where it breaks
        Tensor w_t = fc1.weight.permute({1, 0});
        check("FC1 Weight Transposed", w_t); // [400, 120]
        
        std::cout << "Executing MatMul(Input, W_T)...\n";
        // Explicitly print dims being checked inside matmul_mp
        size_t K_input = x.impl->shape[x.impl->ndim - 1];
        size_t K_weight = w_t.impl->shape[w_t.impl->ndim - 2];
        std::cout << "Checking Inner Dims: Input Last=" << K_input << " vs Weight 2ndLast=" << K_weight << "\n";
        
        x = matmul_mp(x, w_t); // <--- This is likely where it crashes
        check("FC1 Output", x);

        std::cout << "\n=== SUCCESS: Forward pass dimensions are consistent. ===\n";

    } catch (const std::exception& e) {
        std::cerr << "\n!!! FAILURE !!!\nError message: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}