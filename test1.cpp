#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "tensor1.h"
#include "autograd.h"
#include "opsmp.h"
#include "layer.h"
#include "conv.h"

// Helper to print first few elements
void print_head(const Tensor& t, const std::string& name) {
    std::cout << name << " [";
    for(int i=0; i< std::min((size_t)5, t.numel()); ++i) 
        std::cout << t.read_scalar(i) << ", ";
    std::cout << "...]\n";
}

// TEST 1: Basic Autograd & Permute (Fixing the Linear Layer bug)
void test_permute_grad() {
    std::cout << "\n=== TEST 1: Permute Gradient ===\n";
    // x: [2, 3] -> x.t(): [3, 2]
    Tensor x = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3}, DType::Float32, true);
    
    Tensor y = x.t_(); // Permute
    Tensor loss = sum_mp(y); // Sum all elements. Grad should be 1.0 everywhere.
    
    backward(loss);
    
    std::cout << "Backward run complete.\n";
    
    // Check gradients of x
    // If working, all grads should be 1.0
    // If broken (stride bug), they will be scrambled or zero.
    if (x.impl->storage->grad) {
        print_head(tensor_from_grad(x), "x.grad");
        double g0 = read_scalar_at(x.impl->storage->grad.get(), 0, x._dtype());
        if (std::abs(g0 - 1.0) < 1e-5) std::cout << "[PASS] Permute Gradient correct.\n";
        else std::cout << "[FAIL] Permute Gradient incorrect (Expected 1.0).\n";
    } else {
        std::cout << "[FAIL] x has no grad!\n";
    }
}

// TEST 2: Linear Layer Update
void test_linear_update() {
    std::cout << "\n=== TEST 2: Linear Layer Update ===\n";
    Linear fc(10, 1, false, DType::Float32); // 10 inputs, 1 output, no bias
    fc.weight.requires_grad_(true);
    
    // Initialize weight to 0.5
    fc.weight = Tensor::full({1, 10}, 0.5, DType::Float32, true); 
    
    Tensor input = Tensor::ones({1, 10}, DType::Float32, false);
    Tensor output = fc(input); // output = input @ weight.t()
    
    // Loss = output (scalar). dLoss/dOut = 1.
    // dOut/dW = input. 
    // So dLoss/dW should be 1.0 for all weights.
    backward(output);
    
    print_head(tensor_from_grad(fc.weight), "Weight Grad");
    
    double wg = read_scalar_at(fc.weight.impl->storage->grad.get(), 0, fc.weight._dtype());
    if (std::abs(wg - 1.0) < 1e-5) std::cout << "[PASS] Linear Weight Gradient correct.\n";
    else std::cout << "[FAIL] Linear Weight Gradient incorrect.\n";
}

// TEST 3: Conv2d Forward/Backward
void test_conv2d() {
    std::cout << "\n=== TEST 3: Conv2d Gradients ===\n";
    // 1 input channel, 1 output channel, 3x3 kernel
    Conv2d conv(1, 1, 3, 3, 1, 1, 1, 1, DType::Float32); 
    conv.weight.requires_grad_(true);
    
    Tensor input = Tensor::ones({1, 1, 5, 5}, DType::Float32, false);
    Tensor output = conv(input);
    
    Tensor loss = sum_mp(output);
    backward(loss);
    
    if (conv.weight.impl->storage->grad) {
        print_head(tensor_from_grad(conv.weight), "Conv Weight Grad");
        std::cout << "[PASS] Conv2d backward ran without crashing.\n";
    } else {
        std::cout << "[FAIL] Conv2d weight has no grad.\n";
    }
}

int main() {
    try {
        test_permute_grad();
        test_linear_update();
        test_conv2d();
    } catch (const std::exception& e) {
        std::cerr << "\nCRITICAL ERROR: " << e.what() << std::endl;
    }
    return 0;
}
