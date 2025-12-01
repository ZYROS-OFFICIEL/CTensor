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
void test_ensure_grad_buffer() {
    Tensor A = Tensor::ones({3,3}, DType::Float32, true);
    ensure_grad_buffer(A, true);

    std::cout << "\n=== TEST ensure_grad_buffer ===\n";
    print_(tensor_from_grad(A)); // MUST be all zeros
}
void test_accumulate_grad_scalar() {
    std::cout << "\n=== TEST accumulate_grad (scalar) ===\n";

    Tensor A = Tensor::zeros({1}, DType::Float32, true);
    Tensor g = Tensor::full({1}, 5.0, DType::Float32, false);

    accumulate_grad(A, g);

    print_(tensor_from_grad(A));   // MUST be [5]
}
void test_accumulate_grad_broadcast() {
    std::cout << "\n=== TEST accumulate_grad broadcast ===\n";

    Tensor A = Tensor::zeros({3}, DType::Float32, true);
    Tensor g = Tensor::full({1}, 2.0, DType::Float32, false);

    accumulate_grad(A, g);

    print_(tensor_from_grad(A)); // MUST be [2,2,2]
}
void test_tensor_from_grad() {
    std::cout << "\n=== TEST tensor_from_grad ===\n";

    Tensor A = Tensor::zeros({2,3}, DType::Float32, true);
    ensure_grad_buffer(A, false);

    float* p = (float*)A.impl->storage->grad.get();
    for (int i=0;i<6;i++) p[i] = float(i+1);

    Tensor G = tensor_from_grad(A);
    print_(G);   // MUST print 1 2 3 4 5 6
}
void test_add_backward() {
    std::cout << "\n=== TEST add backward ===\n";

    Tensor A = Tensor::ones({1,10}, DType::Float32, true);
    Tensor B = Tensor::full({10,1}, 2.0, DType::Float32, true);

    Tensor Y = add_mp(A, B); // Y = A + B
    backward(Y);

    std::cout << "grad A: "; print_(tensor_from_grad(A));
    std::cout << "grad B: "; print_(tensor_from_grad(B));

    // Expected:
    // grad A = [1,1,1]
    // grad B = [1,1,1]
}

//it's an add test 
void test_matmul_grad() {    
    std::cout << "\n=== TEST matmul backward ===\n";

    Tensor A = Tensor::ones({1,10}, DType::Float32, true);
    Tensor B = Tensor::ones({10,1}, DType::Float32, true);

    Tensor Y = matmul_mp(A, B); 
    backward(Y);

    std::cout << "grad A: "; print_(tensor_from_grad(A));
    std::cout << "grad B: "; print_(tensor_from_grad(B));

    // Expected:
    // grad A = [1,1,1]
    // grad B = [1,1,1]
}
void test_matmul_with_permute() {
    std::cout << "\n=== TEST: add with permute ===\n";

    Tensor A = Tensor::ones({1,10}, DType::Float32, true);   // contiguous
    Tensor B = Tensor::ones({10,1}, DType::Float32, true);   // contiguous
    Tensor Y1 = add_mp(A, B); // expected scalar 10
    
    std::cout << "Y1 (A @ B): "; print_(Y1);
    backward(Y1);
    print_(tensor_from_grad(Y1), "Grad Y1");
    print_(tensor_from_grad(A), "Grad A after Y1");
    print_(tensor_from_grad(B), "Grad B after Y1");
    Tensor BT = B.permute({1,0});   // likely non-contiguous view [1,10]
    Tensor AT = A.permute({1,0});   // [10,1] view

    Tensor Y2 = matmul_mp(AT, BT);   // expected scalar 10 also
    std::cout << "Y2 (AT @ BT): "; print_(Y2);
}

// TEST 2: Linear Layer Update
void test_linear_update() {
    std::cout << "\n=== TEST 2: Linear Layer Update ===\n";
    Linear fc(10, 1, false, DType::Float32); // 10 inputs, 1 output, no bias
    fc.weight.requires_grad_(true);
    
    // Initialize weight to 0.5
    float* ptr = (float*)fc.weight.impl->storage->data.get();
    for (size_t i = 0; i < 10; ++i) ptr[i] = 0.5f;

    
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
        test_matmul_with_permute();
        test_matmul_grad();
        test_ensure_grad_buffer();
        test_accumulate_grad_scalar();
        test_accumulate_grad_broadcast();
        test_tensor_from_grad();
        test_add_backward();
        test_permute_grad();
        test_linear_update();
        test_conv2d();
    } catch (const std::exception& e) {
        std::cerr << "\nCRITICAL ERROR: " << e.what() << std::endl;
    }
    return 0;
}
