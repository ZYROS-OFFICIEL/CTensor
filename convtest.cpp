#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iomanip> // For setting precision
#include <cstdlib> // For srand
#include <ctime>   // For time

#include "tensor1.h"
#include "ops1.h"
#include "autograd.h"
#include "conv.h"

// Helper function to print a tensor (from tensor1.h, good for debugging)
void print_tensor(const std::string& name, const Tensor& t) {
    std::cout << "Tensor: " << name << " Shape: (";
    auto s = t.shape();
    for(size_t i=0; i<s.size(); ++i) {
        std::cout << s[i] << (i == s.size() - 1 ? "" : ", ");
    }
    std::cout << ")" << std::endl;
    std::cout << "Data: ";
    print_t(t); // Uses the flat print from tensor1.h
}

// Helper function to compare two scalars
bool check_near(const std::string& name, double val, double expected, double tolerance = 1e-5) {
    if (std::abs(val - expected) > tolerance) {
        std::cout << "FAIL: " << name << ". Expected: " << expected << ", Got: " << val << std::endl;
        return false;
    }
    std::cout << "PASS: " << name << std::endl;
    return true;
}

// Helper function to compare two gradient tensors
bool check_gradients(const std::string& name, const Tensor& param, const Tensor& grad_numeric, double tolerance = 1e-4) {
    // Get the analytical gradient computed by backward()
    Tensor grad_analytic = tensor_from_grad(param);

    if (grad_analytic.numel() != grad_numeric.numel()) {
        std::cout << "FAIL: " << name << ". Shape mismatch!" << std::endl;
        print_tensor(name + "_analytic", grad_analytic);
        print_tensor(name + "_numeric", grad_numeric);
        return false;
    }

    double max_rel_error = 0.0;
    for (size_t i = 0; i < grad_analytic.numel(); ++i) {
        double a = grad_analytic.read_scalar(i);
        double n = grad_numeric.read_scalar(i);
        double diff = std::abs(a - n);
        double denom = std::max(std::abs(a), std::abs(n));
        
        // Avoid division by zero if both are tiny
        if (denom < tolerance) {
            continue;
        }

        double rel_error = diff / denom;
        if (rel_error > max_rel_error) {
            max_rel_error = rel_error;
        }
    }

    if (max_rel_error > tolerance) {
        std::cout << "FAIL: " << name << ". Max relative error: " << max_rel_error << " (Tolerance: " << tolerance << ")" << std::endl;
        // Optional: print tensors for debugging
        // print_tensor(name + "_analytic", grad_analytic);
        // print_tensor(name + "_numeric", grad_numeric);
        return false;
    }

    std::cout << "PASS: " << name << " (Max rel error: " << max_rel_error << ")" << std::endl;
    return true;
}

// --- Test Conv1d ---
bool test_conv1d() {
    bool ok = true;
    std::cout << "\n--- Testing Conv1d ---" << std::endl;

    // --- 1. Forward Pass Check ---
    // Input: [1, 1, 4] (Batch=1, C_in=1, W=4) -> [1, 1, 1, 1]
    Tensor input = Tensor::ones({1, 1, 4});
    // Conv: C_in=1, C_out=1, K=2, S=1, P=0
    Conv1d conv(1, 1, 2, 1, 0); 
    // Manually set weights to 1 and bias to 0
    conv.weight = Tensor::ones(conv.weight.shape());
    conv.bias = Tensor::zeros(conv.bias.shape());

    // Output shape should be (4 - 2)/1 + 1 = 3 -> [1, 1, 3]
    Tensor output = conv.forward(input);

    // Manual calculation:
    // out[0,0,0] = (in[0]*w[0] + in[1]*w[1]) + bias = (1*1 + 1*1) + 0 = 2
    // out[0,0,1] = (in[1]*w[0] + in[2]*w[1]) + bias = (1*1 + 1*1) + 0 = 2
    ok &= check_near("Conv1d Forward [0,0,0]", output[0][0][0], 2.0);
    ok &= check_near("Conv1d Forward [0,0,1]", output[0][0][1], 2.0);
    ok &= check_near("Conv1d Forward [0,0,2]", output[0][0][2], 2.0);

    // --- 2. Gradient Check ---
    // Use a simple sum() as the loss function
    auto compute_loss_1d = [&](Tensor& inp, Conv1d& layer) {
        return sum(layer.forward(inp), -1); // Sum all elements
    };

    // Re-create layer with requires_grad=true
    Tensor input_grad = Tensor::rand({1, 1, 4}, DType::Float32, true);
    Conv1d conv_grad(1, 1, 2, 1, 0); // weight/bias have requires_grad=true by default

    // Compute analytical grads
    Tensor loss = compute_loss_1d(input_grad, conv_grad);
    backward(loss);

    double h = 1e-5;

    // Check input gradient
    Tensor grad_numeric_input = Tensor::zeros(input_grad.shape());
    for (size_t i = 0; i < input_grad.numel(); ++i) {
        double orig_val = input_grad.read_scalar(i);
        input_grad.write_scalar(i, orig_val + h);
        double loss_plus = compute_loss_1d(input_grad, conv_grad).read_scalar(0);
        input_grad.write_scalar(i, orig_val - h);
        double loss_minus = compute_loss_1d(input_grad, conv_grad).read_scalar(0);
        grad_numeric_input.write_scalar(i, (loss_plus - loss_minus) / (2 * h));
        input_grad.write_scalar(i, orig_val); // Restore
    }

    // Check weight gradient
    Tensor grad_numeric_weight = Tensor::zeros(conv_grad.weight.shape());
    for (size_t i = 0; i < conv_grad.weight.numel(); ++i) {
        double orig_val = conv_grad.weight.read_scalar(i);
        conv_grad.weight.write_scalar(i, orig_val + h);
        double loss_plus = compute_loss_1d(input_grad, conv_grad).read_scalar(0);
        conv_grad.weight.write_scalar(i, orig_val - h);
        double loss_minus = compute_loss_1d(input_grad, conv_grad).read_scalar(0);
        grad_numeric_weight.write_scalar(i, (loss_plus - loss_minus) / (2 * h));
        conv_grad.weight.write_scalar(i, orig_val); // Restore
    }
    
    // Check bias gradient
    Tensor grad_numeric_bias = Tensor::zeros(conv_grad.bias.shape());
    for (size_t i = 0; i < conv_grad.bias.numel(); ++i) {
        double orig_val = conv_grad.bias.read_scalar(i);
        conv_grad.bias.write_scalar(i, orig_val + h);
        double loss_plus = compute_loss_1d(input_grad, conv_grad).read_scalar(0);
        conv_grad.bias.write_scalar(i, orig_val - h);
        double loss_minus = compute_loss_1d(input_grad, conv_grad).read_scalar(0);
        grad_numeric_bias.write_scalar(i, (loss_plus - loss_minus) / (2 * h));
        conv_grad.bias.write_scalar(i, orig_val); // Restore
    }

    // Compare gradients
    ok &= check_gradients("Conv1d Grad Input", input_grad, grad_numeric_input);
    ok &= check_gradients("Conv1d Grad Weight", conv_grad.weight, grad_numeric_weight);
    ok &= check_gradients("Conv1d Grad Bias", conv_grad.bias, grad_numeric_bias);
    return ok;
}

// --- Test Conv2d ---
bool test_conv2d() {
    bool ok = true;
    std::cout << "\n--- Testing Conv2d ---" << std::endl;

    // --- 1. Forward Pass Check ---
    // Input: [1, 1, 3, 3] (Batch=1, C_in=1, H=3, W=3) -> all 1s
    Tensor input = Tensor::ones({1, 1, 3, 3});
    // Conv: C_in=1, C_out=1, K=2x2, S=1x1, P=0x0
    Conv2d conv(1, 1, 2, 2, 1, 1, 0, 0); 
    conv.weight = Tensor::ones(conv.weight.shape());
    conv.bias = Tensor::zeros(conv.bias.shape());

    // Output shape: (3 - 2)/1 + 1 = 2 -> [1, 1, 2, 2]
    Tensor output = conv.forward(input);

    // Manual calculation:
    // out[0,0,0,0] = (1*1 + 1*1 + 1*1 + 1*1) + 0 = 4
    ok &= check_near("Conv2d Forward [0,0,0,0]", output[0][0][0][0], 4.0);
    ok &= check_near("Conv2d Forward [0,0,1,1]", output[0][0][1][1], 4.0);
    
    // --- 2. Gradient Check ---
    auto compute_loss_2d = [&](Tensor& inp, Conv2d& layer) {
        return sum(layer.forward(inp), -1);
    };

    Tensor input_grad = Tensor::rand({1, 2, 4, 4}, DType::Float32, true); // B=1, C_in=2, H=4, W=4
    Conv2d conv_grad(2, 3, 2, 2, 1, 1, 0, 0); // C_in=2, C_out=3, K=2, S=1, P=0

    Tensor loss = compute_loss_2d(input_grad, conv_grad);
    backward(loss);
    double h = 1e-5;

    // Check input gradient
    Tensor grad_numeric_input = Tensor::zeros(input_grad.shape());
    for (size_t i = 0; i < input_grad.numel(); ++i) {
        double orig_val = input_grad.read_scalar(i);
        input_grad.write_scalar(i, orig_val + h);
        double loss_plus = compute_loss_2d(input_grad, conv_grad).read_scalar(0);
        input_grad.write_scalar(i, orig_val - h);
        double loss_minus = compute_loss_2d(input_grad, conv_grad).read_scalar(0);
        grad_numeric_input.write_scalar(i, (loss_plus - loss_minus) / (2 * h));
        input_grad.write_scalar(i, orig_val);
    }

    // Check weight gradient
    Tensor grad_numeric_weight = Tensor::zeros(conv_grad.weight.shape());
    for (size_t i = 0; i < conv_grad.weight.numel(); ++i) {
        double orig_val = conv_grad.weight.read_scalar(i);
        conv_grad.weight.write_scalar(i, orig_val + h);
        double loss_plus = compute_loss_2d(input_grad, conv_grad).read_scalar(0);
        conv_grad.weight.write_scalar(i, orig_val - h);
        double loss_minus = compute_loss_2d(input_grad, conv_grad).read_scalar(0);
        grad_numeric_weight.write_scalar(i, (loss_plus - loss_minus) / (2 * h));
        conv_grad.weight.write_scalar(i, orig_val);
    }
    
    // Check bias gradient
    Tensor grad_numeric_bias = Tensor::zeros(conv_grad.bias.shape());
    for (size_t i = 0; i < conv_grad.bias.numel(); ++i) {
        double orig_val = conv_grad.bias.read_scalar(i);
        conv_grad.bias.write_scalar(i, orig_val + h);
        double loss_plus = compute_loss_2d(input_grad, conv_grad).read_scalar(0);
        conv_grad.bias.write_scalar(i, orig_val - h);
        double loss_minus = compute_loss_2d(input_grad, conv_grad).read_scalar(0);
        grad_numeric_bias.write_scalar(i, (loss_plus - loss_minus) / (2 * h));
        conv_grad.bias.write_scalar(i, orig_val);
    }

    // Compare gradients
    ok &= check_gradients("Conv2d Grad Input", input_grad, grad_numeric_input);
    ok &= check_gradients("Conv2d Grad Weight", conv_grad.weight, grad_numeric_weight);
    ok &= check_gradients("Conv2d Grad Bias", conv_grad.bias, grad_numeric_bias);
    return ok;
}

// --- Test Conv3d ---
bool test_conv3d() {
    bool ok = true;
    std::cout << "\n--- Testing Conv3d ---" << std::endl;

    // --- 1. Forward Pass Check ---
    // Input: [1, 1, 3, 3, 3] (B=1, C=1, D=3, H=3, W=3) -> all 1s
    Tensor input = Tensor::ones({1, 1, 3, 3, 3});
    // Conv: C_in=1, C_out=1, K=2x2x2, S=1, P=0
    Conv3d conv(1, 1, 2, 2, 2, 1, 1, 1, 0, 0, 0); 
    conv.weight = Tensor::ones(conv.weight.shape());
    conv.bias = Tensor::zeros(conv.bias.shape());

    // Output shape: (3 - 2)/1 + 1 = 2 -> [1, 1, 2, 2, 2]
    Tensor output = conv.forward(input);

    // Manual calculation:
    // out[0,0,0,0,0] = (1*1 + 1*1 + ... 8 times) + 0 = 8
    ok &= check_near("Conv3d Forward [0,0,0,0,0]", output[0][0][0][0][0], 8.0);
    ok &= check_near("Conv3d Forward [0,0,1,1,1]", output[0][0][1][1][1], 8.0);
    
    // --- 2. Gradient Check ---
    auto compute_loss_3d = [&](Tensor& inp, Conv3d& layer) {
        return sum(layer.forward(inp), -1);
    };

    // B=1, C_in=2, D=3, H=3, W=3
    Tensor input_grad = Tensor::rand({1, 2, 3, 3, 3}, DType::Float32, true); 
    // C_in=2, C_out=3, K=2, S=1, P=0
    Conv3d conv_grad(2, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0); 

    Tensor loss = compute_loss_3d(input_grad, conv_grad);
    backward(loss);
    double h = 1e-5;

    // Check input gradient
    Tensor grad_numeric_input = Tensor::zeros(input_grad.shape());
    for (size_t i = 0; i < input_grad.numel(); ++i) {
        double orig_val = input_grad.read_scalar(i);
        input_grad.write_scalar(i, orig_val + h);
        double loss_plus = compute_loss_3d(input_grad, conv_grad).read_scalar(0);
        input_grad.write_scalar(i, orig_val - h);
        double loss_minus = compute_loss_3d(input_grad, conv_grad).read_scalar(0);
        grad_numeric_input.write_scalar(i, (loss_plus - loss_minus) / (2 * h));
        input_grad.write_scalar(i, orig_val);
    }

    // Check weight gradient
    Tensor grad_numeric_weight = Tensor::zeros(conv_grad.weight.shape());
    for (size_t i = 0; i < conv_grad.weight.numel(); ++i) {
        double orig_val = conv_grad.weight.read_scalar(i);
        conv_grad.weight.write_scalar(i, orig_val + h);
        double loss_plus = compute_loss_3d(input_grad, conv_grad).read_scalar(0);
        conv_grad.weight.write_scalar(i, orig_val - h);
        double loss_minus = compute_loss_3d(input_grad, conv_grad).read_scalar(0);
        grad_numeric_weight.write_scalar(i, (loss_plus - loss_minus) / (2 * h));
        conv_grad.weight.write_scalar(i, orig_val);
    }
    
    // Check bias gradient
    Tensor grad_numeric_bias = Tensor::zeros(conv_grad.bias.shape());
    for (size_t i = 0; i < conv_grad.bias.numel(); ++i) {
        double orig_val = conv_grad.bias.read_scalar(i);
        conv_grad.bias.write_scalar(i, orig_val + h);
        double loss_plus = compute_loss_3d(input_grad, conv_grad).read_scalar(0);
        conv_grad.bias.write_scalar(i, orig_val - h);
        double loss_minus = compute_loss_3d(input_grad, conv_grad).read_scalar(0);
        grad_numeric_bias.write_scalar(i, (loss_plus - loss_minus) / (2 * h));
        conv_grad.bias.write_scalar(i, orig_val);
    }

    // Compare gradients
    ok &= check_gradients("Conv3d Grad Input", input_grad, grad_numeric_input);
    ok &= check_gradients("Conv3d Grad Weight", conv_grad.weight, grad_numeric_weight);
    ok &= check_gradients("Conv3d Grad Bias", conv_grad.bias, grad_numeric_bias);
    return ok;
}


int main() {
    // Seed the random number generator
    std::srand((unsigned int)std::time(nullptr));
    
    // Set output precision for floats
    std::cout << std::fixed << std::setprecision(8);

    try {
        
        if (!test_conv1d()) return 1;
        // test_conv1d(); // <-- This was a redundant call
        if (!test_conv2d()) return 1;
        if (!test_conv3d()) return 1;

    } catch (const std::exception& e) {
        std::cerr << "!!! TEST FAILED (Exception) !!!\n" << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nAll convolution tests completed." << std::endl;
    return 0;
}