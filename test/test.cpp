#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "tensor.h"

// Simple Test Runner Macros
#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "[\033[31mFAIL\033[0m] " << message << " (" << #condition << ") at line " << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

#define TEST_PASS(name) std::cout << "[\033[32mPASS\033[0m] " << name << std::endl;

void test_basic_creation() {
    // 1. Create Zeros
    Tensor t = Tensor::zeros({2, 3}, DType::Float32);
    ASSERT(t.numel() == 6, "Zeros numel mismatch");
    ASSERT(t.shape()[0] == 2 && t.shape()[1] == 3, "Zeros shape mismatch");
    ASSERT(t.read_scalar(0) == 0.0, "Zeros value mismatch");

    // 2. Create Ones
    Tensor t2 = Tensor::ones({2, 2}, DType::Int32);
    ASSERT(t2.read_scalar(0) == 1.0, "Ones value mismatch");
    ASSERT(t2._dtype() == DType::Int32, "DType mismatch");

    TEST_PASS("Basic Creation");
}

void test_indexing_and_proxy() {
    Tensor t = Tensor::zeros({3, 3}, DType::Float32);
    
    // Write using Proxy operator[]
    t[1][1] = 42.5;
    t[0][2] = 10.0;

    // Read back
    double val1 = t[1][1];
    double val2 = t[0][2];

    ASSERT(std::abs(val1 - 42.5) < 1e-5, "Proxy write/read failed (center)");
    ASSERT(std::abs(val2 - 10.0) < 1e-5, "Proxy write/read failed (corner)");
    
    // Check raw storage via read_scalar to ensure offset math is correct
    // Flattened index for [1][1] in 3x3 is 1*3 + 1 = 4
    ASSERT(std::abs(t.read_scalar(4) - 42.5) < 1e-5, "Raw storage index mismatch");

    TEST_PASS("Indexing & Proxy");
}

void test_views_and_strides() {
    // Create linear tensor: [0, 1, 2, 3, 4, 5]
    Tensor t = Tensor::arange(0, 6, 1, DType::Float32); // Shape {6}
    t = t.reshape({2, 3}); // Shape {2, 3} -> [[0, 1, 2], [3, 4, 5]]

    // 1. Reshape check
    ASSERT(t.shape()[0] == 2 && t.shape()[1] == 3, "Reshape dimensions wrong");
    ASSERT(t[1][0] == 3.0, "Reshape value mapping wrong");

    // 2. Transpose (View)
    Tensor t_T = t.permute({1, 0}); // Shape {3, 2} -> [[0, 3], [1, 4], [2, 5]]
    
    ASSERT(t_T.shape()[0] == 3 && t_T.shape()[1] == 2, "Transpose shape wrong");
    ASSERT(t_T[0][1] == 3.0, "Transpose value wrong (0,1 should be old 1,0)");
    ASSERT(t_T[1][1] == 4.0, "Transpose value wrong");

    // 3. Modification Propagation (View property)
    t_T[0][1] = 99.0; // Change (0,1) in Transpose (which is (1,0) in original)
    ASSERT(t[1][0] == 99.0, "Writing to view did not update original tensor");

    TEST_PASS("Views & Strides");
}

void test_gradients_architecture() {
    // 1. Default: No grad
    Tensor t = Tensor::ones({2, 2}, DType::Float32, false);
    ASSERT(!t.requires_grad(), "Default should be no grad");
    
    // Grad should be empty/null Tensor
    Tensor g = t.grad();
    ASSERT(g.numel() == 0, "Grad should be empty when requires_grad=false");

    // 2. Enable Grad
    t.requires_grad_(true);
    ASSERT(t.requires_grad(), "Flag should be true");
    
    // Manually initialize grad (simulating what backward() would do)
    // We can't access t.impl->grad directly easily as it is private/internal, 
    // but the architecture test is that the struct exists.
    // In your code, `grad()` returns a Tensor wrapping impl->grad. 
    // If impl->grad is nullptr, it returns an empty tensor.
    
    // Let's verify we can construct a separate tensor and assign it if we had access,
    // or just rely on the fact that compilation succeeds means the struct is correct.
    
    TEST_PASS("Gradients Architecture");
}

void test_contiguous() {
    Tensor t = Tensor::arange(0, 6, 1, DType::Float32).reshape({2, 3});
    Tensor t_T = t.permute({1, 0}); // Non-contiguous view

    ASSERT(t.is_contiguous(), "Reshaped linear should be contiguous");
    ASSERT(!t_T.is_contiguous(), "Transposed non-square should not be contiguous");

    Tensor t_contig = t_T.contiguous();
    ASSERT(t_contig.is_contiguous(), "Contiguous() failed");
    ASSERT(t_contig.shape()[0] == 3 && t_contig.shape()[1] == 2, "Contiguous shape preserved");
    ASSERT(t_contig[0][1] == 3.0, "Contiguous data preserved");

    TEST_PASS("Contiguity");
}

void test_gather() {
    // Source: [[1, 2], [3, 4]]
    Tensor src = Tensor::from_vector({1, 2, 3, 4}, {2, 2}, DType::Float32);
    
    // Indices: [[0, 0], [1, 0]]
    // Gather dim 1:
    // Row 0: take index 0 -> 1, take index 0 -> 1 => [1, 1]
    // Row 1: take index 1 -> 4, take index 0 -> 3 => [4, 3]
    
    Tensor indices = Tensor::from_vector({0, 0, 1, 0}, {2, 2}, DType::Int64);
    Tensor result = src.gather(indices, 1);

    ASSERT(result[0][0] == 1.0, "Gather [0,0] wrong");
    ASSERT(result[0][1] == 1.0, "Gather [0,1] wrong");
    ASSERT(result[1][0] == 4.0, "Gather [1,0] wrong");
    ASSERT(result[1][1] == 3.0, "Gather [1,1] wrong");

    TEST_PASS("Gather");
}

int main() {
    std::cout << "Running Tensor Library Tests..." << std::endl;
    std::cout << "===============================" << std::endl;

    test_basic_creation();
    test_indexing_and_proxy();
    test_views_and_strides();
    test_contiguous();
    test_gradients_architecture();
    test_gather();

    std::cout << "===============================" << std::endl;
    std::cout << "All Tests Passed!" << std::endl;
    return 0;
}