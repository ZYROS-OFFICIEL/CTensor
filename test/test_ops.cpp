#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>

#include "tensor.h"
#include "ops_dispatch.h"
#include "autograd.h" // Required if you are testing backward()

// --- Simple Test Framework ---
#define ASSERT_CLOSE(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            std::cerr << "Assertion failed: " << (a) << " != " << (b) \
                      << " (diff: " << std::abs((a)-(b)) << ") at line " \
                      << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while(0)

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: " << #cond << " at line " << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while(0)
void log_test(const std::string& name) {
    std::cout << "[TEST] " << name << "..." << std::endl;
}
void passed() {
    std::cout << " -> PASSED\n" << std::endl;
}
// --- Test Cases ---

void test_initialization() {
    log_test("Initialization & Shapes");
    
    // Test Ones
    Tensor a = Tensor::ones({2, 3});
    ASSERT_TRUE(a.numel() == 6);
    ASSERT_CLOSE(a[{0, 0}], 1.0, 1e-5);
    ASSERT_CLOSE(a[{1, 2}], 1.0, 1e-5);

    // Test Full
    Tensor b = Tensor::full({2, 2}, 3.14);
    ASSERT_CLOSE(b[{0, 0}], 3.14, 1e-5);
    ASSERT_CLOSE(b[{1, 1}], 3.14, 1e-5);

    // Test Range
    Tensor c = Tensor::arange(0, 5, 1, DType::Float32);
    ASSERT_TRUE(c.numel() == 5);
    ASSERT_CLOSE(c[{4}], 4.0, 1e-5);

    passed();
}
void test_arithmetic() {
    log_test("Basic Arithmetic (Add, Sub, Mul, Div)");

    Tensor a = Tensor::full({2, 2}, 2.0);
    Tensor b = Tensor::full({2, 2}, 3.0);

    // Add
    Tensor c = add(a, b);
    ASSERT_CLOSE(c[{0, 0}], 5.0, 1e-5);

    // Sub
    Tensor d = sub(b, a);
    ASSERT_CLOSE(d[{0, 0}], 1.0, 1e-5);

    // Mul
    Tensor e = mul(a, b);
    ASSERT_CLOSE(e[{0, 0}], 6.0, 1e-5);

    // Div
    Tensor f = div(a, b);
    ASSERT_CLOSE(f[{0, 0}], 2.0/3.0, 1e-5);

    passed();
}
void test_matmul() {
    log_test("Matrix Multiplication");

    // A = [[1, 2], [3, 4]]
    Tensor A = Tensor::from_vector({1, 2, 3, 4}, {2, 2});
    
    // B = [[1, 0], [0, 1]] (Identity)
    Tensor B = Tensor::from_vector({1, 0, 0, 1}, {2, 2});

    Tensor C = matmul(A, B);

    ASSERT_CLOSE(C[{0, 0}], 1.0, 1e-5);
    ASSERT_CLOSE(C[{0, 1}], 2.0, 1e-5);
    ASSERT_CLOSE(C[{1, 0}], 3.0, 1e-5);
    ASSERT_CLOSE(C[{1, 1}], 4.0, 1e-5);

    // Dot product style: [1, 2] @ [2, 1]^T -> 1*2 + 2*1 = 4
    Tensor v1 = Tensor::from_vector({1, 2}, {1, 2});
    Tensor v2 = Tensor::from_vector({2, 1}, {2, 1}); // Transposed
    Tensor D = matmul(v1, v2);
    
    ASSERT_TRUE(D.numel() == 1);
    ASSERT_CLOSE(D[{0}], 4.0, 1e-5);

    passed();
}
void test_reductions() {
    log_test("Reductions (Sum, Mean, Max)");

    Tensor t = Tensor::from_vector({1, 2, 3, 4}, {2, 2});

    Tensor s = sum(t);
    ASSERT_CLOSE(s[{0}], 10.0, 1e-5);

    Tensor m = mean(t);
    ASSERT_CLOSE(m[{0}], 2.5, 1e-5);

    Tensor mx = max(t);
    ASSERT_CLOSE(mx[{0}], 4.0, 1e-5);

    passed();
}