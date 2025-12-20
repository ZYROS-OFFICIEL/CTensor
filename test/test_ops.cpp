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