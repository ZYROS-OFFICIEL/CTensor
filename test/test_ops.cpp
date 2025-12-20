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