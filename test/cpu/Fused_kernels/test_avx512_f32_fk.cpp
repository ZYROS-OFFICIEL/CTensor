#include <iostream>
#include <cassert>
#include <cmath>
#include "core.h"
#include "cpu.h"

#if defined(__GNUC__) || defined(__clang__)
  #if defined(__x86_64__) || defined(__i386__)
    #define HAS_BUILTIN_CPU_SUPPORTS 1
  #endif
#endif

bool has_avx512() {
#ifdef HAS_BUILTIN_CPU_SUPPORTS
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

void test_fma_f32() {
    if (!has_avx512()) return;
    Tensor a = Tensor::full({4}, 2.0, DType::Float32);
    Tensor b = Tensor::full({4}, 3.0, DType::Float32);
    Tensor c = Tensor::full({4}, 1.0, DType::Float32);
    Tensor res = fma_avx512_f32(a, b, c);
    assert(std::abs(res.read_scalar(0) - 7.0) < 1e-5);
}

void test_add_relu_f32() {
    if (!has_avx512()) return;
    Tensor a = Tensor::full({4}, -2.0, DType::Float32);
    Tensor b = Tensor::full({4}, 1.0, DType::Float32);
    Tensor res = add_relu_avx512_f32(a, b);
    assert(std::abs(res.read_scalar(0)) < 1e-5);
}

int main() {
    test_fma_f32();
    test_add_relu_f32();
    return 0;
}