// ops.cpp
#include "ops.h"
#include "tensor.h"
#include <stdexcept>
#include <cmath>
#include <functional>
#include <omp.h>
#include <type_traits>
#include <cstring> // memcpy
#include <iostream>

#if defined(__GNUC__) || defined(__clang__)
  #if defined(__x86_64__) || defined(__i386__)
    #define HAS_BUILTIN_CPU_SUPPORTS 1
  #endif
#endif

// ----------------------------- CPU feature detection -----------------------------
static bool cpu_supports(const char* feat) {
#if defined(HAS_BUILTIN_CPU_SUPPORTS)
    // Available on GCC/Clang for x86/x64
    return __builtin_cpu_supports(feat);
#else
    (void)feat;
    return false;
#endif
}

static bool has_avx2() {
    return cpu_supports("avx2");
}
static bool has_avx512f() {
    return cpu_supports("avx512f");
}

// ----------------------------- Helpers -----------------------------
inline void check_same_shape_or_throw(const Tensor& a, const Tensor& b) {
    auto as = a.shape();
    auto bs = b.shape();
    if (as != bs) throw std::invalid_argument("shape mismatch in binary op");
}

inline size_t dtype_size_bytes(DType dt) {
    return dtype_size(dt);
}

inline bool is_cpu_device(const Tensor& t) {
    if (!t.impl) throw std::runtime_error("Empty tensor in is_cpu_device");
    return (t.impl->data->device.type == DeviceType::CPU);
}

inline bool is_cuda_device(const Tensor& t) {
    if (!t.impl) throw std::runtime_error("Empty tensor in is_cuda_device");
    return (t.impl->data->device.type == DeviceType::CUDA);
}
