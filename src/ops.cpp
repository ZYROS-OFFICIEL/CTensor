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
