#include <iostream>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

struct SIMDInfo {
    bool sse42 = false;
    bool avx = false;
    bool avx2 = false;
    bool avx512f = false;

    static SIMDInfo detect() {
        SIMDInfo info;

    #if defined(__GNUC__) || defined(__clang__)
        info.avx512f = __builtin_cpu_supports("avx512f");
    #endif

        return info;
    }
};

int main() {
    SIMDInfo simd = SIMDInfo::detect();

    if (simd.avx512f)
        std::cout << "✅ AVX-512 supported\n";
    else
        std::cout << "❌ No SIMD support\n";
}
