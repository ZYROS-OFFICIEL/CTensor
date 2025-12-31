#include <cstdio>

#if defined(__GNUC__) || defined(__clang__)
int main() {
    __builtin_cpu_init();

    bool avx2     = __builtin_cpu_supports("avx2");
    bool avx512f  = __builtin_cpu_supports("avx512f");

    std::printf("CPU feature detection:\n");
    std::printf("  AVX2     : %s\n", avx2    ? "YES" : "NO");
    std::printf("  AVX-512F : %s\n", avx512f ? "YES" : "NO");

    return 0;
}
#else
#error "This test requires GCC or Clang on x86"
#endif
