cmake_minimum_required(VERSION 4.1.0-rc1)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------------
# Scalar code
# -------------------------------
add_library(opsmp OBJECT src/opsmp.cpp)

# AVX2 code
add_library(ops_avx2 OBJECT src/ops_avx2.cpp)
target_compile_options(ops_avx2 PRIVATE
    -O0 -mavx2 -mfma
)

# AVX-512 code
add_library(ops_avx512 OBJECT src/ops_avx512.cpp)
target_compile_options(ops_avx512 PRIVATE
    -O0 -mavx512f
)

# -------------------------------
# Final executable
# -------------------------------
add_executable(app
    src/main.cpp
    $<TARGET_OBJECTS:ops_scalar>
    $<TARGET_OBJECTS:ops_avx2>
    $<TARGET_OBJECTS:ops_avx512>
)
