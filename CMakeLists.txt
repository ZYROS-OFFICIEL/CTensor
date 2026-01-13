cmake_minimum_required(VERSION 3.26)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PROJECT_INCLUDES
    src
    src/core
    src/cpu/AVX2
    src/cpu/AVX512
)
# -------------------------------
# Scalar code
# -------------------------------
add_library(opsmp OBJECT src/core/opsmp.cpp)
target_include_directories(opsmp PRIVATE ${PROJECT_INCLUDES})
add_library(core OBJECT src/core/autograd.cpp src/core/ops_dispatch.cpp src/core/tensor.cpp src/core/data.cpp)
target_include_directories(core PRIVATE ${PROJECT_INCLUDES})


# AVX2 code
add_library(ops_avx2 OBJECT src/cpu/AVX2/ops_avx2_d64.cpp src/cpu/AVX2/ops_avx2_f32.cpp)
target_include_directories(ops_avx2 PRIVATE ${PROJECT_INCLUDES})
target_compile_options(ops_avx2 PRIVATE
    -O0 -mavx2 -mfma
)

# AVX-512 code
add_library(ops_avx512 OBJECT src/cpu/AVX512/ops_avx512_d64.cpp src/cpu/AVX512/ops_avx512_f32.cpp)
target_include_directories(ops_avx512 PRIVATE ${PROJECT_INCLUDES})
target_compile_options(ops_avx512 PRIVATE
    -O0 -mavx512f -mavx512dq
)

# -------------------------------
# Final executable
# -------------------------------

add_executable(app
    test/test_ops.cpp
    $<TARGET_OBJECTS:opsmp>
    $<TARGET_OBJECTS:ops_avx2>
    $<TARGET_OBJECTS:ops_avx512>
    $<TARGET_OBJECTS:core>
)
target_include_directories(app PRIVATE
    src
    src/core
    src/cpu/AVX2
    src/cpu/AVX512
)
target_compile_options(app PRIVATE -O0)