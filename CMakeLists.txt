cmake_minimum_required(VERSION 3.26)
project(MyProject VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_INCLUDES
    src
    src/core
    src/cpu/AVX2
    src/cpu/AVX512
    src/neuralnet
    src/neuralnet/conv
    src/neuralnet/dataset
    src/neuralnet/pooling
    src/neuralnet/dropout
    src/neuralnet/dataloader
    src/neuralnet/batchnorm
)

# --------------------------------------------------
# OBJECT libraries
# --------------------------------------------------

# Scalar
add_library(opsmp OBJECT src/core/opsmp.cpp)
target_include_directories(opsmp PRIVATE ${PROJECT_INCLUDES})

add_library(core OBJECT
    src/core/autograd.cpp
    src/core/ops_dispatch.cpp
    src/core/tensor.cpp
    src/core/data.cpp
    src/core/transforme.cpp
)
target_include_directories(core PRIVATE ${PROJECT_INCLUDES})

# Neural net
add_library(neuralnet OBJECT
    src/neuralnet/loss.cpp
    src/neuralnet/layer.cpp
    src/neuralnet/Relu.cpp
    src/neuralnet/check.cpp
    src/neuralnet/dropout/dropout.cpp
    src/neuralnet/pooling/pooling.cpp
    src/neuralnet/batchnorm/batchnorm.cpp
)
target_include_directories(neuralnet PRIVATE ${PROJECT_INCLUDES})

# Convolution
add_library(conv OBJECT src/neuralnet/conv/conv.cpp)
target_include_directories(conv PRIVATE ${PROJECT_INCLUDES})

# AVX2
add_library(ops_avx2 OBJECT
    src/cpu/AVX2/ops_avx2_d64.cpp
    src/cpu/AVX2/ops_avx2_f32.cpp
)
target_include_directories(ops_avx2 PRIVATE ${PROJECT_INCLUDES})
target_compile_options(ops_avx2 PRIVATE -O3 -mavx2 -mfma -fopenmp)

# AVX512
add_library(ops_avx512 OBJECT
    src/cpu/AVX512/ops_avx512_d64.cpp
    src/cpu/AVX512/ops_avx512_f32.cpp
)
target_include_directories(ops_avx512 PRIVATE ${PROJECT_INCLUDES})
target_compile_options(ops_avx512 PRIVATE -O3 -mavx512f -mavx512dq)

add_library(myproject STATIC
    $<TARGET_OBJECTS:opsmp>
    $<TARGET_OBJECTS:ops_avx2>
    $<TARGET_OBJECTS:ops_avx512>
    $<TARGET_OBJECTS:core>
    $<TARGET_OBJECTS:neuralnet>
    $<TARGET_OBJECTS:conv>
)

target_compile_features(myproject PUBLIC cxx_std_20)

target_include_directories(myproject
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)

target_compile_options(myproject PUBLIC -fopenmp)
target_link_libraries(myproject PUBLIC gomp)


option(MYPROJECT_BUILD_TESTS "Build benchmarks/tests" ON)

if (MYPROJECT_BUILD_TESTS)
    add_executable(app test/time_avx.cpp)
    add_executable(testmnist test/testmnist.cpp)
    add_executable(trainmnist test/train_mnist.cpp)
    add_executable(transformetest test/transforme_test.cpp)
    add_executable(convnet test/convnet_test.cpp)
    target_link_libraries(app PRIVATE myproject)
    target_link_libraries(testmnist PRIVATE myproject)
    target_link_libraries(trainmnist PRIVATE myproject)
    target_link_libraries(transformetest PRIVATE myproject)
    target_link_libraries(convnet PRIVATE myproject)
endif()

install(
    DIRECTORY src/
    DESTINATION include/myproject
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(
    TARGETS myproject
    EXPORT MyProjectTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(
    EXPORT MyProjectTargets
    NAMESPACE MyProject::
    DESTINATION lib/cmake/MyProject
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    cmake/MyProjectConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MyProjectConfig.cmake
    INSTALL_DESTINATION lib/cmake/MyProject
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/MyProjectConfig.cmake
    DESTINATION lib/cmake/MyProject
)
