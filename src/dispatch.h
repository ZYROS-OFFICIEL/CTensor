#pragma once
#include "tensor1.h"
#include <stdexcept>
#include <string>

// Helper macro to define the case statement
#define DISPATCH_CASE(ENUM, TYPE, ...) \
    case ENUM: { \
        using scalar_t = TYPE; \
        __VA_ARGS__(); \
        break; \
    }

// The Main Dispatcher
// Usage: DISPATCH_ALL_TYPES(tensor.dtype(), "function_name", [&] { ... code using scalar_t ... });
#define DISPATCH_ALL_TYPES(DTYPE, NAME, ...) \
    switch (DTYPE) { \
        DISPATCH_CASE(DType::Float32,  float,    __VA_ARGS__) \
        DISPATCH_CASE(DType::Int32,    int32_t,  __VA_ARGS__) \
        DISPATCH_CASE(DType::Double64, double,   __VA_ARGS__) \
        DISPATCH_CASE(DType::UInt8,    uint8_t,  __VA_ARGS__) \
        DISPATCH_CASE(DType::Int8,     int8_t,   __VA_ARGS__) \
        DISPATCH_CASE(DType::Int16,    int16_t,  __VA_ARGS__) \
        DISPATCH_CASE(DType::Int64,    int64_t,  __VA_ARGS__) \
        DISPATCH_CASE(DType::Bool,     bool,     __VA_ARGS__) \
        /* Float16 TODO: Add half-precision class support later */ \
        default: throw std::runtime_error(std::string(NAME) + ": unsupported dtype"); \
    }