#pragma once
#include <iostream>
#include <cstddef>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdlib>      // malloc/free
#include <ctime>        // time for rand seed
#include <stdexcept>    // exceptions
#include <cassert>

enum class DType { Float32, Int32, Double64 };
static size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return sizeof(float);
        case DType::Int32:   return sizeof(int);
        case DType::Double64:return sizeof(double);
    }
    return sizeof(float);
}