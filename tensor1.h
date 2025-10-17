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

using namespace std;
enum class DType { Float32, Int32, Double64 };
static size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return sizeof(float);
        case DType::Int32:   return sizeof(int);
        case DType::Double64:return sizeof(double);
    }
    return sizeof(float);
}

struct Storage{
    std::shared_ptr<void> data;
    size_t size;
    static shared_ptr<void> allocate(size_t n, DType dt){
        size = n * dtype_size(dt);
        data = std::shared_ptr<void>(malloc(size), free);
        if(!data && size) throw std::bad_alloc();
        memset(data.get(), 0, size);
        return std::shared_ptr<void>(p, std::free);
    }
}