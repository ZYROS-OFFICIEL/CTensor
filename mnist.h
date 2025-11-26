#pragma once
#include "tensor1.h"
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm> // for std::reverse

// --- MNIST Loader Helper ---

// MNIST files use big-endian integers. We need to swap to little-endian (on x86).
inline uint32_t read_uint32(std::ifstream& is) {
    uint32_t val;
    is.read(reinterpret_cast<char*>(&val), 4);
    // Swap bytes
    return ((val << 24) & 0xFF000000) |
           ((val << 8)  & 0x00FF0000) |
           ((val >> 8)  & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

