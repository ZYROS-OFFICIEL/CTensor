#pragma once
#include "tensor1.h"
#include "mnist.h"
#include <vector>
#include <algorithm>
#include <random>
#include <cstring>

class DataLoader {
    MNISTData& data;
    size_t batch_size;
    std::vector<size_t> indices;
    size_t current_idx;
    bool shuffle;
    
    // Target DTypes
    DType img_dtype;
    DType lbl_dtype;
};