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
public:
    // Default to Float32/Int32 if not specified, or match dataset?
    // Let's allow user to specify.
    DataLoader(MNISTData& dataset, size_t bs, bool shuffle_data = true, 
               DType image_type = DType::Float32, DType label_type = DType::Int32) 
        : data(dataset), batch_size(bs), current_idx(0), shuffle(shuffle_data),
          img_dtype(image_type), lbl_dtype(label_type)
    {
        size_t n = data.images.shape()[0];
        indices.resize(n);
        for (size_t i = 0; i < n; ++i) indices[i] = i;
        reset();
    }
    void reset() {
        current_idx = 0;
        if (shuffle) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
        }
    }

};