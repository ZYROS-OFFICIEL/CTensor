#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <functional>
#include <numeric>
#include "core.h"
#include "neuralnet.h" 
#include "loss.h"      

// =======================================================================
//                              UTILITIES
// =======================================================================

// 1. Robust Initialization
inline void robust_weight_init(std::vector<Tensor*>& params, float scale = 0.05f) {
    std::cout << "Initializing weights (Robust Normal, scale=" << scale << ")..." << std::endl;
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (auto* p : params) {
        if (!p->impl) continue;
        size_t n = p->numel();
        float* ptr = (float*)p->impl->data->data.get();
        for (size_t i = 0; i < n; ++i) {
            float r = static_cast<float>(std::rand()) / RAND_MAX; 
            ptr[i] = (r * 2.0f - 1.0f) * scale; 
        }
    }
}

// 2. Corruption Check
inline bool check_weights_corrupted(const std::vector<Tensor*>& params) {
    for(auto* p : params) {
        if(!p->impl) continue;
        float* ptr = (float*)p->impl->data->data.get();
        size_t n = p->numel();
        // Check a sample for speed
        size_t check_limit = std::min(n, (size_t)1000);
        for(size_t i=0; i< check_limit; ++i) { 
            if (std::isnan(ptr[i]) || std::isinf(ptr[i]) || std::abs(ptr[i]) > 50.0f) return true;
        }
    }
    return false;
}

// =======================================================================
//                              DATA LOADING
// =======================================================================

struct TensorDataset {
    Tensor images; // [N, C, H, W] or [N, D]
    Tensor labels; // [N, 1]
    
    // Transform function: (src_pointer, dst_pointer, num_elements)
    std::function<void(const void*, float*, size_t)> transform;
    
    bool is_u8_images;

    TensorDataset(Tensor imgs, Tensor lbls) : images(imgs), labels(lbls) {
        is_u8_images = (images._dtype() == DType::UInt8);
    }
};

class SimpleDataLoader {
    TensorDataset& dataset;
    size_t batch_size;
    bool shuffle;
    std::vector<size_t> indices;
    size_t current_idx = 0;
    std::mt19937 rng;

public:
    SimpleDataLoader(TensorDataset& ds, size_t bs, bool shuffle_ = true) 
        : dataset(ds), batch_size(bs), shuffle(shuffle_), rng(std::random_device{}()) {
        indices.resize(dataset.images.shape()[0]);
        std::iota(indices.begin(), indices.end(), 0);
    }

    void reset() {
        current_idx = 0;
        if (shuffle) {
            std::shuffle(indices.begin(), indices.end(), rng);
        }
    }

    bool has_next() const {
        return current_idx < indices.size();
    }

    size_t size() const { return indices.size(); }

    // Returns {BatchImages, BatchLabels}
    std::pair<Tensor, Tensor> next() {
        if (current_idx >= indices.size()) return {Tensor(), Tensor()};

        size_t end_idx = std::min(current_idx + batch_size, indices.size());
        size_t actual_batch_size = end_idx - current_idx;

        // 1. Prepare Output Tensors
        std::vector<size_t> img_shape = dataset.images.shape();
        img_shape[0] = actual_batch_size;
        Tensor batch_img(img_shape, DType::Float32, false);

        std::vector<size_t> lbl_shape = dataset.labels.shape();
        lbl_shape[0] = actual_batch_size;
        Tensor batch_lbl(lbl_shape, DType::Int32, false);

        // 2. Fill Data
        uint8_t* u8_src_base = nullptr;
        float* f32_src_base = nullptr;
        if (dataset.is_u8_images) u8_src_base = (uint8_t*)dataset.images.impl->data->data.get();
        else f32_src_base = (float*)dataset.images.impl->data->data.get();

        int32_t* lbl_src_base = (int32_t*)dataset.labels.impl->data->data.get();
        
        float* dst_img_ptr = (float*)batch_img.impl->data->data.get();
        int32_t* dst_lbl_ptr = (int32_t*)batch_lbl.impl->data->data.get();

        size_t sample_size = 1;
        for(size_t i=1; i<img_shape.size(); ++i) sample_size *= img_shape[i];

        // Parallel copy/transform
        #pragma omp parallel for
        for (int i = 0; i < (int)actual_batch_size; ++i) {
            size_t real_idx = indices[current_idx + i];
            
            // Labels
            dst_lbl_ptr[i] = lbl_src_base[real_idx];

            // Images
            float* current_dst = dst_img_ptr + i * sample_size;
            const void* src = dataset.is_u8_images ? (void*)(u8_src_base + real_idx * sample_size) 
                                                   : (void*)(f32_src_base + real_idx * sample_size);

            if (dataset.transform) {
                dataset.transform(src, current_dst, sample_size);
            } else {
                // Default: normalize 0-255 to 0-1
                if (dataset.is_u8_images) {
                    uint8_t* s = (uint8_t*)src;
                    for(size_t k=0; k<sample_size; ++k) current_dst[k] = (float)s[k] / 255.0f;
                } else {
                    float* s = (float*)src;
                    for(size_t k=0; k<sample_size; ++k) current_dst[k] = s[k]; 
                }
            }
        }

        current_idx += batch_size;
        return {batch_img, batch_lbl};
    }
};

// =======================================================================
//                              TRAINER
// =======================================================================

// Included Optimizers here for convenience if separate header not available
// (Assuming the user has train_utils.h containing Optimizers, 
//  we just need the training loop function)

template <typename Model, typename Optimizer>
void train_one_epoch(int epoch, Model& model, SimpleDataLoader& loader, Optimizer& optim) {
    loader.reset();
    double total_loss = 0.0;
    int batches = 0;
    size_t processed = 0;

    std::cout << "Epoch " << epoch << " starting..." << std::endl;

    while(loader.has_next()) {
        auto [data, target] = loader.next();
        
        optim.zero_grad();
        Tensor output = model.forward(data);
        Tensor loss = Loss::CrossEntropy(output, target);
        
        backward(loss);
        
        // Stability: Clip gradients
        clip_grad_norm(model.parameters(), 1.0);
        
        optim.step();

        double l = loss.read_scalar(0);
        if (std::isfinite(l)) {
            total_loss += l;
        } else {
            std::cout << "\n[WARN] NaN Loss at batch " << batches << ". Skipping stats.\n";
        }

        batches++;
        processed += data.shape()[0];

        if (batches % 100 == 0) {
            std::cout << "Batch " << std::setw(4) << batches 
                      << " [" << std::setw(6) << processed << "/" << loader.size() << "]"
                      << " Loss: " << std::fixed << std::setprecision(5) << l << "\r" << std::flush;
        }
    }
    std::cout << "\nEpoch " << epoch << " Finished. Avg Loss: " 
              << (batches > 0 ? total_loss/batches : 0.0) << std::endl;
}