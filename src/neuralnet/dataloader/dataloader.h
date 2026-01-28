#pragma once
#include "core/tensor.h"
#include "neuralnet/dataset/mnist.h"
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
    
    // Returns a pair {Images, Labels}
    std::pair<Tensor, Tensor> next() {
        if (current_idx >= indices.size()) {
            return {Tensor(), Tensor()}; // End of epoch
        }

        size_t end_idx = std::min(current_idx + batch_size, indices.size());
        size_t actual_batch_size = end_idx - current_idx;

        // Allocate Tensors with TARGET DType
        std::vector<size_t> img_shape = { actual_batch_size, 1, 28, 28 };
        std::vector<size_t> lbl_shape = { actual_batch_size, 1 };
        
        Tensor batch_imgs(img_shape, img_dtype, false);
        Tensor batch_lbls(lbl_shape, lbl_dtype, false);

        // Source Data (From MNIST Loader - usually Float32/Int32)
        // We use void* and dtype_bytes to be generic if needed, 
        // but we know MNISTData struct has Tensors.
        
        const void* src_img_data = data.images.impl->data->data.get();
        const void* src_lbl_data = data.labels.impl->data->data.get();
        DType src_img_dt = data.images._dtype();
        DType src_lbl_dt = data.labels._dtype();

        void* dst_img_data = batch_imgs.impl->data->data.get();
        void* dst_lbl_data = batch_lbls.impl->data->data.get();

        size_t pixels_per_img = 28 * 28;

        // Optimization: If types match, use fast memcpy
        bool fast_img = (src_img_dt == img_dtype);
        bool fast_lbl = (src_lbl_dt == lbl_dtype);
        
        size_t src_img_size = dtype_size(src_img_dt);
        size_t img_bytes_per_sample = pixels_per_img * src_img_size;
        size_t src_lbl_size = dtype_size(src_lbl_dt);

        #pragma omp parallel for
        for (size_t i = 0; i < actual_batch_size; ++i) {
            size_t original_idx = indices[current_idx + i];
            
            // --- IMAGE COPY ---
            if (fast_img) {
                // Direct memcpy
                std::memcpy(
                    (char*)dst_img_data + i * img_bytes_per_sample,
                    (char*)src_img_data + original_idx * img_bytes_per_sample,
                    img_bytes_per_sample
                );
            } else {
                // Type Conversion Loop (Slow but supports quantization/casting)
                size_t src_offset = original_idx * pixels_per_img;
                size_t dst_offset = i * pixels_per_img;
                for (size_t p = 0; p < pixels_per_img; ++p) {
                    double val = read_scalar_at(src_img_data, src_offset + p, src_img_dt);
                    write_scalar_at(dst_img_data, dst_offset + p, img_dtype, val);
                }
            }

            // --- LABEL COPY ---
            if (fast_lbl) {
                std::memcpy(
                    (char*)dst_lbl_data + i * src_lbl_size,
                    (char*)src_lbl_data + original_idx * src_lbl_size,
                    src_lbl_size
                );
            } else {
                double val = read_scalar_at(src_lbl_data, original_idx, src_lbl_dt);
                write_scalar_at(dst_lbl_data, i, lbl_dtype, val);
            }
        }

        current_idx += actual_batch_size;
        return {batch_imgs, batch_lbls};
    }
    
    size_t size() const { return indices.size(); }
    size_t num_batches() const { return (indices.size() + batch_size - 1) / batch_size; }
};