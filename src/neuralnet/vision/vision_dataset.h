#pragma once
#include "core.h"
#include "neuralnet.h"

namespace torch {
namespace vision {
    namespace datasets {
        inline TensorDataset MNIST(const std::string& img_path, const std::string& lbl_path) {
            MNISTData raw_data = load_mnist(img_path, lbl_path);
            TensorDataset ds(raw_data.images, raw_data.labels);
            
            // Standardize dataset internally.
            // CAUTION: raw_data.images is likely UInt8 because of the .idx3-ubyte format!
            // We must handle both UInt8 and Float32 properly to avoid memory corruption.
            bool is_u8 = ds.is_u8_images;
            
            ds.transform = [is_u8](const void* src, float* dst, size_t n) {
                if (is_u8) {
                    const uint8_t* s = (const uint8_t*)src; 
                    for(size_t i = 0; i < n; ++i) {
                        // 1. Convert 8-bit pixel (0-255) to float (0-1)
                        float val = (float)s[i] / 255.0f;
                        // 2. Standardize using MNIST mean and std
                        dst[i] = (val - 0.1307f) / 0.3081f;
                    }
                } else {
                    const float* s = (const float*)src; 
                    for(size_t i = 0; i < n; ++i) {
                        // Already float, just standardize
                        dst[i] = (s[i] - 0.1307f) / 0.3081f;
                    }
                }
            };
            return ds;
        }
    }
}
}