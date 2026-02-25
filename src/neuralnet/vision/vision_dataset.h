#pragma once
#include "core.h"
#include "neuralnet.h"

namespace torch {
namespace vision {
    namespace datasets {
        inline TensorDataset MNIST(const std::string& img_path, const std::string& lbl_path) {
            MNISTData raw_data = load_mnist(img_path, lbl_path);
            TensorDataset ds(raw_data.images, raw_data.labels);
            
            // --- CRITICAL FIX: Safe Memory Casting ---
            // Detects if the data is 8-bit bytes (0-255) vs 32-bit floats.
            bool is_u8 = ds.is_u8_images;
            
            ds.transform = [is_u8](const void* src, float* dst, size_t n) {
                if (is_u8) {
                    // Safe Cast: Read 1 byte, convert to float, then normalize
                    const uint8_t* s = (const uint8_t*)src; 
                    for (size_t i = 0; i < n; ++i) {
                        float val = (float)s[i] / 255.0f;
                        dst[i] = (val - 0.1307f) / 0.3081f;
                    }
                } else {
                    // Already float, just apply MNIST mean/std
                    const float* s = (const float*)src; 
                    for (size_t i = 0; i < n; ++i) {
                        dst[i] = (s[i] - 0.1307f) / 0.3081f;
                    }
                }
            };
            return ds;
        }
    }
}
}