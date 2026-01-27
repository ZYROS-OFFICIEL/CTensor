#pragma once
#include <string>
#include <vector>
#include "tensor.h"

// Simple Checkpoint System
// Format:
// [MAGIC_NUMBER (4 bytes)]
// [NUM_TENSORS (4 bytes)]
// For each tensor:
//   [NDIM (4 bytes)]
//   [SHAPE (NDIM * 8 bytes)]
//   [DATA_SIZE_BYTES (8 bytes)]
//   [RAW_DATA (DATA_SIZE_BYTES)]

namespace checkpoints {

    // Save all parameters from the list to a file
    void save_weights(const std::vector<Tensor*>& params, const std::string& filename);

    // Load weights from file into the parameter list (must match order and shape)
    void load_weights(std::vector<Tensor*>& params, const std::string& filename);

}