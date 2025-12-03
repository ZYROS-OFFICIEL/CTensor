#include "check.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>

namespace checkpoints {

    constexpr uint32_t MAGIC_NUMBER = 0xCAFEBABE;

    void save_weights(const std::vector<Tensor*>& params, const std::string& filename) {
        std::ofstream outfile(filename, std::ios::binary);
        if (!outfile) throw std::runtime_error("Could not open file for saving: " + filename);

        // 1. Write Magic Number
        uint32_t magic = MAGIC_NUMBER;
        outfile.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

        // 2. Write Number of Tensors
        uint32_t num_tensors = (uint32_t)params.size();
        outfile.write(reinterpret_cast<const char*>(&num_tensors), sizeof(num_tensors));

        // 3. Write Tensors
        for (const auto* t : params) {
            if (!t->impl) throw std::runtime_error("Attempting to save null tensor");

            // Meta: NDIM
            uint32_t ndim = (uint32_t)t->impl->ndim;
            outfile.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));

            // Meta: Shape
            // We use uint64_t for dimensions to be safe
            for (size_t i = 0; i < ndim; ++i) {
                uint64_t dim = (uint64_t)t->impl->shape[i];
                outfile.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
            }

            // Data
            // Ensure contiguous memory before writing!
            // In your library, 'data' is a raw pointer. If strides are not standard, 
            // saving raw bytes is dangerous.
            // Assumption: Parameters (weights) are always contiguous.
            
            size_t numel = t->numel();
            size_t type_size = t->dtype_bytes();
            uint64_t data_bytes = numel * type_size;

            outfile.write(reinterpret_cast<const char*>(&data_bytes), sizeof(data_bytes));
            outfile.write(reinterpret_cast<const char*>(t->impl->storage->data.get()), data_bytes);
        }

        outfile.close();
        std::cout << "Saved " << num_tensors << " tensors to " << filename << "\n";
    }

    void load_weights(std::vector<Tensor*>& params, const std::string& filename) {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) throw std::runtime_error("Could not open file for loading: " + filename);

        // 1. Check Magic
        uint32_t magic;
        infile.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != MAGIC_NUMBER) throw std::runtime_error("Invalid checkpoint format or endianness mismatch.");

        // 2. Check Count
        uint32_t num_tensors;
        infile.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));

        if (num_tensors != params.size()) {
            std::cerr << "Warning: Checkpoint has " << num_tensors << " params but model has " << params.size() << "\n";
            // We proceed, but mismatch usually implies wrong model arch
        }

        // 3. Read Tensors
        size_t limit = std::min((size_t)num_tensors, params.size());
        
        for (size_t i = 0; i < limit; ++i) {
            Tensor* t = params[i];

            // Read NDIM
            uint32_t ndim;
            infile.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

            // Read Shape & Verify
            std::vector<size_t> loaded_shape;
            for (uint32_t d = 0; d < ndim; ++d) {
                uint64_t dim;
                infile.read(reinterpret_cast<char*>(&dim), sizeof(dim));
                loaded_shape.push_back((size_t)dim);
            }

            // Verify Shape
            if (loaded_shape != t->shape()) {
                std::cerr << "Shape mismatch at param " << i << ". Loaded: (";
                for(auto s : loaded_shape) std::cerr << s << ",";
                std::cerr << ") Expected: (";
                t->print_shape();
                std::cerr << ")\n";
                throw std::runtime_error("Checkpoint shape mismatch");
            }

            // Read Data Size
            uint64_t data_bytes;
            infile.read(reinterpret_cast<char*>(&data_bytes), sizeof(data_bytes));

            // Verify Size
            size_t expected_bytes = t->numel() * t->dtype_bytes();
            if (data_bytes != expected_bytes) {
                throw std::runtime_error("Data size mismatch. DType change?");
            }

            // Read Raw Data directly into tensor storage
            infile.read(reinterpret_cast<char*>(t->impl->storage->data.get()), data_bytes);
        }

        infile.close();
        std::cout << "Loaded weights successfully from " << filename << "\n";
    }
}