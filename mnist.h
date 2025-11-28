#pragma once
#include "tensor1.h"
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <cstdint>
#include <stdexcept>

// --- MNIST Loader Helper ---
// MNIST files use big-endian integers. Convert to little-endian on x86.
inline uint32_t read_uint32(std::ifstream& is) {
    uint32_t val = 0;
    is.read(reinterpret_cast<char*>(&val), 4);
    // file is big-endian, convert to host (little-endian on x86)
    return ((val << 24) & 0xFF000000) |
           ((val << 8)  & 0x00FF0000) |
           ((val >> 8)  & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

struct MNISTData {
    Tensor images; // [N, 1, 28, 28] normalized to 0-1 (Float32)
    Tensor labels; // [N, 1] Int32
};

// Optional: enable extra debug prints by defining MNIST_DEBUG (e.g. -DMNIST_DEBUG)

inline MNISTData load_mnist(const std::string& image_path, const std::string& label_path) {
    std::ifstream img_file(image_path, std::ios::binary);
    std::ifstream lbl_file(label_path, std::ios::binary);

    if (!img_file.is_open()) throw std::runtime_error("Cannot open MNIST images: " + image_path);
    if (!lbl_file.is_open()) throw std::runtime_error("Cannot open MNIST labels: " + label_path);

    // Read headers (big-endian -> convert)
    uint32_t magic_img = read_uint32(img_file);
    uint32_t num_imgs  = read_uint32(img_file);
    uint32_t rows      = read_uint32(img_file);
    uint32_t cols      = read_uint32(img_file);

    uint32_t magic_lbl = read_uint32(lbl_file);
    uint32_t num_lbls  = read_uint32(lbl_file);

    if (magic_img != 2051) throw std::runtime_error("Invalid MNIST image file magic number");
    if (magic_lbl != 2049) throw std::runtime_error("Invalid MNIST label file magic number");
    if (num_imgs != num_lbls) throw std::runtime_error("MNIST image/label count mismatch");

    std::cout << "Loading " << num_imgs << " images (" << rows << "x" << cols << ")...\n";

    // Allocate Tensors
    std::vector<size_t> img_shape = { (size_t)num_imgs, 1, (size_t)rows, (size_t)cols };
    Tensor images(img_shape, DType::Float32, false);

    // Labels shaped [N,1] as Int32 so gather will accept them directly
    std::vector<size_t> lbl_shape = { (size_t)num_imgs, 1 };
    Tensor labels(lbl_shape, DType::Int32, false);

    // Read Data buffers
    size_t num_pixels = (size_t)num_imgs * rows * cols;
    std::vector<unsigned char> img_buffer;
    std::vector<unsigned char> lbl_buffer;

    // Try bulk read first
    try {
        img_buffer.resize(num_pixels);
        lbl_buffer.resize(num_imgs);

        img_file.read(reinterpret_cast<char*>(img_buffer.data()), static_cast<std::streamsize>(num_pixels));
        lbl_file.read(reinterpret_cast<char*>(lbl_buffer.data()), static_cast<std::streamsize>(num_imgs));

        // Verify read succeeded
        std::streamsize img_gcount = img_file.gcount();
        std::streamsize lbl_gcount = lbl_file.gcount();

        if (img_gcount != static_cast<std::streamsize>(num_pixels)) {
            std::cerr << "Warning: img_file.gcount() = " << img_gcount
                      << " expected " << num_pixels << ". Falling back to safe per-image read.\n";
            img_buffer.clear();
        }
        if (lbl_gcount != static_cast<std::streamsize>(num_imgs)) {
            std::cerr << "Warning: lbl_file.gcount() = " << lbl_gcount
                      << " expected " << num_imgs << ".\n";
            // keep going; label fallback handled below if needed
        }
    } catch (...) {
        img_buffer.clear();
        lbl_buffer.clear();
    }

    // Pointers to underlying storage (may be non-float-layout in some builds)
    auto* img_storage_ptr = images.impl->storage->data.get();
    auto* lbl_storage_ptr = labels.impl->storage->data.get();

    // Use dtype-aware helpers to write into tensor storage to avoid layout assumptions
    if (!img_buffer.empty() && img_buffer.size() == num_pixels) {
        // Bulk conversion (safe cast to float)
        #pragma omp parallel for
        for (size_t i = 0; i < num_pixels; ++i) {
            float v = static_cast<float>(img_buffer[i]) / 255.0f; // ensure float division
            write_scalar_at(img_storage_ptr, i, DType::Float32, static_cast<double>(v));
        }

        // Labels (bulk if available)
        if (!lbl_buffer.empty() && lbl_buffer.size() == num_imgs) {
            #pragma omp parallel for
            for (size_t i = 0; i < (size_t)num_imgs; ++i) {
                int32_t lab = static_cast<int32_t>(lbl_buffer[i]);
                // labels shape [N,1] -> linear index i
                write_scalar_at(lbl_storage_ptr, i, DType::Int32, static_cast<double>(lab));
            }
        } else {
            // fallback to per-label read from file (seek to labels data)
            lbl_file.clear();
            lbl_file.seekg(8, std::ios::beg); // 2 uint32 header = 8 bytes
            for (size_t i = 0; i < (size_t)num_imgs; ++i) {
                unsigned char lb = 0;
                lbl_file.read(reinterpret_cast<char*>(&lb), 1);
                int32_t lab = static_cast<int32_t>(lb);
                write_scalar_at(lbl_storage_ptr, i, DType::Int32, static_cast<double>(lab));
            }
        }

    } else {
        // Bulk read failed or was truncated: do safe per-image read
        std::cerr << "Performing safe per-image read (slower)...\n";

        // Reset file to the position after header (16 bytes)
        img_file.clear();
        img_file.seekg(16, std::ios::beg);

        size_t offset = 0;
        std::vector<unsigned char> tmp((size_t)rows * (size_t)cols);
        for (size_t img = 0; img < (size_t)num_imgs; ++img) {
            img_file.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(rows * cols));
            if (!img_file) {
                throw std::runtime_error("Failed reading image " + std::to_string(img));
            }
            for (size_t p = 0; p < (size_t)rows * (size_t)cols; ++p) {
                float v = static_cast<float>(tmp[p]) / 255.0f;
                write_scalar_at(img_storage_ptr, offset + p, DType::Float32, static_cast<double>(v));
            }
            offset += (size_t)rows * (size_t)cols;
        }

        // Labels: safe per-label read
        lbl_file.clear();
        lbl_file.seekg(8, std::ios::beg);
        for (size_t i = 0; i < (size_t)num_imgs; ++i) {
            unsigned char lb = 0;
            lbl_file.read(reinterpret_cast<char*>(&lb), 1);
            if (!lbl_file) throw std::runtime_error("Failed reading label " + std::to_string(i));
            int32_t lab = static_cast<int32_t>(lb);
            write_scalar_at(lbl_storage_ptr, i, DType::Int32, static_cast<double>(lab));
        }
    }

#ifdef MNIST_DEBUG
    // Debug: print raw and converted values for the first items
    std::cout << "--- MNIST DEBUG DUMP ---\n";
    // print first 40 raw bytes if available
    if (!img_buffer.empty()) {
        std::cout << "Raw first 40 image bytes: ";
        for (size_t i = 0; i < 40 && i < img_buffer.size(); ++i) std::cout << (int)img_buffer[i] << " ";
        std::cout << "\n";
    }

    std::cout << "First 40 converted image floats: ";
    for (size_t i = 0; i < 40 && i < num_pixels; ++i) {
        double val = read_scalar_at(img_storage_ptr, i, DType::Float32);
        std::cout << val << " ";
    }
    std::cout << "\n";

    std::cout << "First 20 converted labels: ";
    for (size_t i = 0; i < 20 && i < (size_t)num_imgs; ++i) {
        double labv = read_scalar_at(lbl_storage_ptr, i, DType::Int32);
        std::cout << (int)labv << " ";
    }
    std::cout << "\n";
    std::cout << "--- END DEBUG DUMP ---\n";
#endif

    return { images, labels };
}
