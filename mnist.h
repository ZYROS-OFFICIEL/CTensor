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

struct MNISTData {
    Tensor images; // [N, 1, 28, 28] normalized to 0-1
    Tensor labels; // [N] (indices 0-9)
};

inline MNISTData load_mnist(const std::string& image_path, const std::string& label_path) {
    std::ifstream img_file(image_path, std::ios::binary);
    std::ifstream lbl_file(label_path, std::ios::binary);

    if (!img_file.is_open()) throw std::runtime_error("Cannot open MNIST images: " + image_path);
    if (!lbl_file.is_open()) throw std::runtime_error("Cannot open MNIST labels: " + label_path);

    // Read Headers
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
    // Images: [N, 1, 28, 28]
    Tensor images({(size_t)num_imgs, 1, (size_t)rows, (size_t)cols}, DType::Float32, false);
    // Labels: [N] (stored as Float32 for now to match your tensor type, usually integers)
    Tensor labels({(size_t)num_imgs}, DType::Float32, false);

    // Read Data buffers
    size_t num_pixels = num_imgs * rows * cols;
    std::vector<unsigned char> img_buffer(num_pixels);
    std::vector<unsigned char> lbl_buffer(num_imgs);

    img_file.read(reinterpret_cast<char*>(img_buffer.data()), num_pixels);
    lbl_file.read(reinterpret_cast<char*>(lbl_buffer.data()), num_imgs);

    // Fill Tensors (Convert uint8 to float 0.0-1.0)
    // Parallelize this conversion loop
    auto* img_data = images.impl->storage->data.get();
    auto* lbl_data = labels.impl->storage->data.get();

    #pragma omp parallel for
    for (size_t i = 0; i < num_pixels; ++i) {
        float val = static_cast<float>(img_buffer[i]) / 255.0f;
        write_scalar_at(img_data, i, DType::Float32, val);
    }

    #pragma omp parallel for
    for (size_t i = 0; i < num_imgs; ++i) {
        float val = static_cast<float>(lbl_buffer[i]);
        write_scalar_at(lbl_data, i, DType::Float32, val);
    }

    return {images, labels};
}