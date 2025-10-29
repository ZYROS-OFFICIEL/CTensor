#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <stdexcept>
#include "tensor1.h"
#include "stb_image.h"
#include "stb_image_write.h"
// ---------- flat vector -> tensor ----------
template<typename T>
Tensor from_flat_vector(const std::vector<T>& data, const std::vector<size_t>& shape, DType dtype = DType::Float32, bool requires_grad = false) {
    // compute expected size
    size_t expected = 1;
    for (auto d : shape) expected *= d;
    if (expected != data.size()) throw std::invalid_argument("from_flat_vector: shape does not match data.size()");

    Tensor out(shape, dtype, requires_grad);
    for (size_t i = 0; i < expected; ++i) {
        // use write_scalar_at to handle dtype conversions
        write_scalar_at(out.impl->storage->data.get(), i, out.impl->dtype, static_cast<double>(data[i]));
    }
    return out;
}
// ---------- raw pointer -> tensor ----------
template<typename T>
Tensor from_raw_ptr(const T* ptr, size_t count, const std::vector<size_t>& shape, DType dtype = DType::Float32, bool requires_grad = false) {
    size_t expected = 1;
    for (auto d : shape) expected *= d;
    if (expected != count) throw std::invalid_argument("from_raw_ptr: shape does not match count");

    Tensor out(shape, dtype, requires_grad);
    for (size_t i = 0; i < expected; ++i) {
        write_scalar_at(out.impl->storage->data.get(), i, out.impl->dtype, static_cast<double>(ptr[i]));
    }
    return out;
}
// ---------- 2D nested vector -> tensor ----------
template<typename T>
Tensor from_2d_vector(const std::vector<std::vector<T>>& v2, DType dtype = DType::Float32, bool requires_grad = false) {
    size_t rows = v2.size();
    size_t cols = (rows == 0) ? 0 : v2[0].size();
    for (size_t r = 0; r < rows; ++r) {
        if (v2[r].size() != cols) throw std::invalid_argument("from_2d_vector: ragged rows not supported");
    }
    Tensor out({rows, cols}, dtype, requires_grad);
    size_t idx = 0;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            write_scalar_at(out.impl->storage->data.get(), idx++, out.impl->dtype, static_cast<double>(v2[r][c]));
        }
    }
    return out;
}
// ---------- 3D nested vector -> tensor ----------
template<typename T>
Tensor from_3d_vector(const std::vector<std::vector<std::vector<T>>>& v3, DType dtype = DType::Float32, bool requires_grad = false) {
    size_t d0 = v3.size();
    size_t d1 = (d0 == 0) ? 0 : v3[0].size();
    size_t d2 = (d1 == 0) ? 0 : v3[0][0].size();
    for (size_t i = 0; i < d0; ++i) {
        if (v3[i].size() != d1) throw std::invalid_argument("from_3d_vector: ragged dims not supported (dim1 mismatch)");
        for (size_t j = 0; j < d1; ++j) {
            if (v3[i][j].size() != d2) throw std::invalid_argument("from_3d_vector: ragged dims not supported (dim2 mismatch)");
        }
    }
    Tensor out({d0, d1, d2}, dtype, requires_grad);
    size_t idx = 0;
    for (size_t i = 0; i < d0; ++i)
        for (size_t j = 0; j < d1; ++j)
            for (size_t k = 0; k < d2; ++k)
                write_scalar_at(out.impl->storage->data.get(), idx++, out.impl->dtype, static_cast<double>(v3[i][j][k]));
    return out;
}
// ---------- CSV (numeric) -> 2D tensor ----------
inline Tensor from_csv(const std::string& filename, DType dtype = DType::Float32, bool has_header = false, char sep = ',') {
    std::ifstream ifs(filename);
    if (!ifs) throw std::runtime_error("from_csv: cannot open file");

    std::string line;
    std::vector<std::vector<double>> rows;
    size_t cols = 0;
    bool first_line = true;

    while (std::getline(ifs, line)) {
        if (first_line && has_header) { first_line = false; continue; }
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string item;
        std::vector<double> row;
        while (std::getline(ss, item, sep)) {
            // trim spaces (quick)
            size_t a = item.find_first_not_of(" \t\r\n");
            size_t b = item.find_last_not_of(" \t\r\n");
            if (a == std::string::npos) { row.push_back(0.0); continue; }
            std::string trimmed = item.substr(a, b - a + 1);
            row.push_back(std::stod(trimmed));
        }
        if (first_line) { cols = row.size(); first_line = false; }
        if (row.size() != cols) throw std::runtime_error("from_csv: inconsistent column count");
        rows.push_back(std::move(row));
    }

    // build tensor
    size_t R = rows.size();
    size_t C = cols;
    Tensor out({R, C}, dtype, false);
    size_t idx = 0;
    for (size_t r = 0; r < R; ++r)
        for (size_t c = 0; c < C; ++c)
            write_scalar_at(out.impl->storage->data.get(), idx++, out.impl->dtype, rows[r][c]);

    return out;
}
// ---------- binary file (.bin) -> tensor ----------
inline Tensor from_binary(const std::string& filename,
                          const std::vector<size_t>& shape,
                          DType dtype,
                          bool requires_grad = false) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) throw std::runtime_error("from_binary: cannot open file");

    size_t numel = 1;
    for (auto d : shape) numel *= d;

    size_t type_size = 0;
    switch (dtype) {
        case DType::Int32: type_size = sizeof(int32_t); break;
        case DType::Float32: type_size = sizeof(float); break;
        case DType::Double64: type_size = sizeof(double); break;
        default: throw std::runtime_error("from_binary: unsupported dtype");
    }

    std::vector<char> buffer(numel * type_size);
    ifs.read(buffer.data(), buffer.size());
    if (ifs.gcount() != static_cast<std::streamsize>(buffer.size()))
        throw std::runtime_error("from_binary: unexpected EOF");

    Tensor out(shape, dtype, requires_grad);
    for (size_t i = 0; i < numel; ++i) {
        double val = 0.0;
        switch (dtype) {
            case DType::Int32:   val = static_cast<double>(reinterpret_cast<int32_t*>(buffer.data())[i]); break;
            case DType::Float32: val = static_cast<double>(reinterpret_cast<float*>(buffer.data())[i]); break;
            case DType::Double64:val = reinterpret_cast<double*>(buffer.data())[i]; break;
        }
        write_scalar_at(out.impl->storage->data.get(), i, dtype, val);
    }

    return out;
}
// ---------- NumPy .npy file -> tensor ----------
inline Tensor from_npy(const std::string& filename, bool requires_grad = false) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) throw std::runtime_error("from_npy: cannot open file");

    // magic string
    char magic[6];
    ifs.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY")
        throw std::runtime_error("from_npy: invalid header");

    // version
    uint8_t major, minor;
    ifs.read(reinterpret_cast<char*>(&major), 1);
    ifs.read(reinterpret_cast<char*>(&minor), 1);

    uint16_t header_len16;
    uint32_t header_len32;
    size_t header_len = 0;
    if (major <= 1) {
        ifs.read(reinterpret_cast<char*>(&header_len16), 2);
        header_len = header_len16;
    } else {
        ifs.read(reinterpret_cast<char*>(&header_len32), 4);
        header_len = header_len32;
    }

    std::string header(header_len, ' ');
    ifs.read(header.data(), header_len);

    // parse dtype
    std::regex descr_re("'descr': *'([<>=|])([fiu])(\\d+)'");
    std::smatch m;
    if (!std::regex_search(header, m, descr_re))
        throw std::runtime_error("from_npy: cannot parse dtype");
    char endian = m[1].str()[0];
    char typechar = m[2].str()[0];
    int bits = std::stoi(m[3].str());

    if (endian != '<' && endian != '|')
        throw std::runtime_error("from_npy: only little-endian supported");

    DType dtype;
    if (typechar == 'f' && bits == 4) dtype = DType::Float32;
    else if (typechar == 'f' && bits == 8) dtype = DType::Double64;
    else if (typechar == 'i' && bits == 4) dtype = DType::Int32;
    else throw std::runtime_error("from_npy: unsupported dtype");

    // parse shape
    std::regex shape_re("'shape': *\\(([^\\)]*)\\)");
    std::smatch s;
    if (!std::regex_search(header, s, shape_re))
        throw std::runtime_error("from_npy: cannot parse shape");

    std::string shape_str = s[1].str();
    std::stringstream ss(shape_str);
    std::vector<size_t> shape;
    while (ss.good()) {
        std::string dim;
        std::getline(ss, dim, ',');
        if (!dim.empty()) {
            size_t val = std::stoul(dim);
            shape.push_back(val);
        }
    }

    size_t numel = 1;
    for (auto d : shape) numel *= d;

    // read data
    size_t type_size = 0;
    switch (dtype) {
        case DType::Float32: type_size = 4; break;
        case DType::Double64: type_size = 8; break;
        case DType::Int32: type_size = 4; break;
        default: throw std::runtime_error("from_npy: unsupported dtype size");
    }

    std::vector<char> buffer(numel * type_size);
    ifs.read(buffer.data(), buffer.size());
    if (ifs.gcount() != static_cast<std::streamsize>(buffer.size()))
        throw std::runtime_error("from_npy: truncated data");

    Tensor out(shape, dtype, requires_grad);
    for (size_t i = 0; i < numel; ++i) {
        double val = 0.0;
        switch (dtype) {
            case DType::Float32: val = static_cast<double>(reinterpret_cast<float*>(buffer.data())[i]); break;
            case DType::Double64: val = reinterpret_cast<double*>(buffer.data())[i]; break;
            case DType::Int32: val = static_cast<double>(reinterpret_cast<int32_t*>(buffer.data())[i]); break;
        }
        write_scalar_at(out.impl->storage->data.get(), i, dtype, val);
    }

    return out;
}
//---------------tensor -> image  ---------------
void tensorio::to_image(const Tensor& t, const std::string& path) {
    assert(t.ndim == 3 && "Expected [C,H,W] tensor");

    size_t C = t.shape[0];
    size_t H = t.shape[1];
    size_t W = t.shape[2];

    std::vector<unsigned char> buffer(W * H * C);

    for (size_t i = 0; i < buffer.size(); ++i) {
        double val = read_scalar_at(t.impl->storage->data.get(), i, t.impl->dtype);
        buffer[i] = static_cast<unsigned char>(std::clamp(val * 255.0, 0.0, 255.0));
    }

    stbi_write_png(path.c_str(), (int)W, (int)H, (int)C, buffer.data(), (int)(W * C));
}
//---------------from image file -------------------
Tensor tensorio::from_image(const std::string& path, DType dtype) {
    int w, h, c;
    unsigned char* img_data = stbi_load(path.c_str(), &w, &h, &c, 0);
    if (!img_data) {
        throw std::runtime_error("Failed to load image: " + path);
    }

    size_t numel = (size_t)w * h * c;
    Tensor t({(size_t)c, (size_t)h, (size_t)w}, dtype);

    // Copy and convert
    for (size_t i = 0; i < numel; ++i) {
        write_scalar_at(t.impl->storage->data.get(), i, dtype , static_cast<double>(img_data[i]) / 255.0);
    }

    stbi_image_free(img_data);
    return t;
}
