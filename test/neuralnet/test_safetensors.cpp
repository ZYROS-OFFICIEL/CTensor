#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include "core.h"
#include "neuralnet.h"
#include "models/gpt2.h"

static std::string tmp_path(const std::string& name) {
    return "/tmp/" + name;
}

void test_basic_roundtrip() {
    std::string path = tmp_path("st_basic.safetensors");

    Tensor a = Tensor::arange(0.0, 12.0, 1.0).reshape({3, 4});
    Tensor b = Tensor::full({2, 2}, 7.0, DType::Int32);

    std::vector<std::pair<std::string, Tensor>> tensors = {
        {"a", a}, {"b", b}
    };
    std::unordered_map<std::string, std::string> meta = {{"format", "ctensor-test"}};

    safetensors::save(path, tensors, meta);

    std::unordered_map<std::string, std::string> loaded_meta;
    auto loaded = safetensors::load(path, &loaded_meta);

    assert(loaded.size() == 2);
    assert(loaded_meta.at("format") == "ctensor-test");

    Tensor& la = loaded.at("a");
    assert(la.shape() == std::vector<size_t>({3, 4}));
    assert(la._dtype() == DType::Float32);
    for (size_t i = 0; i < 12; ++i) assert(std::abs(la.read_scalar(i) - (double)i) < 1e-6);

    Tensor& lb = loaded.at("b");
    assert(lb.shape() == std::vector<size_t>({2, 2}));
    assert(lb._dtype() == DType::Int32);
    for (size_t i = 0; i < 4; ++i) assert(lb.read_scalar(i) == 7.0);

    std::remove(path.c_str());
}

void test_sliced_tensor_roundtrip() {
    std::string path = tmp_path("st_sliced.safetensors");

    Tensor full = Tensor::arange(0.0, 24.0, 1.0).reshape({4, 6});
    Tensor row = full.select(0, 2);

    std::vector<std::pair<std::string, Tensor>> tensors = { {"row", row} };
    safetensors::save(path, tensors);

    auto loaded = safetensors::load(path);
    Tensor& lrow = loaded.at("row");
    assert(lrow.shape() == std::vector<size_t>({6}));
    for (size_t i = 0; i < 6; ++i) assert(std::abs(lrow.read_scalar(i) - (12.0 + (double)i)) < 1e-6);

    std::remove(path.c_str());
}

void test_gpt2_named_parameters_roundtrip() {
    std::string path = tmp_path("st_gpt2.safetensors");

    GPT2Config cfg = GPT2Config::toy(50);
    GPT2Model model(cfg);

    auto params = model.named_parameters();
    safetensors::save(path, params);

    GPT2Model model2(cfg);
    auto params2 = model2.named_parameters();

    safetensors::LoadReport report = safetensors::load_into(path, params2, true);
    assert(report.ok());

    for (size_t i = 0; i < params.size(); ++i) {
        Tensor* p1 = params[i].second;
        Tensor* p2 = nullptr;
        for (auto& [name, ptr] : params2) if (name == params[i].first) { p2 = ptr; break; }
        assert(p2 != nullptr);
        assert(p1->numel() == p2->numel());
        for (size_t k = 0; k < p1->numel(); ++k)
            assert(std::abs(p1->read_scalar(k) - p2->read_scalar(k)) < 1e-6);
    }

    std::remove(path.c_str());
}

void test_load_into_non_strict_report() {
    std::string path = tmp_path("st_partial.safetensors");

    Tensor w = Tensor::full({2, 2}, 1.0);
    Tensor extra = Tensor::full({3}, 2.0);
    safetensors::save(path, std::vector<std::pair<std::string, Tensor>>{
        {"weight", w}, {"extra_unused", extra}
    });

    Tensor model_weight = Tensor::zeros({2, 2});
    Tensor model_bias = Tensor::zeros({4});
    Module::NamedParams params = {
        {"weight", &model_weight},
        {"bias", &model_bias}
    };

    auto report = safetensors::load_into(path, params, false);
    assert(!report.ok());
    assert(report.missing.size() == 1 && report.missing[0] == "bias");
    assert(report.unexpected.size() == 1 && report.unexpected[0] == "extra_unused");
    assert(report.shape_mismatch.empty());

    for (size_t i = 0; i < 4; ++i) assert(model_weight.read_scalar(i) == 1.0);

    bool threw = false;
    try {
        safetensors::load_into(path, params, true);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    std::remove(path.c_str());
}

void test_hand_built_file() {
    std::string path = tmp_path("st_handbuilt.safetensors");

    std::string header = "{\"x\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}";
    size_t pad = (8 - (8 + header.size()) % 8) % 8;
    header.append(pad, ' ');

    std::ofstream f(path, std::ios::binary);
    uint64_t len = header.size();
    f.write(reinterpret_cast<const char*>(&len), 8);
    f.write(header.data(), (std::streamsize)header.size());
    float vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    f.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    f.close();

    auto loaded = safetensors::load(path);
    Tensor& x = loaded.at("x");
    assert(x.shape() == std::vector<size_t>({2, 2}));
    assert(x._dtype() == DType::Float32);
    for (int i = 0; i < 4; ++i) assert(std::abs(x.read_scalar(i) - (double)(i + 1)) < 1e-6);

    std::remove(path.c_str());
}

void test_bf16_native() {
    std::string path = tmp_path("st_bf16.safetensors");

    std::string header = "{\"y\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}";
    size_t pad = (8 - (8 + header.size()) % 8) % 8;
    header.append(pad, ' ');

    std::ofstream f(path, std::ios::binary);
    uint64_t len = header.size();
    f.write(reinterpret_cast<const char*>(&len), 8);
    f.write(header.data(), (std::streamsize)header.size());

    float a = 1.5f, b = -2.0f;
    uint16_t bf_a, bf_b;
    std::memcpy(&bf_a, reinterpret_cast<const char*>(&a) + 2, 2);
    std::memcpy(&bf_b, reinterpret_cast<const char*>(&b) + 2, 2);
    f.write(reinterpret_cast<const char*>(&bf_a), 2);
    f.write(reinterpret_cast<const char*>(&bf_b), 2);
    f.close();

    auto loaded = safetensors::load(path);
    Tensor& y = loaded.at("y");
    assert(y._dtype() == DType::BFloat16);
    assert(std::abs(y.read_scalar(0) - 1.5) < 1e-3);
    assert(std::abs(y.read_scalar(1) - (-2.0)) < 1e-3);

    std::remove(path.c_str());
}

void test_bf16_writer_roundtrip() {
    std::string path = tmp_path("st_bf16_write.safetensors");

    Tensor t = Tensor::empty({3}, DType::BFloat16);
    t.write_scalar(0, 3.25);
    t.write_scalar(1, -0.5);
    t.write_scalar(2, 100.0);

    safetensors::save(path, std::vector<std::pair<std::string, Tensor>>{ {"bf", t} });

    auto loaded = safetensors::load(path);
    Tensor& bf = loaded.at("bf");
    assert(bf._dtype() == DType::BFloat16);
    assert(bf.numel() == 3);
    for (size_t i = 0; i < 3; ++i)
        assert(std::abs(bf.read_scalar(i) - t.read_scalar(i)) < 1e-6);

    std::remove(path.c_str());
}

void test_float16_roundtrip() {
    std::string path = tmp_path("st_f16.safetensors");

    Tensor t = Tensor::empty({4}, DType::Float16);
    t.write_scalar(0, 1.5);
    t.write_scalar(1, -3.25);
    t.write_scalar(2, 0.0);
    t.write_scalar(3, 65504.0);

    safetensors::save(path, std::vector<std::pair<std::string, Tensor>>{ {"h", t} });

    auto loaded = safetensors::load(path);
    Tensor& h = loaded.at("h");
    assert(h._dtype() == DType::Float16);
    for (size_t i = 0; i < 4; ++i)
        assert(std::abs(h.read_scalar(i) - t.read_scalar(i)) < 1e-6);

    std::remove(path.c_str());
}

int main() {
    test_basic_roundtrip();
    test_sliced_tensor_roundtrip();
    test_gpt2_named_parameters_roundtrip();
    test_load_into_non_strict_report();
    test_hand_built_file();
    test_bf16_native();
    test_bf16_writer_roundtrip();
    test_float16_roundtrip();
    std::cout << "test_safetensors passed\n";
    return 0;
}
