#pragma once
#include "core/tensor.h"
#include "neuralnet/module.h"
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace safetensors {

struct LoadReport {
    std::vector<std::string> missing;
    std::vector<std::string> unexpected;
    std::vector<std::string> shape_mismatch;

    bool ok() const { return missing.empty() && unexpected.empty() && shape_mismatch.empty(); }
};

std::unordered_map<std::string, Tensor> load(
    const std::string& path,
    std::unordered_map<std::string, std::string>* metadata_out = nullptr);

void save(
    const std::string& path,
    const std::vector<std::pair<std::string, Tensor>>& tensors,
    const std::unordered_map<std::string, std::string>& metadata = {});

void save(
    const std::string& path,
    const Module::NamedParams& params,
    const std::unordered_map<std::string, std::string>& metadata = {});

LoadReport load_into(
    const std::string& path,
    const Module::NamedParams& params,
    bool strict = true);

}
