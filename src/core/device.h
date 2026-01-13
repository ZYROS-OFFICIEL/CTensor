#pragma once
#include <string>
#include <stdexcept>

enum class DeviceType { CPU, CUDA };

struct Device {
    DeviceType type;
    int index; // e.g., GPU 0, GPU 1

    Device(DeviceType t = DeviceType::CPU, int i = -1) : type(t), index(i) {}
    
    bool is_cpu() const { return type == DeviceType::CPU; }
    bool is_cuda() const { return type == DeviceType::CUDA; }
    
    std::string str() const {
        return (type == DeviceType::CPU ? "cpu" : "cuda") + 
               (index >= 0 ? ":" + std::to_string(index) : "");
    }
    
    bool operator==(const Device& other) const {
        return type == other.type && index == other.index;
    }
    bool operator!=(const Device& other) const { return !(*this == other); }
};