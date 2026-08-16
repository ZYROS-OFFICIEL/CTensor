#pragma once
#include <string>
#include <stdexcept>

#ifdef __CUDACC__
    #define HOST_DEVICE __host__ __device__
#else
    #define HOST_DEVICE
#endif

enum class DeviceType { CPU, CUDA };

struct Device {
    DeviceType type;
    int index; 

    HOST_DEVICE Device(DeviceType t = DeviceType::CPU, int i = -1) : type(t), index(i) {}
    
    HOST_DEVICE bool is_cpu() const { return type == DeviceType::CPU; }
    HOST_DEVICE bool is_cuda() const { return type == DeviceType::CUDA; }
    
    std::string str() const {
        return (type == DeviceType::CPU ? "cpu" : "cuda") + 
               (index >= 0 ? ":" + std::to_string(index) : "");
    }
    
    HOST_DEVICE bool operator==(const Device& other) const {
        return type == other.type && index == other.index;
    }
    HOST_DEVICE bool operator!=(const Device& other) const { return !(*this == other); }
};