
## Hardware & Devices

The `Device` structure specifies the hardware target where a tensor's memory is allocated and where its computations are executed.

---

## Device Initialization

### Definition
Creates a device object representing either the **CPU** or a specific **GPU (CUDA)** index.

### Usage
```cpp
Device cpu_dev(DeviceType::CPU);
Device gpu_dev(DeviceType::CUDA, 0); // GPU 0
```

**Returns:** `Device`

---

## Device Properties (`is_cpu`, `is_cuda`, `str`)

### Definition
Helper methods used to check the hardware type or obtain a readable string representation of the device.

### Usage
```cpp
bool is_gpu = device.is_cuda();
std::string name = device.str(); // "cuda:0" or "cpu"
```

| Method | Return Type |
|------|------|
| `is_cpu()` | `bool` |
| `is_cuda()` | `bool` |
| `str()` | `std::string` |

---

