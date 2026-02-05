#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <ctime>
#include <cmath>
#include "core.h"
#include "neuralnet.h"

// --- CONVNET MODEL ---
// Architecture:
// 1. Conv2d: 1 -> 6 channels, 5x5 kernel
// 2. Relu
// 3. Conv2d: 6 -> 16 channels, 5x5 kernel, Stride 2 (Acts as pooling)
// 4. Flatten
// 5. Linear: 16 * 10 * 10 -> 120
// 6. Relu
// 7. Linear: 120 -> 84
// 8. Relu
// 9. Linear: 84 -> 10
class ConvNet : public Module {
public:
    Conv2d c1;
    Relu relu1;
    Conv2d c2;
    Relu relu2;
    Flatten flat;
    Linear fc1;
    Relu relu3;
    Linear fc2;
    Relu relu4;
    Linear fc3;

    ConvNet() 
        : // Input: 1x28x28
          // Output: 6x24x24 (28 - 5 + 1)
          c1(1, 6, 5, 5, 1, 1, 0, 0, DType::Float32),
          relu1(),
          
          // Input: 6x24x24
          // Output: 16x10x10 ((24 - 5) / 2 + 1) -> Stride 2 for downsampling
          c2(6, 16, 5, 5, 2, 2, 0, 0, DType::Float32),
          relu2(),
          
          flat(),
          
          // Flatten size: 16 channels * 10 height * 10 width = 1600
          fc1(1600, 120, true, DType::Float32),
          relu3(),
          
          fc2(120, 84, true, DType::Float32),
          relu4(),
          
          fc3(84, 10, true, DType::Float32)
    {}

    Tensor forward(const Tensor& x) {
        Tensor out = c1(x);
        out = relu1(out);
        
        out = c2(out);
        out = relu2(out);
        
        out = flat(out);
        
        out = fc1(out);
        out = relu3(out);
        
        out = fc2(out);
        out = relu4(out);
        
        out = fc3(out);
        return out; 
    }

    // Manually register parameters since we don't have auto-registration yet
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> p;
        auto p_c1 = c1.parameters(); p.insert(p.end(), p_c1.begin(), p_c1.end());
        auto p_c2 = c2.parameters(); p.insert(p.end(), p_c2.begin(), p_c2.end());
        auto p_fc1 = fc1.parameters(); p.insert(p.end(), p_fc1.begin(), p_fc1.end());
        auto p_fc2 = fc2.parameters(); p.insert(p.end(), p_fc2.begin(), p_fc2.end());
        auto p_fc3 = fc3.parameters(); p.insert(p.end(), p_fc3.begin(), p_fc3.end());
        return p;
    }
};