#include <iostream>
#include <fstream> 
#include "core.h"
#include "neuralnet.h"
#include "check.h" 
#include "train_utils.h" // Includes generic loader, trainer, optimizer utils

// Define Model
class MLPNet : public Module {
public:
    Flatten flat;
    Linear fc1, fc2, fc3;
    Relu relu1, relu2;

    MLPNet() 
        : flat(),
          fc1(784, 256, true, DType::Float32),
          relu1(),
          fc2(256, 128, true, DType::Float32),
          relu2(),
          fc3(128, 10, true, DType::Float32)
    {}

    Tensor forward(const Tensor& x) {
        Tensor out = flat(x); 
        out = fc1(out);
        out = relu1(out);
        out = fc2(out);
        out = relu2(out);
        out = fc3(out);
        return out; 
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> p;
        auto p1 = fc1.parameters(); p.insert(p.end(), p1.begin(), p1.end());
        auto p2 = fc2.parameters(); p.insert(p.end(), p2.begin(), p2.end());
        auto p3 = fc3.parameters(); p.insert(p.end(), p3.begin(), p3.end());
        return p;
    }
};

int main() {
    try {
        // 1. Load Data
        std::cout << "Loading MNIST data..." << std::endl;
        MNISTData raw_data = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        
        // 2. Wrap in Dataset with Normalization
        TensorDataset dataset(raw_data.images, raw_data.labels);
        
        // Standard MNIST Normalization: (x - 0.1307) / 0.3081
        dataset.transform = [](const void* src, float* dst, size_t n) {
            const uint8_t* s = (const uint8_t*)src; // Assuming UInt8 input
            for(size_t i=0; i<n; ++i) {
                float val = (float)s[i] / 255.0f; 
                dst[i] = (val - 0.1307f) / 0.3081f;
            }
        };

        SimpleDataLoader loader(dataset, 64, true);

        // 3. Setup Model & Weights
        MLPNet model;
        std::vector<Tensor*> params = model.parameters();
        
        std::string ckpt = "mnist_weights.bin";
        std::ifstream infile(ckpt);
        
        if (infile.good()) {
            infile.close();
            checkpoints::load_weights(params, ckpt);
            if (check_weights_corrupted(params)) {
                std::cerr << "Checkpoint corrupted. Re-initializing.\n";
                robust_weight_init(params, 0.01f);
            } else {
                std::cout << "Checkpoint loaded.\n";
            }
        } else {
            robust_weight_init(params, 0.01f);
        }

        // 4. Setup Optimizer
        for(auto* p : params) p->requires_grad_(true);
        AdamW optim(params, 0.0001); // Safe LR

        // 5. Train Loop
        for (int epoch = 0; epoch < 10; ++epoch) {
            train_one_epoch(epoch, model, loader, optim);
            checkpoints::save_weights(params, ckpt);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}