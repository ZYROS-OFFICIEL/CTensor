#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring> // for memcpy
#include <ctime>   // for time
#include <cstdint>
#include "tensor1.h"
#include "opsmp.h"
#include "autograd.h"
#include "conv.h"
#include "pooling.h"
#include "layer.h"
#include "Relu.h"
#include "dropout.h"
#include "loss.h"        // <--- Now uses your updated library loss
#include "train_utils.h"
#include "mnist.h"

// --- Model Definition ---
class ConvNet : public Module {
public:
    Conv2d conv1;
    Relu relu1;
    MaxPool2d pool1;
    
    Conv2d conv2;
    Relu relu2;
    MaxPool2d pool2;
    
    Flatten flat;
    
    Linear fc1;
    Relu relu3;
    Linear fc2;

    ConvNet() 
        // LeNet-5 style architecture
        : conv1(1, 6, 5, 5, 1, 1, 2, 2),
          relu1(),
          pool1(2, 2, 2, 2),
          
          conv2(6, 16, 5, 5),
          relu2(),
          pool2(2, 2, 2, 2),
          
          flat(),
          
          fc1(16 * 5 * 5, 120),
          relu3(),
          fc2(120, 10)
    {}

    Tensor forward(const Tensor& x) {
        Tensor out = x;
        out = conv1(out);
        out = Relu_mp(out);
        out = pool1(out);
        
        out = conv2(out);
        out = Relu_mp(out);
        out = pool2(out);
        
        out = flat(out);
        
        // Safety Reshape for Linear Layer
        if (out.impl->ndim != 2) {
             size_t batch_size = out.impl->shape[0];
             size_t features = out.numel() / batch_size;
             out = out.reshape({batch_size, features});
        }
        
        out = fc1(out);
        out = Relu_mp(out);
        out = fc2(out);
        
        return out; 
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> p;
        auto p1 = conv1.parameters(); p.insert(p.end(), p1.begin(), p1.end());
        auto p2 = conv2.parameters(); p.insert(p.end(), p2.begin(), p2.end());
        auto p3 = fc1.parameters();   p.insert(p.end(), p3.begin(), p3.end());
        auto p4 = fc2.parameters();   p.insert(p.end(), p4.begin(), p4.end());
        return p;
    }
};

int main() {
    try {
        std::cout << "Loading MNIST data..." << std::endl;
        MNISTData train_data = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        
        ConvNet model;

        // --- IMPORTANT: Weight Re-Initialization & Gradient Enable ---
        std::cout << "Initializing weights..." << std::endl;
        std::srand(std::time(nullptr));
        
        for (auto* p : model.parameters()) {
            if (!p->impl) continue;

            // 1. CRITICAL: Enable gradients manually 
            // (in case the constructor defaulted to false)
            p->requires_grad_(true); 

            // 2. Initialize with small random values [-0.1, 0.1]
            size_t n = p->numel();
            float* ptr = (float*)p->impl->storage->data.get();
            
            for (size_t i = 0; i < n; ++i) {
                float r = static_cast<float>(std::rand()) / RAND_MAX; 
                ptr[i] = (r - 0.5f) * 0.2f; 
            }
        }
        // -------------------------------------------------------------

        Optimizer optim(model.parameters(), 0.01); // Learning Rate 0.01
        
        int BATCH_SIZE = 64;
        int EPOCHS = 5;
        size_t num_train = train_data.images.shape()[0];
        size_t num_batches = num_train / BATCH_SIZE;

        std::cout << "Starting training on " << num_train << " images." << std::endl;

        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            model.train();
            double epoch_loss = 0.0;
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int b = 0; b < num_batches; ++b) {
                size_t start_idx = b * BATCH_SIZE;
                
                // --- BATCH PREPARATION ---
                
                // 1. Images: Use Float32
                std::vector<size_t> batch_shape_img = { (size_t)BATCH_SIZE, 1, 28, 28 };
                Tensor batch_imgs(batch_shape_img, DType::Float32, false); 

                float* src_ptr = (float*)train_data.images.impl->storage->data.get() + start_idx * 28*28;
                float* dst_ptr = (float*)batch_imgs.impl->storage->data.get();
                std::memcpy(dst_ptr, src_ptr, BATCH_SIZE * 28 * 28 * sizeof(float));
                
                // 2. Labels: Use Int32 (Indices)
                std::vector<size_t> batch_shape_lbl = { (size_t)BATCH_SIZE, 1 }; 
                Tensor batch_lbls(batch_shape_lbl, DType::Int32, false);
                
                int32_t* src_lbl = (int32_t*)train_data.labels.impl->storage->data.get() + start_idx;
                int32_t* dst_lbl = (int32_t*)batch_lbls.impl->storage->data.get();
                std::memcpy(dst_lbl, src_lbl, BATCH_SIZE * sizeof(int32_t));

                // --- TRAINING STEP ---
                optim.zero_grad();
                
                Tensor output = model.forward(batch_imgs);
                
                // USE LIBRARY LOSS (Now works with indices!)
                Tensor loss = Loss::CrossEntropy(output, batch_lbls);
                
                backward(loss);
                optim.step();
                
                // Logging
                double current_loss = loss.read_scalar(0);
                epoch_loss += current_loss;
                
                if (b % 100 == 0) {
                    std::cout << "Batch " << b << "/" << num_batches 
                              << " Loss: " << current_loss << std::endl;
                }
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            std::cout << "Epoch " << epoch << " Done. Avg Loss: " << epoch_loss / num_batches
                      << " Time: " << std::chrono::duration<double>(end_time - start_time).count() << "s" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}