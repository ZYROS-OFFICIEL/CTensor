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
#include "loss.h"
#include "train_utils.h"
#include "mnist.h"
#include "check.h"

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
        // Input: 1x28x28
        : conv1(1, 6, 5, 5, 1, 1, 2, 2), // Out: 6x28x28 (Padding preserves size here)
          relu1(),
          pool1(2, 2, 2, 2),             // Out: 6x14x14
          
          conv2(6, 16, 5, 5),            // Out: 16x10x10 (Valid padding: 14-5+1 = 10)
          relu2(),
          pool2(2, 2, 2, 2),             // Out: 16x5x5
          
          flat(),                        // Out: 400 (16*5*5)
          
          fc1(16 * 5 * 5, 120),
          relu3(),
          fc2(120, 10)
    {}

    Tensor forward(const Tensor& x) {
        Tensor out = x;
        out = conv1(out);
        out = relu1(out);
        out = pool1(out);
        
        out = conv2(out);
        out = relu2(out);
        out = pool2(out);
        
        out = flat(out);
        
        // Safety Reshape for Linear Layer (Just in case Flatten behaves oddly)
        if (out.impl->ndim != 2) {
             size_t batch_size = out.impl->shape[0];
             size_t features = out.numel() / batch_size;
             out = out.reshape({batch_size, features});
        }
        
        out = fc1(out);
        out = relu3(out);
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

int main(int argc, char** argv) {
    try {
        std::cout << "Loading MNIST data..." << std::endl;
        MNISTData train_data = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        DataLoader loader(train_data, 64, true, DType::Float32, DType::Int32);
        ConvNet model;
        // Check if we should load weights
        std::string checkpoint_path = "mnist_cnn.bin";
        
        // Simple arg parsing to resume training
        if (argc > 1 && std::string(argv[1]) == "--resume") {
            try {
                std::cout << "Loading checkpoint from " << checkpoint_path << "...\n";
                std::vector<Tensor*> params = model.parameters();
                checkpoints::load_weights(params, checkpoint_path);
            } catch (const std::exception& e) {
                std::cout << "Could not load checkpoint: " << e.what() << ". Starting from scratch.\n";
                // If fail, re-init rand
                std::srand(std::time(nullptr));
                for (auto* p : model.parameters()) {
                    if (!p->impl) continue;
                    p->requires_grad_(true); 
                    size_t n = p->numel();
                    float* ptr = (float*)p->impl->storage->data.get();
                    for (size_t i = 0; i < n; ++i) ptr[i] = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.1f;
                }
            }
        } else {
            // Init from scratch
            std::cout << "Initializing weights..." << std::endl;
            std::srand(std::time(nullptr));
            for (auto* p : model.parameters()) {
                if (!p->impl) continue;
                p->requires_grad_(true); 
                size_t n = p->numel();
                float* ptr = (float*)p->impl->storage->data.get();
                for (size_t i = 0; i < n; ++i) ptr[i] = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.1f;
            }
        }
        Optimizer optim(model.parameters(), 0.01);
        
        int BATCH_SIZE = 64;
        int EPOCHS = 10;
        size_t num_train = train_data.images.shape()[0];
        size_t num_batches = num_train / BATCH_SIZE;

        std::cout << "Starting training on " << num_train << " images." << std::endl;

        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            model.train();
            double epoch_loss = 0.0;
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int b = 0; b < num_batches; ++b) {
                size_t start_idx = b * BATCH_SIZE;
                

                // --- TRAINING STEP ---
                optim.zero_grad();
                
                Tensor output = model.forward(batch_imgs);
                Tensor loss = Loss::CrossEntropy(output, batch_lbls);
                
                backward(loss);
                optim.step();
                
                // --- 3. Added NaN Protection (from train_mnist.cpp) ---
                if (std::isnan(loss.read_scalar(0))) {
                    std::cout << "NaN detected at batch " << b << std::endl;
                    break;
                }

                epoch_loss += loss.read_scalar(0);
                
                // --- 4. More Frequent Logging (Every 20 batches) ---
                if (b % 20 == 0) {
                    std::cout << "Batch " << b << " Loss: " << loss.read_scalar(0) << std::endl;
                }
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            std::cout << "Epoch " << epoch+1 << " Done. Avg Loss: " << epoch_loss / num_batches
                      << " Time: " << std::chrono::duration<double>(end_time - start_time).count() << "s" << std::endl;
                      
            // --- SAVE CHECKPOINT ---
            checkpoints::save_weights(model.parameters(), checkpoint_path);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}