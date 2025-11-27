#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

#include "tensor1.h"
#include "opsmp.h"
#include "autograd.h"
#include "conv.h"
#include "pooling.h"
#include "layer.h"
#include "Relu.h"
#include "dropout.h"
#include "loss.h"       // Ensure this exists and has CrossEntropy
#include "train_utils.h" // For Optimizer and set_model_mode
#include "mnist.h"      // The loader above

// Helper to print shapes
void debug_shape(const std::string& name, const Tensor& t) {
    std::cout << name << ": [";
    auto s = t.shape();
    for (size_t i = 0; i < s.size(); ++i) {
        std::cout << s[i] << (i < s.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
}

// --- Define the Model ---
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
        : conv1(1, 6, 5, 5, 1, 1, 2, 2),  // 1->6, 28x28
          relu1(),
          pool1(2, 2),                    // -> 14x14
          
          conv2(6, 16, 5, 5),             // 6->16, -> 10x10
          relu2(),
          pool2(2, 2),                    // -> 5x5
          
          flat(),
          
          fc1(16 * 5 * 5, 120),           // 400 -> 120
          relu3(),
          fc2(120, 10)                    // 120 -> 10
    {}

    Tensor forward(const Tensor& x) {
        // debug_shape("Input", x);
        Tensor out = x;
        
        out = conv1(out);
        // debug_shape("Conv1", out);
        out = Relu_mp(out);
        out = pool1(out);
        // debug_shape("Pool1", out);
        
        out = conv2(out);
        // debug_shape("Conv2", out);
        out = Relu_mp(out);
        out = pool2(out);
        // debug_shape("Pool2", out);
        
        // --- Flattening ---
        out = flat(out); 
        // debug_shape("Flatten", out);

        // CRITICAL: Safety Reshape to ensure [B, 400]
        if (out.impl->ndim != 2) {
             size_t batch_size = out.impl->shape[0];
             size_t features = out.numel() / batch_size;
             out = out.reshape({batch_size, features});
             // debug_shape("Reshape Fix", out);
        }
        
        out = fc1(out);
        // debug_shape("FC1", out);
        out = Relu_mp(out);
        out = fc2(out);
        // debug_shape("FC2", out);
        
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

// Main function (unchanged logic, just ensures correct flow)
int main() {
    try {
        std::cout << "Loading MNIST data..." << std::endl;
        MNISTData train_data = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        
        // Normalize images to [0, 1]
        // (Assuming load_mnist already does the division, if not, do it here)
        
        ConvNet model;
        Optimizer optim(model.parameters(), 0.01); 
        
        int BATCH_SIZE = 64;
        int EPOCHS = 1;
        size_t num_train = train_data.images.shape()[0];
        size_t num_batches = num_train / BATCH_SIZE;

        std::cout << "Starting training on " << num_train << " images." << std::endl;

        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            model.train();
            double epoch_loss = 0.0;
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int b = 0; b < num_batches; ++b) {
                size_t start_idx = b * BATCH_SIZE;
                
                // Batch Images [64, 1, 28, 28]
                std::vector<size_t> batch_shape_img = { (size_t)BATCH_SIZE, 1, 28, 28 };
                Tensor batch_imgs(batch_shape_img, DType::Float32, false);
                float* src_ptr = (float*)train_data.images.impl->storage->data.get() + start_idx * 28*28;
                float* dst_ptr = (float*)batch_imgs.impl->storage->data.get();
                std::memcpy(dst_ptr, src_ptr, BATCH_SIZE * 28 * 28 * sizeof(float));
                
                // Batch Labels [64]
                std::vector<size_t> batch_shape_lbl = { (size_t)BATCH_SIZE };
                Tensor batch_lbls(batch_shape_lbl, DType::Float32, false);
                float* src_lbl = (float*)train_data.labels.impl->storage->data.get() + start_idx;
                float* dst_lbl = (float*)batch_lbls.impl->storage->data.get();
                std::memcpy(dst_lbl, src_lbl, BATCH_SIZE * sizeof(float));

                optim.zero_grad();
                Tensor output = model.forward(batch_imgs);
                
                Tensor loss = Loss::CrossEntropy(output, batch_lbls, "mean"); 
                
                backward(loss);
                optim.step();
                
                epoch_loss += loss.read_scalar(0);
                
                if (b % 50 == 0) {
                    std::cout << "Epoch " << epoch << " Batch " << b << "/" << num_batches 
                              << " Loss: " << loss.read_scalar(0) << std::endl;
                }
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            std::cout << "Epoch " << epoch << " Done. Time: " 
                      << std::chrono::duration<double>(end_time - start_time).count() << "s" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}