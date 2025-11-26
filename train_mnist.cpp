#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

#include "tensor1.h"
#include "opsmp.h"
#include "autograd.h"
#include "conv.h"
#include "pooling.h"
#include "linear.h"
#include "Relu.h"
#include "dropout.h"
#include "loss.h"       // Ensure this exists and has CrossEntropy
#include "train_utils.h" // For Optimizer and set_model_mode
#include "mnist.h"      // The loader above

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
        : conv1(1, 6, 5, 5, 1, 1, 2, 2),  // 1->6 channels, 5x5 kernel, padding 2 (preserves 28x28)
          relu1(),
          pool1(2, 2),                    // 28x28 -> 14x14
          
          conv2(6, 16, 5, 5),             // 6->16 channels, 5x5 kernel, valid pad -> 10x10
          relu2(),
          pool2(2, 2),                    // 10x10 -> 5x5
          
          flat(),
          
          fc1(16 * 5 * 5, 120),           // 400 -> 120
          relu3(),
          fc2(120, 10)                    // 120 -> 10 classes
    {}

    Tensor forward(const Tensor& x) {
        Tensor out = x;
        out = conv1(out);
        out = Relu_mp(out); // Use functional ReLU or class
        out = pool1(out);
        
        out = conv2(out);
        out = Relu_mp(out);
        out = pool2(out);
        
        out = flat(out);
        
        out = fc1(out);
        out = Relu_mp(out);
        out = fc2(out);
        
        return out; // Return raw logits (CrossEntropyLoss handles Softmax)
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

// Helper to calculate accuracy
double compute_accuracy(const Tensor& logits, const Tensor& labels) {
    // Simple argmax check
    size_t batch_size = logits.shape()[0];
    size_t classes = logits.shape()[1];
    int correct = 0;
    
    // We can do this on CPU nicely
    // Ideally use argmax_mp if implemented, but manual loop is fine for check
    for (size_t b = 0; b < batch_size; ++b) {
        double max_val = -1e9;
        int pred = -1;
        for (size_t c = 0; c < classes; ++c) {
            // Access logits[b][c]
            // Assuming contiguous for speed in this helper
            double val = logits.read_scalar(b * classes + c); 
            if (val > max_val) {
                max_val = val;
                pred = c;
            }
        }
        int label = (int)labels.read_scalar(b);
        if (pred == label) correct++;
    }
    return (double)correct / batch_size;
}

int main() {
    try {
        // 1. Load Data
        std::cout << "Loading MNIST data..." << std::endl;
        // Make sure these files are in your folder!
        MNISTData train_data = load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
        MNISTData test_data = load_mnist("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
        
        // Normalize: MNIST is 0-255, loader already does /255.0
        // Standardize? (x - 0.1307) / 0.3081 is common for MNIST but optional.

        // 2. Setup
        ConvNet model;
        Optimizer optim(model.parameters(), 0.01); // LR = 0.01
        
        int BATCH_SIZE = 64;
        int EPOCHS = 2; // Start small
        size_t num_train = train_data.images.shape()[0];
        size_t num_batches = num_train / BATCH_SIZE;

        std::cout << "Starting training on " << num_train << " images." << std::endl;

        // 3. Training Loop
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            model.train();
            double epoch_loss = 0.0;
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int b = 0; b < num_batches; ++b) {
                // a. Get Batch (slicing)
                // We need a 'slice' or 'select' helper usually.
                // Since we don't have a generic slice op exposed in opsmp yet,
                // we can use narrow/select or just pointer arithmetic hacks if contiguous.
                // Ideally: Tensor batch_imgs = train_data.images.slice(b*BS, (b+1)*BS);
                
                // WORKAROUND: Manual copy for batch creation if slice isn't ready
                // This is slow but works.
                // Ideally implement Tensor::slice(start, end) later.
                
                // Let's assume we implemented a basic slice logic or implement one here quickly
                // Actually, let's just implement a quick "get_batch" helper using memcpy for speed
                
                size_t start_idx = b * BATCH_SIZE;
                // Create Tensor views? No, strides are tricky. Copy is safer.
                std::vector<size_t> batch_shape_img = { (size_t)BATCH_SIZE, 1, 28, 28 };
                Tensor batch_imgs(batch_shape_img, DType::Float32, false);
                
                size_t img_size = 28*28;
                // Copy block
                // Using raw pointer copy for speed
                float* src_ptr = (float*)train_data.images.impl->storage->data.get() + start_idx * img_size;
                float* dst_ptr = (float*)batch_imgs.impl->storage->data.get();
                std::memcpy(dst_ptr, src_ptr, BATCH_SIZE * img_size * sizeof(float));
                
                // Labels
                std::vector<size_t> batch_shape_lbl = { (size_t)BATCH_SIZE };
                Tensor batch_lbls(batch_shape_lbl, DType::Float32, false);
                float* src_lbl = (float*)train_data.labels.impl->storage->data.get() + start_idx;
                float* dst_lbl = (float*)batch_lbls.impl->storage->data.get();
                std::memcpy(dst_lbl, src_lbl, BATCH_SIZE * sizeof(float));

                // b. Forward
                optim.zero_grad();
                Tensor output = model.forward(batch_imgs);
                
                // c. Loss
                // CrossEntropy takes logits and indices
                // Assuming your GradCrossEntropy takes (pred, target, reduction)
                // And does Softmax internally (PyTorch style). 
                // If it expects probabilities, add Softmax first.
                // Assuming standard PyTorch-like CrossEntropyLoss (Logits -> Loss)
                
                Tensor loss = cross_entropy(output, batch_lbls, "mean"); 
                
                // d. Backward
                backward(loss);
                
                // e. Update
                optim.step();
                
                epoch_loss += loss.read_scalar(0);
                
                if (b % 100 == 0) {
                    std::cout << "Epoch " << epoch << " Batch " << b << "/" << num_batches 
                              << " Loss: " << loss.read_scalar(0) << std::endl;
                }
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(end_time - start_time).count();
            std::cout << "Epoch " << epoch << " Done. Avg Loss: " << epoch_loss / num_batches 
                      << " Time: " << duration << "s" << std::endl;
            
            // Validation (on subset of test to be fast)
            model.eval();
            // ... validation loop (similar to above) ...
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}