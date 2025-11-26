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
