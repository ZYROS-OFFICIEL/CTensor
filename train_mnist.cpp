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
#include "loss.h"
#include "train_utils.h"
#include "mnist.h"

// --- SAFE LOSS FUNCTION IMPLEMENTATION ---
// Implements LogSoftmax + NLLLoss with explicit reshaping to prevent broadcast errors
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
    // logits: [Batch, Classes]
    // targets: [Batch, 1] (indices)
    
    size_t B = logits.shape()[0];
    size_t C = logits.shape()[1];

    // 1. LogSoftmax Stability Trick: x - max(x)
    // We don't have max(dim) exposed in opsmp header nicely yet for some versions, 
    // so we skip the shift for simplicity (might be slightly unstable for huge values but fine for MNIST).
    // If you have max_mp(t, dim), use it:
    Tensor max_vals = max_mp(logits, 1).reshape({B, 1});
    Tensor shifted = logits - max_vals; 

    // 2. Compute Exp
    Tensor exp_vals = exp_mp(shifted);

    // 3. Sum Exp (keepdims=True manually)
    Tensor sum_exp = sum_mp(exp_vals, 1); // Returns [Batch]
    sum_exp = sum_exp.reshape({B, 1});    // Force [Batch, 1] for broadcasting <--- THE FIX

    // 4. Log(Sum(Exp))
    Tensor log_sum_exp = ln_mp(sum_exp);

    // 5. LogSoftmax = logits - log_sum_exp
    Tensor log_probs = shifted - log_sum_exp; // [B, 10] - [B, 1] -> OK!

    // 6. NLL Loss (Pick correct class probability)
    // targets must be [B, 1]
    Tensor picked = log_probs.gather(targets, 1); // [B, 1]
    
    // 7. Mean and Negate
    Tensor loss = mean_mp(picked) * -1.0;
    
    return loss;
}

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
          pool1(2, 2, 2, 2),  // Fixed Stride=2
          
          conv2(6, 16, 5, 5),
          relu2(),
          pool2(2, 2, 2, 2),  // Fixed Stride=2
          
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
        
        // Safety Reshape
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
        Tensor test_logits = Tensor::from_vector(
            {
                2.0, 1.0, 0.1, 0.5,
                0.1, 0.2, 3.0, 0.3,
                1.0, 2.0, 0.1, 0.4,
                0.5, 0.3, 0.2, 4.0
            },
            {4,4}
        );
        Tensor test_targets = Tensor::from_vector(
            {0, 2, 1, 3},
            {4,1},
            DType::Int32
        );

        Tensor L = cross_entropy_loss(test_logits, test_targets);
        std::cout << "Test loss = " << L.read_scalar(0) << std::endl;

        std::cout << "Loading MNIST data..." << std::endl;
        MNISTData train_data = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        
        ConvNet model;
        // Default rand() gives [0, 1]. We need [-0.1, 0.1] for convergence.
        std::srand(std::time(nullptr));
        for (auto* p : model.parameters()) {
            if (!p->impl) continue;
            size_t n = p->numel();
            // access raw data assuming float (since model weights are float)
            float* ptr = (float*)p->impl->storage->data.get();
            
            // He/Xavier-like initialization (simple version)
            // Random values between -0.1 and 0.1
            for (size_t i = 0; i < n; ++i) {
                float r = static_cast<float>(std::rand()) / RAND_MAX; // 0 to 1
                ptr[i] = (r - 0.5f) * 0.2f; // shift to -0.5..0.5, then scale to -0.1..0.1
            }
        }
        Optimizer optim(model.parameters(), 1e-2); 
        
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
                
                // --- CORRECTED ---
                // Batch Images
                std::vector<size_t> batch_shape_img = { (size_t)BATCH_SIZE, 1, 28, 28 };

                // 1. Change DType to Float32
                Tensor batch_imgs(batch_shape_img, DType::Float32, false); 

                // 2. Cast to float* (assuming Float32 is 4 bytes)
                float* src_ptr = (float*)train_data.images.impl->storage->data.get() + start_idx * 28*28;
                float* dst_ptr = (float*)batch_imgs.impl->storage->data.get();

                // 3. Copy using sizeof(float)
                std::memcpy(dst_ptr, src_ptr, BATCH_SIZE * 28 * 28 * sizeof(float));
                
                // Batch Labels (Reshaped to [B, 1] for Gather)
                std::vector<size_t> batch_shape_lbl = { (size_t)BATCH_SIZE, 1 }; 
                Tensor batch_lbls(batch_shape_lbl, DType::Int32, false);
                int32_t* src_lbl = (int32_t*)train_data.labels.impl->storage->data.get() + start_idx;
                int32_t* dst_lbl = (int32_t*)batch_lbls.impl->storage->data.get();
                std::memcpy(dst_lbl, src_lbl, BATCH_SIZE * sizeof(int32_t));

                optim.zero_grad();
                
                // Forward
                Tensor output = model.forward(batch_imgs);
                
                // Loss (Using safe local implementation)
                Tensor loss = cross_entropy_loss(output, batch_lbls);
                
                // Backward
                backward(loss);

                
                // Update
                optim.step();
                
                epoch_loss += loss.read_scalar(0);
                
                if (b % 50 == 0) {
                    std::cout << "Batch " << b << "/" << num_batches 
                              << " Loss: " << loss.read_scalar(0) << std::endl;
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