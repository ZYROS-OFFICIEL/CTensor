#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <ctime>
#include <cstdint>
#include <algorithm> // For std::max_element
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

// --- Model Definition (Must match training exactly) ---
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
        out = relu1(out);
        out = pool1(out);
        out = conv2(out);
        out = relu2(out);
        out = pool2(out);
        out = flat(out);
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
        std::cout << "Loading MNIST Test Data..." << std::endl;
        MNISTData test_data = load_mnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
        
        ConvNet model;
        std::string checkpoint_path = "mnist_cnn.bin";
        
        if (argc > 1 && std::string(argv[1]) == "--resume") {
            std::cout << "Loading checkpoint from " << checkpoint_path << "...\n";
            std::vector<Tensor*> params = model.parameters();
            checkpoints::load_weights(params, checkpoint_path);
        } else {
            std::cerr << "WARNING: No weights loaded! Run with --resume to load trained model.\n";
        }

        // No Optimizer needed for testing
        
        int BATCH_SIZE = 100; // Larger batch for inference is fine
        size_t num_test = test_data.images.shape()[0];
        size_t num_batches = num_test / BATCH_SIZE;

        std::cout << "Starting evaluation on " << num_test << " images.\n";

        // model.eval(); // If you implement eval mode (disables dropout), call it here.

        double total_loss = 0.0;
        int correct_predictions = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int b = 0; b < num_batches; ++b) {
            size_t start_idx = b * BATCH_SIZE;
            
            // 1. Prepare Batch
            std::vector<size_t> batch_shape_img = { (size_t)BATCH_SIZE, 1, 28, 28 };
            Tensor batch_imgs(batch_shape_img, DType::Float32, false); 

            float* src_ptr = (float*)test_data.images.impl->storage->data.get() + start_idx * 28*28;
            float* dst_ptr = (float*)batch_imgs.impl->storage->data.get();
            std::memcpy(dst_ptr, src_ptr, BATCH_SIZE * 28 * 28 * sizeof(float));
            
            std::vector<size_t> batch_shape_lbl = { (size_t)BATCH_SIZE, 1 }; 
            Tensor batch_lbls(batch_shape_lbl, DType::Int32, false);
            
            int32_t* src_lbl = (int32_t*)test_data.labels.impl->storage->data.get() + start_idx;
            int32_t* dst_lbl = (int32_t*)batch_lbls.impl->storage->data.get();
            std::memcpy(dst_lbl, src_lbl, BATCH_SIZE * sizeof(int32_t));

            // 2. Forward
            Tensor output = model.forward(batch_imgs); // Output is [Batch, 10] (Logits)
            Tensor loss = Loss::CrossEntropy(output, batch_lbls);
            total_loss += loss.read_scalar(0);

            // 3. Calculate Accuracy
            // We need to find argmax of output for each row
            float* out_data = (float*)output.impl->storage->data.get();
            
            for (int i = 0; i < BATCH_SIZE; ++i) {
                // Find max in row i
                float max_val = -1e9;
                int pred_class = -1;
                for (int c = 0; c < 10; ++c) {
                    float val = out_data[i * 10 + c];
                    if (val > max_val) {
                        max_val = val;
                        pred_class = c;
                    }
                }
                
                // Compare with true label
                if (pred_class == src_lbl[i]) {
                    correct_predictions++;
                }
            }

            if (b % 20 == 0) {
                std::cout << "Batch " << b << "/" << num_batches << " Loss: " << loss.read_scalar(0) << "\r" << std::flush;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end_time - start_time).count();
        double avg_loss = total_loss / num_batches;
        double accuracy = (double)correct_predictions / (num_batches * BATCH_SIZE) * 100.0;

        std::cout << "\n=== Test Results ===\n";
        std::cout << "Time: " << duration << "s\n";
        std::cout << "Avg Loss: " << avg_loss << "\n";
        std::cout << "Accuracy: " << accuracy << "% (" << correct_predictions << "/" << (num_batches*BATCH_SIZE) << ")\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}