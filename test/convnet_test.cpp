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
int main() {
    try {
        std::cout << "Loading MNIST data..." << std::endl;
        // Ensure these files are in your build directory
        MNISTData train_data = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        
        ConvNet model;

        std::cout << "Initializing weights (Xavier/Kaiming init recommended, simple random used here)..." << std::endl;
        std::srand(std::time(nullptr));
        
        // Initialize weights
        for (auto* p : model.parameters()) {
            if (!p->impl) continue;
            p->requires_grad_(true);
            size_t n = p->numel();
            if (!p->impl->data || !p->impl->data->data) continue;
            
            // Heavier initialization for Convs usually helps, but keeping simple for consistency
            float* ptr = (float*)p->impl->data->data.get();
            for (size_t i = 0; i < n; ++i) {
                float r = static_cast<float>(std::rand()) / RAND_MAX; 
                ptr[i] = (r - 0.5f) * 0.1f; 
            }
        }

        // Slightly lower learning rate for ConvNets usually helps stability
        SGD optim(model.parameters(), 0.005);
        
        int BATCH_SIZE = 32; // Smaller batch size to keep memory usage low during debugging
        int EPOCHS = 2;
        size_t num_train = train_data.images.shape()[0];
        size_t num_batches = num_train / BATCH_SIZE;

        std::cout << "Starting ConvNet training loop..." << std::endl;

        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            double epoch_loss = 0.0;
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int b = 0; b < num_batches; ++b) {
                size_t start_idx = b * BATCH_SIZE;

                // 1. Prepare Batch Images [BATCH, 1, 28, 28]
                std::vector<size_t> batch_shape_img = { (size_t)BATCH_SIZE, 1, 28, 28 };
                Tensor batch_imgs(batch_shape_img, DType::Float32, false); 
                
                float* src_ptr = (float*)train_data.images.impl->data->data.get() + start_idx * 28*28;
                float* dst_ptr = (float*)batch_imgs.impl->data->data.get();
                std::memcpy(dst_ptr, src_ptr, BATCH_SIZE * 28 * 28 * sizeof(float));
                
                // 2. Prepare Batch Labels [BATCH, 1]
                std::vector<size_t> batch_shape_lbl = { (size_t)BATCH_SIZE, 1 }; 
                Tensor batch_lbls(batch_shape_lbl, DType::Int32, false);
                
                int32_t* src_lbl = (int32_t*)train_data.labels.impl->data->data.get() + start_idx;
                int32_t* dst_lbl = (int32_t*)batch_lbls.impl->data->data.get();
                std::memcpy(dst_lbl, src_lbl, BATCH_SIZE * sizeof(int32_t));

                // 3. Forward & Backward
                optim.zero_grad();
                Tensor output = model.forward(batch_imgs);
                Tensor loss = Loss::CrossEntropy(output, batch_lbls);
                
                backward(loss); 
                optim.step();

                // 4. Logging
                double current_loss = loss.read_scalar(0);
                if (std::isnan(current_loss)) {
                    std::cout << "NaN detected at batch " << b << std::endl;
                    break;
                }

                epoch_loss += current_loss;
                if (b % 50 == 0) { // Log more frequently
                    std::cout << "Epoch " << epoch << " | Batch " << b << "/" << num_batches 
                              << " | Loss: " << current_loss << std::endl;
                }
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            double dur = std::chrono::duration<double>(end_time - start_time).count();
            std::cout << "--- Epoch " << epoch << " Finished in " << dur << "s ---" << std::endl;
            std::cout << "Avg Loss: " << epoch_loss / num_batches << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}