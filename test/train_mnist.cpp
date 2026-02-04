#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <ctime>
#include <cmath>
#include "core.h"
#include "neuralnet.h"

class MLPNet : public Module {
public:
    Flatten flat;
    Linear fc1;
    Relu relu1;
    Linear fc2;
    Relu relu2;
    Linear fc3;

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
        std::cout << "Loading MNIST data..." << std::endl;
        MNISTData train_data = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        
        MLPNet model;

        std::cout << "Initializing weights (Kaiming Uniform)..." << std::endl;
        std::srand(std::time(nullptr));
        
        for (auto* p : model.parameters()) {
            if (!p->impl) continue;
            p->requires_grad_(true);
            size_t n = p->numel();
            if (!p->impl->data || !p->impl->data->data) continue;
            
            float limit = std::sqrt(6.0f / (float)p->shape()[0]); 
            if (p->impl->ndim > 1) limit = std::sqrt(6.0f / (float)p->shape()[1]); 
            
            float* ptr = (float*)p->impl->data->data.get();
            for (size_t i = 0; i < n; ++i) {
                float r = static_cast<float>(std::rand()) / RAND_MAX; 
                ptr[i] = (r * 2.0f - 1.0f) * limit; 
            }
        }

        SGD optim(model.parameters(), 0.01);
        
        int BATCH_SIZE = 64;
        int EPOCHS = 2;
        size_t num_train = train_data.images.shape()[0];
        size_t num_batches = num_train / BATCH_SIZE;

        std::cout << "Starting training loop..." << std::endl;

        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            double epoch_loss = 0.0;
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int b = 0; b < num_batches; ++b) {
                size_t start_idx = b * BATCH_SIZE;

                // 1. Prepare Batch Images
                std::vector<size_t> batch_shape_img = { (size_t)BATCH_SIZE, 1, 28, 28 };
                Tensor batch_imgs(batch_shape_img, DType::Float32, false); 
                
                float* dst_ptr = (float*)batch_imgs.impl->data->data.get();
                size_t batch_elements = BATCH_SIZE * 28 * 28;
                
                // --- ROBUST DATA LOADING ---
                // Check if source is UInt8 (bytes) or Float32
                if (train_data.images._dtype() == DType::UInt8) {
                    // Safe Cast: Interpret as bytes first!
                    uint8_t* src_ptr = (uint8_t*)train_data.images.impl->data->data.get() + start_idx * 28*28;
                    #pragma omp parallel for
                    for (size_t i = 0; i < batch_elements; ++i) {
                        dst_ptr[i] = (float)src_ptr[i] / 255.0f;
                    }
                } else {
                    // Assume Float32
                    float* src_ptr = (float*)train_data.images.impl->data->data.get() + start_idx * 28*28;
                    #pragma omp parallel for
                    for (size_t i = 0; i < batch_elements; ++i) {
                        dst_ptr[i] = src_ptr[i] / 255.0f;
                    }
                }
                
                // 2. Prepare Labels
                std::vector<size_t> batch_shape_lbl = { (size_t)BATCH_SIZE, 1 }; 
                Tensor batch_lbls(batch_shape_lbl, DType::Int32, false);
                int32_t* dst_lbl = (int32_t*)batch_lbls.impl->data->data.get();
                
                if (train_data.labels._dtype() == DType::UInt8) {
                    // Correctly read 1 byte per label and cast to int
                    uint8_t* src_lbl = (uint8_t*)train_data.labels.impl->data->data.get() + start_idx;
                    for(int i=0; i<BATCH_SIZE; ++i) {
                        dst_lbl[i] = static_cast<int32_t>(src_lbl[i]);
                    }
                } else {
                    // Already Int32
                    int32_t* src_lbl = (int32_t*)train_data.labels.impl->data->data.get() + start_idx;
                    std::memcpy(dst_lbl, src_lbl, BATCH_SIZE * sizeof(int32_t));
                }

                optim.zero_grad();
                Tensor output = model.forward(batch_imgs);
                Tensor loss = Loss::CrossEntropy(output, batch_lbls);
                
                backward(loss); 
                optim.step();

                if (std::isnan(loss.read_scalar(0))) {
                    std::cout << "NaN detected at batch " << b << std::endl;
                    break;
                }

                epoch_loss += loss.read_scalar(0);
                if (b % 100 == 0) std::cout << "Batch " << b << " Loss: " << loss.read_scalar(0) << std::endl;
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            std::cout << "Epoch " << epoch << " Time: " << std::chrono::duration<double>(end_time - start_time).count() << "s | Avg Loss: " << epoch_loss/num_batches << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}