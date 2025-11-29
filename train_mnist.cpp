#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <ctime>
#include "tensor1.h"
#include "opsmp.h"
#include "autograd.h"
#include "layer.h"
#include "Relu.h"
#include "loss.h"
#include "train_utils.h"
#include "mnist.h"

// --- SIMPLE MLP MODEL (No Convolutions) ---
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
          // 28x28 = 784 input features
          fc1(784, 256, true, DType::Float32),
          relu1(),
          fc2(256, 128, true, DType::Float32),
          relu2(),
          fc3(128, 10, true, DType::Float32)
    {}

    Tensor forward(const Tensor& x) {
        Tensor out = flat(x); // Reshapes [B, 1, 28, 28] -> [B, 784]
        out = fc1(out);
        out = Relu_mp(out);
        out = fc2(out);
        out = Relu_mp(out);
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
        
        MLPNet model; // <--- USING MLP

        std::cout << "Initializing weights (Float32)..." << std::endl;
        std::srand(std::time(nullptr));
        
        for (auto* p : model.parameters()) {
            if (!p->impl) continue;
            p->requires_grad_(true);
            size_t n = p->numel();
            float* ptr = (float*)p->impl->storage->data.get();
            for (size_t i = 0; i < n; ++i) {
                float r = static_cast<float>(std::rand()) / RAND_MAX; 
                ptr[i] = (r - 0.5f) * 0.1f;
            }
        }

        Optimizer optim(model.parameters(), 0.01);
        
        int BATCH_SIZE = 64;
        int EPOCHS = 2;
        size_t num_train = train_data.images.shape()[0];
        size_t num_batches = num_train / BATCH_SIZE;

        std::cout << "Starting training MLP..." << std::endl;

        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            double epoch_loss = 0.0;
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int b = 0; b < num_batches; ++b) {
                size_t start_idx = b * BATCH_SIZE;

                // Prepare Batch (Float32)
                std::vector<size_t> batch_shape_img = { (size_t)BATCH_SIZE, 1, 28, 28 };
                Tensor batch_imgs(batch_shape_img, DType::Float32, false); 
                float* src_ptr = (float*)train_data.images.impl->storage->data.get() + start_idx * 28*28;
                float* dst_ptr = (float*)batch_imgs.impl->storage->data.get();
                std::memcpy(dst_ptr, src_ptr, BATCH_SIZE * 28 * 28 * sizeof(float));
                
                std::vector<size_t> batch_shape_lbl = { (size_t)BATCH_SIZE, 1 }; 
                Tensor batch_lbls(batch_shape_lbl, DType::Int32, false);
                int32_t* src_lbl = (int32_t*)train_data.labels.impl->storage->data.get() + start_idx;
                int32_t* dst_lbl = (int32_t*)batch_lbls.impl->storage->data.get();
                std::memcpy(dst_lbl, src_lbl, BATCH_SIZE * sizeof(int32_t));

                optim.zero_grad();
                Tensor output = model.forward(batch_imgs);
                Tensor loss = Loss::CrossEntropy(output, batch_lbls);
                
                backward(loss); // <--- If this crashes, bug is in Reshape/Linear
                optim.step();
                
                epoch_loss += loss.read_scalar(0);
                if (b % 100 == 0) std::cout << "Batch " << b << " Loss: " << loss.read_scalar(0) << std::endl;
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            std::cout << "Epoch " << epoch << " Time: " << std::chrono::duration<double>(end_time - start_time).count() << "s" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}