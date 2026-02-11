#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <ctime>
#include <cmath>
#include <fstream> 
#include <numeric>
#include <algorithm> 
#include <random>    

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

void safe_init(std::vector<Tensor*>& params) {
    std::cout << "Initializing weights (Safe Normal)..." << std::endl;
    std::srand(std::time(nullptr));
    for (auto* p : params) {
        if (!p->impl) continue;
        size_t n = p->numel();
        float scale = 0.01f; // Even smaller scale for safety
        
        float* ptr = (float*)p->impl->data->data.get();
        for (size_t i = 0; i < n; ++i) {
            float r = static_cast<float>(std::rand()) / RAND_MAX; 
            ptr[i] = (r * 2.0f - 1.0f) * scale; 
        }
    }
}

bool are_weights_corrupted(const std::vector<Tensor*>& params) {
    for(auto* p : params) {
        if(!p->impl) continue;
        float* ptr = (float*)p->impl->data->data.get();
        size_t n = p->numel();
        for(size_t i=0; i< std::min(n, (size_t)100); ++i) { 
            if (std::isnan(ptr[i]) || std::isinf(ptr[i]) || std::abs(ptr[i]) > 50.0f) return true;
        }
    }
    return false;
}

int main() {
    try {
        std::cout << "Loading MNIST data..." << std::endl;
        MNISTData train_data = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        
        MLPNet model;
        std::vector<Tensor*> params = model.parameters();
        
        std::string checkpoint_path = "mnist_weights.bin";
        std::ifstream infile(checkpoint_path);
        bool loaded = false;

        if (infile.good()) {
            infile.close();
            checkpoints::load_weights(params, checkpoint_path);
            if (are_weights_corrupted(params)) {
                std::cerr << "WARNING: Corrupted weights detected. Resetting.\n";
                safe_init(params);
            } else {
                std::cout << "Checkpoint loaded and verified.\n";
                loaded = true;
            }
        } 
        
        if (!loaded) safe_init(params);

        for(auto* p : params) p->requires_grad_(true);

        // --- STABILITY FIX: Low Learning Rate + Weight Decay ---
        // 1e-4 is very safe. 
        AdamW optim(params, 0.0001, 0.9, 0.999, 1e-8, 0.01);
        
        int BATCH_SIZE = 64;
        int EPOCHS = 5; 
        size_t num_train = train_data.images.shape()[0];
        size_t num_batches = num_train / BATCH_SIZE;

        std::vector<size_t> indices(num_train);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 g(std::random_device{}());

        // Pointers for fast access
        uint8_t* img_u8_base = nullptr;
        float* img_f32_base = nullptr;
        bool is_u8 = (train_data.images._dtype() == DType::UInt8);
        if (is_u8) img_u8_base = (uint8_t*)train_data.images.impl->data->data.get();
        else img_f32_base = (float*)train_data.images.impl->data->data.get();

        int32_t* lbl_base = (int32_t*)train_data.labels.impl->data->data.get();

        std::cout << "Starting training loop (Safe Mode)..." << std::endl;

        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), g);

            double epoch_loss = 0.0;
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int b = 0; b < num_batches; ++b) {
                size_t start_idx = b * BATCH_SIZE;

                // 1. Prepare Batch
                std::vector<size_t> batch_shape_img = { (size_t)BATCH_SIZE, 1, 28, 28 };
                Tensor batch_imgs(batch_shape_img, DType::Float32, false); 
                float* dst_ptr = (float*)batch_imgs.impl->data->data.get();
                
                std::vector<size_t> batch_shape_lbl = { (size_t)BATCH_SIZE, 1 }; 
                Tensor batch_lbls(batch_shape_lbl, DType::Int32, false);
                int32_t* dst_lbl = (int32_t*)batch_lbls.impl->data->data.get();

                #pragma omp parallel for
                for (int i = 0; i < BATCH_SIZE; ++i) {
                    size_t real_idx = indices[start_idx + i];
                    
                    // Normalize MNIST: (x - 0.1307) / 0.3081
                    // This helps gradients flow better than simple 0-1 scaling
                    if (is_u8) {
                        uint8_t* src = img_u8_base + real_idx * 784;
                        float* dst = dst_ptr + i * 784;
                        for (int k = 0; k < 784; ++k) {
                            float val = (float)src[k] / 255.0f;
                            dst[k] = (val - 0.1307f) / 0.3081f; 
                        }
                    } else {
                        float* src = img_f32_base + real_idx * 784;
                        float* dst = dst_ptr + i * 784;
                        for (int k = 0; k < 784; ++k) {
                            float val = src[k] / 255.0f;
                            dst[k] = (val - 0.1307f) / 0.3081f;
                        }
                    }
                    dst_lbl[i] = lbl_base[real_idx];
                }

                // 2. Training Step
                optim.zero_grad();
                Tensor output = model.forward(batch_imgs);
                Tensor loss = Loss::CrossEntropy(output, batch_lbls);
                
                backward(loss); 
                
                // --- GRADIENT CLIPPING (GLOBAL NORM) ---
                double total_norm_sq = 0.0;
                for(auto* p : params) {
                    if(p->impl && p->impl->grad) {
                        float* g = (float*)p->impl->grad->data->data.get();
                        size_t n = p->numel();
                        for(size_t k=0; k<n; ++k) {
                            if(std::isfinite(g[k])) total_norm_sq += g[k]*g[k];
                        }
                    }
                }
                double total_norm = std::sqrt(total_norm_sq);
                double max_norm = 1.0;
                
                if (total_norm > max_norm || std::isnan(total_norm)) {
                    double clip_coef = max_norm / (total_norm + 1e-6);
                    for(auto* p : params) {
                        if(p->impl && p->impl->grad) {
                            float* g = (float*)p->impl->grad->data->data.get();
                            size_t n = p->numel();
                            #pragma omp parallel for
                            for(size_t k=0; k<n; ++k) {
                                if(std::isfinite(g[k])) g[k] *= (float)clip_coef;
                                else g[k] = 0.0f; // Zero out NaNs
                            }
                        }
                    }
                }
                // ---------------------------------------

                optim.step();

                double l = loss.read_scalar(0);
                if (std::isfinite(l)) epoch_loss += l;
                
                if (b % 100 == 0) {
                    std::cout << "Batch " << b << " Loss: " << l << std::endl;
                }
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            std::cout << "Epoch " << epoch << " Time: " << std::chrono::duration<double>(end_time - start_time).count() << "s | Avg Loss: " << epoch_loss/num_batches << std::endl;
            
            checkpoints::save_weights(params, checkpoint_path);
            std::cout << "Saved checkpoint to " << checkpoint_path << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}