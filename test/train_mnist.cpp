#include <iostream>
#include <fstream> 
#include <numeric>
#include <cmath>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <omp.h>
#include "core.h"
#include "neuralnet.h"

// =======================================================================
//                          SAFETY UTILITIES
// =======================================================================

inline void clip_grad_norm(const std::vector<Tensor*>& params, double max_norm) {
    double total_norm_sq = 0.0;
    for (auto* p : params) {
        if (!p->impl || !p->impl->grad) continue;
        float* g = (float*)p->impl->grad->data->data.get();
        size_t n = p->numel();
        double layer_sum = 0.0;
        // Removed OMP here to be absolutely safe during debug
        for (size_t i = 0; i < n; ++i) {
            layer_sum += g[i] * g[i];
        }
        total_norm_sq += layer_sum;
    }
    double total_norm = std::sqrt(total_norm_sq);
    
    if (total_norm > max_norm) {
        double scale = max_norm / (total_norm + 1e-6);
        for (auto* p : params) {
            if (!p->impl || !p->impl->grad) continue;
            float* g = (float*)p->impl->grad->data->data.get();
            size_t n = p->numel();
            for (size_t i = 0; i < n; ++i) g[i] *= (float)scale;
        }
    }
}

// =======================================================================
//                              MODEL
// =======================================================================

class MLPNet : public Module {
public:
    Flatten flat;
    Linear fc1, fc2, fc3;
    Relu relu1, relu2;

    MLPNet() 
        : flat(),
          // 784 -> 128 -> 64 -> 10
          fc1(784, 128, true, DType::Float32),
          relu1(),
          fc2(128, 64, true, DType::Float32),
          relu2(),
          fc3(64, 10, true, DType::Float32)
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

// =======================================================================
//                              MAIN
// =======================================================================

int main() {
    try {
        // --- STABILIZATION FIX: Disable OpenMP for training loop ---
        // This prevents race conditions in gradient accumulation.
        omp_set_num_threads(1); 
        std::cout << "Forcing Single-Threaded Mode for Stability." << std::endl;

        std::cout << "Loading MNIST data..." << std::endl;
        MNISTData raw_data = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        
        TensorDataset dataset(raw_data.images, raw_data.labels);
        
        // Transform: Normalize to [-1, 1]
        dataset.transform = [](const void* src, float* dst, size_t n) {
            const float* s = (const float*)src; 
            for(size_t i=0; i<n; ++i) {
                dst[i] = (s[i] - 0.1307f) / 0.3081f;
            }
        };

        SimpleDataLoader loader(dataset, 32, true); // Reduced Batch Size to 32

        MLPNet model;
        std::vector<Tensor*> params = model.parameters();
        
        // 1. Force fresh initialization with smaller scale
        robust_weight_init(params, 0.02f); 
        for(auto* p : params) p->requires_grad_(true);

        // 2. VERY Conservative LR
        AdamW optim(params, 0.0001); 

        std::cout << "Starting training..." << std::endl;
        std::string ckpt = "mnist_weights.bin";

        for (int epoch = 0; epoch < 5; ++epoch) {
            loader.reset();
            double total_loss = 0.0;
            int batches = 0;
            size_t correct = 0;
            size_t total = 0;
            
            while(loader.has_next()) {
                auto [data, target] = loader.next();
                
                // Sanity check inputs
                if (batches == 0) {
                     float* dptr = (float*)data.impl->data->data.get();
                     if (std::isnan(dptr[0]) || std::abs(dptr[0]) > 100.0f) {
                         throw std::runtime_error("Input data corrupted/not normalized!");
                     }
                }

                optim.zero_grad();
                Tensor output = model.forward(data);
                Tensor loss = Loss::CrossEntropy(output, target);
                
                backward(loss);

                // Clip Gradients aggressively
                clip_grad_norm(params, 0.5); 

                optim.step();
                
                double l_val = loss.read_scalar(0);
                if (std::isnan(l_val) || std::isinf(l_val)) {
                    std::cout << "\n[ERROR] Loss Explosion (NaN/Inf) at Batch " << batches << "! Stopping.\n";
                    return -1;
                }
                total_loss += l_val;
                
                // Calculate Accuracy
                const float* out_data = (const float*)output.impl->data->data.get();
                const int32_t* tgt_data = (const int32_t*)target.impl->data->data.get();
                size_t bs = data.shape()[0];
                for(size_t i=0; i<bs; ++i) {
                    float max_v = -1e9;
                    int pred = -1;
                    for(int c=0; c<10; ++c) {
                        if (out_data[i*10 + c] > max_v) {
                            max_v = out_data[i*10 + c];
                            pred = c;
                        }
                    }
                    if (pred == tgt_data[i]) correct++;
                }
                total += bs;
                batches++;

                if (batches % 100 == 0) {
                     std::cout << "Epoch " << epoch << " Batch " << std::setw(4) << batches 
                               << " Loss: " << std::fixed << std::setprecision(4) << l_val 
                               << " Acc: " << (100.0 * correct / total) << "% \r" << std::flush;
                }
            }
            std::cout << "\nEpoch " << epoch << " Finished. Avg Loss: " << (total_loss / batches) 
                      << " Accuracy: " << (100.0 * correct / total) << "%\n";
            
            checkpoints::save_weights(params, ckpt);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}