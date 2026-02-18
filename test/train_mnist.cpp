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

// Gradient Clipping to prevent explosion
inline void clip_grad_norm(const std::vector<Tensor*>& params, double max_norm) {
    double total_norm_sq = 0.0;
    for (auto* p : params) {
        if (!p->impl || !p->impl->grad) continue;
        float* g = (float*)p->impl->grad->data->data.get();
        size_t n = p->numel();
        double layer_sum = 0.0;
        // Single thread safety
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
        omp_set_num_threads(4); 

        std::cout << "Loading MNIST data..." << std::endl;
        MNISTData raw_data = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        
        TensorDataset dataset(raw_data.images, raw_data.labels);
        
        // Transform: Normalize MNIST to [0, 1] then standardize
        dataset.transform = [](const void* src, float* dst, size_t n) {
            const float* s = (const float*)src; 
            for(size_t i=0; i<n; ++i) {
                // Approximate mean/std for MNIST
                dst[i] = (s[i] - 0.1307f) / 0.3081f;
            }
        };

        // Batch size 64 is often more stable than 128 for simple MLPs initially
        size_t batch_size = 64;
        SimpleDataLoader loader(dataset, batch_size, true); 

        MLPNet model;
        std::vector<Tensor*> params = model.parameters();
        
        kaiming_init(params); 
        for(auto* p : params) p->requires_grad_(true);

        AdamW optim(params, 0.001); 

        std::cout << "Starting training..." << std::endl;
        std::string ckpt = "mnist_weights.bin";

        for (int epoch = 0; epoch < 5; ++epoch) {
            loader.reset();
            double total_loss = 0.0;
            int batches = 0;
            size_t correct = 0;
            size_t total_samples = 0;
            
            while(loader.has_next()) {
                auto [data, target_raw] = loader.next();
                
                // --- FIX 1: Prepare Shape for Loss ---
                // CrossEntropy uses gather(), which requires input & index to have same NDIM.
                // Output is [B, 10] (2D), so Target must be [B, 1] (2D).
                Tensor target_loss = target_raw;
                if (target_loss.shape().size() == 1) {
                    target_loss = target_loss.unsqueeze(1);
                }
                
                optim.zero_grad();
                Tensor output = model.forward(data);
                
                // Pass 2D target to Loss
                Tensor loss = Loss::CrossEntropy(output, target_loss);
                
                // --- FIX 2: HANDLE LOSS NAN/EXPLOSION ---
                double l_val = loss.item<double>(); 
                if (std::isnan(l_val) || std::isinf(l_val)) {
                    std::cout << "\n[ERROR] Loss is NaN/Inf at batch " << batches << ". Skipping step.\n";
                    batches++;
                    continue; 
                }
                
                backward(loss);
                clip_grad_norm(params, 1.0); 
                optim.step();
                
                total_loss += l_val;
                
                // --- FIX 3: Prepare Shape for Accuracy ---
                // argmax(1) returns 1D [Batch].
                // We must flatten target to 1D [Batch] to avoid broadcasting (which caused >100% acc).
                Tensor pred = output.argmax(1); 
                Tensor target_acc = target_raw.flatten();
                
                Tensor match = (pred == target_acc).astype(DType::Float32);
                correct += (size_t)sum(match).item<float>();
                
                total_samples += data.shape()[0];
                batches++;

                if (batches % 50 == 0) {
                     double current_acc = 100.0 * correct / total_samples;
                     double avg_loss = total_loss / batches;
                     
                     std::cout << "Epoch " << epoch << " Batch " << std::setw(4) << batches 
                               << " Loss: " << std::fixed << std::setprecision(4) << avg_loss
                               << " Acc: " << std::setprecision(2) << current_acc << "% \r" << std::flush;
                }
            }
            std::cout << "\nEpoch " << epoch << " Finished. Avg Loss: " << (total_loss / batches) 
                      << " Accuracy: " << (100.0 * correct / total_samples) << "%\n";
            
            checkpoints::save_weights(params, ckpt);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}