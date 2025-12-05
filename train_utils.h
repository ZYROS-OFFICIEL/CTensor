#pragma once
#include "module.h"
#include <vector>
#include "dataloader.h" 
#include "dataloader.h"
// --- Generalized Training/Eval Utilities ---

// If you wrap your layers in the Module class (above), you can use this:
inline void set_train_mode(std::vector<Module*>& layers, bool train_mode) {
    for (auto* layer : layers) {
        if (train_mode) layer->train();
        else layer->eval();
    }
}

// Or, if you use a container like Sequential:
inline void set_model_mode(Module& model, bool train_mode) {
    if (train_mode) model.train();
    else model.eval();
}

// --- Optimizer Interface (Basic SGD) ---
// This fits perfectly here as it interacts with the parameters collected from modules.


// --- Optimizer Base Class ---
class Optimizer {
public:
    std::vector<Tensor*> params;
    double lr;

    Optimizer(const std::vector<Tensor*>& p, double learning_rate) 
        : params(p), lr(learning_rate) {}
    
    virtual ~Optimizer() = default;
    virtual void step() = 0;

    void zero_grad() {
        for (auto* p : params) {
            if (p->impl->storage->grad) {
                size_t nbytes = p->numel() * p->dtype_bytes();
                std::memset(p->impl->storage->grad.get(), 0, nbytes);
            }
        }
    }
};

// --- SGD Optimizer ---
class SGD : public Optimizer {
public:
    SGD(const std::vector<Tensor*>& p, double learning_rate) : Optimizer(p, learning_rate) {}

    void step() override {
        for (auto* p : params) {
            if (!p->impl->storage->grad) continue;
            
            // Fast path for Float32
            if (p->_dtype() == DType::Float32) {
                float* p_ptr = (float*)p->impl->storage->data.get();
                float* g_ptr = (float*)p->impl->storage->grad.get();
                size_t n = p->numel();
                
                #pragma omp parallel for
                for (size_t i = 0; i < n; ++i) {
                    p_ptr[i] -= (float)lr * g_ptr[i];
                }
            } else {
                // Fallback
                size_t n = p->numel();
                for (size_t i = 0; i < n; ++i) {
                    double p_val = read_scalar_at(p->impl->storage->data.get(), i, p->impl->dtype);
                    double g_val = read_scalar_at(p->impl->storage->grad.get(), i, p->impl->dtype);
                    write_scalar_at(p->impl->storage->data.get(), i, p->impl->dtype, p_val - lr * g_val);
                }
            }
        }
    }
};


// --- Generic Training Function ---
// Template allows it to accept any Model class that has a forward() method
template <typename ModelType>
void train_epoch(ModelType& model, DataLoader& loader, Optimizer& optim, int epoch, size_t log_interval = 100) {
    // model.train(); // Uncomment if you implement train/eval modes in Module
    
    double total_loss = 0.0;
    int batch_idx = 0;
    size_t processed = 0;
    
    loader.reset(); // Shuffle for new epoch

    while(true) {
        auto [data, target] = loader.next();
        if (!data.impl) break; // End of epoch

        optim.zero_grad();
        
        // Forward
        Tensor output = model.forward(data);
        
        // Loss (Assuming CrossEntropy for classification)
        Tensor loss = Loss::CrossEntropy(output, target);
        
        // Backward
        backward(loss);
        optim.step();

        // Stats
        double l = loss.read_scalar(0);
        if (std::isnan(l)) {
            std::cerr << "\nERROR: NaN Loss detected at epoch " << epoch << " batch " << batch_idx << "!\n";
            break;
        }
        
        total_loss += l;
        processed += data.shape()[0];

        if (batch_idx % log_interval == 0) {
            std::cout << "Train Epoch: " << epoch << " [" 
                      << std::setw(5) << processed << "/" << loader.size() << "]"
                      << " Loss: " << std::fixed << std::setprecision(6) << l << "\r" << std::flush;
        }
        batch_idx++;
    }
    
    double avg_loss = (batch_idx > 0) ? (total_loss / batch_idx) : 0.0;
    std::cout << "\nTrain Epoch: " << epoch << " Finished. Avg Loss: " << avg_loss << std::endl;
}

// --- Generic Evaluation Function ---
template <typename ModelType>
double evaluate(ModelType& model, DataLoader& loader) {
    // model.eval(); // Uncomment if implemented
    
    double total_loss = 0.0;
    int correct = 0;
    size_t total_samples = 0;
    int batch_count = 0;
    
    loader.reset(); // No shuffle usually needed for test, but resets cursor

    while(true) {
        auto [data, target] = loader.next();
        if (!data.impl) break;

        Tensor output = model.forward(data);
        Tensor loss = Loss::CrossEntropy(output, target); // Sum reduction? Check your Loss impl.
        
        // If loss is mean-reduced inside CrossEntropy:
        total_loss += loss.read_scalar(0);
        
        // --- Calculate Accuracy (Argmax) ---
        // Assuming output is [Batch, Classes] (Float32)
        // Assuming target is [Batch, 1] (Int32)
        
        const float* out_ptr = (const float*)output.impl->storage->data.get();
        const int32_t* tgt_ptr = (const int32_t*)target.impl->storage->data.get();
        
        size_t batch = data.shape()[0];
        size_t classes = output.shape()[1];
        
        for (size_t b = 0; b < batch; ++b) {
            float max_val = -1e9;
            int pred = -1;
            for (size_t c = 0; c < classes; ++c) {
                float val = out_ptr[b * classes + c];
                if (val > max_val) {
                    max_val = val;
                    pred = (int)c;
                }
            }
            
            if (pred == tgt_ptr[b]) {
                correct++;
            }
        }
        
        total_samples += batch;
        batch_count++;
        
        // Simple progress spinner
        if (batch_count % 50 == 0) std::cout << "." << std::flush;
    }
    
    double avg_loss = (batch_count > 0) ? total_loss / batch_count : 0.0;
    double accuracy = (total_samples > 0) ? (100.0 * correct / total_samples) : 0.0;

    std::cout << "\nTest Set: Avg Loss: " << avg_loss << ", Accuracy: " << correct << "/" << total_samples 
              << " (" << std::fixed << std::setprecision(2) << accuracy << "%)\n";
              
    return accuracy;
}