#pragma once
#include "module.h"
#include <vector>
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

class Optimizer {
public:
    std::vector<Tensor*> params;
    double lr;

    Optimizer(const std::vector<Tensor*>& parameters, double learning_rate) 
        : params(parameters), lr(learning_rate) {}

    void step() {
        for (auto* p : params) {
            if (!p->impl->storage->grad) continue; // Skip if no grad
            
            // p = p - lr * grad
            // Simple SGD update
            // Note: p->data -= lr * p->grad
            size_t n = p->numel();
            // We need direct access for speed, or use tensor ops
            // Tensor update = *p - tensor_from_grad(*p) * lr;
            // But we want in-place update.
            
            // Manual loop for now (assumes float32/double consistency)
            // Ideally, add a `sub_in_place` or `add_scaled` op to Tensor.
            // For now, using ops1 functions is safest but creates new tensors.
            // Ideally: p->data[i] -= lr * p->grad[i]
            
            // Since we don't have an easy in-place tensor op exposed yet:
            for (size_t i = 0; i < n; ++i) {
                // Use read_scalar_at / write_scalar_at to be safe with types
                // BUT parameters are usually contiguous weights.
                // Assuming contiguous for optimization here is common, but let's be safe.
                // Wait, params are weights created by us, they are contiguous.
                
                double p_val = read_scalar_at(p->impl->storage->data.get(), i, p->impl->dtype);
                double g_val = read_scalar_at(p->impl->storage->grad.get(), i, p->impl->dtype);
                
                write_scalar_at(p->impl->storage->data.get(), i, p->impl->dtype, p_val - lr * g_val);
            }
        }
    }

    void zero_grad() {
        for (auto* p : params) {
            if (p->impl->storage->grad) {
                // Re-allocate or memset to zero
                // ensure_grad_buffer(*p, true) would do it if exposed
                // Or just memset directly.
                size_t nbytes = p->numel() * p->dtype_bytes();
                std::memset(p->impl->storage->grad.get(), 0, nbytes);
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