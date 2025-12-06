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

// --- Adam Optimizer ---
class Adam : public Optimizer {
    // State for m (momentum) and v (velocity)
    struct State {
        std::vector<float> m;
        std::vector<float> v;
    };
    std::unordered_map<void*, State> states;
    
    double beta1, beta2, eps;
    int t; // timestep

public:
    Adam(const std::vector<Tensor*>& p, double learning_rate = 0.001, 
         double b1 = 0.9, double b2 = 0.999, double epsilon = 1e-8) 
        : Optimizer(p, learning_rate), beta1(b1), beta2(b2), eps(epsilon), t(0) {}

    void step() override {
        t++;
        for (auto* p : params) {
            if (!p->impl->storage->grad) continue;
            
            size_t n = p->numel();
            void* key = p->impl->storage->data.get();

            // Initialize state if missing
            if (states.find(key) == states.end()) {
                states[key] = { std::vector<float>(n, 0.0f), std::vector<float>(n, 0.0f) };
            }
            
            State& s = states[key];
            
            // Only implementing fast path for Float32
            if (p->_dtype() == DType::Float32) {
                float* theta = (float*)p->impl->storage->data.get();
                float* grad  = (float*)p->impl->storage->grad.get();
                float* m = s.m.data();
                float* v = s.v.data();
                
                // Correction factors
                double bias_correction1 = 1.0 - std::pow(beta1, t);
                double bias_correction2 = 1.0 - std::pow(beta2, t);
                
                // PyTorch-style expansion for efficiency:
                // lr_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
                float lr_t = (float)(lr * std::sqrt(bias_correction2) / bias_correction1);

                #pragma omp parallel for
                for (size_t i = 0; i < n; ++i) {
                    float g = grad[i];
                    
                    // Update biased first moment estimate
                    m[i] = (float)beta1 * m[i] + (1.0f - (float)beta1) * g;
                    
                    // Update biased second raw moment estimate
                    v[i] = (float)beta2 * v[i] + (1.0f - (float)beta2) * g * g;
                    
                    // Update parameters
                    theta[i] -= lr_t * m[i] / (std::sqrt(v[i]) + (float)eps);
                }
            }
        }
    }
};

// --- AdamW Optimizer ---
// Adam with decoupled weight decay
class AdamW : public Optimizer {
    struct State {
        std::vector<float> m;
        std::vector<float> v;
    };
    std::unordered_map<void*, State> states;
    
    double beta1, beta2, eps, weight_decay;
    int t; 
};

class RMSprop : public Optimizer {
    struct State {
        std::vector<float> v; // square average
    };
    std::unordered_map<void*, State> states;
    double alpha, eps;
public:
    RMSprop(const std::vector<Tensor*>& p, double learning_rate = 0.01, 
            double alpha_ = 0.99, double epsilon = 1e-8) 
        : Optimizer(p, learning_rate), alpha(alpha_), eps(epsilon) {}

    void step() override {
        for (auto* p : params) {
            if (!p->impl->storage->grad) continue;
            
            size_t n = p->numel();
            void* key = p->impl->storage->data.get();

            if (states.find(key) == states.end()) {
                states[key] = { std::vector<float>(n, 0.0f) };
            }
            State& s = states[key];

            if (p->_dtype() == DType::Float32) {
                float* theta = (float*)p->impl->storage->data.get();
                float* grad  = (float*)p->impl->storage->grad.get();
                float* v = s.v.data();

                #pragma omp parallel for
                for (size_t i = 0; i < n; ++i) {
                    float g = grad[i];
                    // v = alpha * v + (1 - alpha) * g^2
                    v[i] = (float)alpha * v[i] + (1.0f - (float)alpha) * g * g;
                    // theta = theta - lr * g / (sqrt(v) + eps)
                    theta[i] -= (float)lr * g / (std::sqrt(v[i]) + (float)eps);
                }
            }
        }
    }
};
// --- Adagrad Optimizer ---
// Adapts learning rate based on sum of squared gradients so far
// theta = theta - lr * grad / sqrt(sum_sq_grad + eps)
class Adagrad : public Optimizer {
    struct State {
        std::vector<float> sum_sq; 
    };
    std::unordered_map<void*, State> states;
    double eps;
    
public:
    Adagrad(const std::vector<Tensor*>& p, double learning_rate = 0.01, double epsilon = 1e-10) 
        : Optimizer(p, learning_rate), eps(epsilon) {}

    void step() override {
        for (auto* p : params) {
            if (!p->impl->storage->grad) continue;
            
            size_t n = p->numel();
            void* key = p->impl->storage->data.get();

            if (states.find(key) == states.end()) {
                states[key] = { std::vector<float>(n, 0.0f) };
            }
            State& s = states[key];

            if (p->_dtype() == DType::Float32) {
                float* theta = (float*)p->impl->storage->data.get();
                float* grad  = (float*)p->impl->storage->grad.get();
                float* sum_sq = s.sum_sq.data();

                #pragma omp parallel for
                for (size_t i = 0; i < n; ++i) {
                    float g = grad[i];
                    // sum_sq += g^2
                    sum_sq[i] += g * g;
                    // theta = theta - lr * g / (sqrt(sum_sq) + eps)
                    theta[i] -= (float)lr * g / (std::sqrt(sum_sq[i]) + (float)eps);
                }
            }
        }
    }
};
// --- Learning Rate Scheduler ---
class StepLR {
    Optimizer& optim;
    int step_size;
    double gamma;
    int last_epoch;

public:
    StepLR(Optimizer& opt, int step, double g = 0.1) 
        : optim(opt), step_size(step), gamma(g), last_epoch(-1) {}

    void step() {
        last_epoch++;
        if (last_epoch > 0 && last_epoch % step_size == 0) {
            optim.lr *= gamma;
            std::cout << "--- Scheduler: LR set to " << optim.lr << " ---\n";
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