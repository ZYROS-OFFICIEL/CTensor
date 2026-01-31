#pragma once
#include "module.h"
#include <vector>
#include "neuralnet/dataloader/dataloader.h" 
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
        if (!p->impl) continue;

        if (!p->impl->grad) continue;
            if (p->impl->grad->data->data) {
                size_t nbytes = p->numel() * p->dtype_bytes();
                std::memset(p->impl->grad->data->data.get(), 0, nbytes);
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
            if (!p->impl->grad->data->data) continue;
            
            // Fast path for Float32
            if (p->_dtype() == DType::Float32) {
                float* p_ptr = (float*)p->impl->data->data.get();
                float* g_ptr = (float*)p->impl->grad->data->data.get();
                size_t n = p->numel();
                
                #pragma omp parallel for
                for (size_t i = 0; i < n; ++i) {
                    p_ptr[i] -= (float)lr * g_ptr[i];
                }
            } else {
                // Fallback
                size_t n = p->numel();
                for (size_t i = 0; i < n; ++i) {
                    double p_val = read_scalar_at(p->impl->data->data.get(), i, p->impl->dtype);
                    double g_val = read_scalar_at(p->impl->grad->data->data.get(), i, p->impl->dtype);
                    write_scalar_at(p->impl->data->data.get(), i, p->impl->dtype, p_val - lr * g_val);
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
            if (!p->impl->grad->data->data) continue;
            
            size_t n = p->numel();
            void* key = p->impl->data->data.get();

            // Initialize state if missing
            if (states.find(key) == states.end()) {
                states[key] = { std::vector<float>(n, 0.0f), std::vector<float>(n, 0.0f) };
            }
            
            State& s = states[key];
            
            // Only implementing fast path for Float32
            if (p->_dtype() == DType::Float32) {
                float* theta = (float*)p->impl->data->data.get();
                float* grad  = (float*)p->impl->grad->data->data.get();
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
    
public:
    AdamW(const std::vector<Tensor*>& p, double learning_rate = 0.001, 
          double b1 = 0.9, double b2 = 0.999, double epsilon = 1e-8, double decay = 0.01) 
        : Optimizer(p, learning_rate), beta1(b1), beta2(b2), eps(epsilon), weight_decay(decay), t(0) {}

    void step() override {
        t++;
        for (auto* p : params) {
            if (!p->impl->grad->data->data) continue;
            
            size_t n = p->numel();
            void* key = p->impl->data->data.get();

            if (states.find(key) == states.end()) {
                states[key] = { std::vector<float>(n, 0.0f), std::vector<float>(n, 0.0f) };
            }
            State& s = states[key];
            
            if (p->_dtype() == DType::Float32) {
                float* theta = (float*)p->impl->data->data.get();
                float* grad  = (float*)p->impl->grad->data->data.get();
                float* m = s.m.data();
                float* v = s.v.data();
                
                double bias_correction1 = 1.0 - std::pow(beta1, t);
                double bias_correction2 = 1.0 - std::pow(beta2, t);
                float lr_t = (float)(lr * std::sqrt(bias_correction2) / bias_correction1);

                #pragma omp parallel for
                for (size_t i = 0; i < n; ++i) {
                    float g = grad[i];
                    
                    // Decoupled Weight Decay
                    theta[i] -= (float)(lr * weight_decay * theta[i]);

                    m[i] = (float)beta1 * m[i] + (1.0f - (float)beta1) * g;
                    v[i] = (float)beta2 * v[i] + (1.0f - (float)beta2) * g * g;
                    theta[i] -= lr_t * m[i] / (std::sqrt(v[i]) + (float)eps);
                }
            }
        }
    }
};


// --- Adamax Optimizer ---
// Adam based on infinity norm
class Adamax : public Optimizer {
    struct State {
        std::vector<float> m;
        std::vector<float> u; // exp_avg_sq using max
    };
    std::unordered_map<void*, State> states;
    
    double beta1, beta2, eps;
    int t; 
    
public:
    Adamax(const std::vector<Tensor*>& p, double learning_rate = 0.002, 
           double b1 = 0.9, double b2 = 0.999, double epsilon = 1e-8) 
        : Optimizer(p, learning_rate), beta1(b1), beta2(b2), eps(epsilon), t(0) {}

    void step() override {
        t++;
        for (auto* p : params) {
            if (!p->impl->grad->data->data) continue;
            
            size_t n = p->numel();
            void* key = p->impl->data->data.get();

            if (states.find(key) == states.end()) {
                states[key] = { std::vector<float>(n, 0.0f), std::vector<float>(n, 0.0f) };
            }
            State& s = states[key];
            
            if (p->_dtype() == DType::Float32) {
                float* theta = (float*)p->impl->data->data.get();
                float* grad  = (float*)p->impl->grad->data->data.get();
                float* m = s.m.data();
                float* u = s.u.data();
                
                double bias_correction1 = 1.0 - std::pow(beta1, t);
                float step_size = (float)(lr / bias_correction1);

                #pragma omp parallel for
                for (size_t i = 0; i < n; ++i) {
                    float g = grad[i];
                    m[i] = (float)beta1 * m[i] + (1.0f - (float)beta1) * g;
                    u[i] = std::max((float)beta2 * u[i], std::abs(g)); // infinity norm
                    theta[i] -= step_size * m[i] / (u[i] + (float)eps);
                }
            }
        }
    }
};



// --- NAdam Optimizer ---
// Nesterov-accelerated Adaptive Moment Estimation
class NAdam : public Optimizer {
    struct State {
        std::vector<float> m;
        std::vector<float> v;
    };
    std::unordered_map<void*, State> states;
    
    double beta1, beta2, eps, momentum_decay;
    int t; 
public:
    NAdam(const std::vector<Tensor*>& p, double learning_rate = 0.002, 
          double b1 = 0.9, double b2 = 0.999, double epsilon = 1e-8, double momentum_decay_ = 0.004) 
        : Optimizer(p, learning_rate), beta1(b1), beta2(b2), eps(epsilon), momentum_decay(momentum_decay_), t(0) {}

    void step() override {
        t++;
        for (auto* p : params) {
            if (!p->impl->grad->data->data) continue;
            
            size_t n = p->numel();
            void* key = p->impl->data->data.get();

            if (states.find(key) == states.end()) {
                states[key] = { std::vector<float>(n, 0.0f), std::vector<float>(n, 0.0f) };
            }
            State& s = states[key];
            
            if (p->_dtype() == DType::Float32) {
                float* theta = (float*)p->impl->data->data.get();
                float* grad  = (float*)p->impl->grad->data->data.get();
                float* m = s.m.data();
                float* v = s.v.data();
                
                // NAdam complex momentum decay schedule can be simplified
                // Using standard implementation approximation
                double bias_correction1 = 1.0 - std::pow(beta1, t);
                double bias_correction2 = 1.0 - std::pow(beta2, t);
                
                // Calculate mu_t and mu_t+1 for Nesterov
                // Simplified NAdam often uses just standard beta1
                // We will use standard Nesterov update rule on top of Adam
                
                float step_size = (float)(lr * std::sqrt(bias_correction2) / bias_correction1);
                float b1_t = (float)beta1 * (1.0f - 0.5f * std::pow(0.96f, t * momentum_decay));
                float b1_next = (float)beta1 * (1.0f - 0.5f * std::pow(0.96f, (t + 1) * momentum_decay));

                #pragma omp parallel for
                for (size_t i = 0; i < n; ++i) {
                    float g = grad[i];
                    m[i] = (float)beta1 * m[i] + (1.0f - (float)beta1) * g;
                    v[i] = (float)beta2 * v[i] + (1.0f - (float)beta2) * g * g;
                    
                    // Nesterov Momentum Term
                    float m_hat = (float)beta1 * m[i] / (1.0f - std::pow(beta1, t+1)) + (1.0f - (float)beta1) * g / (1.0f - std::pow(beta1, t)); 
                    // Simplified: Use current m and lookahead g
                    // Standard PyTorch implementation effectively does:
                    float m_nesterov = (float)beta1 * m[i] + (1.0f - (float)beta1) * g;
                    
                    theta[i] -= step_size * m_nesterov / (std::sqrt(v[i]) + (float)eps);
                }
            }
        }
    }
};


// --- RAdam Optimizer ---
// Rectified Adam
class RAdam : public Optimizer {
    struct State {
        std::vector<float> m;
        std::vector<float> v;
    };
    std::unordered_map<void*, State> states;
    
    double beta1, beta2, eps;
    int t; 
    
public:
    RAdam(const std::vector<Tensor*>& p, double learning_rate = 0.001, 
          double b1 = 0.9, double b2 = 0.999, double epsilon = 1e-8) 
        : Optimizer(p, learning_rate), beta1(b1), beta2(b2), eps(epsilon), t(0) {}

    void step() override {
        t++;
        // Calculate rho_inf
        float rho_inf = 2.0f / (1.0f - (float)beta2) - 1.0f;

        for (auto* p : params) {
            if (!p->impl->grad->data->data) continue;
            
            size_t n = p->numel();
            void* key = p->impl->data->data.get();

            if (states.find(key) == states.end()) {
                states[key] = { std::vector<float>(n, 0.0f), std::vector<float>(n, 0.0f) };
            }
            State& s = states[key];
            
            if (p->_dtype() == DType::Float32) {
                float* theta = (float*)p->impl->data->data.get();
                float* grad  = (float*)p->impl->grad->data->data.get();
                float* m = s.m.data();
                float* v = s.v.data();
                
                double bias_correction1 = 1.0 - std::pow(beta1, t);
                double bias_correction2 = 1.0 - std::pow(beta2, t);
                
                // Calculate rho_t
                float rho_t = rho_inf - 2.0f * t * std::pow(beta2, t) / (1.0f - std::pow(beta2, t));

                #pragma omp parallel for
                for (size_t i = 0; i < n; ++i) {
                    float g = grad[i];
                    m[i] = (float)beta1 * m[i] + (1.0f - (float)beta1) * g;
                    v[i] = (float)beta2 * v[i] + (1.0f - (float)beta2) * g * g;
                    
                    float m_hat = m[i] / (float)bias_correction1;
                    
                    if (rho_t > 4.0f) {
                        float v_hat = std::sqrt(v[i] / (float)bias_correction2) + (float)eps;
                        float rect = std::sqrt(((rho_t - 4.0f) * (rho_t - 2.0f) * rho_inf) / ((rho_inf - 4.0f) * (rho_inf - 2.0f) * rho_t));
                        theta[i] -= (float)lr * rect * m_hat / v_hat;
                    } else {
                        theta[i] -= (float)lr * m_hat;
                    }
                }
            }
        }
    }
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
            if (!p->impl->grad->data->data) continue;
            
            size_t n = p->numel();
            void* key = p->impl->data->data.get();

            if (states.find(key) == states.end()) {
                states[key] = { std::vector<float>(n, 0.0f) };
            }
            State& s = states[key];

            if (p->_dtype() == DType::Float32) {
                float* theta = (float*)p->impl->data->data.get();
                float* grad  = (float*)p->impl->grad->data->data.get();
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
            if (!p->impl->grad->data->data) continue;
            
            size_t n = p->numel();
            void* key = p->impl->data->data.get();

            if (states.find(key) == states.end()) {
                states[key] = { std::vector<float>(n, 0.0f) };
            }
            State& s = states[key];

            if (p->_dtype() == DType::Float32) {
                float* theta = (float*)p->impl->data->data.get();
                float* grad  = (float*)p->impl->grad->data->data.get();
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

// --- Lion Optimizer (Google, 2023) ---
// "Symbolic Discovery of Optimization Algorithms"
// Uses sign-based updates. Faster compute (no sqrt/div) and 2x memory efficient (1 state).
class Lion : public Optimizer {
    struct State {
        std::vector<float> m; // Only momentum! No 'v' (variance) needed.
    };
    std::unordered_map<void*, State> states;
    
    double beta1, beta2, weight_decay;
    
public:
    Lion(const std::vector<Tensor*>& p, double learning_rate = 0.0001, 
         double b1 = 0.9, double b2 = 0.99, double decay = 0.01) 
        : Optimizer(p, learning_rate), beta1(b1), beta2(b2), weight_decay(decay) {}

    void step() override {
        for (auto* p : params) {
            if (!p->impl->grad->data->data) continue;
            
            size_t n = p->numel();
            void* key = p->impl->data->data.get();

            if (states.find(key) == states.end()) {
                states[key] = { std::vector<float>(n, 0.0f) };
            }
            State& s = states[key];
            
            // Fast Path: Float32 and Contiguous
            if (p->_dtype() == DType::Float32 && p->is_contiguous()) {
                float* theta = (float*)p->impl->data->data.get();
                float* grad  = (float*)p->impl->grad->data->data.get();
                float* m = s.m.data();
                
                // Pre-calculate constants
                float beta1_f = (float)beta1;
                float beta2_f = (float)beta2;
                float lr_decay = (float)(lr * weight_decay);
                float lr_f = (float)lr;

                #pragma omp parallel for
                for (size_t i = 0; i < n; ++i) {
                    float g = grad[i];
                    float m_t = m[i];

                    // 1. Perform Weight Decay
                    if (weight_decay > 0) {
                        theta[i] -= lr_decay * theta[i];
                    }

                    // 2. Symbolic Update (c_t)
                    // c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                    float c_t = beta1_f * m_t + (1.0f - beta1_f) * g;

                    // 3. Update Model Parameters
                    // theta_t = theta_{t-1} - lr * sign(c_t)
                    // Branchless sign optimization: (c_t > 0) - (c_t < 0)
                    float sign_ct = (c_t > 0.0f) ? 1.0f : ((c_t < 0.0f) ? -1.0f : 0.0f);
                    theta[i] -= lr_f * sign_ct;

                    // 4. Update Momentum
                    // m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
                    m[i] = beta2_f * m_t + (1.0f - beta2_f) * g;
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
        
        const float* out_ptr = (const float*)output.impl->data->data.get();
        const int32_t* tgt_ptr = (const int32_t*)target.impl->data->data.get();
        
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