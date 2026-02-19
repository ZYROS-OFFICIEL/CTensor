#pragma once

// Include your engine's headers
#include "core.h"
#include "neuralnet.h"
#include "train_utils.h"
#include "training_utils.h"
#include "check.h"
#include "loss.h"

#include <vector>
#include <string>
#include <iostream>
#include <cmath>

namespace torch {

// =======================================================================
//                          DATALOADER WRAPPER
// =======================================================================
struct Batch {
    Tensor data;
    Tensor target;
};

class DataLoader {
    SimpleDataLoader internal_loader;
public:
    DataLoader(TensorDataset& ds, size_t batch_size, bool shuffle=true) 
        : internal_loader(ds, batch_size, shuffle) {}
        
    size_t size() const { return internal_loader.size(); }

    struct Iterator {
        SimpleDataLoader* ptr;
        Batch current;
        bool done;

        Iterator(SimpleDataLoader* p, bool d) : ptr(p), done(d) {
            if (!done) advance();
        }
        void advance() {
            if (ptr->has_next()) {
                auto p = ptr->next();
                current = {p.first, p.second};
            } else {
                done = true;
            }
        }
        bool operator!=(const Iterator& other) const { return done != other.done; }
        Iterator& operator++() { advance(); return *this; }
        Batch& operator*() { return current; }
    };

    Iterator begin() { internal_loader.reset(); return Iterator(&internal_loader, false); }
    Iterator end()   { return Iterator(&internal_loader, true); }
};

// =======================================================================
//                          TORCH.NN
// =======================================================================

namespace nn {
    using Module = ::Module;
    using Linear = ::Linear;
    using Flatten = ::Flatten;

    // --- Functional API ---
    namespace functional {
        inline Tensor relu(const Tensor& x) { return ::relu(x); }
        inline Tensor sigmoid(const Tensor& x) { return ::sigmoid(x); }
    }

    // --- Loss Functors ---
    struct CrossEntropyLoss {
        Tensor operator()(const Tensor& pred, const Tensor& target) {
            // Automatically fix dimension shape mismatch [B] vs [B, 1] for Loss
            Tensor t_loss = target;
            if (t_loss.shape().size() == 1) t_loss = t_loss.unsqueeze(1);
            return Loss::CrossEntropy(pred, t_loss);
        }
    };

    // --- Parameter Gathering Helper ---
    template<typename... Modules>
    std::vector<Tensor*> combine_params(Modules&... modules) {
        std::vector<Tensor*> all_params;
        auto add_params = [&](auto& m) {
            auto p = m.parameters();
            all_params.insert(all_params.end(), p.begin(), p.end());
        };
        (add_params(modules), ...);
        return all_params;
    }

    // --- Utilities ---
    namespace utils {
        inline void clip_grad_norm_(const std::vector<Tensor*>& params, double max_norm) {
            double total_norm_sq = 0.0;
            for (auto* p : params) {
                if (!p || !p->impl || !p->impl->grad) continue;
                float* g = (float*)p->impl->grad->data->data.get();
                size_t n = p->numel();
                double layer_sum = 0.0;
                for (size_t i = 0; i < n; ++i) layer_sum += g[i] * g[i];
                total_norm_sq += layer_sum;
            }
            double total_norm = std::sqrt(total_norm_sq);
            if (total_norm > max_norm) {
                double scale = max_norm / (total_norm + 1e-6);
                for (auto* p : params) {
                    if (!p || !p->impl || !p->impl->grad) continue;
                    float* g = (float*)p->impl->grad->data->data.get();
                    size_t n = p->numel();
                    for (size_t i = 0; i < n; ++i) g[i] *= (float)scale;
                }
            }
        }
    }

    // --- Init ---
    namespace init {
        inline void kaiming_uniform_(std::vector<Tensor*>& params) {
            ::kaiming_init(params); 
        }
    }
}

// =======================================================================
//                          TORCH.OPTIM
// =======================================================================

namespace optim {
    using SGD = ::SGD;
    using Adam = ::Adam;
    using AdamW = ::AdamW;
    using RMSprop = ::RMSprop;
}

// =======================================================================
//                          METRICS & VISION
// =======================================================================

namespace metrics {
    inline size_t accuracy(const Tensor& pred_logits, const Tensor& target) {
        Tensor p = pred_logits.argmax(1); // [B]
        Tensor t = target;
        // Ensure target is flat for accuracy to avoid broadcasting bugs
        if (t.shape().size() > 1 && t.shape().back() == 1) t = t.flatten();
        
        Tensor match = (p == t).astype(DType::Float32);
        return (size_t)::sum(match).item<float>();
    }
}

namespace vision {
    namespace datasets {
        inline TensorDataset MNIST(const std::string& img_path, const std::string& lbl_path) {
            MNISTData raw_data = load_mnist(img_path, lbl_path);
            TensorDataset ds(raw_data.images, raw_data.labels);
            
            // Standardize dataset internally
            ds.transform = [](const void* src, float* dst, size_t n) {
                const float* s = (const float*)src; 
                for(size_t i=0; i<n; ++i) dst[i] = (s[i] - 0.1307f) / 0.3081f;
            };
            return ds;
        }
    }
}

} 