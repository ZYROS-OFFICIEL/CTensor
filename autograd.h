#pragma once
#include <unordered_map>
#include <functional>
#include <stack>
#include <set>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>
#include "tensor1.h"
#include "opsmp.h"

// -------------------- helpers --------------------
inline void check_index_in_storage(const Tensorimpl* impl, size_t idx, const char* ctx) {
    if (!impl || !impl->storage) {
        std::cerr << ctx << ": missing impl/storage\n";
        return;
    }
    if (idx >= impl->storage->size) {
        std::cerr << "OOB " << ctx << ": idx=" << idx
                  << " offset=" << impl->offset
                  << " storage->size=" << impl->storage->size
                  << " ndim=" << impl->ndim << " dtype_bytes=" << impl->storage->size
                  << "\n";
        throw std::runtime_error("index out of underlying storage bounds");
    }
}

// ensure grad buffer exists on tensor; if zero=true fill with zeros
void ensure_grad_buffer(Tensor &t, bool zero = false);

// Helper: create a tensor whose DATA is copied from self.grad
Tensor tensor_from_grad(const Tensor& self);
// copy .data -> .grad (allocate grad buffer and copy values)
static void copy_data_to_grad(Tensor &t);

// reduce `t` by summing over axes but keeping dims
static Tensor reduce_sum_axes_keepdims(Tensor t, std::vector<int> axes);

// fetch dimension value of `target` as if left-padded to `nd` dimensions
static size_t dim_in_padded(const Tensor& target, size_t nd, size_t idx);

// accumulate gradient from grad_src into target (broadcast-aware)
void accumulate_grad(Tensor& target, const Tensor& grad_src);

// ------------------ GradFn base ------------------
struct GradFn {
    std::vector<Tensor> parents;              // used for DFS/topo traversal
    virtual void backward(const Tensor& self) = 0; // self is the tensor whose grad is in self.impl->storage->grad
    virtual ~GradFn() = default;
};

// ------------------ Backward nodes ------------------
struct GradAdd : GradFn {
    Tensor a, b;
    GradAdd(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a, b}; }
    void backward(const Tensor& self) override {
        if (!self.impl->storage->grad) throw std::runtime_error("GradAdd: missing self grad");

        Tensor grad_self = tensor_from_grad(self); 

        if (a.requires_grad()) accumulate_grad(a, grad_self);
        if (b.requires_grad()) accumulate_grad(b, grad_self);
    }
};

struct GradSub : GradFn {
    Tensor a, b;
    GradSub(const Tensor& a_, const Tensor& b_);
    void backward(const Tensor& self) override;
};

struct GradMul : GradFn {
    Tensor a, b;
    GradMul(const Tensor& a_, const Tensor& b_);
    void backward(const Tensor& self) override;
};


struct GradDiv : GradFn {
    Tensor a, b;
    GradDiv(const Tensor& a_, const Tensor& b_);
    void backward(const Tensor& self) override;
};

struct GradPow : GradFn {
    Tensor a, b;
    GradPow(const Tensor& a_, const Tensor& b_);
    void backward(const Tensor& self) override;
};

struct GradMatMul : GradFn {
    Tensor a, b;
    GradMatMul(const Tensor& a_, const Tensor& b_);
    void backward(const Tensor& self) override;
};

struct GradAbs : GradFn {
    Tensor t;
    GradAbs(const Tensor& t_) : t(t_) { parents = {t_}; }

    void backward(const Tensor& self) override;
};

struct GradSum : GradFn {
    Tensor t;
    int dim;
    GradSum(const Tensor& t_, int dim_) : t(t_), dim(dim_) { parents = {t}; }

    void backward(const Tensor& self) override ;
};
struct GradMean : GradFn {
    Tensor t;
    double scale;
    GradMean(const Tensor& t_, double scale_) : t(t_), scale(scale_) {
        parents = {t};
    }
    void backward(const Tensor& self) override ;
};

struct GradLn : GradFn {
    Tensor t;
    GradLn(const Tensor& t_) : t(t_) { parents = {t}; }

    void backward(const Tensor& self) override;
};
struct GradExp : GradFn {
    Tensor t;
    GradExp(const Tensor& t_) : t(t_) { parents = {t}; }

    void backward(const Tensor& self) override;
};
struct GradSqrt : GradFn {
    Tensor t;
    GradSqrt(const Tensor& t_) : t(t_) { parents = {t}; }

    void backward(const Tensor& self) override;
};
struct GradSin : GradFn {
    Tensor t;
    GradSin(const Tensor& t_) : t(t_) { parents = {t}; }

    void backward(const Tensor& self) override;
};
struct GradASin : GradFn {
    Tensor t;
    GradASin(const Tensor& t_) : t(t_) { parents = {t}; }

    void backward(const Tensor& self) override;
};
struct GradSinH : GradFn {
    Tensor t;
    GradSinH(const Tensor& t_) : t(t_) { parents = {t}; }

    void backward(const Tensor& self) override;
};
struct GradCos : GradFn {
    Tensor t;
    GradCos(const Tensor& t_) : t(t_) { parents = {t}; }

    void backward(const Tensor& self) override;
};
struct GradACos : GradFn {
    Tensor t;
    GradACos(const Tensor& t_) : t(t_) { parents = {t}; }

    void backward(const Tensor& self) override;
};
struct GradCosH : GradFn {
    Tensor t;
    GradCosH(const Tensor& t_) : t(t_) { parents = {t}; }

    void backward(const Tensor& self) override;
};
struct GradTan : GradFn {
    Tensor t;
    GradTan(const Tensor& t_) : t(t_) { parents = {t}; }

    void backward(const Tensor& self) override;
};
struct GradATan : GradFn {
    Tensor t;
    GradATan(const Tensor& t_) : t(t_) { parents = {t}; }

    void backward(const Tensor& self) override;
};
struct GradTanH : GradFn {
    Tensor t;
    GradTanH(const Tensor& t_) : t(t_) { parents = {t}; }

    void backward(const Tensor& self) override;
};
struct GradSigmoid : GradFn {
    Tensor t;
    GradSigmoid(const Tensor& t_) : t(t_) { parents = {t}; }

    void backward(const Tensor& self) override;
};
struct GradRelu : GradFn {
    Tensor t;
    GradRelu(const Tensor& t_) : t(t_) { parents = {t}; }

    void backward(const Tensor& self) override;
};
struct GradSoftPlus : GradFn {
    Tensor t;
    GradSoftPlus(const Tensor& t_) : t(t_) { parents = {t}; }

    void backward(const Tensor& self) override;
};

struct GradAddScalar : GradFn {
    Tensor a;
    double scalar;
    GradAddScalar(const Tensor& a_, double scalar_) : a(a_), scalar(scalar_) { parents = {a}; }
    void backward(const Tensor& self) override;
};
struct GradSubScalar : GradFn {
    Tensor a;
    double scalar;
    GradSubScalar(const Tensor& a_, double scalar_) : a(a_), scalar(scalar_) { parents = {a}; }
    void backward(const Tensor& self) override;
};
struct GradSubAfterScalar : GradFn {
    Tensor a;
    double scalar;
    GradSubAfterScalar(const Tensor& a_, double scalar_ ) : a(a_) , scalar(scalar_)  { parents = {a}; }
    void backward(const Tensor& self) override;
};
struct GradMulScalar : GradFn {
    Tensor a;
    double scalar;
    GradMulScalar(const Tensor& a_, double scalar_) : a(a_), scalar(scalar_) { parents = {a}; }
    void backward(const Tensor& self) override;
};
struct GradDivScalar : GradFn {
    Tensor a;
    double scalar;
    GradDivScalar(const Tensor& a_, double scalar_) : a(a_), scalar(scalar_) {
        parents = {a};
    }

    void backward(const Tensor& self) override {
        if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
            throw std::runtime_error("GradDivScalar: missing self grad");
        if (!a.impl || !a.requires_grad()) return;

        Tensor grad_self = tensor_from_grad(self);
        Tensor grad_input = grad_self * (1.0 / scalar);
        accumulate_grad(a, grad_input);
    }
};
struct GradScalarDiv : GradFn {
    Tensor a;
    double scalar;
    GradScalarDiv(const Tensor& a_, double scalar_) : a(a_), scalar(scalar_) {
        parents = {a};
    }

    void backward(const Tensor& self) override {
        if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
            throw std::runtime_error("GradScalarDiv: missing self grad");
        if (!a.impl || !a.requires_grad()) return;

        Tensor grad_self = tensor_from_grad(self);
        size_t n = a.numel_();
        Tensor grad_input(a.shape(), a._dtype(), false);

        for (size_t i = 0; i < n; ++i) {
            double gv = read_scalar_at(grad_self.impl->storage->data.get(), i, grad_self._dtype());
            double va = read_scalar_at(a.impl->storage->data.get(), i, a._dtype());
            write_scalar_at(grad_input.impl->storage->data.get(), i, grad_input._dtype(), -gv * scalar / (va * va));
        }

        accumulate_grad(a, grad_input);
    }
};
struct GradPowScalar : GradFn {
    Tensor a;
    double scalar;

    GradPowScalar(const Tensor& a_, double scalar_) : a(a_), scalar(scalar_) {
        parents = {a};
    }

    void backward(const Tensor& self) override {
        if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
            throw std::runtime_error("GradPowScalar: missing self grad");
        if (!a.impl || !a.requires_grad()) return;

        Tensor grad_self = tensor_from_grad(self);
        size_t n = a.numel_();
        Tensor grad_input(a.shape(), a._dtype(), false);

        for (size_t i = 0; i < n; ++i) {
            double gv = read_scalar_at(grad_self.impl->storage->data.get(), i, grad_self._dtype());
            double va = read_scalar_at(a.impl->storage->data.get(), i, a._dtype());
            write_scalar_at(grad_input.impl->storage->data.get(), i, grad_input._dtype(),
                            gv * scalar * std::pow(va, scalar - 1.0));
        }

        accumulate_grad(a, grad_input);
    }
};
struct GradScalarPow : GradFn {
    Tensor a;
    double scalar;

    GradScalarPow(const Tensor& a_, double scalar_) : a(a_), scalar(scalar_) {
        parents = {a};
    }

    void backward(const Tensor& self) override {
        if (!self.impl || !self.impl->storage || !self.impl->storage->grad)
            throw std::runtime_error("GradScalarPow: missing self grad");
        if (!a.impl || !a.requires_grad()) return;

        Tensor grad_self = tensor_from_grad(self);
        size_t n = a.numel_();
        Tensor grad_input(a.shape(), a._dtype(), false);

        for (size_t i = 0; i < n; ++i) {
            double gv = read_scalar_at(grad_self.impl->storage->data.get(), i, grad_self._dtype());
            double va = read_scalar_at(a.impl->storage->data.get(), i, a._dtype());
            write_scalar_at(grad_input.impl->storage->data.get(), i, grad_input._dtype(),
                            gv * std::pow(scalar, va) * std::log(scalar));
        }

        accumulate_grad(a, grad_input);
    }
};

struct GradPermute : GradFn {
    Tensor t;
    std::vector<size_t> forward_dims; // The permutation used in forward
    std::vector<size_t> reverse_dims; // The inverse permutation for backward

    GradPermute(const Tensor& t_, std::vector<size_t> dims_) 
        : t(t_), forward_dims(dims_) { 
        parents = {t};
        
        // Calculate inverse permutation to restore original shape
        reverse_dims.resize(dims_.size());
        for (size_t i = 0; i < dims_.size(); ++i) {
            reverse_dims[dims_[i]] = i;
        }
    }

    void backward(const Tensor& self) override;
};
struct GradReshape : GradFn {
    Tensor t;
    std::vector<size_t> old_shape;
    
    GradReshape(const Tensor& t_, std::vector<size_t> old_) : t(t_), old_shape(old_) { 
        parents = {t}; 
    }

    void backward(const Tensor& self) override {
        if (!self.impl->storage->grad) throw std::runtime_error("GradReshape: missing self grad");
        if (!t.requires_grad()) return;
        
        // 1. Get gradient of output (which is the reshaped version)
        Tensor grad_output = tensor_from_grad(self);
        
        // FIX: Verify number of elements matches
        size_t n_out = grad_output.numel();
        size_t n_in = 1;
        for (auto s : old_shape) n_in *= s;
        
        if (n_out != n_in) {
            // This happens if reshape logic was wrong. 
            // Silent corruption check.
            std::cerr << "CRITICAL ERROR: GradReshape size mismatch. Out=" << n_out << " In=" << n_in << "\n";
            return; 
        }

        // 2. Reshape it back to input shape
        // We CANNOT just call .reshape() on grad_output if it's a non-contiguous view 
        // derived from a complex chain.
        // We force a contiguous copy first to be safe.
        // Using opsmp::add_scalar_mp(..., 0.0) is the standard way we are using to copy.
        // BUT to break the cycle, let's manually create the tensor.
        
        Tensor grad_input;
        
        // Option A: If we trust .reshape() handles strides (it usually doesn't for arbitrary views):
        // grad_input = grad_output.reshape(old_shape);
        
        // Option B (Safer): Create new tensor with target shape, copy data.
        grad_input = Tensor::zeros(old_shape, grad_output._dtype(), false);
        
        // Copy data from grad_output to grad_input
        // Since they have same numel, we can flatten copy IF grad_output is contiguous.
        // If grad_output is NOT contiguous, we need a strided copy.
        // Fortunately, accumulate_grad handles this!
        
        // So actually, we just need to view grad_output as the old shape.
        // If reshaping fails, it's because strides are incompatible.
        
        // ROBUST FIX:
        // 1. Force contiguous (deep copy)
        Tensor grad_output_contig = add_scalar_mp(grad_output, 0.0);
        
        // 2. Now safe to reshape metadata
        grad_input = grad_output_contig.reshape(old_shape);

        accumulate_grad(t, grad_input);
    }
};
// ------------------ topo sort helper ------------------
static void topo_sort_from(const Tensor& root, std::vector<Tensor>& topo);

// ------------------ backward ------------------
void backward(Tensor& loss);
Tensor grad_of(const Tensor& t) ;

