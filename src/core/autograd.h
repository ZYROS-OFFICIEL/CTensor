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
#include "tensor.h"
#include "ops_dispatch.h"

// -------------------- helpers --------------------
inline void check_index_in_storage(const Tensorimpl* impl, size_t idx, const char* ctx) {
    if (!impl || !impl->data) {
        std::cerr << ctx << ": missing impl/data\n";
        return;
    }
    if (idx >= impl->data->size) {
        std::cerr << "OOB " << ctx << ": idx=" << idx
                  << " offset=" << impl->offset
                  << " storage->size=" << impl->data->size
                  << " ndim=" << impl->ndim
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

// accumulate gradient from grad_src into target (broadcast-aware)
void accumulate_grad(Tensor& target, const Tensor& grad_src);

// ------------------ GradFn base ------------------
struct GradFn {
    std::vector<Tensor> parents;              // used for DFS/topo traversal
    virtual void backward(const Tensor& self) = 0; 
    virtual std::string name() const { return "GradFn"; }
    virtual ~GradFn() = default;
};

// ------------------ Backward nodes ------------------
struct GradAdd : GradFn {
    Tensor a, b;
    GradAdd(const Tensor& a_, const Tensor& b_) : a(a_), b(b_) { parents = {a, b}; }
    void backward(const Tensor& self) override ;
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
    int dim; 
    GradMean(const Tensor& t_, double scale_, int dim_ = -1) : t(t_), scale(scale_), dim(dim_) {
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
struct GradSinh : GradFn {
    Tensor t;
    GradSinh(const Tensor& t_) : t(t_) { parents = {t}; }

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
struct GradCosh : GradFn {
    Tensor t;
    GradCosh(const Tensor& t_) : t(t_) { parents = {t}; }

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
struct GradTanh : GradFn {
    Tensor t;
    GradTanh(const Tensor& t_) : t(t_) { parents = {t}; }

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
struct GradSoftplus : GradFn {
    Tensor t;
    GradSoftplus(const Tensor& t_) : t(t_) { parents = {t}; }

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
    Tensor a; double s;
    GradDivScalar(const Tensor& a_, double s_) : a(a_), s(s_) { parents = {a}; }
    void backward(const Tensor& self) override;
};
struct GradScalarDiv : GradFn {
    Tensor a; double s;
    GradScalarDiv(const Tensor& a_, double s_) : a(a_), s(s_) { parents = {a}; }
    void backward(const Tensor& self) override;
};
struct GradPowScalar : GradFn {
    Tensor a; double s;
    GradPowScalar(const Tensor& a_, double s_) : a(a_), s(s_) { parents = {a}; }
    void backward(const Tensor& self) override;
};
struct GradScalarPow : GradFn {
    Tensor a;
    double scalar;
    GradScalarPow(const Tensor& a_, double scalar_) : a(a_), scalar(scalar_) {
        parents = {a};
    }
    void backward(const Tensor& self) override; 
};

struct GradPermute : GradFn {
    Tensor t;
    std::vector<size_t> forward_dims; 
    std::vector<size_t> reverse_dims; 

    GradPermute(const Tensor& t_, const std::vector<size_t>& dims_);

    void backward(const Tensor& self) override;
};
struct GradReshape : GradFn {
    Tensor t;
    std::vector<size_t> old_shape;
    
    GradReshape(const Tensor& t_, std::vector<size_t> old_) : t(t_), old_shape(old_) { 
        parents = {t}; 
    }

    void backward(const Tensor& self) override;
};

struct GradGather : GradFn {
    Tensor t;      // The source tensor (embeddings or logits)
    Tensor index;  // The indices used
    size_t dim;    // The dimension gathered along

    GradGather(const Tensor& t_, const Tensor& index_, size_t dim_) 
        : t(t_), index(index_), dim(dim_) {
        parents = {t}; // Index usually doesn't require grad in standard layers
    }

    void backward(const Tensor& self) override;
};

// ------------------ backward ------------------
void backward(Tensor& root);