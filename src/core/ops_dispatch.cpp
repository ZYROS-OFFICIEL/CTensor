#include "ops_dispatch.h"
#include "tensor.h"
#include "opsmp.h" 
#include "autograd.h"
#include "ops_avx2_d64.h"
#include "ops_avx2_f32.h"
#include "ops_avx512_f32.h"
#include "ops_avx512_d64.h"

#include <stdexcept>
#include <string>
#include <array>
#include <iostream>
#include <cmath>

#if defined(__GNUC__) || defined(__clang__)
  #if defined(__x86_64__) || defined(__i386__)
    #define HAS_BUILTIN_CPU_SUPPORTS 1
  #endif
#endif

// CPU Feature Detection
static inline bool cpu_has_avx2() {
#ifdef HAS_BUILTIN_CPU_SUPPORTS
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}
static inline bool cpu_has_avx512f() {
#ifdef HAS_BUILTIN_CPU_SUPPORTS
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

// ========================================================================
//                     STATIC DISPATCH REGISTRY
// ========================================================================

enum class BinaryOp {
    ADD, SUB, MUL, DIV, POW, MATMUL,
    LT, LE, GT, GE, EQ, NE,
    _COUNT
};

using BinaryKernelFn = Tensor(*)(const Tensor&, const Tensor&);

struct DispatchTable {
    // [Op][DType] -> Function Pointer
    std::array<std::array<BinaryKernelFn, 9>, (size_t)BinaryOp::_COUNT> table;

    DispatchTable() {
        // 1. Initialize everything with Multi-Threading (MP) Fallback
        for (auto& row : table) row.fill(nullptr);

        // Float32 Defaults
        table[(int)BinaryOp::ADD][(int)DType::Float32] = add_mp;
        table[(int)BinaryOp::SUB][(int)DType::Float32] = diff_mp;
        table[(int)BinaryOp::MUL][(int)DType::Float32] = mult_mp;
        table[(int)BinaryOp::DIV][(int)DType::Float32] = div_mp;
        table[(int)BinaryOp::POW][(int)DType::Float32] = pow_mp;
        table[(int)BinaryOp::MATMUL][(int)DType::Float32] = matmul_mp;
        
        table[(int)BinaryOp::LT][(int)DType::Float32] = lt_mp;
        table[(int)BinaryOp::LE][(int)DType::Float32] = le_mp;
        table[(int)BinaryOp::GT][(int)DType::Float32] = gt_mp;
        table[(int)BinaryOp::GE][(int)DType::Float32] = ge_mp;
        table[(int)BinaryOp::EQ][(int)DType::Float32] = eq_mp;
        table[(int)BinaryOp::NE][(int)DType::Float32] = neq_mp;

        // Double64 Defaults
        table[(int)BinaryOp::ADD][(int)DType::Double64] = add_mp;
        table[(int)BinaryOp::SUB][(int)DType::Double64] = diff_mp;
        table[(int)BinaryOp::MUL][(int)DType::Double64] = mult_mp;
        table[(int)BinaryOp::DIV][(int)DType::Double64] = div_mp;
        table[(int)BinaryOp::POW][(int)DType::Double64] = pow_mp;
        table[(int)BinaryOp::MATMUL][(int)DType::Double64] = matmul_mp;

        // Int32 Defaults (Generic MP)
        table[(int)BinaryOp::ADD][(int)DType::Int32] = add_mp;
        table[(int)BinaryOp::SUB][(int)DType::Int32] = diff_mp;
        table[(int)BinaryOp::MUL][(int)DType::Int32] = mult_mp;
        table[(int)BinaryOp::DIV][(int)DType::Int32] = div_mp;
        table[(int)BinaryOp::POW][(int)DType::Int32] = pow_mp;

        // 2. AVX2 Overrides
        if (cpu_has_avx2()) {
            // Float32
            table[(int)BinaryOp::ADD][(int)DType::Float32] = add_avx2_f32;
            table[(int)BinaryOp::SUB][(int)DType::Float32] = sub_avx2_f32;
            table[(int)BinaryOp::MUL][(int)DType::Float32] = mul_avx2_f32;
            table[(int)BinaryOp::DIV][(int)DType::Float32] = div_avx2_f32;
            table[(int)BinaryOp::POW][(int)DType::Float32] = pow_avx2_f32;
            table[(int)BinaryOp::MATMUL][(int)DType::Float32] = matmul_avx2_f32;
            
            table[(int)BinaryOp::LT][(int)DType::Float32] = lt_avx2_f32;
            table[(int)BinaryOp::LE][(int)DType::Float32] = le_avx2_f32;
            table[(int)BinaryOp::GT][(int)DType::Float32] = gt_avx2_f32;
            table[(int)BinaryOp::GE][(int)DType::Float32] = ge_avx2_f32;
            table[(int)BinaryOp::EQ][(int)DType::Float32] = eq_avx2_f32;
            table[(int)BinaryOp::NE][(int)DType::Float32] = ne_avx2_f32;

            // Double64
            table[(int)BinaryOp::ADD][(int)DType::Double64] = add_avx2_d64;
            table[(int)BinaryOp::SUB][(int)DType::Double64] = sub_avx2_d64;
            table[(int)BinaryOp::MUL][(int)DType::Double64] = mul_avx2_d64;
            table[(int)BinaryOp::DIV][(int)DType::Double64] = div_avx2_d64;
            table[(int)BinaryOp::POW][(int)DType::Double64] = pow_avx2_d64;
            table[(int)BinaryOp::MATMUL][(int)DType::Double64] = matmul_avx2_d64;
        }

        // 3. AVX-512 Overrides (Highest Priority)
        if (cpu_has_avx512f()) {
            // Float32
            table[(int)BinaryOp::ADD][(int)DType::Float32] = add_avx512_f32;
            table[(int)BinaryOp::SUB][(int)DType::Float32] = sub_avx512_f32;
            table[(int)BinaryOp::MUL][(int)DType::Float32] = mul_avx512_f32;
            table[(int)BinaryOp::DIV][(int)DType::Float32] = div_avx512_f32;
            table[(int)BinaryOp::POW][(int)DType::Float32] = pow_avx512_f32;
            table[(int)BinaryOp::MATMUL][(int)DType::Float32] = matmul_avx512_f32;
            
            table[(int)BinaryOp::LT][(int)DType::Float32] = lt_avx512_f32;
            table[(int)BinaryOp::LE][(int)DType::Float32] = le_avx512_f32;
            table[(int)BinaryOp::GT][(int)DType::Float32] = gt_avx512_f32;
            table[(int)BinaryOp::GE][(int)DType::Float32] = ge_avx512_f32;
            table[(int)BinaryOp::EQ][(int)DType::Float32] = eq_avx512_f32;
            table[(int)BinaryOp::NE][(int)DType::Float32] = ne_avx512_f32;
            
             // Double64
            table[(int)BinaryOp::ADD][(int)DType::Double64] = add_avx512_d64;
            table[(int)BinaryOp::SUB][(int)DType::Double64] = sub_avx512_d64;
            table[(int)BinaryOp::MUL][(int)DType::Double64] = mul_avx512_d64;
            table[(int)BinaryOp::DIV][(int)DType::Double64] = div_avx512_d64;
            table[(int)BinaryOp::POW][(int)DType::Double64] = pow_avx512_d64;
            table[(int)BinaryOp::MATMUL][(int)DType::Double64] = matmul_avx512_d64;
        }
    }
};

static const DispatchTable& get_registry() {
    static DispatchTable table;
    return table;
}

// ---------- Helpers ----------
inline void ensure_same_device(const Tensor &a, const Tensor &b, const char* opname) {
    if (a.device() != b.device()) throw std::runtime_error(std::string(opname) + ": tensors must be on same device");
}

template <typename GradFnType, typename OpFunc>
Tensor run_binary_op(const Tensor& a, const Tensor& b, OpFunc op) {
    Tensor out = op();
    if (a.requires_grad() || b.requires_grad()) {
        out.requires_grad_(true);
        out.impl->grad_fn = std::make_shared<GradFnType>(a, b);
    }
    return out;
}

// ========================================================================
//                     DISPATCHER IMPL
// ========================================================================

Tensor dispatch_binary(BinaryOp op, const Tensor& a, const Tensor& b, const char* name) {
    ensure_same_device(a, b, name);

    // 1. SCALAR SHORT-CIRCUIT
    if (a.numel() == 1 && b.numel() == 1) {
        double va = a.read_scalar(0);
        double vb = b.read_scalar(0);
        double res = 0.0;
        bool is_bool = false;

        switch(op) {
            case BinaryOp::ADD: res = va + vb; break;
            case BinaryOp::SUB: res = va - vb; break;
            case BinaryOp::MUL: res = va * vb; break;
            case BinaryOp::DIV: res = va / vb; break;
            case BinaryOp::POW: res = std::pow(va, vb); break;
            case BinaryOp::LT:  res = (va < vb); is_bool = true; break;
            case BinaryOp::LE:  res = (va <= vb); is_bool = true; break;
            case BinaryOp::GT:  res = (va > vb); is_bool = true; break;
            case BinaryOp::GE:  res = (va >= vb); is_bool = true; break;
            case BinaryOp::EQ:  res = (va == vb); is_bool = true; break;
            case BinaryOp::NE:  res = (va != vb); is_bool = true; break;
            default: break; 
        }

        if (op != BinaryOp::MATMUL) {
            Tensor out({1}, is_bool ? DType::Bool : a._dtype());
            out.write_scalar(0, res);
            return out;
        }
    }

    // 2. Main Dispatch
    if (a.device().is_cpu()) {
        auto fn = get_registry().table[(int)op][(int)a._dtype()];
        if (!fn) throw std::runtime_error(std::string(name) + ": unsupported dtype or op not registered");
        return fn(a, b);
    }

    throw std::runtime_error(std::string(name) + ": unsupported device");
}

// ========================================================================
//                     PUBLIC API
// ========================================================================

Tensor add(const Tensor &a, const Tensor &b) {
    return run_binary_op<GradAdd>(a, b, [&](){ return dispatch_binary(BinaryOp::ADD, a, b, "add"); });
}

Tensor sub(const Tensor &a, const Tensor &b) {
    return run_binary_op<GradSub>(a, b, [&](){ return dispatch_binary(BinaryOp::SUB, a, b, "sub"); });
}

Tensor mul(const Tensor &a, const Tensor &b) {
    return run_binary_op<GradMul>(a, b, [&](){ return dispatch_binary(BinaryOp::MUL, a, b, "mul"); });
}

Tensor div(const Tensor &a, const Tensor &b) {
    return run_binary_op<GradDiv>(a, b, [&](){ return dispatch_binary(BinaryOp::DIV, a, b, "div"); });
}

Tensor pow(const Tensor &a, const Tensor &b) {
    return run_binary_op<GradPow>(a, b, [&](){ return dispatch_binary(BinaryOp::POW, a, b, "pow"); });
}

Tensor matmul(const Tensor &a, const Tensor &b) {
    return run_binary_op<GradMatMul>(a, b, [&](){ return dispatch_binary(BinaryOp::MATMUL, a, b, "matmul"); });
}

// ========================================================================
//                     SCALAR OPS
// ========================================================================

template <typename GradFnType>
Tensor run_scalar_op(const Tensor& a, double s, Tensor(*op)(const Tensor&, double)) {
    Tensor out = op(a, s);
    if (a.requires_grad()) {
        out.requires_grad_(true);
        out.impl->grad_fn = std::make_shared<GradFnType>(a, s);
    }
    return out;
}

Tensor add_scalar(const Tensor &a, double scalar) { return run_scalar_op<GradAddScalar>(a, scalar, add_scalar_mp); }
Tensor sub_scalar(const Tensor &a, double scalar) { return run_scalar_op<GradSubScalar>(a, scalar, sub_scalar_mp); }
Tensor mul_scalar(const Tensor &a, double scalar) { return run_scalar_op<GradMulScalar>(a, scalar, mult_scalar_mp); }
Tensor div_scalar(const Tensor &a, double scalar) { return run_scalar_op<GradDivScalar>(a, scalar, div_scalar_mp); }
Tensor pow_scalar(const Tensor &a, double scalar) { return run_scalar_op<GradPowScalar>(a, scalar, pow_scalar_mp); }

Tensor sub_scalar_rev(double scalar, const Tensor &a) { 
    Tensor out = sub_afterscalar_mp(scalar, a);
    if (a.requires_grad()) {
        out.requires_grad_(true);
        out.impl->grad_fn = std::make_shared<GradSubAfterScalar>(a, scalar);
    }
    return out;
}
Tensor div_scalar_rev(double scalar, const Tensor &a) { 
    Tensor out = scalar_div_mp(scalar, a);
    if (a.requires_grad()) {
        out.requires_grad_(true);
        out.impl->grad_fn = std::make_shared<GradScalarDiv>(a, scalar);
    }
    return out;
}
Tensor pow_scalar_rev(double scalar, const Tensor &a) { 
    Tensor out = scalar_pow_mp(scalar, a);
    if (a.requires_grad()) {
        out.requires_grad_(true);
        out.impl->grad_fn = std::make_shared<GradScalarPow>(a, scalar);
    }
    return out;
}

// ========================================================================
//                     UNARY OPS
// ========================================================================

#define IMPLEMENT_UNARY_OP(NAME, GRAD_CLASS, FUNC_MP, FUNC_AVX2, FUNC_AVX512) \
Tensor NAME(const Tensor &a) { \
    Tensor out; \
    if (a.device().is_cpu()) { \
        switch (a._dtype()) { \
            case DType::Float32: \
                if (cpu_has_avx512f()) out = FUNC_AVX512 ## _f32(a); \
                else if (cpu_has_avx2())    out = FUNC_AVX2 ## _f32(a); \
                else out = FUNC_MP(a); \
                break; \
            case DType::Double64: \
                if (cpu_has_avx512f()) out = FUNC_AVX512 ## _d64(a); \
                else if (cpu_has_avx2())    out = FUNC_AVX2 ## _d64(a); \
                else out = FUNC_MP(a); \
                break; \
            default: out = FUNC_MP(a); break; \
        } \
    } else { \
       throw std::runtime_error(std::string(#NAME) + ": unsupported device"); \
    } \
    if (a.requires_grad()) { \
        out.requires_grad_(true); \
        out.impl->grad_fn = std::make_shared<GRAD_CLASS>(a); \
    } \
    return out; \
}

IMPLEMENT_UNARY_OP(abs, GradAbs, abs_mp, abs_avx2, abs_avx512)
IMPLEMENT_UNARY_OP(log, GradLn, ln_mp, ln_avx2, ln_avx512)
IMPLEMENT_UNARY_OP(ln, GradLn, ln_mp, ln_avx2, ln_avx512)
IMPLEMENT_UNARY_OP(exp, GradExp, exp_mp, exp_avx2, exp_avx512)
IMPLEMENT_UNARY_OP(sqrt, GradSqrt, sqrt_mp, sqrt_avx2, sqrt_avx512)
IMPLEMENT_UNARY_OP(sin, GradSin, sin_mp, sin_avx2, sin_avx512)
IMPLEMENT_UNARY_OP(asin, GradASin, asin_mp, asin_avx2, asin_avx512)
IMPLEMENT_UNARY_OP(cos, GradCos, cos_mp, cos_avx2, cos_avx512)
IMPLEMENT_UNARY_OP(acos, GradACos, acos_mp, acos_avx2, acos_avx512)
IMPLEMENT_UNARY_OP(tan, GradTan, tan_mp, tan_avx2, tan_avx512)
IMPLEMENT_UNARY_OP(atan, GradATan, atan_mp, atan_avx2, atan_avx512)
IMPLEMENT_UNARY_OP(tanh, GradTanh, tanh_mp, tanh_avx2, tanh_avx512)
IMPLEMENT_UNARY_OP(sinh, GradSinh, sinh_mp, sinh_avx2, sinh_avx512)
IMPLEMENT_UNARY_OP(cosh, GradCosh, cosh_mp, cosh_avx2, cosh_avx512)
IMPLEMENT_UNARY_OP(sigmoid, GradSigmoid, sigmoid_mp, sigmoid_avx2, sigmoid_avx512)
IMPLEMENT_UNARY_OP(relu, GradRelu, Relu_mp, relu_avx2, relu_avx512) 
IMPLEMENT_UNARY_OP(softplus, GradSoftplus, softplus_mp, softplus_avx2, softplus_avx512)

Tensor Relu(const Tensor &a){ return relu(a); }

// ========================================================================
//                     COMPARISONS
// ========================================================================

Tensor lt(const Tensor &a, const Tensor &b) { return dispatch_binary(BinaryOp::LT, a, b, "lt"); }
Tensor le(const Tensor &a, const Tensor &b) { return dispatch_binary(BinaryOp::LE, a, b, "le"); }
Tensor gt(const Tensor &a, const Tensor &b) { return dispatch_binary(BinaryOp::GT, a, b, "gt"); }
Tensor ge(const Tensor &a, const Tensor &b) { return dispatch_binary(BinaryOp::GE, a, b, "ge"); }
Tensor eq(const Tensor &a, const Tensor &b) { return dispatch_binary(BinaryOp::EQ, a, b, "eq"); }
Tensor ne(const Tensor &a, const Tensor &b) { return dispatch_binary(BinaryOp::NE, a, b, "ne"); }

Tensor lt(const Tensor &a, double b) { return lt_mp(a, b); }
Tensor le(const Tensor &a, double b) { return le_mp(a, b); }
Tensor gt(const Tensor &a, double b) { return gt_mp(a, b); }
Tensor ge(const Tensor &a, double b) { return ge_mp(a, b); }
Tensor eq(const Tensor &a, double b) { return eq_mp(a, b); }
Tensor ne(const Tensor &a, double b) { return neq_mp(a, b); }

// ========================================================================
//                     REDUCTIONS
// ========================================================================

Tensor sum(const Tensor &a, int dim) {
    Tensor out;
    if (a.device().is_cpu()) {
        switch (a._dtype()) {
            case DType::Float32:
                if (cpu_has_avx512f()) out = sum_avx512_f32(a,dim);
                else if (cpu_has_avx2()) out = sum_avx2_f32(a,dim);
                else out = sum_mp(a,dim);
                break;
            case DType::Double64:
                if (cpu_has_avx512f()) out = sum_avx512_d64(a,dim);
                else if (cpu_has_avx2()) out = sum_avx2_d64(a,dim);
                else out = sum_mp(a,dim);
                break;
            default: out = sum_mp(a,dim); break;
        }
    } else throw std::runtime_error("sum: unsupported device");

    if (a.requires_grad()) {
        out.requires_grad_(true);
        out.impl->grad_fn = std::make_shared<GradSum>(a, dim);
    }
    return out;
}

Tensor mean(const Tensor &a, int dim) {
    Tensor s = sum(a, dim);
    double N = (double)a.numel();
    if (dim != -1 && dim < (int)a.shape().size()) N = (double)a.shape()[dim];
    return mul_scalar(s, 1.0 / N);
}

Tensor max(const Tensor &a, int dim) {
    if (a.device().is_cpu()) {
        switch (a._dtype()) {
            case DType::Float32:
                if (cpu_has_avx512f()) return max_avx512_f32(a,dim);
                if (cpu_has_avx2())    return max_avx2_f32(a,dim);
                return max_mp(a,dim);
            case DType::Double64:
                if (cpu_has_avx512f()) return max_avx512_d64(a,dim);
                if (cpu_has_avx2())    return max_avx2_d64(a,dim);
                return max_mp(a,dim);
            default: return max_mp(a,dim);
        }
    }
    throw std::runtime_error("max: unsupported device");
}

Tensor min(const Tensor &a, int dim) {
    if (a.device().is_cpu()) {
        switch (a._dtype()) {
            case DType::Float32:
                if (cpu_has_avx512f()) return min_avx512_f32(a,dim);
                if (cpu_has_avx2())    return min_avx2_f32(a,dim);
                return min_mp(a,dim);
            case DType::Double64:
                if (cpu_has_avx512f()) return min_avx512_d64(a,dim);
                if (cpu_has_avx2())    return min_avx2_d64(a,dim);
                return min_mp(a,dim);
            default: return min_mp(a,dim);
        }
    }
    throw std::runtime_error("min: unsupported device");
}

Tensor cat(const std::vector<Tensor>& tensors, size_t dim) {
    return cat_mp(tensors, dim);
}

// --- Operators ---

Tensor& operator+=(Tensor& a, const Tensor& b) { a = add(a, b); return a; }
Tensor& operator+=(Tensor& a, double scalar) { a = add_scalar(a, scalar); return a; }
Tensor& operator-=(Tensor& a, const Tensor& b) { a = sub(a, b); return a; }
Tensor& operator-=(Tensor& a, double scalar) { a = sub_scalar(a, scalar); return a; }
Tensor& operator*=(Tensor& a, const Tensor& b) { a = mul(a, b); return a; }
Tensor& operator*=(Tensor& a, double scalar) { a = mul_scalar(a, scalar); return a; }
Tensor& operator/=(Tensor& a, const Tensor& b) { a = div(a, b); return a; }
Tensor& operator/=(Tensor& a, double scalar) { a = div_scalar(a, scalar); return a; }
Tensor& operator^=(Tensor& a, const Tensor& b) { a = pow(a, b); return a; }
Tensor& operator^=(Tensor& a, double scalar) { a = pow_scalar(a, scalar); return a; }

Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
Tensor operator+(const Tensor& a, double scalar) { return add_scalar(a, scalar); }
Tensor operator+(double scalar, const Tensor& a) { return add_scalar(a, scalar); }
Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return div(a, b); }
Tensor operator^(const Tensor& a, const Tensor& b) { return pow(a, b); }