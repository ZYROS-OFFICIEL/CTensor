#include "dispatch.h"
#include "tensor.h"
#include "opsmp.h" 
#include "autograd.h"
#include "ops_avx2_d64.h"
#include "ops_avx2_f32.h"
#include "ops_avx512_f32.h"
#include "ops_avx512_d64.h"

#include <stdexcept>
#include <string>

#if defined(__GNUC__) || defined(__clang__)
  #if defined(__x86_64__) || defined(__i386__)
    #define HAS_BUILTIN_CPU_SUPPORTS 1
  #endif
#endif

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

// ---------- Dispatcher helpers ----------
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
// The main scheme: device -> dtype -> arch

Tensor add(const Tensor &a, const Tensor &b) {
    ensure_same_device(a,b,"add");
    return run_binary_op<GradAdd>(a, b, [&](){
        if (a.device().is_cpu()) {
            switch (a._dtype()) {
                case DType::Float32:
                    if (cpu_has_avx512f()) return add_avx512_f32(a,b);
                    if (cpu_has_avx2())    return add_avx2_f32(a,b);
                    return add_mp(a,b);
                case DType::Int32:    return add_mp(a,b);
                case DType::Double64:
                    if (cpu_has_avx512f()) return add_avx512_d64(a,b);
                    if (cpu_has_avx2())    return add_avx2_d64(a,b);
                    return add_mp(a,b);
                default: throw std::runtime_error("add: unsupported dtype");
            }
        }
        throw std::runtime_error("add: unsupported device");
    });
}

Tensor sub(const Tensor &a, const Tensor &b) {
    ensure_same_device(a,b,"sub");
    return run_binary_op<GradSub>(a, b, [&](){
        if (a.device().is_cpu()) {
            switch (a._dtype()) {
                case DType::Float32:
                    if (cpu_has_avx512f()) return sub_avx512_f32(a,b);
                    if (cpu_has_avx2())    return sub_avx2_f32(a,b);
                    return diff_mp(a,b);
                case DType::Int32:    return diff_mp(a,b);
                case DType::Double64:
                    if (cpu_has_avx512f()) return sub_avx512_d64(a,b);
                    if (cpu_has_avx2())    return sub_avx2_d64(a,b);
                    return diff_mp(a,b);
                default: throw std::runtime_error("sub: unsupported dtype");
            }
        }
        throw std::runtime_error("sub: unsupported device");
    });
}

Tensor mul(const Tensor &a, const Tensor &b) {
    ensure_same_device(a,b,"mul");
    return run_binary_op<GradMul>(a, b, [&](){
        if (a.device().is_cpu()) {
            switch (a._dtype()) {
                case DType::Float32:
                    if (cpu_has_avx512f()) return mul_avx512_f32(a,b);
                    if (cpu_has_avx2())    return mul_avx2_f32(a,b);
                    return mult_mp(a,b);
                case DType::Int32:    return mult_mp(a,b);
                case DType::Double64:
                    if (cpu_has_avx512f()) return mul_avx512_d64(a,b);
                    if (cpu_has_avx2())    return mul_avx2_d64(a,b);
                    return mult_mp(a,b);
                default: throw std::runtime_error("mul: unsupported dtype");
            }
        }
        throw std::runtime_error("mul: unsupported device");
    });
}

Tensor div(const Tensor &a, const Tensor &b) {
    ensure_same_device(a,b,"div");
    return run_binary_op<GradDiv>(a, b, [&](){
        if (a.device().is_cpu()) {
            switch (a._dtype()) {
                case DType::Float32:
                    if (cpu_has_avx512f()) return div_avx512_f32(a,b);
                    if (cpu_has_avx2())    return div_avx2_f32(a,b);
                    return div_mp(a,b);
                case DType::Int32:    return div_mp(a,b);
                case DType::Double64:
                    if (cpu_has_avx512f()) return div_avx512_d64(a,b);
                    if (cpu_has_avx2())    return div_avx2_d64(a,b);
                    return div_mp(a,b);
                default: throw std::runtime_error("div: unsupported dtype");
            }
        }
        throw std::runtime_error("div: unsupported device");
    });
}

Tensor pow(const Tensor &a, const Tensor &b) {
    ensure_same_device(a,b,"pow");
    return run_binary_op<GradPow>(a, b, [&](){
        if (a.device().is_cpu()) {
            switch (a._dtype()) {
                case DType::Float32:
                    if (cpu_has_avx512f()) return pow_avx512_f32(a,b);
                    if (cpu_has_avx2())    return pow_avx2_f32(a,b);
                    return pow_mp(a,b);
                case DType::Int32:    return pow_mp(a,b);
                case DType::Double64:
                    if (cpu_has_avx512f()) return pow_avx512_d64(a,b);
                    if (cpu_has_avx2())    return pow_avx2_d64(a,b);
                    return pow_mp(a,b);
                default: throw std::runtime_error("pow: unsupported dtype");
            }
        }
        throw std::runtime_error("pow: unsupported device");
    });
}


Tensor matmul(const Tensor &a, const Tensor &b) {
    ensure_same_device(a,b,"matmul");
    return run_binary_op<GradMatMul>(a, b, [&](){
        if (a.device().is_cpu()) {
            switch (a._dtype()) {
                case DType::Float32:
                    if (cpu_has_avx512f()) return matmul_avx512_f32(a,b);
                    if (cpu_has_avx2())    return matmul_avx2_f32(a,b);
                    return matmul_mp(a,b);
                case DType::Double64:
                    if (cpu_has_avx512f()) return matmul_avx512_d64(a,b);
                    if (cpu_has_avx2())    return matmul_avx2_d64(a,b);
                    return matmul_mp(a,b);
                default: return matmul_mp(a,b);
            }
        }
        throw std::runtime_error("matmul: unsupported device");
    });
}

// Scalar Ops
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
//                           Unary Operations
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
Tensor Relu(const Tensor &a){
    return relu(a);
}

// ========================================================================
//                           Comparisons (No Grad)
// ========================================================================

#define IMPLEMENT_COMPARE_OP(NAME, FUNC_MP, FUNC_AVX2, FUNC_AVX512) \
Tensor NAME(const Tensor &a, const Tensor &b) { \
    ensure_same_device(a,b,#NAME); \
    if (a.device().is_cpu()) { \
        switch (a._dtype()) { \
            case DType::Float32: \
                if (cpu_has_avx512f()) return FUNC_AVX512 ## _f32(a,b); \
                if (cpu_has_avx2())    return FUNC_AVX2 ## _f32(a,b); \
                return FUNC_MP(a,b); \
            case DType::Double64: \
                if (cpu_has_avx512f()) return FUNC_AVX512 ## _d64(a,b); \
                if (cpu_has_avx2())    return FUNC_AVX2 ## _d64(a,b); \
                return FUNC_MP(a,b); \
            default: return FUNC_MP(a,b); \
        } \
    } \
    throw std::runtime_error(std::string(#NAME) + ": unsupported device"); \
}

IMPLEMENT_COMPARE_OP(lt, lt_mp, lt_avx2, lt_avx512)
IMPLEMENT_COMPARE_OP(le, le_mp, le_avx2, le_avx512)
IMPLEMENT_COMPARE_OP(gt, gt_mp, gt_avx2, gt_avx512)
IMPLEMENT_COMPARE_OP(ge, ge_mp, ge_avx2, ge_avx512)
IMPLEMENT_COMPARE_OP(eq, eq_mp, eq_avx2, eq_avx512)
IMPLEMENT_COMPARE_OP(ne, ne_mp, ne_avx2, ne_avx512)

Tensor lt(const Tensor &a, double b) { return lt_mp(a, b); }
Tensor le(const Tensor &a, double b) { return le_mp(a, b); }
Tensor gt(const Tensor &a, double b) { return gt_mp(a, b); }
Tensor ge(const Tensor &a, double b) { return ge_mp(a, b); }
Tensor eq(const Tensor &a, double b) { return eq_mp(a, b); }
Tensor ne(const Tensor &a, double b) { return neq_mp(a, b); }

// ========================================================================
//                           Reductions
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
    
    // Using composite op (sum * scalar) allows reuse of GradMulScalar and GradSum logic
    return mul_scalar(s, 1.0 / N);
}

// Max/Min (No autograd yet)

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


//                           Utilities
Tensor cat(const std::vector<Tensor>& tensors, size_t dim) {
    // Basic cat without grad for now, or just MP
    return cat_mp(tensors, dim);
}


// --- Compound assignment operators ---

Tensor& operator+=(Tensor& a, const Tensor& b) {
    a = add(a, b);
    return a;
}
Tensor& operator+=(Tensor& a, double scalar) {
    a = add_scalar(a, scalar);
    return a;
}

Tensor& operator-=(Tensor& a, const Tensor& b) {
    a = sub(a, b);
    return a;
}
Tensor& operator-=(Tensor& a, double scalar) {
    a = sub_scalar(a, scalar);
    return a;
}

Tensor& operator*=(Tensor& a, const Tensor& b) {
    a = mul(a, b);
    return a;
}
Tensor& operator*=(Tensor& a, double scalar) {
    a = mul_scalar(a, scalar);
    return a;
}

Tensor& operator/=(Tensor& a, const Tensor& b) {
    a = div(a, b);
    return a;
}
Tensor& operator/=(Tensor& a, double scalar) {
    a = div_scalar(a, scalar);
    return a;
}

Tensor& operator^=(Tensor& a, const Tensor& b) {
    a = pow(a, b);
    return a;
}
Tensor& operator^=(Tensor& a, double scalar) {
    a = pow_scalar(a, scalar);
    return a;
}

Tensor operator+(const Tensor& a, const Tensor& b) {
    return add(a, b);
}

Tensor operator-(const Tensor& a, const Tensor& b) {
    return sub(a, b);
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    return mul(a, b);
}

Tensor operator/(const Tensor& a, const Tensor& b) {
    return div(a, b);
}

Tensor operator^(const Tensor& a, const Tensor& b) {
    return pow(a, b);
}

Tensor operator+(double s, const Tensor& a) {
    return add_scalar(a, s);
}
Tensor operator+(const Tensor& a,double s) {
    return add_scalar(a, s);
}