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

    if (a.device().is_cpu()) {
        DType d = a._dtype();
        switch (d) {
            case DType::Float32: {
                // dtype-level dispatch
                if (cpu_has_avx512f()) return add_avx512_f32(a,b);
                if (cpu_has_avx2()) {
                    return add_avx2_f32(a,b);
                }
                return add_mp(a,b);
            }
            case DType::Int32: {
                // ints: use scalar mp fallback for now
                return add_mp(a,b);
            }
            case DType::Double64: {
                if (cpu_has_avx512f()) return add_avx512_d64(a,b);
                if (cpu_has_avx2())    return add_avx2_d64(a,b);
                return add_mp(a,b);
            }
            default:
                throw std::runtime_error("add: unsupported dtype");
        }
    } else if (a.device().is_cuda()) {
        throw std::runtime_error("add: CUDA not implemented in dispatcher");
    }
    throw std::runtime_error("add: unsupported device");
}

Tensor sub(const Tensor &a, const Tensor &b) {
    ensure_same_device(a,b,"sub");

    if (a.device().is_cpu()) {
        DType d = a._dtype();
        switch (d) {
            case DType::Float32: {
                if (cpu_has_avx512f()) return sub_avx512_f32(a,b);
                if (cpu_has_avx2())    return sub_avx2_f32(a,b);
                return diff_mp(a,b);
            }
            case DType::Int32: return diff_mp(a,b);
            case DType::Double64: {
                if (cpu_has_avx512f()) return sub_avx512_d64(a,b);
                if (cpu_has_avx2())    return sub_avx2_d64(a,b);
                return diff_mp(a,b);
            }
            default: throw std::runtime_error("sub: unsupported dtype");
        }
    }
    throw std::runtime_error("sub: unsupported device");
}

Tensor mul(const Tensor &a, const Tensor &b) {
    ensure_same_device(a,b,"mul");

    if (a.device().is_cpu()) {
        DType d = a._dtype();
        switch (d) {
            case DType::Float32: {
                if (cpu_has_avx512f()) return mul_avx512_f32(a,b);
                if (cpu_has_avx2())    return mul_avx2_f32(a,b);
                return mult_mp(a,b);
            }
            case DType::Int32: return mult_mp(a,b);
            case DType::Double64: {
                if (cpu_has_avx512f()) return mul_avx512_d64(a,b);
                if (cpu_has_avx2())    return mul_avx2_d64(a,b);
                return mult_mp(a,b);
            }
            default: throw std::runtime_error("mul: unsupported dtype");
        }
    }
    throw std::runtime_error("mul: unsupported device");
}

Tensor div(const Tensor &a, const Tensor &b) {
    ensure_same_device(a,b,"div");

    if (a.device().is_cpu()) {
        DType d = a._dtype();
        switch (d) {
            case DType::Float32: {
                if (cpu_has_avx512f()) return div_avx512_f32(a,b);
                if (cpu_has_avx2())    return div_avx2_f32(a,b);
                return div_mp(a,b);
            }
            case DType::Int32: return div_mp(a,b);
            case DType::Double64: {
                if (cpu_has_avx512f()) return div_avx512_d64(a,b);
                if (cpu_has_avx2())    return div_avx2_d64(a,b);
                return div_mp(a,b);
            }
            default: throw std::runtime_error("div: unsupported dtype");
        }
    }
    throw std::runtime_error("div: unsupported device");
}

Tensor pow(const Tensor &a, const Tensor &b) {
    ensure_same_device(a,b,"pow");

    if (a.device().is_cpu()) {
        DType d = a._dtype();
        switch (d) {
            case DType::Float32: {
                if (cpu_has_avx512f()) return pow_avx512_f32(a,b);
                if (cpu_has_avx2())    return pow_avx2_f32(a,b);
                return pow_mp(a,b);
            }
            case DType::Int32: return pow_mp(a,b);
            case DType::Double64: {
                if (cpu_has_avx512f()) return pow_avx512_d64(a,b);
                if (cpu_has_avx2())    return pow_avx2_d64(a,b);
                return pow_mp(a,b);
            }
            default: throw std::runtime_error("pow: unsupported dtype");
        }
    }
    throw std::runtime_error("pow: unsupported device");
}


Tensor matmul(const Tensor &a, const Tensor &b) {
    ensure_same_device(a,b,"matmul");
    // Shape check usually handled inside impl or we can add basic checks here
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
}

Tensor add_scalar(const Tensor &a, double scalar) { return add_scalar_mp(a, scalar); }
Tensor sub_scalar(const Tensor &a, double scalar) { return sub_scalar_mp(a, scalar); }
Tensor sub_scalar_rev(double scalar, const Tensor &a) { return sub_afterscalar_mp(scalar, a); }
Tensor mul_scalar(const Tensor &a, double scalar) { return mult_scalar_mp(a, scalar); }
Tensor div_scalar(const Tensor &a, double scalar) { return div_scalar_mp(a, scalar); }
Tensor div_scalar_rev(double scalar, const Tensor &a) { return scalar_div_mp(scalar, a); }
Tensor pow_scalar(const Tensor &a, double scalar) { return pow_scalar_mp(a, scalar); }
Tensor pow_scalar_rev(double scalar, const Tensor &a) { return scalar_pow_mp(scalar, a); }
// ========================================================================
//                           Unary Operations
// ========================================================================

#define IMPLEMENT_UNARY_OP(NAME, FUNC_MP, FUNC_AVX2, FUNC_AVX512) \
Tensor NAME(const Tensor &a) { \
    if (a.device().is_cpu()) { \
        switch (a._dtype()) { \
            case DType::Float32: \
                if (cpu_has_avx512f()) return FUNC_AVX512 ## _f32(a); \
                if (cpu_has_avx2())    return FUNC_AVX2 ## _f32(a); \
                return FUNC_MP(a); \
            case DType::Double64: \
                if (cpu_has_avx512f()) return FUNC_AVX512 ## _d64(a); \
                if (cpu_has_avx2())    return FUNC_AVX2 ## _d64(a); \
                return FUNC_MP(a); \
            default: return FUNC_MP(a); \
        } \
    } \
    throw std::runtime_error(std::string(#NAME) + ": unsupported device"); \
}

IMPLEMENT_UNARY_OP(abs, abs_mp, abs_avx2, abs_avx512)
IMPLEMENT_UNARY_OP(log, ln_mp, ln_avx2, ln_avx512)   // Maps Ops::log -> ln_mp
IMPLEMENT_UNARY_OP(exp, exp_mp, exp_avx2, exp_avx512)
IMPLEMENT_UNARY_OP(sqrt, sqrt_mp, sqrt_avx2, sqrt_avx512)
IMPLEMENT_UNARY_OP(sin, sin_mp, sin_avx2, sin_avx512)
IMPLEMENT_UNARY_OP(asin, asin_mp, asin_avx2, asin_avx512)
IMPLEMENT_UNARY_OP(cos, cos_mp, cos_avx2, cos_avx512)
IMPLEMENT_UNARY_OP(acos, acos_mp, acos_avx2, acos_avx512)
IMPLEMENT_UNARY_OP(tan, tan_mp, tan_avx2, tan_avx512)
IMPLEMENT_UNARY_OP(atan, atan_mp, atan_avx2, atan_avx512)
IMPLEMENT_UNARY_OP(tanh, tanh_mp, tanh_avx2, tanh_avx512)
IMPLEMENT_UNARY_OP(sinh, sinh_mp, sinh_avx2, sinh_avx512)
IMPLEMENT_UNARY_OP(cosh, cosh_mp, cosh_avx2, cosh_avx512)
IMPLEMENT_UNARY_OP(sigmoid, sigmoid_mp, sigmoid_avx2, sigmoid_avx512)
IMPLEMENT_UNARY_OP(relu, Relu_mp, relu_avx2, relu_avx512) 
IMPLEMENT_UNARY_OP(softplus, softplus_mp, softplus_avx2, softplus_avx512)


// ========================================================================
//                           Comparisons
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

#define Reduction_Op(NAME, FUNC_MP, FUNC_AVX2, FUNC_AVX512) \
Tensor NAME(const Tensor &a,int dim) { \
    if (a.device().is_cpu()) { \
        switch (a._dtype()) { \
            case DType::Float32: \
                if (cpu_has_avx512f()) return FUNC_AVX512 ## _f32(a,dim); \
                if (cpu_has_avx2())    return FUNC_AVX2 ## _f32(a,dim); \
                return FUNC_MP(a,dim); \
            case DType::Double64: \
                if (cpu_has_avx512f()) return FUNC_AVX512 ## _d64(a,dim); \
                if (cpu_has_avx2())    return FUNC_AVX2 ## _d64(a,dim); \
                return FUNC_MP(a,dim); \
            default: return FUNC_MP(a,dim); \
        } \
    } \
    throw std::runtime_error(std::string(#NAME) + ": unsupported device"); \
}
Reduction_Op(sum, sum_mp, sum_avx2, sum_avx512)
Reduction_Op(max, max_mp, max_avx2, max_avx512)
Reduction_Op(min, min_mp, min_avx2, min_avx512)

Tensor mean(const Tensor &a, int dim) {
    Tensor s = sum(a, dim);
    double n = static_cast<double>(a.numel()); 
    return mul_scalar(s, 1.0 / n);
}


//                           Utilities
Tensor cat(const std::vector<Tensor>& tensors, size_t dim) {
    return cat_mp(tensors, dim);
}