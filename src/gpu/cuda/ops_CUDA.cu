#include "ops_cuda.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <type_traits>
#include <stdexcept>
#include <vector>
#include <limits>

#define BLOCK 256
#define TM_F  8
#define TN_F  8
#define BM_F  128
#define BN_F  128
#define BK_F  8
#define TM_D  4
#define TN_D  4
#define BM_D  64
#define BN_D  64
#define BK_D  8

namespace {


template<typename T> struct is_floating              : std::false_type {};
template<>           struct is_floating<float>       : std::true_type  {};
template<>           struct is_floating<double>      : std::true_type  {};

template<typename T> struct accum_traits             { using type = T;        };
template<>           struct accum_traits<int8_t>     { using type = int32_t;  };
template<>           struct accum_traits<uint8_t>    { using type = uint32_t; };
template<>           struct accum_traits<int16_t>    { using type = int32_t;  };
template<>           struct accum_traits<uint16_t>   { using type = uint32_t; };
template<typename T> using  accum_t = typename accum_traits<T>::type;


template<typename T>
inline T* cuda_ptr(const Tensor& t) {
    return reinterpret_cast<T*>(t.impl->data->data.get()) + t.impl->offset;
}

inline void check_launch(const char* msg) {
#ifndef NDEBUG
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(msg) + " (sync): " + cudaGetErrorString(e));
#else
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
#endif
}

inline void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
}


static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a,
                                           const std::vector<size_t>& b) {
    size_t na = a.size(), nb = b.size(), n = na > nb ? na : nb;
    std::vector<size_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        size_t ai = (i < n - na) ? 1 : a[i - (n - na)];
        size_t bi = (i < n - nb) ? 1 : b[i - (n - nb)];
        if (ai != 1 && bi != 1 && ai != bi)
            throw std::runtime_error("broadcast: incompatible shapes");
        out[i] = ai > bi ? ai : bi;
    }
    return out;
}

static std::vector<size_t> build_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> st(shape.size());
    if (shape.empty()) return st;
    st.back() = 1;
    for (int i = (int)shape.size() - 2; i >= 0; --i)
        st[i] = st[i + 1] * shape[i + 1];
    return st;
}

struct BcastMeta {
    static constexpr int MAXDIM = 8;
    int    ndim;
    size_t out_shape  [MAXDIM];
    size_t out_strides[MAXDIM];
    size_t a_strides  [MAXDIM];
    size_t b_strides  [MAXDIM];
};

static BcastMeta make_meta(const std::vector<size_t>& a_shape,
                           const std::vector<size_t>& b_shape,
                           const std::vector<size_t>& out_shape) {
    BcastMeta m;
    m.ndim        = (int)out_shape.size();
    auto out_st   = build_strides(out_shape);
    auto a_st     = build_strides(a_shape);
    auto b_st     = build_strides(b_shape);
    size_t pad_a  = out_shape.size() - a_shape.size();
    size_t pad_b  = out_shape.size() - b_shape.size();
    for (int i = 0; i < m.ndim; ++i) {
        m.out_shape[i]   = out_shape[i];
        m.out_strides[i] = out_st[i];
        m.a_strides[i]   = (i < (int)pad_a || a_shape[i-pad_a] == 1) ? 0 : a_st[i-pad_a];
        m.b_strides[i]   = (i < (int)pad_b || b_shape[i-pad_b] == 1) ? 0 : b_st[i-pad_b];
    }
    return m;
}

__device__ inline void unravel(size_t idx, const BcastMeta& m,
                               size_t& a_off, size_t& b_off) {
    a_off = 0; b_off = 0;
    for (int d = 0; d < m.ndim; ++d) {
        size_t coord  = idx / m.out_strides[d] % m.out_shape[d];
        a_off        += coord * m.a_strides[d];
        b_off        += coord * m.b_strides[d];
    }
}

template<typename T, typename Op>
__global__ void binary_flat_kernel(const T* __restrict__ a,
                                   const T* __restrict__ b,
                                   T*       __restrict__ out,
                                   size_t n, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(a[i], b[i]);
}

template<typename T, typename Op>
__global__ void binary_bcast_kernel(const T* __restrict__ a,
                                    const T* __restrict__ b,
                                    T*       __restrict__ out,
                                    size_t n, BcastMeta m, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t ai, bi;
    unravel(i, m, ai, bi);
    out[i] = op(a[ai], b[bi]);
}

template<typename T, typename Op>
Tensor binary_op_typed(const Tensor& A, const Tensor& B, DType dt, Op op) {
    auto a_shape   = A.shape();
    auto b_shape   = B.shape();
    auto out_shape = broadcast_shape(a_shape, b_shape);
    size_t n = 1;
    for (size_t s : out_shape) n *= s;

    Tensor out(out_shape, dt);
    out.impl->data = Storage::allocate(n, dt, Device(DeviceType::CUDA));

    Tensor A_gpu = A.device().is_cuda() ? A : A.to(Device(DeviceType::CUDA));
    Tensor B_gpu = B.device().is_cuda() ? B : B.to(Device(DeviceType::CUDA));

    const T* ap   = cuda_ptr<T>(A_gpu);
    const T* bp   = cuda_ptr<T>(B_gpu);
    T*       outp = cuda_ptr<T>(out);

    dim3 grid((unsigned)((n + BLOCK - 1) / BLOCK));
    bool flat = A.is_contiguous() && B.is_contiguous()
             && a_shape == out_shape && b_shape == out_shape;
    if (flat)
        binary_flat_kernel<T><<<grid, BLOCK>>>(ap, bp, outp, n, op);
    else
        binary_bcast_kernel<T><<<grid, BLOCK>>>(ap, bp, outp, n, make_meta(a_shape, b_shape, out_shape), op);
    check_launch("binary_op");
    return out;
}


template<typename T, typename Op>
__global__ void unary_kernel(const T* __restrict__ in,
                             T*       __restrict__ out,
                             size_t n, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(in[i]);
}

template<typename T, typename Op>
Tensor unary_op_typed(const Tensor& A, DType dt, Op op) {
    size_t n = A.numel();
    Tensor out(A.shape(), dt);
    out.impl->data = Storage::allocate(n, dt, Device(DeviceType::CUDA));

    Tensor A_gpu  = A.device().is_cuda() ? A : A.to(Device(DeviceType::CUDA));
    const T* ap   = cuda_ptr<T>(A_gpu);
    T*       outp = cuda_ptr<T>(out);

    dim3 grid((unsigned)((n + BLOCK - 1) / BLOCK));
    unary_kernel<T><<<grid, BLOCK>>>(ap, outp, n, op);
    check_launch("unary_op");
    return out;
}


template<typename T, int PRED>
__device__ inline T cmp_result(T a, T b) {
    bool r;
    if constexpr (PRED == 0) r = a <  b;
    else if constexpr (PRED == 1) r = a <= b;
    else if constexpr (PRED == 2) r = a >  b;
    else if constexpr (PRED == 3) r = a >= b;
    else if constexpr (PRED == 4) r = a == b;
    else                          r = a != b;
    return static_cast<T>(r ? 1 : 0);
}

template<typename T, int PRED>
__global__ void cmp_flat_kernel(const T* __restrict__ a,
                                const T* __restrict__ b,
                                T*       __restrict__ out,
                                size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = cmp_result<T, PRED>(a[i], b[i]);
}

template<typename T, int PRED>
__global__ void cmp_bcast_kernel(const T* __restrict__ a,
                                 const T* __restrict__ b,
                                 T*       __restrict__ out,
                                 size_t n, BcastMeta m) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t ai, bi;
    unravel(i, m, ai, bi);
    out[i] = cmp_result<T, PRED>(a[ai], b[bi]);
}

template<typename T, int PRED>
Tensor cmp_op_typed(const Tensor& A, const Tensor& B, DType dt) {
    auto a_shape   = A.shape();
    auto b_shape   = B.shape();
    auto out_shape = broadcast_shape(a_shape, b_shape);
    size_t n = 1;
    for (size_t s : out_shape) n *= s;

    Tensor out(out_shape, dt);
    out.impl->data = Storage::allocate(n, dt, Device(DeviceType::CUDA));

    Tensor A_gpu = A.device().is_cuda() ? A : A.to(Device(DeviceType::CUDA));
    Tensor B_gpu = B.device().is_cuda() ? B : B.to(Device(DeviceType::CUDA));

    dim3 grid((unsigned)((n + BLOCK - 1) / BLOCK));
    bool flat = A.is_contiguous() && B.is_contiguous()
             && a_shape == out_shape && b_shape == out_shape;
    if (flat)
        cmp_flat_kernel<T, PRED><<<grid, BLOCK>>>(cuda_ptr<T>(A_gpu), cuda_ptr<T>(B_gpu), cuda_ptr<T>(out), n);
    else
        cmp_bcast_kernel<T, PRED><<<grid, BLOCK>>>(cuda_ptr<T>(A_gpu), cuda_ptr<T>(B_gpu), cuda_ptr<T>(out), n,
                                                    make_meta(a_shape, b_shape, out_shape));
    check_launch("cmp_op");
    return out;
}


template<int BM, int BN, int BK, int TM, int TN, typename T>
__global__ void matmul_kernel(const T* __restrict__ A,
                              const T* __restrict__ B,
                              T*       __restrict__ C,
                              int M, int K, int N) {
    __shared__ T As[BM][BK];
    __shared__ T Bs[BK][BN];

    int block_row  = blockIdx.y * BM;
    int block_col  = blockIdx.x * BN;
    int thread_row = threadIdx.x / (BN / TN);
    int thread_col = threadIdx.x % (BN / TN);
    constexpr int THREADS = (BM / TM) * (BN / TN);

    T acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int s = 0; s < BM * BK; s += THREADS) {
            int idx  = s + threadIdx.x;
            int r    = idx / BK, c = idx % BK;
            int gRow = block_row + r, gCol = k0 + c;
            As[r][c] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : T(0);
        }
        for (int s = 0; s < BK * BN; s += THREADS) {
            int idx  = s + threadIdx.x;
            int r    = idx / BN, c = idx % BN;
            int gRow = k0 + r, gCol = block_col + c;
            Bs[r][c] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : T(0);
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            T b_reg[TN];
            #pragma unroll
            for (int n = 0; n < TN; ++n) b_reg[n] = Bs[k][thread_col * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                T a_reg = As[thread_row * TM + m][k];
                #pragma unroll
                for (int n = 0; n < TN; ++n) acc[m][n] += a_reg * b_reg[n];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        int gRow = block_row + thread_row * TM + m;
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            int gCol = block_col + thread_col * TN + n;
            if (gRow < M && gCol < N) C[gRow * N + gCol] = acc[m][n];
        }
    }
}

template<typename T, int BM, int BN, int BK, int TM, int TN>
Tensor matmul_typed(const Tensor& A, const Tensor& B, DType dt) {
    if (A.shape().size() != 2 || B.shape().size() != 2)
        throw std::runtime_error("matmul_cuda: only 2D tensors");
    int M = (int)A.shape()[0], K = (int)A.shape()[1], N = (int)B.shape()[1];
    if (K != (int)B.shape()[0]) throw std::runtime_error("matmul_cuda: shape mismatch");

    Tensor A_gpu = A.device().is_cuda() ? A : A.to(Device(DeviceType::CUDA));
    Tensor B_gpu = B.device().is_cuda() ? B : B.to(Device(DeviceType::CUDA));

    Tensor C({(size_t)M, (size_t)N}, dt);
    C.impl->data = Storage::allocate((size_t)M * N, dt, Device(DeviceType::CUDA));

    constexpr int THREADS = (BM / TM) * (BN / TN);
    dim3 block(THREADS);
    dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((M + BM - 1) / BM));
    matmul_kernel<BM, BN, BK, TM, TN, T><<<grid, block>>>(
        cuda_ptr<T>(A_gpu), cuda_ptr<T>(B_gpu), cuda_ptr<T>(C), M, K, N);
    check_launch("matmul_cuda");
    return C;
}


template<typename T>
static T gpu_sum(const T* d, size_t n) {
    using A = accum_t<T>;
    A* d_out;
    check(cudaMalloc(&d_out, sizeof(A)), "cudaMalloc sum");
    void* d_tmp = nullptr; size_t tmp_bytes = 0;
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaMalloc(&d_tmp, tmp_bytes), "cudaMalloc sum tmp");
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaDeviceSynchronize(), "sum sync");
    A r; check(cudaMemcpy(&r, d_out, sizeof(A), cudaMemcpyDeviceToHost), "sum copy");
    cudaFree(d_tmp); cudaFree(d_out);
    return static_cast<T>(r);
}

template<typename T>
static T gpu_max(const T* d, size_t n) {
    T* d_out;
    check(cudaMalloc(&d_out, sizeof(T)), "cudaMalloc max");
    void* d_tmp = nullptr; size_t tmp_bytes = 0;
    cub::DeviceReduce::Max(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaMalloc(&d_tmp, tmp_bytes), "cudaMalloc max tmp");
    cub::DeviceReduce::Max(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaDeviceSynchronize(), "max sync");
    T r; check(cudaMemcpy(&r, d_out, sizeof(T), cudaMemcpyDeviceToHost), "max copy");
    cudaFree(d_tmp); cudaFree(d_out);
    return r;
}

template<typename T>
static T gpu_min(const T* d, size_t n) {
    T* d_out;
    check(cudaMalloc(&d_out, sizeof(T)), "cudaMalloc min");
    void* d_tmp = nullptr; size_t tmp_bytes = 0;
    cub::DeviceReduce::Min(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaMalloc(&d_tmp, tmp_bytes), "cudaMalloc min tmp");
    cub::DeviceReduce::Min(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaDeviceSynchronize(), "min sync");
    T r; check(cudaMemcpy(&r, d_out, sizeof(T), cudaMemcpyDeviceToHost), "min copy");
    cudaFree(d_tmp); cudaFree(d_out);
    return r;
}

} 

Tensor add_cuda(const Tensor& a, const Tensor& b) {
    DType dt = a._dtype();
    DISPATCH_ALL_TYPES(dt, "add_cuda", [&] {
        return binary_op_typed<scalar_t>(a, b, dt,
            [] __device__(scalar_t x, scalar_t y) { return static_cast<scalar_t>(x + y); });
    });
}

Tensor sub_cuda(const Tensor& a, const Tensor& b) {
    DType dt = a._dtype();
    DISPATCH_ALL_TYPES(dt, "sub_cuda", [&] {
        return binary_op_typed<scalar_t>(a, b, dt,
            [] __device__(scalar_t x, scalar_t y) { return static_cast<scalar_t>(x - y); });
    });
}

Tensor mul_cuda(const Tensor& a, const Tensor& b) {
    DType dt = a._dtype();
    DISPATCH_ALL_TYPES(dt, "mul_cuda", [&] {
        return binary_op_typed<scalar_t>(a, b, dt,
            [] __device__(scalar_t x, scalar_t y) { return static_cast<scalar_t>(x * y); });
    });
}

Tensor div_cuda(const Tensor& a, const Tensor& b) {
    DType dt = a._dtype();
    DISPATCH_ALL_TYPES(dt, "div_cuda", [&] {
        return binary_op_typed<scalar_t>(a, b, dt,
            [] __device__(scalar_t x, scalar_t y) { return static_cast<scalar_t>(x / y); });
    });
}

Tensor pow_cuda(const Tensor& a, const Tensor& b) {
    DType dt = a._dtype();
    DISPATCH_ALL_TYPES(dt, "pow_cuda", [&] {
        return binary_op_typed<scalar_t>(a, b, dt,
            [] __device__(scalar_t x, scalar_t y) {
                if constexpr (std::is_same_v<scalar_t, float>)  return powf(x, y);
                if constexpr (std::is_same_v<scalar_t, double>) return pow(x, y);
                return static_cast<scalar_t>(powf((float)x, (float)y));
            });
    });
}

Tensor matmul_cuda(const Tensor& A, const Tensor& B) {
    DType dt = A._dtype();
    switch (dt) {
        case DType::Float32:
            return matmul_typed<float,  BM_F, BN_F, BK_F, TM_F, TN_F>(A, B, dt);
        case DType::Double64:
            return matmul_typed<double, BM_D, BN_D, BK_D, TM_D, TN_D>(A, B, dt);
        default:
            throw std::runtime_error("matmul_cuda: only Float32 and Double64 supported");
    }
}

Tensor lt_cuda(const Tensor& a, const Tensor& b) {
    DType dt = a._dtype();
    DISPATCH_ALL_TYPES(dt, "lt_cuda", [&] { return cmp_op_typed<scalar_t, 0>(a, b, dt); });
}
Tensor le_cuda(const Tensor& a, const Tensor& b) {
    DType dt = a._dtype();
    DISPATCH_ALL_TYPES(dt, "le_cuda", [&] { return cmp_op_typed<scalar_t, 1>(a, b, dt); });
}
Tensor gt_cuda(const Tensor& a, const Tensor& b) {
    DType dt = a._dtype();
    DISPATCH_ALL_TYPES(dt, "gt_cuda", [&] { return cmp_op_typed<scalar_t, 2>(a, b, dt); });
}
Tensor ge_cuda(const Tensor& a, const Tensor& b) {
    DType dt = a._dtype();
    DISPATCH_ALL_TYPES(dt, "ge_cuda", [&] { return cmp_op_typed<scalar_t, 3>(a, b, dt); });
}
Tensor eq_cuda(const Tensor& a, const Tensor& b) {
    DType dt = a._dtype();
    DISPATCH_ALL_TYPES(dt, "eq_cuda", [&] { return cmp_op_typed<scalar_t, 4>(a, b, dt); });
}
Tensor ne_cuda(const Tensor& a, const Tensor& b) {
    DType dt = a._dtype();
    DISPATCH_ALL_TYPES(dt, "ne_cuda", [&] { return cmp_op_typed<scalar_t, 5>(a, b, dt); });
}

Tensor abs_cuda(const Tensor& a) {
    DType dt = a._dtype();
    DISPATCH_ALL_TYPES(dt, "abs_cuda", [&] {
        return unary_op_typed<scalar_t>(a, dt, [] __device__(scalar_t x) {
            if constexpr (std::is_same_v<scalar_t, float>)  return fabsf(x);
            if constexpr (std::is_same_v<scalar_t, double>) return fabs(x);
            return x < scalar_t(0) ? static_cast<scalar_t>(-x) : x;
        });
    });
}

Tensor sqrt_cuda(const Tensor& a) {
    DType dt = a._dtype();
    switch (dt) {
        case DType::Float32:
            return unary_op_typed<float> (a, dt, [] __device__(float  x) { return sqrtf(x); });
        case DType::Double64:
            return unary_op_typed<double>(a, dt, [] __device__(double x) { return sqrt(x); });
        default: throw std::runtime_error("sqrt_cuda: floating-point types only");
    }
}

Tensor relu_cuda(const Tensor& a) {
    DType dt = a._dtype();
    DISPATCH_ALL_TYPES(dt, "relu_cuda", [&] {
        return unary_op_typed<scalar_t>(a, dt, [] __device__(scalar_t x) {
            return x > scalar_t(0) ? x : scalar_t(0);
        });
    });
}

Tensor ln_cuda(const Tensor& a) {
    DType dt = a._dtype();
    switch (dt) {
        case DType::Float32:
            return unary_op_typed<float> (a, dt, [] __device__(float  x) { return logf(x); });
        case DType::Double64:
            return unary_op_typed<double>(a, dt, [] __device__(double x) { return log(x); });
        default: throw std::runtime_error("ln_cuda: floating-point types only");
    }
}

Tensor exp_cuda(const Tensor& a) {
    DType dt = a._dtype();
    switch (dt) {
        case DType::Float32:
            return unary_op_typed<float> (a, dt, [] __device__(float  x) { return expf(x); });
        case DType::Double64:
            return unary_op_typed<double>(a, dt, [] __device__(double x) { return exp(x); });
        default: throw std::runtime_error("exp_cuda: floating-point types only");
    }
}

Tensor sin_cuda(const Tensor& a) {
    DType dt = a._dtype();
    switch (dt) {
        case DType::Float32:  return unary_op_typed<float> (a, dt, [] __device__(float  x) { return sinf(x); });
        case DType::Double64: return unary_op_typed<double>(a, dt, [] __device__(double x) { return sin(x);  });
        default: throw std::runtime_error("sin_cuda: floating-point types only");
    }
}
Tensor asin_cuda(const Tensor& a) {
    DType dt = a._dtype();
    switch (dt) {
        case DType::Float32:  return unary_op_typed<float> (a, dt, [] __device__(float  x) { return asinf(x); });
        case DType::Double64: return unary_op_typed<double>(a, dt, [] __device__(double x) { return asin(x);  });
        default: throw std::runtime_error("asin_cuda: floating-point types only");
    }
}
Tensor cos_cuda(const Tensor& a) {
    DType dt = a._dtype();
    switch (dt) {
        case DType::Float32:  return unary_op_typed<float> (a, dt, [] __device__(float  x) { return cosf(x); });
        case DType::Double64: return unary_op_typed<double>(a, dt, [] __device__(double x) { return cos(x);  });
        default: throw std::runtime_error("cos_cuda: floating-point types only");
    }
}
Tensor acos_cuda(const Tensor& a) {
    DType dt = a._dtype();
    switch (dt) {
        case DType::Float32:  return unary_op_typed<float> (a, dt, [] __device__(float  x) { return acosf(x); });
        case DType::Double64: return unary_op_typed<double>(a, dt, [] __device__(double x) { return acos(x);  });
        default: throw std::runtime_error("acos_cuda: floating-point types only");
    }
}
Tensor tan_cuda(const Tensor& a) {
    DType dt = a._dtype();
    switch (dt) {
        case DType::Float32:  return unary_op_typed<float> (a, dt, [] __device__(float  x) { return tanf(x); });
        case DType::Double64: return unary_op_typed<double>(a, dt, [] __device__(double x) { return tan(x);  });
        default: throw std::runtime_error("tan_cuda: floating-point types only");
    }
}
Tensor atan_cuda(const Tensor& a) {
    DType dt = a._dtype();
    switch (dt) {
        case DType::Float32:  return unary_op_typed<float> (a, dt, [] __device__(float  x) { return atanf(x); });
        case DType::Double64: return unary_op_typed<double>(a, dt, [] __device__(double x) { return atan(x);  });
        default: throw std::runtime_error("atan_cuda: floating-point types only");
    }
}
Tensor tanh_cuda(const Tensor& a) {
    DType dt = a._dtype();
    switch (dt) {
        case DType::Float32:  return unary_op_typed<float> (a, dt, [] __device__(float  x) { return tanhf(x); });
        case DType::Double64: return unary_op_typed<double>(a, dt, [] __device__(double x) { return tanh(x);  });
        default: throw std::runtime_error("tanh_cuda: floating-point types only");
    }
}
Tensor sinh_cuda(const Tensor& a) {
    DType dt = a._dtype();
    switch (dt) {
        case DType::Float32:  return unary_op_typed<float> (a, dt, [] __device__(float  x) { return sinhf(x); });
        case DType::Double64: return unary_op_typed<double>(a, dt, [] __device__(double x) { return sinh(x);  });
        default: throw std::runtime_error("sinh_cuda: floating-point types only");
    }
}
Tensor cosh_cuda(const Tensor& a) {
    DType dt = a._dtype();
    switch (dt) {
        case DType::Float32:  return unary_op_typed<float> (a, dt, [] __device__(float  x) { return coshf(x); });
        case DType::Double64: return unary_op_typed<double>(a, dt, [] __device__(double x) { return cosh(x);  });
        default: throw std::runtime_error("cosh_cuda: floating-point types only");
    }
}
Tensor sigmoid_cuda(const Tensor& a) {
    DType dt = a._dtype();
    switch (dt) {
        case DType::Float32:
            return unary_op_typed<float> (a, dt, [] __device__(float  x) { return 1.0f / (1.0f + expf(-x)); });
        case DType::Double64:
            return unary_op_typed<double>(a, dt, [] __device__(double x) { return 1.0  / (1.0  + exp(-x));  });
        default: throw std::runtime_error("sigmoid_cuda: floating-point types only");
    }
}
Tensor softplus_cuda(const Tensor& a) {
    DType dt = a._dtype();
    switch (dt) {
        case DType::Float32:
            return unary_op_typed<float> (a, dt, [] __device__(float  x) { return logf(1.0f + expf(x)); });
        case DType::Double64:
            return unary_op_typed<double>(a, dt, [] __device__(double x) { return log(1.0  + exp(x));   });
        default: throw std::runtime_error("softplus_cuda: floating-point types only");
    }
}

Tensor sum_cuda(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("sum_cuda: only dim=-1 supported");
    Tensor t_gpu = t.device().is_cuda() ? t : t.to(Device(DeviceType::CUDA));
    DType dt = t._dtype();
    Tensor out({1}, dt);
    DISPATCH_ALL_TYPES(dt, "sum_cuda", [&] {
        scalar_t val = gpu_sum<scalar_t>(cuda_ptr<scalar_t>(t_gpu), t.numel());
        reinterpret_cast<scalar_t*>(out.impl->data->data.get())[0] = val;
    });
    return out;
}

Tensor mean_cuda(const Tensor& t, int dim) {
    Tensor s = sum_cuda(t, dim);
    DType dt = t._dtype();
    DISPATCH_ALL_TYPES(dt, "mean_cuda", [&] {
        reinterpret_cast<scalar_t*>(s.impl->data->data.get())[0] /=
            static_cast<scalar_t>(t.numel());
    });
    return s;
}

Tensor max_cuda(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("max_cuda: only dim=-1 supported");
    Tensor t_gpu = t.device().is_cuda() ? t : t.to(Device(DeviceType::CUDA));
    DType dt = t._dtype();
    Tensor out({1}, dt);
    DISPATCH_ALL_TYPES(dt, "max_cuda", [&] {
        scalar_t val = gpu_max<scalar_t>(cuda_ptr<scalar_t>(t_gpu), t.numel());
        reinterpret_cast<scalar_t*>(out.impl->data->data.get())[0] = val;
    });
    return out;
}

Tensor min_cuda(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("min_cuda: only dim=-1 supported");
    Tensor t_gpu = t.device().is_cuda() ? t : t.to(Device(DeviceType::CUDA));
    DType dt = t._dtype();
    Tensor out({1}, dt);
    DISPATCH_ALL_TYPES(dt, "min_cuda", [&] {
        scalar_t val = gpu_min<scalar_t>(cuda_ptr<scalar_t>(t_gpu), t.numel());
        reinterpret_cast<scalar_t*>(out.impl->data->data.get())[0] = val;
    });
    return out;
}

#endif
