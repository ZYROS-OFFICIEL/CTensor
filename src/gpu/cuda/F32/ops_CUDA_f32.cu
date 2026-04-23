#include "ops_cuda_f32.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <limits>
#include <cstring>

#define BLOCK      256
#define WARP       32
#define TM         8
#define TN         8
#define BM         128
#define BN         128
#define BK         8

namespace {

inline float* cuda_ptr_f32(const Tensor& t) {
    return reinterpret_cast<float*>(t.impl->data->data.get()) + t.impl->offset;
}

inline void check_launch(const char* msg) {
#ifndef NDEBUG
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) throw std::runtime_error(std::string(msg) + " (sync): " + cudaGetErrorString(e));
#else
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
#endif
}

inline void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
}

static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    size_t na = a.size(), nb = b.size();
    size_t n  = na > nb ? na : nb;
    std::vector<size_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        size_t ai = (i < n - na) ? 1 : a[i - (n - na)];
        size_t bi = (i < n - nb) ? 1 : b[i - (n - nb)];
        if (ai != 1 && bi != 1 && ai != bi) throw std::runtime_error("broadcast: incompatible shapes");
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
    size_t out_shape[MAXDIM];
    size_t out_strides[MAXDIM];
    size_t a_strides[MAXDIM];
    size_t b_strides[MAXDIM];
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
        m.a_strides[i]   = (i < (int)pad_a || a_shape[i - pad_a] == 1) ? 0 : a_st[i - pad_a];
        m.b_strides[i]   = (i < (int)pad_b || b_shape[i - pad_b] == 1) ? 0 : b_st[i - pad_b];
    }
    return m;
}

__device__ inline void unravel(size_t idx, const BcastMeta& m, size_t& a_off, size_t& b_off) {
    a_off = 0; b_off = 0;
    for (int d = 0; d < m.ndim; ++d) {
        size_t coord  = idx / m.out_strides[d] % m.out_shape[d];
        a_off        += coord * m.a_strides[d];
        b_off        += coord * m.b_strides[d];
    }
}

template<typename Op>
__global__ void binary_flat_kernel(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float*       __restrict__ out,
                                   size_t n, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(a[i], b[i]);
}

template<typename Op>
__global__ void binary_bcast_kernel(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float*       __restrict__ out,
                                    size_t n, BcastMeta m, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t ai, bi;
    unravel(i, m, ai, bi);
    out[i] = op(a[ai], b[bi]);
}

template<typename Op>
__global__ void unary_kernel(const float* __restrict__ in,
                             float*       __restrict__ out,
                             size_t n, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(in[i]);
}

template<typename Op>
Tensor binary_op(const Tensor& A, const Tensor& B, Op op) {
    auto a_shape   = A.shape();
    auto b_shape   = B.shape();
    auto out_shape = broadcast_shape(a_shape, b_shape);
    size_t n = 1;
    for (size_t s : out_shape) n *= s;

    Tensor out(out_shape, DType::Float32);
    out.impl->data = Storage::allocate(n, DType::Float32, Device(DeviceType::CUDA));

    Tensor A_gpu = A.device().is_cuda() ? A : A.to(Device(DeviceType::CUDA));
    Tensor B_gpu = B.device().is_cuda() ? B : B.to(Device(DeviceType::CUDA));

    const float* ap   = cuda_ptr_f32(A_gpu);
    const float* bp   = cuda_ptr_f32(B_gpu);
    float*       outp = cuda_ptr_f32(out);

    dim3 grid((unsigned)((n + BLOCK - 1) / BLOCK));

    bool flat = A.is_contiguous() && B.is_contiguous() && a_shape == out_shape && b_shape == out_shape;
    if (flat) {
        binary_flat_kernel<<<grid, BLOCK>>>(ap, bp, outp, n, op);
    } else {
        auto meta = make_meta(a_shape, b_shape, out_shape);
        binary_bcast_kernel<<<grid, BLOCK>>>(ap, bp, outp, n, meta, op);
    }
    check_launch("binary_op");
    return out;
}

template<typename Op>
Tensor unary_op(const Tensor& A, Op op) {
    size_t n = A.numel();
    Tensor out(A.shape(), DType::Float32);
    out.impl->data = Storage::allocate(n, DType::Float32, Device(DeviceType::CUDA));

    Tensor A_gpu  = A.device().is_cuda() ? A : A.to(Device(DeviceType::CUDA));
    const float* ap   = cuda_ptr_f32(A_gpu);
    float*       outp = cuda_ptr_f32(out);

    dim3 grid((unsigned)((n + BLOCK - 1) / BLOCK));
    unary_kernel<<<grid, BLOCK>>>(ap, outp, n, op);
    check_launch("unary_op");
    return out;
}

template<int PRED>
__device__ inline bool cmp_pred(float a, float b) {
    if constexpr (PRED == 0) return a <  b;
    if constexpr (PRED == 1) return a <= b;
    if constexpr (PRED == 2) return a >  b;
    if constexpr (PRED == 3) return a >= b;
    if constexpr (PRED == 4) return a == b;
    if constexpr (PRED == 5) return a != b;
    return false;
}

template<int PRED>
__global__ void cmp_flat_kernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float*       __restrict__ out,
                                size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = cmp_pred<PRED>(a[i], b[i]) ? 1.0f : 0.0f;
}

template<int PRED>
__global__ void cmp_bcast_kernel(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float*       __restrict__ out,
                                 size_t n, BcastMeta m) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t ai, bi;
    unravel(i, m, ai, bi);
    out[i] = cmp_pred<PRED>(a[ai], b[bi]) ? 1.0f : 0.0f;
}

template<int PRED>
Tensor cmp_op(const Tensor& A, const Tensor& B) {
    auto a_shape   = A.shape();
    auto b_shape   = B.shape();
    auto out_shape = broadcast_shape(a_shape, b_shape);
    size_t n = 1;
    for (size_t s : out_shape) n *= s;

    Tensor out(out_shape, DType::Float32);
    out.impl->data = Storage::allocate(n, DType::Float32, Device(DeviceType::CUDA));

    Tensor A_gpu = A.device().is_cuda() ? A : A.to(Device(DeviceType::CUDA));
    Tensor B_gpu = B.device().is_cuda() ? B : B.to(Device(DeviceType::CUDA));

    dim3 grid((unsigned)((n + BLOCK - 1) / BLOCK));

    bool flat = A.is_contiguous() && B.is_contiguous() && a_shape == out_shape && b_shape == out_shape;
    if (flat) {
        cmp_flat_kernel<PRED><<<grid, BLOCK>>>(cuda_ptr_f32(A_gpu), cuda_ptr_f32(B_gpu), cuda_ptr_f32(out), n);
    } else {
        auto meta = make_meta(a_shape, b_shape, out_shape);
        cmp_bcast_kernel<PRED><<<grid, BLOCK>>>(cuda_ptr_f32(A_gpu), cuda_ptr_f32(B_gpu), cuda_ptr_f32(out), n, meta);
    }
    check_launch("cmp_op");
    return out;
}


__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float*       __restrict__ C,
                              int M, int K, int N) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int thread_row = threadIdx.x / (BN / TN);
    int thread_col = threadIdx.x % (BN / TN);

    float acc[TM][TN] = {};

    int load_a_row = threadIdx.x / BK;
    int load_a_col = threadIdx.x % BK;
    int load_b_row = threadIdx.x / BN;
    int load_b_col = threadIdx.x % BN;

    int threads_for_a = BM * BK / (BM * BK / blockDim.x);
    (void)threads_for_a;

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int s = 0; s < BM * BK; s += blockDim.x) {
            int idx = s + threadIdx.x;
            int r   = idx / BK;
            int c   = idx % BK;
            int gRow = block_row + r;
            int gCol = k0 + c;
            As[r][c] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
        }
        for (int s = 0; s < BK * BN; s += blockDim.x) {
            int idx  = s + threadIdx.x;
            int r    = idx / BN;
            int c    = idx % BN;
            int gRow = k0 + r;
            int gCol = block_col + c;
            Bs[r][c] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float b_reg[TN];
            #pragma unroll
            for (int n = 0; n < TN; ++n)
                b_reg[n] = Bs[k][thread_col * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                float a_reg = As[thread_row * TM + m][k];
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += a_reg * b_reg[n];
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


template<typename CubOp>
static float cub_reduce(const float* d_in, size_t n, CubOp op, float init) {
    float* d_out;
    check(cudaMalloc(&d_out, sizeof(float)), "cudaMalloc d_out");
    check(cudaMemcpy(d_out, &init, sizeof(float), cudaMemcpyHostToDevice), "init d_out");

    void*  d_tmp      = nullptr;
    size_t tmp_bytes  = 0;
    check(op(d_tmp, tmp_bytes, d_in, d_out, (int)n, 0, false), "cub size query");
    check(cudaMalloc(&d_tmp, tmp_bytes), "cudaMalloc cub tmp");
    check(op(d_tmp, tmp_bytes, d_in, d_out, (int)n, 0, false), "cub reduce");
    check(cudaDeviceSynchronize(), "cub sync");

    float result;
    check(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy result");
    cudaFree(d_tmp);
    cudaFree(d_out);
    return result;
}

static float gpu_sum(const float* d, size_t n) {
    float* d_out;
    check(cudaMalloc(&d_out, sizeof(float)), "cudaMalloc sum");
    void* d_tmp = nullptr; size_t tmp_bytes = 0;
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaMalloc(&d_tmp, tmp_bytes), "cudaMalloc sum tmp");
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaDeviceSynchronize(), "sum sync");
    float r; check(cudaMemcpy(&r, d_out, sizeof(float), cudaMemcpyDeviceToHost), "sum copy");
    cudaFree(d_tmp); cudaFree(d_out);
    return r;
}

static float gpu_max(const float* d, size_t n) {
    float* d_out;
    check(cudaMalloc(&d_out, sizeof(float)), "cudaMalloc max");
    void* d_tmp = nullptr; size_t tmp_bytes = 0;
    cub::DeviceReduce::Max(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaMalloc(&d_tmp, tmp_bytes), "cudaMalloc max tmp");
    cub::DeviceReduce::Max(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaDeviceSynchronize(), "max sync");
    float r; check(cudaMemcpy(&r, d_out, sizeof(float), cudaMemcpyDeviceToHost), "max copy");
    cudaFree(d_tmp); cudaFree(d_out);
    return r;
}

static float gpu_min(const float* d, size_t n) {
    float* d_out;
    check(cudaMalloc(&d_out, sizeof(float)), "cudaMalloc min");
    void* d_tmp = nullptr; size_t tmp_bytes = 0;
    cub::DeviceReduce::Min(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaMalloc(&d_tmp, tmp_bytes), "cudaMalloc min tmp");
    cub::DeviceReduce::Min(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaDeviceSynchronize(), "min sync");
    float r; check(cudaMemcpy(&r, d_out, sizeof(float), cudaMemcpyDeviceToHost), "min copy");
    cudaFree(d_tmp); cudaFree(d_out);
    return r;
}

} 
Tensor add_cuda_f32(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [] __device__(float x, float y) { return x + y; });
}
Tensor sub_cuda_f32(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [] __device__(float x, float y) { return x - y; });
}
Tensor mul_cuda_f32(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [] __device__(float x, float y) { return x * y; });
}
Tensor div_cuda_f32(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [] __device__(float x, float y) { return x / y; });
}
Tensor pow_cuda_f32(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [] __device__(float x, float y) { return powf(x, y); });
}

Tensor matmul_cuda_f32(const Tensor& A, const Tensor& B) {
    if (A.shape().size() != 2 || B.shape().size() != 2)
        throw std::runtime_error("matmul_cuda_f32: only 2D");
    int M = (int)A.shape()[0], K = (int)A.shape()[1], N = (int)B.shape()[1];
    if (K != (int)B.shape()[0]) throw std::runtime_error("matmul_cuda_f32: shape mismatch");

    Tensor A_gpu = A.device().is_cuda() ? A : A.to(Device(DeviceType::CUDA));
    Tensor B_gpu = B.device().is_cuda() ? B : B.to(Device(DeviceType::CUDA));

    Tensor C({(size_t)M, (size_t)N}, DType::Float32);
    C.impl->data = Storage::allocate((size_t)M * N, DType::Float32, Device(DeviceType::CUDA));

    static_assert(BM % TM == 0 && BN % TN == 0, "tile dims must divide block dims");
    constexpr int THREADS = (BM / TM) * (BN / TN);
    dim3 block(THREADS);
    dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((M + BM - 1) / BM));
    matmul_kernel<<<grid, block>>>(cuda_ptr_f32(A_gpu), cuda_ptr_f32(B_gpu), cuda_ptr_f32(C), M, K, N);
    check_launch("matmul_cuda_f32");
    return C;
}

Tensor lt_cuda_f32(const Tensor& a, const Tensor& b) { return cmp_op<0>(a, b); }
Tensor le_cuda_f32(const Tensor& a, const Tensor& b) { return cmp_op<1>(a, b); }
Tensor gt_cuda_f32(const Tensor& a, const Tensor& b) { return cmp_op<2>(a, b); }
Tensor ge_cuda_f32(const Tensor& a, const Tensor& b) { return cmp_op<3>(a, b); }
Tensor eq_cuda_f32(const Tensor& a, const Tensor& b) { return cmp_op<4>(a, b); }
Tensor ne_cuda_f32(const Tensor& a, const Tensor& b) { return cmp_op<5>(a, b); }

Tensor abs_cuda_f32(const Tensor& a)      { return unary_op(a, [] __device__(float x) { return fabsf(x); }); }
Tensor sqrt_cuda_f32(const Tensor& a)     { return unary_op(a, [] __device__(float x) { return sqrtf(x); }); }
Tensor relu_cuda_f32(const Tensor& a)     { return unary_op(a, [] __device__(float x) { return fmaxf(x, 0.0f); }); }
Tensor ln_cuda_f32(const Tensor& a)       { return unary_op(a, [] __device__(float x) { return logf(x); }); }
Tensor exp_cuda_f32(const Tensor& a)      { return unary_op(a, [] __device__(float x) { return expf(x); }); }
Tensor sin_cuda_f32(const Tensor& a)      { return unary_op(a, [] __device__(float x) { return sinf(x); }); }
Tensor asin_cuda_f32(const Tensor& a)     { return unary_op(a, [] __device__(float x) { return asinf(x); }); }
Tensor cos_cuda_f32(const Tensor& a)      { return unary_op(a, [] __device__(float x) { return cosf(x); }); }
Tensor acos_cuda_f32(const Tensor& a)     { return unary_op(a, [] __device__(float x) { return acosf(x); }); }
Tensor tan_cuda_f32(const Tensor& a)      { return unary_op(a, [] __device__(float x) { return tanf(x); }); }
Tensor atan_cuda_f32(const Tensor& a)     { return unary_op(a, [] __device__(float x) { return atanf(x); }); }
Tensor tanh_cuda_f32(const Tensor& a)     { return unary_op(a, [] __device__(float x) { return tanhf(x); }); }
Tensor sinh_cuda_f32(const Tensor& a)     { return unary_op(a, [] __device__(float x) { return sinhf(x); }); }
Tensor cosh_cuda_f32(const Tensor& a)     { return unary_op(a, [] __device__(float x) { return coshf(x); }); }
Tensor sigmoid_cuda_f32(const Tensor& a)  { return unary_op(a, [] __device__(float x) { return 1.0f / (1.0f + expf(-x)); }); }
Tensor softplus_cuda_f32(const Tensor& a) { return unary_op(a, [] __device__(float x) { return logf(1.0f + expf(x)); }); }

Tensor sum_cuda_f32(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("sum_cuda_f32: only dim=-1");
    Tensor t_gpu = t.device().is_cuda() ? t : t.to(Device(DeviceType::CUDA));
    float val = gpu_sum(cuda_ptr_f32(t_gpu), t.numel());
    Tensor out({1}, DType::Float32);
    reinterpret_cast<float*>(out.impl->data->data.get())[0] = val;
    return out;
}

Tensor mean_cuda_f32(const Tensor& t, int dim) {
    Tensor s = sum_cuda_f32(t, dim);
    reinterpret_cast<float*>(s.impl->data->data.get())[0] /= (float)t.numel();
    return s;
}

Tensor max_cuda_f32(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("max_cuda_f32: only dim=-1");
    Tensor t_gpu = t.device().is_cuda() ? t : t.to(Device(DeviceType::CUDA));
    float val = gpu_max(cuda_ptr_f32(t_gpu), t.numel());
    Tensor out({1}, DType::Float32);
    reinterpret_cast<float*>(out.impl->data->data.get())[0] = val;
    return out;
}

Tensor min_cuda_f32(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("min_cuda_f32: only dim=-1");
    Tensor t_gpu = t.device().is_cuda() ? t : t.to(Device(DeviceType::CUDA));
    float val = gpu_min(cuda_ptr_f32(t_gpu), t.numel());
    Tensor out({1}, DType::Float32);
    reinterpret_cast<float*>(out.impl->data->data.get())[0] = val;
    return out;
}

#endif
