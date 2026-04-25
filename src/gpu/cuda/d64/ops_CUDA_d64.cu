#include "ops_cuda_d64.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <limits>

#define BLOCK 256
#define TM    4
#define TN    4
#define BM    64
#define BN    64
#define BK    8

namespace {

inline double* cuda_ptr_d64(const Tensor& t) {
    return reinterpret_cast<double*>(t.impl->data->data.get()) + t.impl->offset;
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
    m.ndim       = (int)out_shape.size();
    auto out_st  = build_strides(out_shape);
    auto a_st    = build_strides(a_shape);
    auto b_st    = build_strides(b_shape);
    size_t pad_a = out_shape.size() - a_shape.size();
    size_t pad_b = out_shape.size() - b_shape.size();
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
__global__ void binary_flat_kernel(const double* __restrict__ a,
                                   const double* __restrict__ b,
                                   double*       __restrict__ out,
                                   size_t n, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(a[i], b[i]);
}

template<typename Op>
__global__ void binary_bcast_kernel(const double* __restrict__ a,
                                    const double* __restrict__ b,
                                    double*       __restrict__ out,
                                    size_t n, BcastMeta m, Op op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t ai, bi;
    unravel(i, m, ai, bi);
    out[i] = op(a[ai], b[bi]);
}

template<typename Op>
__global__ void unary_kernel(const double* __restrict__ in,
                             double*       __restrict__ out,
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

    Tensor out(out_shape, DType::Double64);
    out.impl->data = Storage::allocate(n, DType::Double64, Device(DeviceType::CUDA));

    Tensor A_gpu = A.device().is_cuda() ? A : A.to(Device(DeviceType::CUDA));
    Tensor B_gpu = B.device().is_cuda() ? B : B.to(Device(DeviceType::CUDA));

    const double* ap   = cuda_ptr_d64(A_gpu);
    const double* bp   = cuda_ptr_d64(B_gpu);
    double*       outp = cuda_ptr_d64(out);

    dim3 grid((unsigned)((n + BLOCK - 1) / BLOCK));

    bool flat = A.is_contiguous() && B.is_contiguous() && a_shape == out_shape && b_shape == out_shape;
    if (flat) {
        binary_flat_kernel<<<grid, BLOCK>>>(ap, bp, outp, n, op);
    } else {
        auto meta = make_meta(a_shape, b_shape, out_shape);
        binary_bcast_kernel<<<grid, BLOCK>>>(ap, bp, outp, n, meta, op);
    }
    check_launch("binary_op_d64");
    return out;
}

template<typename Op>
Tensor unary_op(const Tensor& A, Op op) {
    size_t n = A.numel();
    Tensor out(A.shape(), DType::Double64);
    out.impl->data = Storage::allocate(n, DType::Double64, Device(DeviceType::CUDA));

    Tensor A_gpu  = A.device().is_cuda() ? A : A.to(Device(DeviceType::CUDA));
    const double* ap   = cuda_ptr_d64(A_gpu);
    double*       outp = cuda_ptr_d64(out);

    dim3 grid((unsigned)((n + BLOCK - 1) / BLOCK));
    unary_kernel<<<grid, BLOCK>>>(ap, outp, n, op);
    check_launch("unary_op_d64");
    return out;
}


template<int PRED>
__device__ inline bool cmp_pred(double a, double b) {
    if constexpr (PRED == 0) return a <  b;
    if constexpr (PRED == 1) return a <= b;
    if constexpr (PRED == 2) return a >  b;
    if constexpr (PRED == 3) return a >= b;
    if constexpr (PRED == 4) return a == b;
    if constexpr (PRED == 5) return a != b;
    return false;
}

template<int PRED>
__global__ void cmp_flat_kernel(const double* __restrict__ a,
                                const double* __restrict__ b,
                                double*       __restrict__ out,
                                size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = cmp_pred<PRED>(a[i], b[i]) ? 1.0 : 0.0;
}

template<int PRED>
__global__ void cmp_bcast_kernel(const double* __restrict__ a,
                                 const double* __restrict__ b,
                                 double*       __restrict__ out,
                                 size_t n, BcastMeta m) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t ai, bi;
    unravel(i, m, ai, bi);
    out[i] = cmp_pred<PRED>(a[ai], b[bi]) ? 1.0 : 0.0;
}

template<int PRED>
Tensor cmp_op(const Tensor& A, const Tensor& B) {
    auto a_shape   = A.shape();
    auto b_shape   = B.shape();
    auto out_shape = broadcast_shape(a_shape, b_shape);
    size_t n = 1;
    for (size_t s : out_shape) n *= s;

    Tensor out(out_shape, DType::Double64);
    out.impl->data = Storage::allocate(n, DType::Double64, Device(DeviceType::CUDA));

    Tensor A_gpu = A.device().is_cuda() ? A : A.to(Device(DeviceType::CUDA));
    Tensor B_gpu = B.device().is_cuda() ? B : B.to(Device(DeviceType::CUDA));

    dim3 grid((unsigned)((n + BLOCK - 1) / BLOCK));

    bool flat = A.is_contiguous() && B.is_contiguous() && a_shape == out_shape && b_shape == out_shape;
    if (flat) {
        cmp_flat_kernel<PRED><<<grid, BLOCK>>>(cuda_ptr_d64(A_gpu), cuda_ptr_d64(B_gpu), cuda_ptr_d64(out), n);
    } else {
        auto meta = make_meta(a_shape, b_shape, out_shape);
        cmp_bcast_kernel<PRED><<<grid, BLOCK>>>(cuda_ptr_d64(A_gpu), cuda_ptr_d64(B_gpu), cuda_ptr_d64(out), n, meta);
    }
    check_launch("cmp_op_d64");
    return out;
}


__global__ void matmul_kernel(const double* __restrict__ A,
                              const double* __restrict__ B,
                              double*       __restrict__ C,
                              int M, int K, int N) {
    __shared__ double As[BM][BK];
    __shared__ double Bs[BK][BN];

    int block_row  = blockIdx.y * BM;
    int block_col  = blockIdx.x * BN;
    int thread_row = threadIdx.x / (BN / TN);
    int thread_col = threadIdx.x % (BN / TN);

    double acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int s = 0; s < BM * BK; s += BLOCK) {
            int idx  = s + threadIdx.x;
            int r    = idx / BK;
            int c    = idx % BK;
            int gRow = block_row + r;
            int gCol = k0 + c;
            As[r][c] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0;
        }
        for (int s = 0; s < BK * BN; s += BLOCK) {
            int idx  = s + threadIdx.x;
            int r    = idx / BN;
            int c    = idx % BN;
            int gRow = k0 + r;
            int gCol = block_col + c;
            Bs[r][c] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            double b_reg[TN];
            #pragma unroll
            for (int n = 0; n < TN; ++n)
                b_reg[n] = Bs[k][thread_col * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                double a_reg = As[thread_row * TM + m][k];
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


static double gpu_sum(const double* d, size_t n) {
    double* d_out;
    check(cudaMalloc(&d_out, sizeof(double)), "cudaMalloc sum d64");
    void* d_tmp = nullptr; size_t tmp_bytes = 0;
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaMalloc(&d_tmp, tmp_bytes), "cudaMalloc sum tmp d64");
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaDeviceSynchronize(), "sum sync d64");
    double r; check(cudaMemcpy(&r, d_out, sizeof(double), cudaMemcpyDeviceToHost), "sum copy d64");
    cudaFree(d_tmp); cudaFree(d_out);
    return r;
}

static double gpu_max(const double* d, size_t n) {
    double* d_out;
    check(cudaMalloc(&d_out, sizeof(double)), "cudaMalloc max d64");
    void* d_tmp = nullptr; size_t tmp_bytes = 0;
    cub::DeviceReduce::Max(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaMalloc(&d_tmp, tmp_bytes), "cudaMalloc max tmp d64");
    cub::DeviceReduce::Max(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaDeviceSynchronize(), "max sync d64");
    double r; check(cudaMemcpy(&r, d_out, sizeof(double), cudaMemcpyDeviceToHost), "max copy d64");
    cudaFree(d_tmp); cudaFree(d_out);
    return r;
}

static double gpu_min(const double* d, size_t n) {
    double* d_out;
    check(cudaMalloc(&d_out, sizeof(double)), "cudaMalloc min d64");
    void* d_tmp = nullptr; size_t tmp_bytes = 0;
    cub::DeviceReduce::Min(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaMalloc(&d_tmp, tmp_bytes), "cudaMalloc min tmp d64");
    cub::DeviceReduce::Min(d_tmp, tmp_bytes, d, d_out, (int)n);
    check(cudaDeviceSynchronize(), "min sync d64");
    double r; check(cudaMemcpy(&r, d_out, sizeof(double), cudaMemcpyDeviceToHost), "min copy d64");
    cudaFree(d_tmp); cudaFree(d_out);
    return r;
}

} /

Tensor add_cuda_d64(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [] __device__(double x, double y) { return x + y; });
}
Tensor sub_cuda_d64(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [] __device__(double x, double y) { return x - y; });
}
Tensor mul_cuda_d64(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [] __device__(double x, double y) { return x * y; });
}
Tensor div_cuda_d64(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [] __device__(double x, double y) { return x / y; });
}
Tensor pow_cuda_d64(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [] __device__(double x, double y) { return pow(x, y); });
}

Tensor matmul_cuda_d64(const Tensor& A, const Tensor& B) {
    if (A.shape().size() != 2 || B.shape().size() != 2)
        throw std::runtime_error("matmul_cuda_d64: only 2D");
    int M = (int)A.shape()[0], K = (int)A.shape()[1], N = (int)B.shape()[1];
    if (K != (int)B.shape()[0]) throw std::runtime_error("matmul_cuda_d64: shape mismatch");

    Tensor A_gpu = A.device().is_cuda() ? A : A.to(Device(DeviceType::CUDA));
    Tensor B_gpu = B.device().is_cuda() ? B : B.to(Device(DeviceType::CUDA));

    Tensor C({(size_t)M, (size_t)N}, DType::Double64);
    C.impl->data = Storage::allocate((size_t)M * N, DType::Double64, Device(DeviceType::CUDA));

    static_assert(BM % TM == 0 && BN % TN == 0, "tile dims must divide block dims");
    constexpr int THREADS = (BM / TM) * (BN / TN);
    dim3 block(THREADS);
    dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((M + BM - 1) / BM));
    matmul_kernel<<<grid, block>>>(cuda_ptr_d64(A_gpu), cuda_ptr_d64(B_gpu), cuda_ptr_d64(C), M, K, N);
    check_launch("matmul_cuda_d64");
    return C;
}

Tensor lt_cuda_d64(const Tensor& a, const Tensor& b) { return cmp_op<0>(a, b); }
Tensor le_cuda_d64(const Tensor& a, const Tensor& b) { return cmp_op<1>(a, b); }
Tensor gt_cuda_d64(const Tensor& a, const Tensor& b) { return cmp_op<2>(a, b); }
Tensor ge_cuda_d64(const Tensor& a, const Tensor& b) { return cmp_op<3>(a, b); }
Tensor eq_cuda_d64(const Tensor& a, const Tensor& b) { return cmp_op<4>(a, b); }
Tensor ne_cuda_d64(const Tensor& a, const Tensor& b) { return cmp_op<5>(a, b); }

Tensor abs_cuda_d64(const Tensor& a)      { return unary_op(a, [] __device__(double x) { return fabs(x); }); }
Tensor sqrt_cuda_d64(const Tensor& a)     { return unary_op(a, [] __device__(double x) { return sqrt(x); }); }
Tensor relu_cuda_d64(const Tensor& a)     { return unary_op(a, [] __device__(double x) { return fmax(x, 0.0); }); }
Tensor ln_cuda_d64(const Tensor& a)       { return unary_op(a, [] __device__(double x) { return log(x); }); }
Tensor exp_cuda_d64(const Tensor& a)      { return unary_op(a, [] __device__(double x) { return exp(x); }); }
Tensor sin_cuda_d64(const Tensor& a)      { return unary_op(a, [] __device__(double x) { return sin(x); }); }
Tensor asin_cuda_d64(const Tensor& a)     { return unary_op(a, [] __device__(double x) { return asin(x); }); }
Tensor cos_cuda_d64(const Tensor& a)      { return unary_op(a, [] __device__(double x) { return cos(x); }); }
Tensor acos_cuda_d64(const Tensor& a)     { return unary_op(a, [] __device__(double x) { return acos(x); }); }
Tensor tan_cuda_d64(const Tensor& a)      { return unary_op(a, [] __device__(double x) { return tan(x); }); }
Tensor atan_cuda_d64(const Tensor& a)     { return unary_op(a, [] __device__(double x) { return atan(x); }); }
Tensor tanh_cuda_d64(const Tensor& a)     { return unary_op(a, [] __device__(double x) { return tanh(x); }); }
Tensor sinh_cuda_d64(const Tensor& a)     { return unary_op(a, [] __device__(double x) { return sinh(x); }); }
Tensor cosh_cuda_d64(const Tensor& a)     { return unary_op(a, [] __device__(double x) { return cosh(x); }); }
Tensor sigmoid_cuda_d64(const Tensor& a)  { return unary_op(a, [] __device__(double x) { return 1.0 / (1.0 + exp(-x)); }); }
Tensor softplus_cuda_d64(const Tensor& a) { return unary_op(a, [] __device__(double x) { return log(1.0 + exp(x)); }); }

Tensor sum_cuda_d64(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("sum_cuda_d64: only dim=-1");
    Tensor t_gpu = t.device().is_cuda() ? t : t.to(Device(DeviceType::CUDA));
    double val = gpu_sum(cuda_ptr_d64(t_gpu), t.numel());
    Tensor out({1}, DType::Double64);
    reinterpret_cast<double*>(out.impl->data->data.get())[0] = val;
    return out;
}

Tensor mean_cuda_d64(const Tensor& t, int dim) {
    Tensor s = sum_cuda_d64(t, dim);
    reinterpret_cast<double*>(s.impl->data->data.get())[0] /= (double)t.numel();
    return s;
}

Tensor max_cuda_d64(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("max_cuda_d64: only dim=-1");
    Tensor t_gpu = t.device().is_cuda() ? t : t.to(Device(DeviceType::CUDA));
    double val = gpu_max(cuda_ptr_d64(t_gpu), t.numel());
    Tensor out({1}, DType::Double64);
    reinterpret_cast<double*>(out.impl->data->data.get())[0] = val;
    return out;
}

Tensor min_cuda_d64(const Tensor& t, int dim) {
    if (dim != -1) throw std::runtime_error("min_cuda_d64: only dim=-1");
    Tensor t_gpu = t.device().is_cuda() ? t : t.to(Device(DeviceType::CUDA));
    double val = gpu_min(cuda_ptr_d64(t_gpu), t.numel());
    Tensor out({1}, DType::Double64);
    reinterpret_cast<double*>(out.impl->data->data.get())[0] = val;
    return out;
}

#endif
