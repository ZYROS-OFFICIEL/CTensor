#include <vector>
#include <stdexcept>
#include <cmath>
#include "tensor1.h"
#include "autograd.h"
#include <immintrin.h>
#include <cstring>

// helper: produce result shape for elementwise binary op (a and b already padded to same ndim)
static std::vector<size_t> compute_result_shape_padded(const Tensor& a, const Tensor& b) {
    size_t ndim = std::max(a.impl->ndim, b.impl->ndim);
    std::vector<size_t> result(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        result[i] = std::max(a.impl->shape[i], b.impl->shape[i]);
    }
    return result;
}

//------------------Helper ---------------------------------------------
static bool broadcastable(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    size_t na = a.size(), nb = b.size();
    size_t ndim = std::max(na, nb);

    for (size_t i = 0; i < ndim; ++i) {
        size_t da = (i < ndim - na) ? 1 : a[i - (ndim - na)];
        size_t db = (i < ndim - nb) ? 1 : b[i - (ndim - nb)];
        if (da != db && da != 1 && db != 1)
            return false;
    }
    return true;
}

// --- helper: broadcast batch shapes ---
static std::vector<size_t> broadcast_batch_shape_from_vectors(
    const std::vector<size_t>& a,
    const std::vector<size_t>& b) {

    size_t ndimA = a.size();
    size_t ndimB = b.size();
    size_t ndim = std::max(ndimA, ndimB);

    std::vector<size_t> result(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        size_t dimA = (i < ndim - ndimA) ? 1 : a[i - (ndim - ndimA)];
        size_t dimB = (i < ndim - ndimB) ? 1 : b[i - (ndim - ndimB)];
        if (dimA != dimB && dimA != 1 && dimB != 1)
            throw std::invalid_argument("Incompatible batch shapes for broadcasting");
        result[i] = std::max(dimA, dimB);
    }
    return result;
}

static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    size_t na = a.size(), nb = b.size();
    size_t ndim = std::max(na, nb);
    std::vector<size_t> result(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        size_t da = (i < ndim - na) ? 1 : a[i - (ndim - na)];
        size_t db = (i < ndim - nb) ? 1 : b[i - (ndim - nb)];
        if (da != db && da != 1 && db != 1)
            throw std::runtime_error("Incompatible shapes for broadcasting");
        result[i] = std::max(da, db);
    }
    return result;
}

Tensor apply_scalar_op(
    const Tensor& a,
    double scalar,
    std::function<double(double, double)> forward_op,
    std::function<std::shared_ptr<GradFn>(const Tensor&, double)> grad_fn_ctor
) {
    if (!a.impl)
        throw std::runtime_error("apply_scalar_op: null tensor implementation");

    Tensor result(a.shape(), a.impl->dtype, a.requires_grad());

    size_t n = a.numel_();
    for (size_t i = 0; i < n; ++i) {
        double va = read_scalar_at(a.impl->storage->data.get(), i, a.impl->dtype);
        write_scalar_at(result.impl->storage->data.get(), i, result.impl->dtype, forward_op(va, scalar));
    }

    if (a.requires_grad())
        result.impl->grad_fn = grad_fn_ctor(a, scalar);

    return result;
}


Tensor add(const Tensor& a_, const Tensor& b_) {
    if (!a_.impl || !b_.impl)
        throw std::runtime_error("add_: null tensor implementation");

    // --- Step 1: pad shapes to same ndim ---
    size_t ndim_result = std::max(a_.impl->ndim, b_.impl->ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    // --- Step 2: wrap shapes and strides ---
    std::vector<size_t> shape_a(a.impl->shape, a.impl->shape + a.impl->ndim);
    std::vector<size_t> shape_b(b.impl->shape, b.impl->shape + b.impl->ndim);
    std::vector<size_t> strides_a(a.impl->strides, a.impl->strides + a.impl->ndim);
    std::vector<size_t> strides_b(b.impl->strides, b.impl->strides + b.impl->ndim);

    // --- Step 3: check broadcastable & compute result shape ---
    std::vector<size_t> result_shape(ndim_result);
    for (size_t i = 0; i < ndim_result; ++i) {
        size_t da = (i < ndim_result - shape_a.size()) ? 1 : shape_a[i - (ndim_result - shape_a.size())];
        size_t db = (i < ndim_result - shape_b.size()) ? 1 : shape_b[i - (ndim_result - shape_b.size())];
        if (da != db && da != 1 && db != 1)
            throw std::runtime_error("add_: shapes are not broadcastable");
        result_shape[i] = std::max(da, db);
    }

    // --- Step 4: precompute broadcast strides ---
    std::vector<size_t> stride_a_bc(ndim_result);
    std::vector<size_t> stride_b_bc(ndim_result);
    for (size_t i = 0; i < ndim_result; ++i) {
        stride_a_bc[i] = (shape_a[i] == 1 ? 0 : strides_a[i]);
        stride_b_bc[i] = (shape_b[i] == 1 ? 0 : strides_b[i]);
    }

    // --- Step 5: create result tensor ---
    bool req = a.requires_grad() || b.requires_grad();
    Tensor result(result_shape, a.impl->dtype, req);

    // --- Step 6: iterate using flat index and precomputed strides ---
    size_t n = result.numel_();
    for (size_t flat = 0; flat < n; ++flat) {
        size_t index_a = 0, index_b = 0;
        size_t tmp = flat;
        for (int i = (int)ndim_result - 1; i >= 0; --i) {
            size_t idx = tmp % result_shape[i];
            tmp /= result_shape[i];
            index_a += idx * stride_a_bc[i];
            index_b += idx * stride_b_bc[i];
        }

        double va = read_scalar_at(a.impl->storage->data.get(), index_a, a.impl->dtype);
        double vb = read_scalar_at(b.impl->storage->data.get(), index_b, b.impl->dtype);
        write_scalar_at(result.impl->storage->data.get(), flat, result.impl->dtype, va + vb);
    }

    return result;
}

Tensor add_(const Tensor& a_, const Tensor& b_) {
    if (!a_.impl || !b_.impl)
        throw std::runtime_error("add_: null tensor implementation");

    // --- Step 1: pad shapes to same ndim ---
    size_t ndim_result = std::max(a_.impl->ndim, b_.impl->ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    // --- Step 2: wrap shape pointers into vectors ---
    std::vector<size_t> shape_a(a.impl->shape, a.impl->shape + a.impl->ndim);
    std::vector<size_t> shape_b(b.impl->shape, b.impl->shape + b.impl->ndim);

    // --- Step 3: compute broadcasted shape ---
    if (!broadcastable(shape_a, shape_b))
        throw std::runtime_error("add_: shapes are not broadcastable");
    std::vector<size_t> result_shape = broadcast_shape(shape_a, shape_b);

    // --- Step 4: create result tensor ---
    bool req = a_.requires_grad() || b_.requires_grad();
    Tensor result(result_shape, a.impl->dtype, req);
    
    // attach grad_fn if needed (so autograd can traverse)
    if (req) {
        result.impl->grad_fn = std::make_shared<GradAdd>(a_, b_);
    }
    // --- Step 5: iterate over result elements ---
    size_t n = result.numel_();
    std::vector<size_t> idx(ndim_result, 0);

    for (size_t flat = 0; flat < n; ++flat) {
        // convert flat index -> multi-dimensional index
        size_t rem = flat;
        for (int i = (int)ndim_result - 1; i >= 0; --i) {
            idx[i] = rem % result_shape[i];
            rem /= result_shape[i];
        }

        // compute flat index for a and b using broadcasting
        size_t index_a = 0, index_b = 0;
        for (size_t i = 0; i < ndim_result; ++i) {
            size_t idx_a = (a.impl->shape[i] == 1 ? 0 : idx[i]);
            size_t idx_b = (b.impl->shape[i] == 1 ? 0 : idx[i]);
            index_a += idx_a * a.impl->strides[i];
            index_b += idx_b * b.impl->strides[i];
        }

        double va = read_scalar_at(a.impl->storage->data.get(), index_a, a.impl->dtype);
        double vb = read_scalar_at(b.impl->storage->data.get(), index_b, b.impl->dtype);
        write_scalar_at(result.impl->storage->data.get(), flat, result.impl->dtype, va + vb);
    }

    return result;
}
Tensor add_scalar(const Tensor& a, double scalar) {
    if (!a.impl)
        throw std::runtime_error("add_scalar: null tensor implementation");

    Tensor result(a.shape(), a.impl->dtype, a.requires_grad());

    size_t n = a.numel_();
    for (size_t i = 0; i < n; ++i) {
        double va = read_scalar_at(a.impl->storage->data.get(), i, a.impl->dtype);
        write_scalar_at(result.impl->storage->data.get(), i, result.impl->dtype, va + scalar);
    }

    // attach grad_fn if needed (so autograd can traverse)
    /*if (a.requires_grad()) {
        result.impl->grad_fn = std::make_shared<GradMulScalar>(a, scalar);
    }*/

    return result;
}

Tensor diff_(const Tensor& a_, const Tensor& b_) {
    if (!a_.impl || !b_.impl)
        throw std::runtime_error("add_: null tensor implementation");

    // --- Step 1: pad shapes to same ndim ---
    size_t ndim_result = std::max(a_.impl->ndim, b_.impl->ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    // --- Step 2: wrap shape pointers into vectors ---
    std::vector<size_t> shape_a(a.impl->shape, a.impl->shape + a.impl->ndim);
    std::vector<size_t> shape_b(b.impl->shape, b.impl->shape + b.impl->ndim);

    // --- Step 3: compute broadcasted shape ---
    if (!broadcastable(shape_a, shape_b))
        throw std::runtime_error("add_: shapes are not broadcastable");
    std::vector<size_t> result_shape = broadcast_shape(shape_a, shape_b);

    // --- Step 4: create result tensor ---
    // --- Step 4: create result tensor ---
    bool req = a_.requires_grad() || b_.requires_grad();
    Tensor result(result_shape, a.impl->dtype, req);
    
    // attach grad_fn if needed (so autograd can traverse)
    if (req) {
        result.impl->grad_fn = std::make_shared<GradSub>(a_, b_);
    }

    // --- Step 5: iterate over result elements ---
    size_t n = result.numel_();
    std::vector<size_t> idx(ndim_result, 0);

    for (size_t flat = 0; flat < n; ++flat) {
        // convert flat index -> multi-dimensional index
        size_t rem = flat;
        for (int i = (int)ndim_result - 1; i >= 0; --i) {
            idx[i] = rem % result_shape[i];
            rem /= result_shape[i];
        }

        // compute flat index for a and b using broadcasting
        size_t index_a = 0, index_b = 0;
        for (size_t i = 0; i < ndim_result; ++i) {
            size_t idx_a = (a.impl->shape[i] == 1 ? 0 : idx[i]);
            size_t idx_b = (b.impl->shape[i] == 1 ? 0 : idx[i]);
            index_a += idx_a * a.impl->strides[i];
            index_b += idx_b * b.impl->strides[i];
        }

        double va = read_scalar_at(a.impl->storage->data.get(), index_a, a.impl->dtype);
        double vb = read_scalar_at(b.impl->storage->data.get(), index_b, b.impl->dtype);
        write_scalar_at(result.impl->storage->data.get(), flat, result.impl->dtype, va - vb);
    }

    return result;
}
Tensor mult_(const Tensor& a_, const Tensor& b_) {
    if (!a_.impl || !b_.impl)
        throw std::runtime_error("add_: null tensor implementation");

    // --- Step 1: pad shapes to same ndim ---
    size_t ndim_result = std::max(a_.impl->ndim, b_.impl->ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    // --- Step 2: wrap shape pointers into vectors ---
    std::vector<size_t> shape_a(a.impl->shape, a.impl->shape + a.impl->ndim);
    std::vector<size_t> shape_b(b.impl->shape, b.impl->shape + b.impl->ndim);

    // --- Step 3: compute broadcasted shape ---
    if (!broadcastable(shape_a, shape_b))
        throw std::runtime_error("add_: shapes are not broadcastable");
    std::vector<size_t> result_shape = broadcast_shape(shape_a, shape_b);

    // --- Step 4: create result tensor ---
    bool req = a_.requires_grad() || b_.requires_grad();
    Tensor result(result_shape, a.impl->dtype, req);
    
    // attach grad_fn if needed (so autograd can traverse)
    if (req) {
        result.impl->grad_fn = std::make_shared<GradMul>(a_, b_);
    }

    // --- Step 5: iterate over result elements ---
    size_t n = result.numel_();
    std::vector<size_t> idx(ndim_result, 0);

    for (size_t flat = 0; flat < n; ++flat) {
        // convert flat index -> multi-dimensional index
        size_t rem = flat;
        for (int i = (int)ndim_result - 1; i >= 0; --i) {
            idx[i] = rem % result_shape[i];
            rem /= result_shape[i];
        }

        // compute flat index for a and b using broadcasting
        size_t index_a = 0, index_b = 0;
        for (size_t i = 0; i < ndim_result; ++i) {
            size_t idx_a = (a.impl->shape[i] == 1 ? 0 : idx[i]);
            size_t idx_b = (b.impl->shape[i] == 1 ? 0 : idx[i]);
            index_a += idx_a * a.impl->strides[i];
            index_b += idx_b * b.impl->strides[i];
        }

        double va = read_scalar_at(a.impl->storage->data.get(), index_a, a.impl->dtype);
        double vb = read_scalar_at(b.impl->storage->data.get(), index_b, b.impl->dtype);
        write_scalar_at(result.impl->storage->data.get(), flat, result.impl->dtype, va * vb);
    }

    return result;
}
Tensor mult_scalar(const Tensor& a, double scalar) {
    if (!a.impl)
        throw std::runtime_error("mult_scalar: null tensor implementation");

    Tensor result(a.shape(), a.impl->dtype, a.requires_grad());

    size_t n = a.numel_();
    for (size_t i = 0; i < n; ++i) {
        double va = read_scalar_at(a.impl->storage->data.get(), i, a.impl->dtype);
        write_scalar_at(result.impl->storage->data.get(), i, result.impl->dtype, va * scalar);
    }

    // attach grad_fn if needed (so autograd can traverse)
    if (a.requires_grad()) {
        result.impl->grad_fn = std::make_shared<GradMulScalar>(a, scalar);
    }

    return result;
}
Tensor div_(const Tensor& a_, const Tensor& b_) {
    if (!a_.impl || !b_.impl)
        throw std::runtime_error("add_: null tensor implementation");

    // --- Step 1: pad shapes to same ndim ---
    size_t ndim_result = std::max(a_.impl->ndim, b_.impl->ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    // --- Step 2: wrap shape pointers into vectors ---
    std::vector<size_t> shape_a(a.impl->shape, a.impl->shape + a.impl->ndim);
    std::vector<size_t> shape_b(b.impl->shape, b.impl->shape + b.impl->ndim);

    // --- Step 3: compute broadcasted shape ---
    if (!broadcastable(shape_a, shape_b))
        throw std::runtime_error("add_: shapes are not broadcastable");
    std::vector<size_t> result_shape = broadcast_shape(shape_a, shape_b);

    // --- Step 4: create result tensor ---
    bool req = a_.requires_grad() || b_.requires_grad();
    Tensor result(result_shape, a.impl->dtype, req);
    
    // attach grad_fn if needed (so autograd can traverse)
    if (req) {
        result.impl->grad_fn = std::make_shared<GradDiv>(a_, b_);
    }

    // --- Step 5: iterate over result elements ---
    size_t n = result.numel_();
    std::vector<size_t> idx(ndim_result, 0);

    for (size_t flat = 0; flat < n; ++flat) {
        // convert flat index -> multi-dimensional index
        size_t rem = flat;
        for (int i = (int)ndim_result - 1; i >= 0; --i) {
            idx[i] = rem % result_shape[i];
            rem /= result_shape[i];
        }

        // compute flat index for a and b using broadcasting
        size_t index_a = 0, index_b = 0;
        for (size_t i = 0; i < ndim_result; ++i) {
            size_t idx_a = (a.impl->shape[i] == 1 ? 0 : idx[i]);
            size_t idx_b = (b.impl->shape[i] == 1 ? 0 : idx[i]);
            index_a += idx_a * a.impl->strides[i];
            index_b += idx_b * b.impl->strides[i];
        }

        double va = read_scalar_at(a.impl->storage->data.get(), index_a, a.impl->dtype);
        double vb = read_scalar_at(b.impl->storage->data.get(), index_b, b.impl->dtype);
        write_scalar_at(result.impl->storage->data.get(), flat, result.impl->dtype, va / vb);
    }

    return result;
}

Tensor pow_(const Tensor& a_, const Tensor& b_) {
    if (!a_.impl || !b_.impl)
        throw std::runtime_error("add_: null tensor implementation");

    // --- Step 1: pad shapes to same ndim ---
    size_t ndim_result = std::max(a_.impl->ndim, b_.impl->ndim);
    Tensor a = pad_to_ndim(a_, ndim_result);
    Tensor b = pad_to_ndim(b_, ndim_result);

    // --- Step 2: wrap shape pointers into vectors ---
    std::vector<size_t> shape_a(a.impl->shape, a.impl->shape + a.impl->ndim);
    std::vector<size_t> shape_b(b.impl->shape, b.impl->shape + b.impl->ndim);

    // --- Step 3: compute broadcasted shape ---
    if (!broadcastable(shape_a, shape_b))
        throw std::runtime_error("add_: shapes are not broadcastable");
    std::vector<size_t> result_shape = broadcast_shape(shape_a, shape_b);

    // --- Step 4: create result tensor ---
    bool req = a_.requires_grad() || b_.requires_grad();
    Tensor result(result_shape, a.impl->dtype, req);
    
    // attach grad_fn if needed (so autograd can traverse)
    if (req) {
        result.impl->grad_fn = std::make_shared<GradPow>(a_, b_);
    }
    // --- Step 5: iterate over result elements ---
    size_t n = result.numel_();
    std::vector<size_t> idx(ndim_result, 0);

    for (size_t flat = 0; flat < n; ++flat) {
        // convert flat index -> multi-dimensional index
        size_t rem = flat;
        for (int i = (int)ndim_result - 1; i >= 0; --i) {
            idx[i] = rem % result_shape[i];
            rem /= result_shape[i];
        }

        // compute flat index for a and b using broadcasting
        size_t index_a = 0, index_b = 0;
        for (size_t i = 0; i < ndim_result; ++i) {
            size_t idx_a = (a.impl->shape[i] == 1 ? 0 : idx[i]);
            size_t idx_b = (b.impl->shape[i] == 1 ? 0 : idx[i]);
            index_a += idx_a * a.impl->strides[i];
            index_b += idx_b * b.impl->strides[i];
        }

        double va = read_scalar_at(a.impl->storage->data.get(), index_a, a.impl->dtype);
        double vb = read_scalar_at(b.impl->storage->data.get(), index_b, b.impl->dtype);
        write_scalar_at(result.impl->storage->data.get(), flat, result.impl->dtype, pow(va, vb));
    }

    return result;
}
Tensor matmul_(const Tensor& A, const Tensor& B) {
    if (A.impl->ndim < 1 || B.impl->ndim < 1)
        throw std::invalid_argument("matmul: tensors must have at least 1 dimension");

    // shapes & strides pointers available in Tensor
    // extract m, k, n
    size_t kA = A.impl->shape[A.impl->ndim - 1];
    size_t kB = (B.impl->ndim >= 2) ? B.impl->shape[B.impl->ndim - 2] : 1;
    if (kA != kB) throw std::invalid_argument("matmul: inner dimensions do not match.");

    size_t m = (A.impl->ndim >= 2) ? A.impl->shape[A.impl->ndim - 2] : 1;
    size_t n = (B.impl->ndim >= 2) ? B.impl->shape[B.impl->ndim - 1] : 1;

    // batch dims vectors (left to right)
    std::vector<size_t> batchA;
    if (A.impl->ndim > 2) batchA = std::vector<size_t>(A.impl->shape, A.impl->shape + (A.impl->ndim - 2));
    std::vector<size_t> batchB;
    if (B.impl->ndim > 2) batchB = std::vector<size_t>(B.impl->shape, B.impl->shape + (B.impl->ndim - 2));

    std::vector<size_t> batchShape = broadcast_batch_shape_from_vectors(batchA, batchB);

    // output shape
    std::vector<size_t> outShape = batchShape;
    outShape.push_back(m);
    outShape.push_back(n);
    bool req = A.requires_grad() || B.requires_grad();
    Tensor C(outShape, A.impl->dtype, req);
    
    // attach grad_fn if needed (so autograd can traverse)
    if (req) {
        C.impl->grad_fn = std::make_shared<GradMatMul>(A, B);
    }
    // convenience
    size_t batchRank = batchShape.size();
    size_t totalBatch = 1;
    for (auto s : batchShape) totalBatch *= s;

    // precompute multipliers for batch multi-index conversion
    std::vector<size_t> batchMul(batchRank, 1);
    for (int i = (int)batchRank - 2; i >= 0; --i) batchMul[i] = batchMul[i+1] * batchShape[i+1];

    // compute base offset (in elements) for A/B given broadcast-aware batchIndex
    auto compute_base_offset = [&](const Tensor& T, const std::vector<size_t>& T_batch_shape,
                                   const std::vector<size_t>& fullBatchShape,
                                   const std::vector<size_t>& batchIndex) -> size_t {
        size_t offset = 0;
        size_t pad = (fullBatchShape.size() > T_batch_shape.size()) ? (fullBatchShape.size() - T_batch_shape.size()) : 0;
        for (size_t d = 0; d < fullBatchShape.size(); ++d) {
            size_t dimT = (d < pad) ? 1 : T_batch_shape[d - pad];
            size_t idxT = (dimT == 1) ? 0 : batchIndex[d];
            if (d >= pad) offset += idxT * T.impl->strides[d - pad];
        }
        return offset;
    };

    std::vector<size_t> batchIndex(batchRank, 0);
    for (size_t batch = 0; batch < totalBatch; ++batch) {
        // batch -> multi-index
        size_t rem = batch;
        for (size_t d = 0; d < batchRank; ++d) {
            batchIndex[d] = rem / batchMul[d];
            rem = rem % batchMul[d];
        }

        size_t baseA = compute_base_offset(A, batchA, batchShape, batchIndex);
        size_t baseB = compute_base_offset(B, batchB, batchShape, batchIndex);

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t t = 0; t < kA; ++t) {
                    size_t offA = baseA + i * A.impl->strides[A.impl->ndim - 2] + t * A.impl->strides[A.impl->ndim - 1];
                    size_t offB = baseB + t * B.impl->strides[B.impl->ndim - 2] + j * B.impl->strides[B.impl->ndim - 1];
                    double va = read_scalar_at(A.impl->storage->data.get(), offA, A.impl->dtype);
                    double vb = read_scalar_at(B.impl->storage->data.get(), offB, B.impl->dtype);
                    sum += va * vb;
                }
                // write to C at (batchIndex..., i, j)
                size_t offC = 0;
                for (size_t d = 0; d < batchRank; ++d) offC += batchIndex[d] * C.impl->strides[d];
                offC += i * C.impl->strides[batchRank];
                offC += j * C.impl->strides[batchRank + 1];
                write_scalar_at(C.impl->storage->data.get(), offC, C.impl->dtype, sum);
            }
        }
    }

    return C;
}
Tensor sum(const Tensor& t, int dim ) {
    if (dim == -1) {
        // reduce all elements
        double s = 0.0;
        size_t n = t.numel_();
        for (size_t i = 0; i < n; ++i)
            s += read_scalar_at(t.impl->storage->data.get(), i, t.impl->dtype);
        bool req = t.requires_grad();
        Tensor out({1}, t.impl->dtype, req);
        write_scalar_at(out.impl->storage->data.get(), 0, t.impl->dtype, s);
        if (req)
            out.impl->grad_fn = std::make_shared<GradSum>(t, -1);
        return out;
    } else {
        if (dim >= (int)t.impl->ndim)
            throw std::invalid_argument("sum: invalid dimension");

        std::vector<size_t> new_shape(t.impl->shape, t.impl->shape + t.impl->ndim);
        new_shape.erase(new_shape.begin() + dim);
        if (new_shape.empty()) new_shape.push_back(1);
        bool req = t.requires_grad();
        Tensor out(new_shape, t.impl->dtype, req);
        memset(out.impl->storage->data.get(), 0, out.numel_() * dtype_size(t.impl->dtype));

        // manual reduction
        size_t outer = 1, inner = 1, reduce_dim = t.impl->shape[dim];
        for (size_t i = 0; i < (size_t)dim; ++i) outer *= t.impl->shape[i];
        for (size_t i = dim + 1; i < t.impl->ndim; ++i) inner *= t.impl->shape[i];

        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                double s = 0.0;
                for (size_t r = 0; r < reduce_dim; ++r) {
                    size_t idx = o * reduce_dim * inner + r * inner + i;
                    s += read_scalar_at(t.impl->storage->data.get(), idx, t.impl->dtype);
                }
                size_t out_idx = o * inner + i;
                write_scalar_at(out.impl->storage->data.get(), out_idx, out.impl->dtype, s);
            }
        }

        return out;
    }
}
Tensor mean(const Tensor& t, int dim ) {
    Tensor s = sum(t, dim);
    double denom = (dim == -1) ? (double)t.numel_() : (double)t.impl->shape[dim];
    size_t n = s.numel_();
    for (size_t i = 0; i < n; ++i) {
        double v = read_scalar_at(s.impl->storage->data.get(), i, s.impl->dtype);
        write_scalar_at(s.impl->storage->data.get(), i, s.impl->dtype, v / denom);
    }
    return s;
}
Tensor max(const Tensor& t, int dim ) {
    if (dim == -1) {
        double m = read_scalar_at(t.impl->storage->data.get(), 0, t.impl->dtype);
        for (size_t i = 1; i < t.numel_(); ++i) {
            double v = read_scalar_at(t.impl->storage->data.get(), i, t.impl->dtype);
            if (v > m) m = v;
        }
        Tensor out({1}, t.impl->dtype, false);
        write_scalar_at(out.impl->storage->data.get(), 0, t.impl->dtype, m);
        return out;
    } else {
        if (dim >= (int)t.impl->ndim)
            throw std::invalid_argument("max: invalid dimension");

        std::vector<size_t> new_shape(t.impl->shape, t.impl->shape + t.impl->ndim);
        new_shape.erase(new_shape.begin() + dim);
        if (new_shape.empty()) new_shape.push_back(1);
        Tensor out(new_shape, t.impl->dtype, false);

        size_t outer = 1, inner = 1, reduce_dim = t.impl->shape[dim];
        for (size_t i = 0; i < (size_t)dim; ++i) outer *= t.impl->shape[i];
        for (size_t i = dim + 1; i < t.impl->ndim; ++i) inner *= t.impl->shape[i];

        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                double m = -INFINITY;
                for (size_t r = 0; r < reduce_dim; ++r) {
                    size_t idx = o * reduce_dim * inner + r * inner + i;
                    double v = read_scalar_at(t.impl->storage->data.get(), idx, t.impl->dtype);
                    if (v > m) m = v;
                }
                size_t out_idx = o * inner + i;
                write_scalar_at(out.impl->storage->data.get(), out_idx, out.impl->dtype, m);
            }
        }

        return out;
    }
}
Tensor min(const Tensor& t, int dim ) {
    if (dim == -1) {
        double m = read_scalar_at(t.impl->storage->data.get(), 0, t.impl->dtype);
        for (size_t i = 1; i < t.numel_(); ++i) {
            double v = read_scalar_at(t.impl->storage->data.get(), i, t.impl->dtype);
            if (v < m) m = v;
        }
        Tensor out({1}, t.impl->dtype, false);
        write_scalar_at(out.impl->storage->data.get(), 0, t.impl->dtype, m);
        return out;
    } else {
        if (dim >= (int)t.impl->ndim)
            throw std::invalid_argument("min: invalid dimension");

        std::vector<size_t> new_shape(t.impl->shape, t.impl->shape + t.impl->ndim);
        new_shape.erase(new_shape.begin() + dim);
        if (new_shape.empty()) new_shape.push_back(1);
        Tensor out(new_shape, t.impl->dtype, false);

        size_t outer = 1, inner = 1, reduce_dim = t.impl->shape[dim];
        for (size_t i = 0; i < (size_t)dim; ++i) outer *= t.impl->shape[i];
        for (size_t i = dim + 1; i < t.impl->ndim; ++i) inner *= t.impl->shape[i];

        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                double m = INFINITY;
                for (size_t r = 0; r < reduce_dim; ++r) {
                    size_t idx = o * reduce_dim * inner + r * inner + i;
                    double v = read_scalar_at(t.impl->storage->data.get(), idx, t.impl->dtype);
                    if (v < m) m = v;
                }
                size_t out_idx = o * inner + i;
                write_scalar_at(out.impl->storage->data.get(), out_idx, out.impl->dtype, m);
            }
        }

        return out;
    }
}
static Tensor cat(const std::vector<Tensor>& tensors, size_t dim) {
    if (tensors.empty()) throw std::invalid_argument("cat: empty tensor list.");

    // Check compatibility
    size_t* ref_shape = tensors[0].impl->shape;
    size_t ndim = tensors[0].impl->ndim;
    for (size_t i = 1; i < tensors.size(); ++i) {
        for (size_t d = 0; d < ndim; ++d) {
            if (d != dim && tensors[i].impl->shape[d] != ref_shape[d])
                throw std::invalid_argument("cat: shape mismatch except along concatenation dim.");
        }
    }

    // Compute new shape
    std::vector<size_t> new_shape(ndim);
    for (size_t d = 0; d < ndim; ++d)
        new_shape[d] = ref_shape[d];
    new_shape[dim] = 0;
    for (auto& t : tensors)
        new_shape[dim] += t.impl->shape[dim];

    // Create output
    Tensor out(new_shape, tensors[0].impl->dtype);
    size_t offset = 0;

    for (auto& t : tensors) {
        size_t n = t.numel_();  // Use your helper if defined
        size_t bytes = n * dtype_size(t.impl->dtype);
        std::memcpy((char*)out.impl->storage->data.get() + offset, t.impl->storage->data.get(), bytes);
        offset += bytes;
    }

    return out;
}

Tensor operator+(const Tensor& a, const Tensor& b) { return add_(a,b); }
Tensor operator+(const Tensor& a, double scalar) { return add_scalar(a,scalar); }
Tensor operator+(double scalar,const Tensor& a ) { return add_scalar(a,scalar); }

Tensor operator-(const Tensor& a, const Tensor& b) { return diff_(a,b); }

// Multiplication
Tensor operator*(const Tensor& a, const Tensor& b) { return mult_(a,b); }
Tensor operator*(const Tensor& a, double scalar) { return mult_scalar(a,scalar); }
Tensor operator*(double scalar,const Tensor& a ) { return mult_scalar(a,scalar); }

Tensor operator/(const Tensor& a, const Tensor& b) { return div_(a,b); }
Tensor operator^(const Tensor& a, const Tensor& b) { return pow_(a,b); }
