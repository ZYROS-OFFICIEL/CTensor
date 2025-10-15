#pragma once
#include "tensors.h"
#include "ops.h"

// ---------- view / reshape / indexing ----------
// create 1D range [start, start+step*(n-1)] -> shape {n}
static Tensor Tensor::arange(double start, double end, double step = 1.0, DType dtype = DType::Float32) {
    if (step == 0.0) throw std::invalid_argument("arange: step must be non-zero");
    double span = end - start;
    if ((span > 0 && step < 0) || (span < 0 && step > 0)) return Tensor(); // empty
    size_t n = 0;
    if ( (step > 0 && start < end) || (step < 0 && start > end) ) {
        n = static_cast<size_t>(std::ceil(std::abs(span) / std::abs(step)));
    }
    Tensor t({n}, dtype, false);
    for (size_t i = 0; i < n; ++i) {
        double v = start + i * step;
        write_scalar_at(t.data.get(), i, dtype, v);
    }
    return t;
}
// reshape (returns a view with new shape but assumes contiguous layout)
Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_numel = 1;
    for (auto s : new_shape) new_numel *= s;
    if (new_numel != numel_()) throw std::invalid_argument("reshape: total number of elements must remain constant.");
    Tensor out = *this; // shallow copy
    out.shape = new_shape;
    out.ndim = out.shape.size();
    // recompute contiguous strides for new shape
    out.strides.assign(out.ndim, 0);
    if (out.ndim > 0) {
        out.strides[out.ndim - 1] = 1;
        for (int i = (int)out.ndim - 2; i >= 0; --i)
            out.strides[i] = out.strides[i + 1] * out.shape[i + 1];
    }
    return out;
}
// select (index along dimension) -> returns view with that dim removed and data pointer alias shifted
Tensor Tensor::select(size_t dim, size_t index) const {
    if (dim >= ndim) throw std::out_of_range("select: dim out of range");
    if (index >= shape[dim]) throw std::out_of_range("select: index out of range");
    // compute element offset in elements
    size_t elem_offset = 0;
    for (size_t d = 0; d < dim; ++d) elem_offset += 0 * strides[d]; // no-op kept for clarity
    elem_offset += index * strides[dim];
    // create aliasing shared_ptr that points to base + bytes offset, but shares ownership with original
    size_t byte_offset = elem_offset * dtype_size(dtype);
    std::shared_ptr<void> new_data;
    if (data) {
        new_data = std::shared_ptr<void>(data, static_cast<char*>(data.get()) + byte_offset);
    } else {
        new_data.reset();
    }
    Tensor out = *this; // shallow copy
    // remove the selected dimension
    std::vector<size_t> new_shape;
    std::vector<size_t> new_strides;
    new_shape.reserve(ndim ? ndim - 1 : 0);
    new_strides.reserve(ndim ? ndim - 1 : 0);
    for (size_t d = 0; d < ndim; ++d) {
        if (d == dim) continue;
        new_shape.push_back(shape[d]);
        new_strides.push_back(strides[d]);
    }
    // special case: if result is 0-dim, keep shape empty (ndim=0) or shape {1}? we'll allow ndim=0.
    out.shape = std::move(new_shape);
    out.strides = std::move(new_strides);
    out.ndim = out.shape.size();
    out.data = std::move(new_data);
    // grad: not adjusted (leave null or share original - for now keep no grad)
    out.grad.reset();
    return out;
}
// remove all dims == 1
Tensor Tensor::squeeze() const {
    std::vector<size_t> new_shape;
    std::vector<size_t> new_strides;
    for (size_t i = 0; i < ndim; ++i) {
        if (shape[i] != 1) {
            new_shape.push_back(shape[i]);
            new_strides.push_back(strides[i]);
        }
    }
    if (new_shape.empty()) {
        // scalar view: shape vector empty, ndim = 0
        Tensor out = *this;
        out.shape.clear();
        out.strides.clear();
        out.ndim = 0;
        return out;
    }
    Tensor out = *this;
    out.shape = std::move(new_shape);
    out.strides = std::move(new_strides);
    out.ndim = out.shape.size();
    return out;
}
// insert a dim of size 1 at position dim (0..ndim inclusive)
Tensor Tensor::unsqueeze(size_t dim) const {
    if (dim > ndim) throw std::out_of_range("unsqueeze: dim out of range");
    std::vector<size_t> new_shape;
    std::vector<size_t> new_strides;
    new_shape.reserve(ndim + 1);
    new_strides.reserve(ndim + 1);
    for (size_t i = 0; i < dim; ++i) {
        new_shape.push_back(shape[i]);
        new_strides.push_back(strides[i]);
    }
    // inserted dim: size 1. Its stride can be anything; to remain consistent with view semantics, set stride to (dim < ndim ? strides[dim] : 1)
    size_t ins_stride = (dim < ndim ? strides[dim] : (ndim ? strides.back() * shape.back() : 1));
    new_shape.push_back(1);
    new_strides.push_back(ins_stride);
    for (size_t i = dim; i < ndim; ++i) {
        new_shape.push_back(shape[i]);
        new_strides.push_back(strides[i]);
    }
    Tensor out = *this;
    out.shape = std::move(new_shape);
    out.strides = std::move(new_strides);
    out.ndim = out.shape.size();
    return out;
}
// flatten -> 1D view
Tensor Tensor::flatten() const {
    Tensor out = *this;
    size_t n = numel_();
    out.shape = { n };
    out.strides = { 1 };
    out.ndim = 1;
    return out;
}