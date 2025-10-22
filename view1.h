#pragma once
#include "tensor1.h"

Tensor Tensor::permute(const std::vector<size_t>& dims) const {
if (!impl)
    throw std::runtime_error("permute: tensor has no implementation");
if (dims.size() != impl->ndim)
    throw std::invalid_argument("permute: dims size must match ndim.");
std::vector<bool> seen(impl->ndim, false);
for (auto d : dims) {
    if (d >= impl->ndim || seen[d])
        throw std::invalid_argument("permute: invalid or duplicate dim.");
    seen[d] = true;
}
std::vector<size_t> new_shape(impl->ndim);
std::vector<size_t> new_strides(impl->ndim);
for (size_t i = 0; i < impl->ndim; ++i) {
    new_shape[i] = impl->shape[dims[i]];
    new_strides[i] = impl->strides[dims[i]];
}
// use the new "view constructor"
Tensor out;
out.impl = std::make_shared<Tensorimpl>(
    impl->storage,
    impl->offset,
    new_shape,
    new_strides,
    impl->dtype,
    impl->requires_grad);
return out;
}
    // ------------- arange -------------
static Tensor Tensor::arange(double start, double end, double step = 1.0, DType dtype = DType::Float32) {
    if (step == 0.0) throw std::invalid_argument("step must be non-zero");
    std::vector<double> vals;
    if (step > 0) {
        for (double x = start; x < end; x += step) vals.push_back(x);
    } else {
        for (double x = start; x > end; x += step) vals.push_back(x);
    }
    Tensor t({vals.size()}, dtype, false);
    for (size_t i = 0; i < vals.size(); ++i) write_scalar_at(t.impl->storage->data.get(), i, dtype, vals[i]);
    return t;
}
// ------------- reshape (returns view sharing storage but with contiguous strides) -------------
Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    if (!impl) throw std::runtime_error("Empty tensor");
    size_t old_n = numel();
    size_t new_n = 1;
    for (auto v: new_shape) new_n *= v;
    if (old_n != new_n) throw std::invalid_argument("reshape: number of elements mismatch");
    // compute contiguous strides for new shape (C-contiguous)
    std::vector<size_t> nst(new_shape.size());
    if (!new_shape.empty()) {
        nst.back() = 1;
        for (int i = (int)new_shape.size()-2; i >= 0; --i)
            nst[i] = nst[i+1] * new_shape[i+1];
    }
    // share storage and same offset
    Tensor out;
    out.impl = std::make_shared<Tensorimpl>(impl->storage, impl->offset, new_shape, nst, impl->dtype, impl->requires_grad);
    return out;
}
// ------------- select: remove dimension dim by indexing index -------------
Tensor Tensor::select(size_t dim, size_t index) const {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (dim >= impl->ndim) throw std::out_of_range("select: dim out of range");
    if (index >= impl->shape[dim]) throw std::out_of_range("select: index out of range");
    std::vector<size_t> nsh;
    std::vector<size_t> nst;
    for (size_t i = 0; i < impl->ndim; ++i) {
        if (i == dim) continue;
        nsh.push_back(impl->shape[i]);
        nst.push_back(impl->strides[i]);
    }
    size_t noffset = impl->offset + index * impl->strides[dim];
    Tensor out;
    out.impl = std::make_shared<Tensorimpl>(impl->storage, noffset, nsh, nst, impl->dtype, impl->requires_grad);
    return out;
}
// ------------- squeeze / unsqueeze / flatten -------------
Tensor Tensor::squeeze() const {
    if (!impl) throw std::runtime_error("Empty tensor");
    std::vector<size_t> nsh;
    std::vector<size_t> nst;
    for (size_t i = 0; i < impl->ndim; ++i) {
        if (impl->shape[i] == 1) continue;
        nsh.push_back(impl->shape[i]);
        nst.push_back(impl->strides[i]);
    }
    if (nsh.empty()) { nsh.push_back(1); nst.push_back(1); } // keep at least 1-d tensor
    Tensor out;
    out.impl = std::make_shared<Tensorimpl>(impl->storage, impl->offset, nsh, nst, impl->dtype, impl->requires_grad);
    return out;
}
Tensor Tensor::unsqueeze(size_t dim) const {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (dim > impl->ndim) throw std::out_of_range("unsqueeze: dim out of range");
    std::vector<size_t> nsh;
    std::vector<size_t> nst;
    // naive approach: create contiguous strides for new shape to be safe
    nsh = shape();
    nsh.insert(nsh.begin() + dim, 1);
    // compute contiguous strides for nsh
    nst.resize(nsh.size());
    nst.back() = 1;
    for (int i = (int)nst.size()-2; i >= 0; --i) nst[i] = nst[i+1] * nsh[i+1];
    Tensor out;
    out.impl = std::make_shared<Tensorimpl>(impl->storage, impl->offset, nsh, nst, impl->dtype, impl->requires_grad);
    return out;
}
Tensor Tensor::flatten() const {
    std::vector<size_t> nsh = { numel() };
    std::vector<size_t> nst = { 1 };
    Tensor out;
    out.impl = std::make_shared<Tensorimpl>(impl->storage, impl->offset, nsh, nst, impl->dtype, impl->requires_grad);
    return out;
}
        // ---------- conversion: return new tensor ----------
Tensor Tensor::astype(DType new_dtype) const {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (new_dtype == impl->dtype) return Tensor(*this); // copy
    Tensor out(shape(), new_dtype, impl->requires_grad);
    size_t n = numel_();
    // straightforward convert elementwise
    for (size_t i = 0; i < n; ++i) {
        double v = read_scalar_at(impl->storage->data.get(), i, impl->dtype);
        write_scalar_at(out.impl->storage->data.get(), i, out.impl->dtype, v);
    }
    // grad not copied by default; if you want to copy grad, convert similarly:
    if (impl->requires_grad && impl->storage->grad) {
        if (!out.impl->storage->grad && n) throw std::bad_alloc();
        for (size_t i = 0; i < n; ++i) {
            double gv = read_scalar_at(impl->storage->grad.get(), i, impl->dtype);
            write_scalar_at(out.impl->storage->grad.get(), i, out.impl->dtype, gv);
        }
    }
    return out;
}