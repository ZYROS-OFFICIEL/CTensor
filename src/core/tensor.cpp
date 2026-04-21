#include "tensor.h"
#include "autograd.h"
#include <cstring>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cstdlib>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

Tensorimpl::Tensorimpl(const std::vector<size_t>& shape_, DType dtype_,
                       bool requires_grad_, Device dev)
    : shape(shape_), dtype(dtype_), requires_grad(requires_grad_)
{
    ndim = shape_.size();
    numel_ = 1;
    for (size_t s : shape_) numel_ *= s;
    strides = calc_strides(shape);
    data = Storage::allocate(numel_, dtype_, dev);
}

Tensorimpl::Tensorimpl(std::shared_ptr<Storage>      storage,
                       size_t                         offset_,
                       const SmallVector<size_t, 5>& shape_,
                       const SmallVector<size_t, 5>& strides_,
                       DType                          dtype_,
                       bool                           requires_grad_)
    : data(std::move(storage)), offset(offset_), shape(shape_), strides(strides_),
      dtype(dtype_), requires_grad(requires_grad_)
{
    ndim = shape.size();
    numel_ = 1;
    for (size_t i = 0; i < ndim; ++i) numel_ *= shape[i];
}

void* CudaMemoryPool::allocate(size_t n) {
#ifdef USE_CUDA
    std::lock_guard<std::mutex> lock(mtx_);
    size_t bucket = next_pow2(n);
    auto& blocks = pool_[bucket];
    for (auto& b : blocks) {
        if (!b.in_use && b.size >= n) { b.in_use = true; return b.ptr; }
    }
    void* ptr = nullptr;
    cudaMalloc(&ptr, bucket);
    blocks.push_back({ptr, bucket, true});
    return ptr;
#else
    throw std::runtime_error("CUDA not available");
#endif
}

void CudaMemoryPool::deallocate(void* ptr, size_t n) noexcept {
#ifdef USE_CUDA
    std::lock_guard<std::mutex> lock(mtx_);
    size_t bucket = next_pow2(n);
    auto it = pool_.find(bucket);
    if (it == pool_.end()) return;
    for (auto& b : it->second) { if (b.ptr == ptr) { b.in_use = false; return; } }
#else
    (void)ptr; (void)n;
#endif
}

void CudaMemoryPool::release_all() noexcept {
#ifdef USE_CUDA
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& [_, blocks] : pool_)
        for (auto& b : blocks) cudaFree(b.ptr);
    pool_.clear();
#endif
}

std::shared_ptr<Storage> Storage::allocate(size_t n, DType dt, Device dev) {
    auto s   = std::make_shared<Storage>();
    s->bytes  = n * dtype_size(dt);
    s->device = dev;
    if (dev.is_cuda()) {
#ifdef USE_CUDA
        size_t nb = s->bytes;
        void* ptr = CudaMemoryPool::instance().allocate(nb);
        s->data   = std::shared_ptr<void>(ptr, [nb](void* p) {
            CudaMemoryPool::instance().deallocate(p, nb);
        });
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        void* ptr = std::calloc(n, dtype_size(dt));
        if (!ptr && n > 0) throw std::runtime_error("Memory allocation failed");
        s->data = std::shared_ptr<void>(ptr, std::free);
    }
    return s;
}

Tensor::Tensor(const std::vector<size_t>& shape_, DType dtype_, bool requires_grad_)
    : impl(make_intrusive<Tensorimpl>(shape_, dtype_, requires_grad_, Device(DeviceType::CPU)))
{}

double Tensor::read_scalar(size_t idx) const {
    require_impl();
    if (impl->data->device.is_cuda()) {
#ifdef USE_CUDA
        char buf[8] = {};
        const char* src = static_cast<const char*>(impl->data->data.get())
                          + idx * dtype_size(impl->dtype);
        cudaMemcpy(buf, src, dtype_size(impl->dtype), cudaMemcpyDeviceToHost);
        return read_scalar_at(buf, 0, impl->dtype);
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
    return read_scalar_at(impl->data->data.get(), idx, impl->dtype);
}

void Tensor::write_scalar(size_t idx, double val) {
    require_impl();
    if (impl->data->device.is_cuda()) {
#ifdef USE_CUDA
        char buf[8] = {};
        write_scalar_at(buf, 0, impl->dtype, val);
        char* dst = static_cast<char*>(impl->data->data.get())
                    + idx * dtype_size(impl->dtype);
        cudaMemcpy(dst, buf, dtype_size(impl->dtype), cudaMemcpyHostToDevice);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        write_scalar_at(impl->data->data.get(), idx, impl->dtype, val);
    }
}

Tensor::Proxy Tensor::operator[](size_t i) {
    require_impl();
    if (i >= impl->shape[0]) throw std::out_of_range("Index out of bounds");
    return Proxy(impl, impl->offset + i * impl->strides[0], 1);
}

Tensor::ConstProxy Tensor::operator[](size_t i) const {
    require_impl();
    if (i >= impl->shape[0]) throw std::out_of_range("Index out of bounds");
    return ConstProxy(impl, impl->offset + i * impl->strides[0], 1);
}

Tensor Tensor::to(Device target) const {
    if (device() == target) return *this;
    Tensor out(impl->shape.to_vector(), impl->dtype, impl->requires_grad);
    size_t n   = numel();
    size_t esz = dtype_size(impl->dtype);
    if (target.is_cuda()) {
#ifdef USE_CUDA
        out.impl->data = Storage::allocate(n, impl->dtype, target);
        cudaMemcpy(out.impl->data->data.get(), impl->data->data.get(),
                   n * esz, cudaMemcpyHostToDevice);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
#ifdef USE_CUDA
        cudaMemcpy(out.impl->data->data.get(), impl->data->data.get(),
                   n * esz, cudaMemcpyDeviceToHost);
#else
        std::memcpy(out.impl->data->data.get(), impl->data->data.get(), n * esz);
#endif
    }
    return out;
}

Tensor Tensor::clone() const {
    if (!impl) return Tensor();
    Tensor out(impl->shape.to_vector(), impl->dtype, impl->requires_grad);
    size_t n       = numel();
    size_t type_sz = dtype_size(impl->dtype);
    if (is_contiguous()) {
        const char* src = static_cast<const char*>(impl->data->data.get()) + impl->offset * type_sz;
        std::memcpy(out.impl->data->data.get(), src, n * type_sz);
    } else {
        char*       dst      = static_cast<char*>(out.impl->data->data.get());
        const char* inp_base = static_cast<const char*>(impl->data->data.get());
        std::vector<size_t> coords(impl->ndim, 0);
        for (size_t i = 0; i < n; ++i) {
            size_t src_off = impl->offset;
            for (size_t d = 0; d < impl->ndim; ++d) src_off += coords[d] * impl->strides[d];
            std::memcpy(dst + i * type_sz, inp_base + src_off * type_sz, type_sz);
            for (int d = (int)impl->ndim - 1; d >= 0; --d) {
                if (++coords[d] < impl->shape[d]) break;
                coords[d] = 0;
            }
        }
    }
    return out;
}

Tensor Tensor::detach() const {
    if (!impl) return Tensor();
    Tensor out;
    out.impl = intrusive_ptr<Tensorimpl>(new Tensorimpl(
        impl->data, impl->offset, impl->shape, impl->strides, impl->dtype, false));
    return out;
}

void Tensor::backward() {
    if (impl && impl->requires_grad) ::backward(*this);
}

void Tensor::zero_grad() {
    if (!impl || !impl->grad) return;
    std::memset(impl->grad->data->data.get(), 0, numel() * dtype_size(impl->dtype));
}

bool Tensor::is_contiguous() const {
    if (!impl) return true;
    size_t expected = 1;
    for (int i = (int)impl->ndim - 1; i >= 0; --i) {
        if (impl->shape[i] > 1) {
            if (impl->strides[i] != expected) return false;
            expected *= impl->shape[i];
        }
    }
    return true;
}

Tensor Tensor::contiguous() const {
    return is_contiguous() ? *this : clone();
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    if (!impl) throw std::runtime_error("Reshape on empty tensor");
    size_t new_n = 1;
    for (size_t s : new_shape) new_n *= s;
    if (new_n != numel()) throw std::runtime_error("Reshape size mismatch");
    if (!is_contiguous()) return contiguous().reshape(new_shape);
    SmallVector<size_t, 5> ns(new_shape);
    Tensor out;
    out.impl = intrusive_ptr<Tensorimpl>(new Tensorimpl(
        impl->data, impl->offset, ns, calc_strides(ns), impl->dtype, impl->requires_grad));
    if (impl->requires_grad)
        out.impl->grad_fn = std::make_shared<GradReshape>(*this, impl->shape.to_vector());
    return out;
}

Tensor Tensor::permute(const std::vector<size_t>& dims) const {
    if (!impl) throw std::runtime_error("Permute on empty tensor");
    if (dims.size() != impl->ndim) throw std::runtime_error("Permute dim mismatch");
    SmallVector<size_t, 5> ns, nst;
    for (size_t d : dims) {
        if (d >= impl->ndim) throw std::out_of_range("Permute dimension out of range");
        ns.push_back(impl->shape[d]);
        nst.push_back(impl->strides[d]);
    }
    Tensor out;
    out.impl = intrusive_ptr<Tensorimpl>(new Tensorimpl(
        impl->data, impl->offset, ns, nst, impl->dtype, impl->requires_grad));
    if (impl->requires_grad)
        out.impl->grad_fn = std::make_shared<GradPermute>(*this, dims);
    return out;
}

Tensor& Tensor::t_() {
    if (!impl) return *this;
    if (impl->ndim != 2) throw std::runtime_error("t_() only supports 2D tensors");
    std::swap(impl->shape[0],   impl->shape[1]);
    std::swap(impl->strides[0], impl->strides[1]);
    return *this;
}

Tensor Tensor::squeeze() const {
    if (!impl) throw std::runtime_error("squeeze on empty tensor");
    std::vector<size_t> ns;
    for (size_t s : impl->shape) if (s != 1) ns.push_back(s);
    if (ns.empty()) ns.push_back(1);
    return reshape(ns);
}

Tensor Tensor::unsqueeze(size_t dim) const {
    std::vector<size_t> ns = shape();
    if (dim > ns.size()) dim = ns.size();
    ns.insert(ns.begin() + dim, 1);
    return reshape(ns);
}

Tensor Tensor::flatten() const { return reshape({numel()}); }

Tensor Tensor::select(size_t dim, size_t index) const {
    if (!impl) throw std::runtime_error("select on empty tensor");
    if (dim >= impl->ndim)        throw std::out_of_range("select: dim out of range");
    if (index >= impl->shape[dim]) throw std::out_of_range("select: index out of range");
    SmallVector<size_t, 5> ns, nst;
    size_t new_off = impl->offset + index * impl->strides[dim];
    for (size_t d = 0; d < impl->ndim; ++d) {
        if (d == dim) continue;
        ns.push_back(impl->shape[d]);
        nst.push_back(impl->strides[d]);
    }
    if (ns.empty()) { ns.push_back(1); nst.push_back(1); }
    Tensor out;
    out.impl = intrusive_ptr<Tensorimpl>(new Tensorimpl(
        impl->data, new_off, ns, nst, impl->dtype, impl->requires_grad));
    return out;
}

Tensor Tensor::gather(const Tensor& index, size_t dim) const {
    if (!impl)       throw std::runtime_error("Gather on empty tensor");
    if (!index.impl) throw std::runtime_error("Gather with empty index");
    if (dim >= impl->ndim) throw std::out_of_range("Gather dimension out of range");
    if (impl->ndim != index.impl->ndim)
        throw std::runtime_error("Input and index must have same number of dimensions");
    DType idx_dt = index._dtype();
    if (idx_dt == DType::Float32 || idx_dt == DType::Double64 || idx_dt == DType::Float16)
        throw std::runtime_error("Index tensor must be integer type");

    Tensor out(index.shape(), impl->dtype, impl->requires_grad);
    size_t N = out.numel();
    if (N == 0) return out;

    Tensor index_cont = index.contiguous();
    void*       out_ptr   = out.impl->data->data.get();
    const void* inp_ptr   = impl->data->data.get();
    const void* idx_ptr   = index_cont.impl->data->data.get();
    const auto& out_shape = out.impl->shape;
    const auto& inp_shape = impl->shape;
    const auto& inp_st    = impl->strides;
    size_t ndim           = impl->ndim;
    std::vector<size_t> coords(ndim, 0);

    for (size_t i = 0; i < N; ++i) {
        int64_t idx = static_cast<int64_t>(read_scalar_at(idx_ptr, i, idx_dt));
        if (idx < 0) idx += (int64_t)inp_shape[dim];
        if (idx < 0 || (size_t)idx >= inp_shape[dim])
            throw std::out_of_range("Gather index out of bounds");
        size_t inp_off = impl->offset;
        for (size_t d = 0; d < ndim; ++d) {
            size_t c = (d == dim) ? (size_t)idx : coords[d];
            if (d != dim && c >= inp_shape[d])
                throw std::runtime_error("Index shape dimension mismatch with input");
            inp_off += c * inp_st[d];
        }
        write_scalar_at(out_ptr, i, impl->dtype, read_scalar_at(inp_ptr, inp_off, impl->dtype));
        for (int d = (int)ndim - 1; d >= 0; --d) {
            if (++coords[d] < out_shape[d]) break;
            coords[d] = 0;
        }
    }
    if (impl->requires_grad)
        out.impl->grad_fn = std::make_shared<GradGather>(*this, index, dim);
    return out;
}

Tensor Tensor::argmax(int dim) const { return ::argmax(*this, dim); }

Tensor Tensor::astype(DType new_dtype) const {
    if (_dtype() == new_dtype) return *this;
    Tensor out(shape(), new_dtype, requires_grad());
    size_t n = numel();
    for (size_t i = 0; i < n; ++i) out.write_scalar(i, read_scalar(i));
    return out;
}

void Tensor::to_(DType new_dtype) {
    if (!impl || impl->dtype == new_dtype) return;
    *this = astype(new_dtype);
}

Tensor Tensor::from_image(const std::string&, DType) {
    std::cerr << "Image loading not implemented\n";
    return Tensor();
}

void Tensor::save_image(const std::string&) const {
    std::cerr << "Image saving not implemented\n";
}

void Tensor::print_shape() const {
    std::cout << "(";
    auto s = shape();
    for (size_t i = 0; i < s.size(); ++i)
        std::cout << s[i] << (i + 1 < s.size() ? ", " : "");
    std::cout << ")\n";
}

Tensor Tensor::ones(const std::vector<size_t>& shape, DType dt, bool rg) {
    Tensor t(shape, dt, rg);
    size_t n = t.numel();
    if (dt == DType::Float32) {
        float* p = static_cast<float*>(t.impl->data->data.get());
        std::fill(p, p + n, 1.0f);
    } else {
        for (size_t i = 0; i < n; ++i) t.write_scalar(i, 1.0);
    }
    return t;
}

Tensor Tensor::zeros(const std::vector<size_t>& shape, DType dt, bool rg) {
    return Tensor(shape, dt, rg);
}

Tensor Tensor::rand(const std::vector<size_t>& shape, DType dt, bool rg) {
    Tensor t(shape, dt, rg);
    size_t n = t.numel();
    if (dt == DType::Float32) {
        float* p = static_cast<float*>(t.impl->data->data.get());
        for (size_t i = 0; i < n; ++i) p[i] = static_cast<float>(std::rand()) / RAND_MAX;
    } else {
        for (size_t i = 0; i < n; ++i)
            t.write_scalar(i, static_cast<double>(std::rand()) / RAND_MAX);
    }
    return t;
}

Tensor Tensor::full(const std::vector<size_t>& shape, double value, DType dt, bool rg) {
    Tensor t(shape, dt, rg);
    size_t n = t.numel();
    for (size_t i = 0; i < n; ++i) t.write_scalar(i, value);
    return t;
}

Tensor Tensor::empty(const std::vector<size_t>& shape, DType dt, bool rg) {
    return Tensor(shape, dt, rg);
}

Tensor Tensor::arange(double start, double end, double step, DType dtype) {
    size_t n = static_cast<size_t>(std::ceil((end - start) / step));
    Tensor t({n}, dtype);
    for (size_t i = 0; i < n; ++i) t.write_scalar(i, start + i * step);
    return t;
}

Tensor Tensor::from_vector(const std::vector<double>& data,
                           const std::vector<size_t>& shape,
                           DType dtype, bool rg) {
    Tensor t(shape, dtype, rg);
    if (t.numel() != data.size()) throw std::runtime_error("from_vector size mismatch");
    for (size_t i = 0; i < data.size(); ++i) t.write_scalar(i, data[i]);
    return t;
}