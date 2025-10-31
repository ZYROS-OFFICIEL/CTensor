#include "tensor1.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <algorithm>

// Storage implementation
std::shared_ptr<Storage> Storage::allocate(size_t n, DType dt, bool requires_grad) {
    auto s = std::make_shared<Storage>();
    s->size = n * dtype_size(dt);

    // allocate data
    void* p = std::malloc(s->size);
    if (!p && s->size) throw std::bad_alloc();
    std::memset(p, 0, s->size);
    s->data = std::shared_ptr<void>(p, std::free);

    // optional grad
    if (requires_grad) {
        void* g = std::malloc(s->size);
        if (!g && s->size) throw std::bad_alloc();
        std::memset(g, 0, s->size);
        s->grad = std::shared_ptr<void>(g, std::free);
    } else {
        s->grad = nullptr;
    }
    return s;
}

// Tensorimpl constructors / destructor
Tensorimpl::Tensorimpl(const std::vector<size_t>& shape_, DType dtype_, bool requires_grad_)
    : offset(0), ndim(shape_.size()), requires_grad(requires_grad_), dtype(dtype_) 
{
    shape = static_cast<size_t*>(std::malloc(ndim * sizeof(size_t)));
    strides = static_cast<size_t*>(std::malloc(ndim * sizeof(size_t)));
    if ((!shape && ndim) || (!strides && ndim)) {
        std::free(shape); std::free(strides);
        throw std::bad_alloc();
    }
    for (size_t i = 0; i < ndim; ++i) shape[i] = shape_[i];
    if (ndim > 0) {
        strides[ndim - 1] = 1;
        for (int i = (int)ndim - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];
    }
    size_t numel = 1;
    for (auto v : shape_) numel *= v;
    storage = Storage::allocate(numel, dtype, requires_grad);
}

Tensorimpl::Tensorimpl(std::shared_ptr<Storage> storage_,
           size_t offset_,
           const std::vector<size_t>& shape_,
           const std::vector<size_t>& strides_,
           DType dtype_,
           bool requires_grad_)
    : storage(std::move(storage_)),
      offset(offset_),
      ndim(shape_.size()),
      requires_grad(requires_grad_),
      dtype(dtype_)
{
    shape = static_cast<size_t*>(std::malloc(ndim * sizeof(size_t)));
    strides = static_cast<size_t*>(std::malloc(ndim * sizeof(size_t)));
    if ((!shape && ndim) || (!strides && ndim)) {
        std::free(shape); std::free(strides);
        throw std::bad_alloc();
    }
    for (size_t i = 0; i < ndim; ++i) {
        shape[i] = shape_[i];
        strides[i] = strides_[i];
    }
}

Tensorimpl::~Tensorimpl() {
    std::free(shape);
    std::free(strides);
}

// Tensor methods implementations

Tensor::Tensor(const std::vector<size_t>& shape_, DType dtype_, bool requires_grad_)
    : impl(std::make_shared<Tensorimpl>(shape_, dtype_, requires_grad_))
{}

size_t Tensor::numel() const {
    if (!impl) return 0;
    size_t n = 1;
    for (size_t i = 0; i < impl->ndim; ++i) n *= impl->shape[i];
    return n;
}

std::vector<size_t> Tensor::shape() const {
    if (!impl) return {};
    return std::vector<size_t>(impl->shape, impl->shape + impl->ndim);
}

Tensor Tensor::clone() const {
    Tensor out(shape(), _dtype(), impl->requires_grad);
    size_t n = numel();
    for (size_t i = 0; i < n; ++i)
        out[i] = (*this)[i];
    return out;
}

void Tensor::print_shape() const {
    if (!impl) { std::cout << "()\n"; return; }
    std::cout << "(";
    for (size_t i = 0; i < impl->ndim; ++i) {
        std::cout << impl->shape[i];
        if (i < impl->ndim - 1) std::cout << ", ";
    }
    std::cout << ")\n";
}

// Indexing entry points (non-template wrappers)
Tensor::Proxy Tensor::operator[](size_t i) {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (impl->ndim == 0) throw std::out_of_range("Tensor has no dimensions");
    if (i >= impl->shape[0]) throw std::out_of_range("Index out of bounds");
    return Proxy(impl, i * impl->strides[0], 1);
}

Tensor::ConstProxy Tensor::operator[](size_t i) const {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (impl->ndim == 0) throw std::out_of_range("Tensor has no dimensions");
    if (i >= impl->shape[0]) throw std::out_of_range("Index out of bounds");
    return ConstProxy(impl, i * impl->strides[0], 1);
}

// Convenience constructors
Tensor Tensor::ones(const std::vector<size_t>& shape_, DType dt, bool requires_grad_){
    Tensor t(shape_, dt, requires_grad_);
    size_t n = t.numel();
    for (size_t i = 0; i < n; ++i) write_scalar_at(t.impl->storage->data.get(), i, dt, 1.0);
    return t;
}

Tensor Tensor::zeros(const std::vector<size_t>& shape_, DType dt, bool requires_grad_){
    Tensor t(shape_, dt, requires_grad_);
    size_t n = t.numel();
    for (size_t i = 0; i < n; ++i) write_scalar_at(t.impl->storage->data.get(), i, dt, 0.0);
    return t;
}

Tensor Tensor::full(const std::vector<size_t>& shape_, double value, DType dt, bool requires_grad_){
    Tensor t(shape_, dt, requires_grad_);
    size_t n = t.numel();
    for (size_t i = 0; i < n; ++i) write_scalar_at(t.impl->storage->data.get(), i, dt, value);
    return t;
}

Tensor Tensor::rand(const std::vector<size_t>& shape_, DType dt, bool requires_grad_) {
    Tensor t(shape_, dt, requires_grad_);
    size_t n = t.numel_();
    std::srand((unsigned int)std::time(nullptr));
    for (size_t i = 0; i < n; ++i)
        write_scalar_at(t.impl->storage->data.get(), i, dt, static_cast<double>(std::rand()) / RAND_MAX);
    return t;
}

Tensor Tensor::empty(const std::vector<size_t>& shape_, DType dt, bool requires_grad_){
    Tensor t(shape_,dt,requires_grad_);
    return t;
}

Tensor Tensor::from_vector(const std::vector<double>& data,const std::vector<size_t>& shape,DType dtype, bool requires_grad)
{
    size_t n = 1;
    for (auto s : shape) n *= s;
    if (data.size() != n)
        throw std::invalid_argument("from_vector: data size does not match shape");
    Tensor t(shape, dtype, requires_grad);
    for (size_t i = 0; i < n; ++i)
        write_scalar_at(t.impl->storage->data.get(), i, dtype, data[i]);
    return t;
}

// dtype conversion
Tensor Tensor::astype(DType new_dtype) const {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (new_dtype == impl->dtype) return Tensor(*this);
    Tensor out(shape(), new_dtype, impl->requires_grad);
    size_t n = numel_();
    for (size_t i = 0; i < n; ++i) {
        double v = read_scalar_at(impl->storage->data.get(), i, impl->dtype);
        write_scalar_at(out.impl->storage->data.get(), i, out.impl->dtype, v);
    }
    if (impl->requires_grad && impl->storage->grad) {
        if (!out.impl->storage->grad && n) throw std::bad_alloc();
        for (size_t i = 0; i < n; ++i) {
            double gv = read_scalar_at(impl->storage->grad.get(), i, impl->dtype);
            write_scalar_at(out.impl->storage->grad.get(), i, out.impl->dtype, gv);
        }
    }
    return out;
}

void Tensor::to_(DType new_dtype) {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (new_dtype == impl->dtype) return;
    size_t n = numel_();
    auto new_storage = Storage::allocate(n, new_dtype, impl->requires_grad);
    for (size_t i = 0; i < n; ++i) {
        double v = read_scalar_at(impl->storage->data.get(), i, impl->dtype);
        write_scalar_at(new_storage->data.get(), i, new_dtype, v);
    }
    if (impl->requires_grad && impl->storage->grad) {
        for (size_t i = 0; i < n; ++i) {
            double gv = read_scalar_at(impl->storage->grad.get(), i, impl->dtype);
            write_scalar_at(new_storage->grad.get(), i, new_dtype, gv);
        }
    }
    impl->storage = new_storage;
    impl->dtype = new_dtype;
}

Tensor& Tensor::t_() {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (impl->ndim < 2)
        throw std::invalid_argument("t_: tensor must have at least 2 dimensions");
    std::swap(impl->shape[impl->ndim - 2], impl->shape[impl->ndim - 1]);
    std::swap(impl->strides[impl->ndim - 2], impl->strides[impl->ndim - 1]);
    return *this;
}

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

Tensor Tensor::arange(double start, double end, double step, DType dtype) {
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

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    if (!impl) throw std::runtime_error("Empty tensor");
    size_t old_n = numel();
    size_t new_n = 1;
    for (auto v: new_shape) new_n *= v;
    if (old_n != new_n) throw std::invalid_argument("reshape: number of elements mismatch");
    std::vector<size_t> nst(new_shape.size());
    if (!new_shape.empty()) {
        nst.back() = 1;
        for (int i = (int)new_shape.size()-2; i >= 0; --i)
            nst[i] = nst[i+1] * new_shape[i+1];
    }
    Tensor out;
    out.impl = std::make_shared<Tensorimpl>(impl->storage, impl->offset, new_shape, nst, impl->dtype, impl->requires_grad);
    return out;
}

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

Tensor Tensor::squeeze() const {
    if (!impl) throw std::runtime_error("Empty tensor");
    std::vector<size_t> nsh;
    std::vector<size_t> nst;
    for (size_t i = 0; i < impl->ndim; ++i) {
        if (impl->shape[i] == 1) continue;
        nsh.push_back(impl->shape[i]);
        nst.push_back(impl->strides[i]);
    }
    if (nsh.empty()) { nsh.push_back(1); nst.push_back(1); }
    Tensor out;
    out.impl = std::make_shared<Tensorimpl>(impl->storage, impl->offset, nsh, nst, impl->dtype, impl->requires_grad);
    return out;
}

Tensor Tensor::unsqueeze(size_t dim) const {
    if (!impl) throw std::runtime_error("Empty tensor");
    if (dim > impl->ndim) throw std::out_of_range("unsqueeze: dim out of range");
    std::vector<size_t> nsh = shape();
    nsh.insert(nsh.begin() + dim, 1);
    std::vector<size_t> nst(nsh.size());
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

// convenience from_image/save_image are best delegated to data.cpp (tensorio)
// but implement wrappers that call tensorio if available (link time)
// The declarations exist in header; implementations can simply call external functions if present.

void Tensor::save_image(const std::string& path) const {
    // implemented in data.cpp via tensorio::to_image, but keep wrapper
    // If tensorio not present at link, remove or provide alternative.
    extern void tensorio_to_image_wrapper(const Tensor&, const std::string&); // weak forward
    // try-catch in case not available; or call directly if you have tensorio.
    tensorio_to_image_wrapper(*this, path);
}

// simple backward convenience (you also have autograd free functions, choose one)
void Tensor::backward() {
    // keep simple: delegate to autograd::backward if implemented (free function)
    extern void backward(Tensor&); // forward reference to free function
    backward(*this);
}
