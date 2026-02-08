#include "tensor.h"
#include "autograd.h"
#include <cstring>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cstdlib>

// ==========================================================
// Storage Implementation
// ==========================================================
std::shared_ptr<Storage> Storage::allocate(size_t n, DType dt, bool requires_grad, Device dev) {
    auto s = std::make_shared<Storage>();
    s->size = n;
    s->device = dev;
    size_t bytes = n * dtype_size(dt);
    
    if (dev.type == DeviceType::CPU) {
        // Allocate zero-initialized memory
        void* ptr = std::calloc(n, dtype_size(dt));
        if (!ptr && n > 0) throw std::runtime_error("Memory allocation failed");
        s->data = std::shared_ptr<void>(ptr, std::free);
    }
    // CUDA allocation would go here
    return s;
}

// ==========================================================
// Tensor Methods
// ==========================================================

Device Tensor::device() const {
    if(!impl) return Device(DeviceType::CPU);
    return impl->data->device;
}

Tensor Tensor::to(Device target_device) {
    if (device().type == target_device.type) return *this;
    // Placeholder: Clone for now (real impl would move data between CPU/GPU)
    return clone(); 
}

Tensor Tensor::clone() const {
    if (!impl) return Tensor();
    // Create new tensor with same properties
    Tensor out(impl->shape.to_vector(), impl->dtype, impl->requires_grad);
    
    // Deep copy data
    size_t bytes = numel() * dtype_bytes();
    if (impl->data && impl->data->data && out.impl->data && out.impl->data->data) {
        std::memcpy(out.impl->data->data.get(), impl->data->data.get(), bytes);
    }
    return out;
}

Tensor Tensor::detach() const {
    if (!impl) return Tensor();
    Tensor out;
    // Create new impl pointing to SAME storage, but no grad history
    out.impl = intrusive_ptr<Tensorimpl>(new Tensorimpl(
        impl->data,
        impl->offset,
        impl->shape,
        impl->strides,
        impl->dtype,
        false // requires_grad = false
    ));
    return out;
}

Tensor Tensor::detach() {
   return static_cast<const Tensor*>(this)->detach();
}

void Tensor::backward() {
    if (impl && impl->requires_grad) {
        ::backward(*this); 
    }
}

void Tensor::zero_grad() {
    if (!impl || !impl->grad) return;
    // Zero out grad buffer
    size_t bytes = numel() * dtype_bytes();
    if (impl->grad->data && impl->grad->data->data) {
        std::memset(impl->grad->data->data.get(), 0, bytes);
    }
}

// --- Shape Manipulations ---

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    if (!impl) throw std::runtime_error("Reshape on empty tensor");
    
    size_t new_n = 1;
    for(auto s : new_shape) new_n *= s;
    if (new_n != numel()) throw std::runtime_error("Reshape size mismatch");
    
    // If not contiguous, we must clone to pack data before reshaping
    if (!is_contiguous()) {
        return contiguous().reshape(new_shape);
    }

    Tensor out;
    out.impl = intrusive_ptr<Tensorimpl>(new Tensorimpl(
        impl->data,
        impl->offset,
        SmallVector<size_t, 5>(new_shape),
        calc_strides(SmallVector<size_t, 5>(new_shape)), // Recalculate strides for standard C-order
        impl->dtype,
        impl->requires_grad
    ));
    if(impl->requires_grad) {
        out.impl->grad_fn = std::make_shared<GradReshape>(*this, new_shape);
    }
    return out;
}

Tensor Tensor::permute(const std::vector<size_t>& dims) const {
    if (!impl) throw std::runtime_error("Permute on empty tensor");
    if (dims.size() != impl->ndim) throw std::runtime_error("Permute dim mismatch");
    
    SmallVector<size_t, 5> new_shape;
    SmallVector<size_t, 5> new_strides;
    
    for(auto d : dims) {
        if (d >= impl->ndim) throw std::out_of_range("Permute dimension out of range");
        new_shape.push_back(impl->shape[d]);
        new_strides.push_back(impl->strides[d]);
    }
    
    Tensor out;
    out.impl = intrusive_ptr<Tensorimpl>(new Tensorimpl(
        impl->data,
        impl->offset,
        new_shape,
        new_strides,
        impl->dtype,
        impl->requires_grad
    ));

    if(impl->requires_grad) {
        out.impl->grad_fn = std::make_shared<GradPermute>(*this, dims);
    }
    return out;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    return clone(); // clone() performs a packed deep copy
}

bool Tensor::is_contiguous() const {
    if (!impl) return true;
    size_t z = 1;
    for (int i = (int)impl->ndim - 1; i >= 0; --i) {
        if (impl->shape[i] > 1) {
            if (impl->strides[i] != z) return false;
            z *= impl->shape[i];
        }
    }
    return true;
}

// --- Constructors ---

Tensor Tensor::ones(const std::vector<size_t>& shape, DType dt, bool req) {
    Tensor t(shape, dt, req);
    size_t n = t.numel();
    
    // Simple fill loop
    if (dt == DType::Float32) {
        float* ptr = (float*)t.impl->data->data.get();
        std::fill(ptr, ptr+n, 1.0f);
    } else {
        // Fallback for double
        double* ptr = (double*)t.impl->data->data.get();
        std::fill(ptr, ptr+n, 1.0);
    }
    return t;
}

Tensor Tensor::zeros(const std::vector<size_t>& shape, DType dt, bool req) {
     // calloc in Storage::allocate already handles zero initialization
     return Tensor(shape, dt, req);
}

Tensor Tensor::rand(const std::vector<size_t>& shape, DType dt, bool req) {
    Tensor t(shape, dt, req);
    size_t n = t.numel();
    if (dt == DType::Float32) {
        float* ptr = (float*)t.impl->data->data.get();
        for(size_t i=0; i<n; ++i) ptr[i] = (float)std::rand() / RAND_MAX;
    } else {
        double* ptr = (double*)t.impl->data->data.get();
        for(size_t i=0; i<n; ++i) ptr[i] = (double)std::rand() / RAND_MAX;
    }
    return t;
}

Tensor Tensor::astype(DType new_dtype) const {
    if (_dtype() == new_dtype) return *this;
    
    // Create new tensor
    Tensor out(shape(), new_dtype, requires_grad());
    size_t n = numel();
    
    // Slow element-wise cast
    for(size_t i=0; i<n; ++i) {
        double v = read_scalar(i);
        out.write_scalar(i, v);
    }
    return out;
}

void Tensor::to_(DType new_dtype) {
    if (!impl) return;
    if (impl->dtype == new_dtype) return;
    *this = astype(new_dtype); // Replace self with converted version
}

Tensor& Tensor::requires_grad_(bool b) {
    if (impl) impl->requires_grad = b;
    return *this;
}

Tensor& Tensor::t_() {
    if (!impl) return *this;
    if (impl->ndim != 2) throw std::runtime_error("t_() only supports 2D tensors");
    
    // Swap dims and strides in place
    std::swap(impl->shape[0], impl->shape[1]);
    std::swap(impl->strides[0], impl->strides[1]);
    return *this;
}

// Placeholder implementations for image ops to prevent linker errors
Tensor Tensor::from_image(const std::string& path, DType dt) {
    std::cerr << "Image loading not implemented\n";
    return Tensor();
}
void Tensor::save_image(const std::string& path) const {
    std::cerr << "Image saving not implemented\n";
}
Tensor Tensor::gather(const Tensor& index, size_t dim) const {
    if (!impl) throw std::runtime_error("Gather on empty tensor");
    if (!index.impl) throw std::runtime_error("Gather with empty index");
    if (dim >= impl->ndim) throw std::out_of_range("Gather dimension out of range");
    if (impl->ndim != index.impl->ndim) throw std::runtime_error("Input and index must have same number of dimensions");

    // 1. Validate index dtype
    DType idx_dt = index._dtype();
    if (idx_dt == DType::Float32 || idx_dt == DType::Double64 || idx_dt == DType::Float16) {
         throw std::runtime_error("Index tensor must be integer type");
    }

    // 2. Prepare Output
    // Output tensor has the same shape as 'index' and same dtype as 'this' (input)
    Tensor out(index.shape(), impl->dtype, impl->requires_grad);
    size_t N = out.numel();
    if (N == 0) return out;

    // 3. Optimize Index Access
    // Ensure index is contiguous so we can iterate it linearly alongside 'out'
    Tensor index_cont = index.contiguous();
    
    // 4. Setup Pointers and Metadata
    // Get raw pointers for performance (bypassing shared_ptr/intrusive_ptr checks in loop)
    void* out_ptr = out.impl->data->data.get();
    void* inp_ptr = impl->data->data.get();
    void* idx_ptr = index_cont.impl->data->data.get();

    const auto& out_shape = out.impl->shape; // Same as index shape
    const auto& inp_shape = impl->shape;
    const auto& inp_strides = impl->strides;
    
    size_t ndim = impl->ndim;
    std::vector<size_t> coords(ndim, 0);

    // 5. Main Loop
    // Iterate linearly through the Output/Index tensor (0 to N-1)
    for (size_t i = 0; i < N; ++i) {
        
        // A. Read the index value (linear access because index_cont is contiguous)
        double idx_val_raw = read_scalar_at(idx_ptr, i, idx_dt);
        int64_t idx = static_cast<int64_t>(idx_val_raw);

        // B. Handle negative indexing and bounds checking
        if (idx < 0) idx += (int64_t)inp_shape[dim];
        
        if (idx < 0 || (size_t)idx >= inp_shape[dim]) {
             throw std::out_of_range("Gather index out of bounds");
        }

        // C. Calculate Offset in Input Tensor
        // We use the current 'coords', but replace the coordinate at 'dim' with 'idx'
        size_t inp_offset = impl->offset;
        for (size_t d = 0; d < ndim; ++d) {
            size_t coord_at_d = (d == dim) ? (size_t)idx : coords[d];
            
            // Safety check: Index shape must be <= Input shape on non-gather dims
            if (coord_at_d >= inp_shape[d]) {
                 throw std::runtime_error("Index shape dimension mismatch with input");
            }
            inp_offset += coord_at_d * inp_strides[d];
        }

        // D. Transfer Value
        double val = read_scalar_at(inp_ptr, inp_offset, impl->dtype);
        write_scalar_at(out_ptr, i, impl->dtype, val);

        // E. Update Coordinates (Odometer)
        // Increment the last dimension, carry over if needed
        for (int d = (int)ndim - 1; d >= 0; --d) {
            coords[d]++;
            if (coords[d] < out_shape[d]) {
                break;
            }
            coords[d] = 0;
        }
    }

    return out;
}
Tensor Tensor::select(size_t dim, size_t index) const {
    throw std::runtime_error("Select not implemented yet");
}
Tensor Tensor::squeeze() const {
    // Basic impl: remove dims of size 1
    std::vector<size_t> new_shape;
    for(auto s : impl->shape) if (s != 1) new_shape.push_back(s);
    if(new_shape.empty()) new_shape.push_back(1);
    return reshape(new_shape);
}
Tensor Tensor::unsqueeze(size_t dim) const {
    std::vector<size_t> new_shape = shape();
    if(dim > new_shape.size()) dim = new_shape.size();
    new_shape.insert(new_shape.begin() + dim, 1);
    return reshape(new_shape);
}
Tensor Tensor::flatten() const {
    return reshape({numel()});
}
void Tensor::print_shape() const {
    std::cout << "(";
    auto s = shape();
    for(size_t i=0; i<s.size(); ++i) {
        std::cout << s[i] << (i<s.size()-1 ? ", " : "");
    }
    std::cout << ")\n";
}

// Ranges
Tensor Tensor::arange(double start, double end, double step, DType dtype) {
    size_t steps = (size_t)std::ceil((end - start) / step);
    Tensor t({steps}, dtype);
    for(size_t i=0; i<steps; ++i) {
        t.write_scalar(i, start + i * step);
    }
    return t;
}

Tensor Tensor::full(const std::vector<size_t>& shape_, double value, DType dt, bool requires_grad_) {
    Tensor t(shape_, dt, requires_grad_);
    size_t n = t.numel();
    // Manual fill
    for(size_t i=0; i<n; ++i) t.write_scalar(i, value);
    return t;
}

Tensor Tensor::empty(const std::vector<size_t>& shape_, DType dt, bool requires_grad_) {
    return Tensor(shape_, dt, requires_grad_);
}

Tensor Tensor::from_vector(const std::vector<double>& data, const std::vector<size_t>& shape, DType dtype, bool requires_grad) {
    Tensor t(shape, dtype, requires_grad);
    if (t.numel() != data.size()) throw std::runtime_error("from_vector size mismatch");
    for(size_t i=0; i<data.size(); ++i) t.write_scalar(i, data[i]);
    return t;
}