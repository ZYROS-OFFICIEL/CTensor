#include "tensor.h"
#include "autograd.h"
#include <cstring>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <cuda>

// Storage Implementation

/*
 Allocates raw, untyped memory for the tensor.
 We use std::calloc instead of std::malloc because it automatically 
 zero-initializes the memory. This acts as a nice default and makes 
 functions like `Tensor::zeros` practically free.
 */
std::shared_ptr<Storage> Storage::allocate(size_t n, DType dt, bool requires_grad, Device dev) {
    auto s = std::make_shared<Storage>();
    s->size = n;
    s->device = dev;
    size_t bytes = n * dtype_size(dt);
    
    if (dev.type == DeviceType::CUDA) {
        //Allocate in the GPU
        void* ptr = CudaMemoryPool::allocate(bytes);
        s->data = std::shared_ptr<void>(ptr, [](void* p) {
            CudaMemoryPool::instance().deallocate(p, bytes);
        });

    }else if(dev.type == DeviceType::CPU){
        // Allocate zero-initialized memory safely.
        // The custom deleter (std::free) ensures memory is freed when the ref-count hits 0.
        void* ptr = std::calloc(n, dtype_size(dt));
        if (!ptr && n > 0) throw std::runtime_error("Memory allocation failed");
        s->data = std::shared_ptr<void>(ptr, std::free);

    }
    
    return s;
}

// Tensor Methods

Device Tensor::device() const {
    if(!impl) return Device(DeviceType::CPU);
    return impl->data->device;
}

Tensor Tensor::to(Device target_device) {
    if (device().type == target_device.type) return *this;
    // Placeholder: Clone for now (real impl would allocate new storage on the target device
    // and perform a cudaMemcpy or equivalent data transfer).
    return clone(); 
}

/*
 clone() performs a DEEP COPY of the tensor's memory.
 It guarantees that the returned tensor is contiguous (densely packed in memory),
 even if the source tensor was sliced or permuted.
 */
Tensor Tensor::clone() const {
    if (!impl) return Tensor();
    
    // 1. Create a dense, contiguous output tensor based on the current logical shape
    Tensor out(impl->shape.to_vector(), impl->dtype, impl->requires_grad);
    
    size_t n = numel();
    size_t type_sz = dtype_bytes();
    
    // 2. Fast Path: Contiguous Copy
    // If the data is already tightly packed in a standard C-order layout, 
    // we can just blast the bytes over using memcpy. This is extremely fast.
    if (is_contiguous()) {
        if (impl->data && impl->data->data && out.impl->data && out.impl->data->data) {
            // Calculate correct offset for source (in case it's a view of a larger tensor)
            char* src = (char*)impl->data->data.get() + impl->offset * type_sz;
            char* dst = (char*)out.impl->data->data.get();
            std::memcpy(dst, src, n * type_sz);
        }
    } 
    // 3. Slow Path: Strided Copy (Repacking)
    // If the tensor has been permuted or sliced, the elements are scattered in memory.
    // We must visit them one by one according to their logical coordinates and 
    // pack them tightly into the new output tensor.
    else {
        char* out_ptr = (char*)out.impl->data->data.get();
        char* inp_base = (char*)impl->data->data.get();
        
        const auto& strides = impl->strides;
        const auto& shape = impl->shape;
        size_t ndim = impl->ndim;
        
        // "Odometer Pattern": Array to keep track of current N-dimensional coordinates.
        // It rolls over just like a car's odometer (e.g., [0,0,0] -> [0,0,1] -> [0,1,0])
        std::vector<size_t> coords(ndim, 0);
        
        for (size_t i = 0; i < n; ++i) {
            // Calculate exact flat index in the scattered source memory
            size_t src_offset = impl->offset;
            for(size_t d = 0; d < ndim; ++d) {
                src_offset += coords[d] * strides[d];
            }
            
            // Copy exactly 1 element to the dense output
            std::memcpy(out_ptr + i * type_sz, inp_base + src_offset * type_sz, type_sz);
            
            // Increment the odometer, starting from the innermost (last) dimension
            for(int d = (int)ndim - 1; d >= 0; --d) {
                coords[d]++;
                if (coords[d] < shape[d]) break; // No carry-over needed, stop incrementing
                coords[d] = 0; // Carry over to the next dimension
            }
        }
    }
    
    return out;
}

/*
 detach() creates a new logical "View" of the SAME underlying physical memory.
 Crucially, it breaks the autograd graph by setting requires_grad to false.
 Any mathematical operations on the detached tensor will not be tracked for backpropagation.
 */
Tensor Tensor::detach() const {
    if (!impl) return Tensor();
    Tensor out;
    // Create a new Tensorimpl that shares the SAME storage (impl->data)
    out.impl = intrusive_ptr<Tensorimpl>(new Tensorimpl(
        impl->data,
        impl->offset,
        impl->shape,
        impl->strides,
        impl->dtype,
        false // requires_grad = false cuts it off from the computation graph
    ));
    return out;
}

Tensor Tensor::detach() {
   return static_cast<const Tensor*>(this)->detach();
}

void Tensor::backward() {
    // Triggers the reverse-mode auto-differentiation engine.
    if (impl && impl->requires_grad) {
        ::backward(*this); 
    }
}

void Tensor::zero_grad() {
    if (!impl || !impl->grad) return;
    // Clears the gradient accumulation buffer. 
    // Necessary in training loops before calling .backward() again, 
    // otherwise gradients from the previous batch will add up.
    size_t bytes = numel() * dtype_bytes();
    if (impl->grad->data && impl->grad->data->data) {
        std::memset(impl->grad->data->data.get(), 0, bytes);
    }
}

// Shape Manipulations (Metadata Operations)

/*
 Reshapes the tensor without copying data, if possible.
 It simply changes the "lens" (shape & strides) through which we view the flat memory.
 */
Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    if (!impl) throw std::runtime_error("Reshape on empty tensor");
    
    size_t new_n = 1;
    for(auto s : new_shape) new_n *= s;
    if (new_n != numel()) throw std::runtime_error("Reshape size mismatch");
    
    // If the tensor elements are scattered (not contiguous), viewing them with a new 
    // shape linearly will grab the wrong elements. We must repack the memory first.
    if (!is_contiguous()) {
        return contiguous().reshape(new_shape);
    }

    Tensor out;
    // Share the same storage, but update the shape and recalculate default strides
    out.impl = intrusive_ptr<Tensorimpl>(new Tensorimpl(
        impl->data,
        impl->offset,
        SmallVector<size_t, 5>(new_shape),
        calc_strides(SmallVector<size_t, 5>(new_shape)), 
        impl->dtype,
        impl->requires_grad
    ));
    
    // Register the operation in the autograd graph
    if(impl->requires_grad) {
        out.impl->grad_fn = std::make_shared<GradReshape>(*this, impl->shape.to_vector());
    }
    return out;
}

/*
 permute() rearranges the dimensions (e.g., Matrix Transpose).
 This NEVER copies data. It simply shuffles the `shape` and `strides` arrays.
 Example: Transposing a shape [2, 3] with strides [3, 1] 
          results in shape [3, 2] with strides [1, 3].
 */
Tensor Tensor::permute(const std::vector<size_t>& dims) const {
    if (!impl) throw std::runtime_error("Permute on empty tensor");
    if (dims.size() != impl->ndim) throw std::runtime_error("Permute dim mismatch");
    
    SmallVector<size_t, 5> new_shape;
    SmallVector<size_t, 5> new_strides;
    
    // Map the old shapes/strides to the new axis order
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
    if (is_contiguous()) return *this; // Already tightly packed, do nothing.
    return clone(); // clone() inherently packs memory into a C-contiguous layout.
}

/*
 Checks if the physical layout of memory matches standard C-order (Row-Major).
 A tensor is contiguous if jumping to the next element in the last dimension 
 requires a step of exactly 1, and jumping in previous dimensions requires 
 stepping over the accumulated sizes of all inner dimensions.
 */
bool Tensor::is_contiguous() const {
    if (!impl) return true;
    size_t expected_stride = 1;
    for (int i = (int)impl->ndim - 1; i >= 0; --i) {
        if (impl->shape[i] > 1) { // Dimensions of size 1 don't break contiguity
            if (impl->strides[i] != expected_stride) return false;
            expected_stride *= impl->shape[i];
        }
    }
    return true;
}

// Constructors & Factories

Tensor Tensor::ones(const std::vector<size_t>& shape, DType dt, bool req) {
    Tensor t(shape, dt, req);
    size_t n = t.numel();
    
    // Direct pointer access for performance. std::fill is highly optimized.
    if (dt == DType::Float32) {
        float* ptr = (float*)t.impl->data->data.get();
        std::fill(ptr, ptr+n, 1.0f);
    } else {
        // Fallback for double (Note: a real framework would use templates to support all types)
        double* ptr = (double*)t.impl->data->data.get();
        std::fill(ptr, ptr+n, 1.0);
    }
    return t;
}

Tensor Tensor::zeros(const std::vector<size_t>& shape, DType dt, bool req) {
     // calloc in Storage::allocate already handles zero initialization automatically.
     return Tensor(shape, dt, req);
}

Tensor Tensor::rand(const std::vector<size_t>& shape, DType dt, bool req) {
    Tensor t(shape, dt, req);
    size_t n = t.numel();
    // Generates uniformly distributed random numbers between 0.0 and 1.0
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
    
    Tensor out(shape(), new_dtype, requires_grad());
    size_t n = numel();
    
    // Slower, generic element-wise cast using our read/write abstraction
    for(size_t i=0; i<n; ++i) {
        double v = read_scalar(i);
        out.write_scalar(i, v);
    }
    return out;
}

void Tensor::to_(DType new_dtype) {
    if (!impl) return;
    if (impl->dtype == new_dtype) return;
    *this = astype(new_dtype); // Replace self with newly allocated converted version
}

Tensor& Tensor::requires_grad_(bool b) {
    if (impl) impl->requires_grad = b;
    return *this;
}

Tensor& Tensor::t_() {
    if (!impl) return *this;
    if (impl->ndim != 2) throw std::runtime_error("t_() only supports 2D tensors");
    
    // In-place 2D Matrix transpose.
    // Swapping dims and strides achieves this in O(1) time without moving data.
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

/*
 The Gather operation.
 Conceptually: out[i][j][k] = input[index[i][j][k]][j][k] (if dim == 0)
 It collects values from the input tensor along a specific dimension `dim`, 
 using the indices provided in the `index` tensor. Used heavily in NLP (e.g. Embedding lookups).
 */
Tensor Tensor::gather(const Tensor& index, size_t dim) const {
    if (!impl) throw std::runtime_error("Gather on empty tensor");
    if (!index.impl) throw std::runtime_error("Gather with empty index");
    if (dim >= impl->ndim) throw std::out_of_range("Gather dimension out of range");
    if (impl->ndim != index.impl->ndim) throw std::runtime_error("Input and index must have same number of dimensions");

    // 1. Validate index dtype (Indices represent positions, so they must be integers)
    DType idx_dt = index._dtype();
    if (idx_dt == DType::Float32 || idx_dt == DType::Double64 || idx_dt == DType::Float16) {
         throw std::runtime_error("Index tensor must be integer type");
    }

    // 2. Prepare Output
    // Output tensor always takes the exact shape of the 'index' tensor.
    Tensor out(index.shape(), impl->dtype, impl->requires_grad);
    size_t N = out.numel();
    if (N == 0) return out;

    // 3. Optimize Index Access
    // Force index to be contiguous so we can iterate through it and our output 
    // simultaneously in a fast, linear loop.
    Tensor index_cont = index.contiguous();
    
    // 4. Setup Pointers and Metadata
    // Grab raw void* pointers to bypass safety checks for raw loop speed.
    void* out_ptr = out.impl->data->data.get();
    void* inp_ptr = impl->data->data.get();
    void* idx_ptr = index_cont.impl->data->data.get();

    const auto& out_shape = out.impl->shape; 
    const auto& inp_shape = impl->shape;
    const auto& inp_strides = impl->strides;
    
    size_t ndim = impl->ndim;
    std::vector<size_t> coords(ndim, 0); // The odometer

    // 5. Main Loop
    for (size_t i = 0; i < N; ++i) {
        
        // A. Read the target index value. Since index_cont is contiguous, 
        // `i` directly maps to the correct memory location.
        double idx_val_raw = read_scalar_at(idx_ptr, i, idx_dt);
        int64_t idx = static_cast<int64_t>(idx_val_raw);

        // B. Handle Python-style negative indexing (e.g., -1 means last element)
        if (idx < 0) idx += (int64_t)inp_shape[dim];
        
        if (idx < 0 || (size_t)idx >= inp_shape[dim]) {
             throw std::out_of_range("Gather index out of bounds");
        }

        // C. Calculate physical offset in the Input Tensor.
        // We use the current logical coordinates (`coords`), BUT we replace the 
        // coordinate at the gathering dimension (`dim`) with our fetched `idx`.
        size_t inp_offset = impl->offset;
        for (size_t d = 0; d < ndim; ++d) {
            size_t coord_at_d = (d == dim) ? (size_t)idx : coords[d];
            
            // Safety check: The index tensor shouldn't be larger than the input 
            // tensor on non-gather dimensions.
            if (coord_at_d >= inp_shape[d]) {
                 throw std::runtime_error("Index shape dimension mismatch with input");
            }
            inp_offset += coord_at_d * inp_strides[d];
        }

        // D. Transfer the Value
        double val = read_scalar_at(inp_ptr, inp_offset, impl->dtype);
        write_scalar_at(out_ptr, i, impl->dtype, val);

        // E. Update Coordinates (Odometer)
        for (int d = (int)ndim - 1; d >= 0; --d) {
            coords[d]++;
            if (coords[d] < out_shape[d]) break;
            coords[d] = 0;
        }
    }

    // Register node for backpropagation
    if (impl->requires_grad) {
        out.impl->grad_fn = std::make_shared<GradGather>(*this, index, dim);
    }

    return out;
}

Tensor Tensor::select(size_t dim, size_t index) const {
    throw std::runtime_error("Select not implemented yet");
}

/*
 Removes dimensions of size 1 from the tensor's shape.
 E.g., shape [1, 3, 1, 4] becomes [3, 4].
 No data is copied, it just changes metadata via reshape.
 */
Tensor Tensor::squeeze() const {
    std::vector<size_t> new_shape;
    for(auto s : impl->shape) if (s != 1) new_shape.push_back(s);
    if(new_shape.empty()) new_shape.push_back(1); // Handle scalar edge case
    return reshape(new_shape);
}

/*
 Inserts a dimension of size 1 at the specified index.
 E.g., unsqueezing shape [3, 4] at dim 0 becomes [1, 3, 4].
 */
Tensor Tensor::unsqueeze(size_t dim) const {
    std::vector<size_t> new_shape = shape();
    if(dim > new_shape.size()) dim = new_shape.size();
    new_shape.insert(new_shape.begin() + dim, 1);
    return reshape(new_shape);
}

Tensor Tensor::flatten() const {
    return reshape({numel()});
}

Tensor Tensor::argmax(int dim) const { 
    return ::argmax(*this, dim); 
}

void Tensor::print_shape() const {
    std::cout << "(";
    auto s = shape();
    for(size_t i=0; i<s.size(); ++i) {
        std::cout << s[i] << (i<s.size()-1 ? ", " : "");
    }
    std::cout << ")\n";
}

// Initialization Utils

/*
 Returns a 1-D tensor of size ceil((end - start) / step) with values from 
 the interval [start, end) taken with common difference step.
 */
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
    // Manual loop because value isn't a simple 0 or 1, and relies on our generic scalar writer
    for(size_t i=0; i<n; ++i) t.write_scalar(i, value);
    return t;
}

Tensor Tensor::empty(const std::vector<size_t>& shape_, DType dt, bool requires_grad_) {
    // Memory is currently 0-initialized anyway due to calloc in Storage. 
    // In a real framework, empty() would use malloc to skip the zeroing overhead.
    return Tensor(shape_, dt, requires_grad_);
}

Tensor Tensor::from_vector(const std::vector<double>& data, const std::vector<size_t>& shape, DType dtype, bool requires_grad) {
    Tensor t(shape, dtype, requires_grad);
    if (t.numel() != data.size()) throw std::runtime_error("from_vector size mismatch");
    // Copies C++ std::vector elements into our raw Storage buffer
    for(size_t i=0; i<data.size(); ++i) t.write_scalar(i, data[i]);
    return t;
}