#include "autograd.h"
#include "ops.h"       
#include "tensor.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <set>
#include <functional>
#include <iostream>
#include <omp.h>

//-------------------- helpers --------------------
void ensure_grad_buffer(Tensor &t, bool zero_existing) {
    if (!t.impl) throw std::runtime_error("ensure_grad_buffer: undefined tensor");

    // Allocate if missing
    // NOTE: 'grad' is now a shared_ptr<Tensorimpl>, so we create a new Tensorimpl
    if (!t.impl->grad) {
        // Create a new Tensorimpl with same shape/dtype/device
        t.impl->grad = std::make_shared<Tensorimpl>(
            t.shape(), 
            t._dtype(), 
            false, 
            t.device()
        );
        
        // Zero it out
        size_t nbytes = t.numel() * t.dtype_bytes();
        // Access: impl->grad (Tensorimpl) -> data (Storage) -> data (void*)
        if (nbytes > 0 && t.impl->grad->data && t.impl->grad->data->data) {
            std::memset(t.impl->grad->data->data.get(), 0, nbytes);
        }
    } 
    // Zero existing if requested (e.g., at start of backward pass)
    else if (zero_existing) {
        size_t nbytes = t.numel() * t.dtype_bytes();
        if (nbytes > 0 && t.impl->grad->data && t.impl->grad->data->data) {
            std::memset(t.impl->grad->data->data.get(), 0, nbytes);
        }
    }
}

// Convert the raw gradient buffer into a usable Tensor object
Tensor tensor_from_grad(const Tensor& self) {
    if (!self.impl || !self.impl->grad)
        throw std::runtime_error("tensor_from_grad: missing grad buffer");

    // Since 'grad' is now a Tensorimpl, we can just wrap it in a Tensor object.
    // This avoids the manual copy loop entirely!
    Tensor g;
    g.impl = self.impl->grad;
    return g;
}