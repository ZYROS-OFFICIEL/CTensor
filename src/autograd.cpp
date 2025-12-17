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
    if (!t.impl->storage->grad) {
        size_t nbytes = t.numel() * t.dtype_bytes();
        void* gptr = std::malloc(nbytes);
        if (!gptr && nbytes > 0) throw std::bad_alloc();
        
        // Always zero-init new buffers
        if (nbytes > 0) std::memset(gptr, 0, nbytes);
        
        t.impl->storage->grad = std::shared_ptr<void>(gptr, std::free);
    } 
    // Zero existing if requested (e.g., at start of backward pass)
    else if (zero_existing) {
        size_t nbytes = t.numel() * t.dtype_bytes();
        if (nbytes > 0) std::memset(t.impl->storage->grad.get(), 0, nbytes);
    }
}