// Numeric gradient check for Conv2d weight using your existing Tensor/Conv classes.
// Includes extra diagnostics to help find where autograd stops propagating.
//
// Compile:
// g++ -g -std=c++17 convnum.cpp tensor1.cpp ops1.cpp autograd.cpp data.cpp conv.cpp -o conv_ -I.

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#include "tensor1.h"
#include "ops1.h"
#include "autograd.h"
#include "conv.h"

// ---------- helpers (stride-aware access & sum) ----------------

static size_t numel_from_impl(const Tensor& t) {
    return t.numel_();
}

// Convert flat index (0..numel-1) to strided storage index using shape/strides/offset
static size_t flat_to_strided_index(const Tensor& t, size_t flat) {
    size_t ndim = t.impl->ndim;
    size_t rem = flat;
    size_t idx = t.impl->offset;
    // build multi-index
    std::vector<size_t> multi(ndim, 0);
    for (int d = (int)ndim - 1; d >= 0; --d) {
        multi[d] = rem % t.impl->shape[d];
        rem /= t.impl->shape[d];
    }
    for (size_t d = 0; d < ndim; ++d) {
        idx += multi[d] * t.impl->strides[d];
    }
    return idx;
}

static double read_data_strided(const Tensor& t, size_t strided_idx) {
    return read_scalar_at(t.impl->storage->data.get(), strided_idx, t._dtype());
}

static double read_grad_strided(const Tensor& t, size_t strided_idx) {
    if (!t.impl->storage->grad) throw std::runtime_error("read_grad_strided: grad buffer missing");
    return read_scalar_at(t.impl->storage->grad.get(), strided_idx, t._dtype());
}

static void write_data_strided(Tensor& t, size_t strided_idx, double v) {
    write_scalar_at(t.impl->storage->data.get(), strided_idx, t._dtype(), v);
}

// Sum tensor data (stride-aware)
static double tensor_sum_data(const Tensor& t) {
    size_t N = numel_from_impl(t);
    double tot = 0.0;
    for (size_t i = 0; i < N; ++i) {
        size_t sid = flat_to_strided_index(t, i);
        tot += read_data_strided(t, sid);
    }
    return tot;
}

// Print small tensor (contiguous view assumed or will print by mapping flat->strided)
static void print_tensor_brief(const Tensor& t, const std::string& name, size_t max_elems = 32) {
    std::cout << name << " shape: [";
    for (size_t i = 0; i < t.impl->ndim; ++i) {
        std::cout << t.impl->shape[i] << (i+1 < t.impl->ndim ? "," : "");
    }
    std::cout << "] values: ";
    size_t N = numel_from_impl(t);
    size_t to = std::min(N, max_elems);
    std::cout << "[";
    for (size_t i = 0; i < to; ++i) {
        size_t sid = flat_to_strided_index(t, i);
        double v = read_data_strided(t, sid);
        std::cout << v << (i+1 < to ? ", " : "");
    }
    if (N > to) std::cout << ", ...";
    std::cout << "]\n";
}

static void print_grad_brief_if_exists(const Tensor& t, const std::string& name, size_t max_elems = 32) {
    std::cout << name << " grad: ";
    if (!t.impl->storage->grad) {
        std::cout << "(no grad buffer)\n";
        return;
    }
    size_t N = numel_from_impl(t);
    size_t to = std::min(N, max_elems);
    std::cout << "[";
    for (size_t i = 0; i < to; ++i) {
        size_t sid = flat_to_strided_index(t, i);
        double v = read_grad_strided(t, sid);
        std::cout << v << (i+1 < to ? ", " : "");
    }
    if (N > to) std::cout << ", ...";
    std::cout << "]\n";
}

// Compute max abs and relative error between two tensors (same shapes)
static void compare_tensors(const Tensor& a, const Tensor& b) {
    assert(a.numel_() == b.numel_());
    size_t N = a.numel_();
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double va = read_data_strided(a, flat_to_strided_index(a, i));
        double vb = read_data_strided(b, flat_to_strided_index(b, i));
        double ad = std::fabs(va - vb);
        double rel = ad / (std::max(1e-12, std::fabs(vb)));
        if (ad > max_abs) max_abs = ad;
        if (rel > max_rel) max_rel = rel;
    }
    std::cout << "Compare: max_abs = " << max_abs << ", max_rel = " << max_rel << "\n";
}

// ---------------- numeric-grad test ------------------
// This tests Conv2d weight gradient for a tiny case.
int main() {
    std::cout << "Conv2d numeric grad check (finite differences) \n\n";

    // small deterministic case
    size_t batch = 1;
    int in_c = 1;
    int out_c = 1;
    int H = 3;
    int W = 3;
    int kh = 2;
    int kw = 2;
    int sh = 1;
    int sw = 1;
    int ph = 0;
    int pw = 0;

    // Create input, weight, bias
    Tensor input(std::vector<size_t>{(size_t)batch, (size_t)in_c, (size_t)H, (size_t)W}, DType::Float32, false);
    Tensor weight(std::vector<size_t>{(size_t)out_c, (size_t)in_c, (size_t)kh, (size_t)kw}, DType::Float32, false);
    Tensor bias(std::vector<size_t>{(size_t)out_c}, DType::Float32, false);

    // fill input and weight with small values for determinism
    for (size_t i = 0; i < input.numel_(); ++i) {
        write_scalar_at(input.impl->storage->data.get(), flat_to_strided_index(input, i), input._dtype(), double(i+1));
    }
    for (size_t i = 0; i < weight.numel_(); ++i) {
        write_scalar_at(weight.impl->storage->data.get(), flat_to_strided_index(weight, i), weight._dtype(), double(i+1));
    }
    for (size_t i = 0; i < bias.numel_(); ++i) write_scalar_at(bias.impl->storage->data.get(), flat_to_strided_index(bias, i), bias._dtype(), 0.0);

    Conv2d conv(in_c, out_c, kh, kw, sh, sw, ph, pw);
    conv.weight = weight;
    conv.bias = bias;

    print_tensor_brief(input, "input");
    print_tensor_brief(weight, "weight");
    print_tensor_brief(bias, "bias");

    // ---------- Analytic gradient via autograd ----------
    // We need tensors requiring grad for analytic path
    input.impl->requires_grad = false; // we only test weight grad here
    weight.impl->requires_grad = true;
    bias.impl->requires_grad = true;

    // Forward -> loss = sum(output)
    Tensor out = conv.forward(input);
    print_tensor_brief(out, "forward_out");

    // Make sure out has grad_fn (diagnostic)
    std::cout << "diagnostic: out.impl->grad_fn " << (out.impl->grad_fn ? "SET" : "NULL") << "\n";

    // Reduce out to scalar while keeping autograd graph
    Tensor loss = out;
    // reduce over dimension 0 repeatedly until scalar
    while (loss.numel_() > 1) {
        // NOTE: sum(t, dim) must be an autograd-aware op in your ops1 implementation
        loss = sum(loss, 0);
    }
    std::cout << "diagnostic: loss.numel = " << loss.numel_() << "\n";
    std::cout << "diagnostic: loss.impl->grad_fn " << (loss.impl->grad_fn ? "SET" : "NULL") << "\n";

    // ensure scalar requires_grad (should be true automatically if sum preserved graph)
    loss.impl->requires_grad = true;

    // If the sum op in your implementation is not autograd-aware, you'll see loss.impl->grad_fn == NULL above.
    // In this case you must fix your sum implementation to attach a GradSum node.
    if (!loss.impl->grad_fn) {
        std::cerr << "Warning: loss has no grad_fn — sum() did not create a gradient node. Backprop will not flow to conv output.\n";
    }

    // Run backward to get analytic grads
    backward(loss);

    
    // Debug: show whether weight.grad buffer exists
    std::cout << "diagnostic: conv.weight.impl->storage->grad " << (conv.weight.impl->storage->grad ? "ALLOCATED" : "NULL") << "\n";
    if (conv.weight.impl->storage->grad) {
        print_grad_brief_if_exists(conv.weight, "weight (after backward)"); 
    } else {
        std::cerr << "Error: weight.grad not created. Dumping some internals for debugging:\n";
        std::cout << " - out.impl->grad_fn: " << (out.impl->grad_fn ? "SET" : "NULL") << "\n";
        std::cout << " - loss.impl->grad_fn: " << (loss.impl->grad_fn ? "SET" : "NULL") << "\n";
        std::cout << " - conv.forward output requires_grad: " << (out.impl->requires_grad ? "true" : "false") << "\n";
        if (out.impl->storage && out.impl->storage->grad)
            std::cout << " - out.grad exists\n";
        else
            std::cout << " - out.grad missing\n";

        // Try to print a small slice of out.grad if exists
        try {
            print_grad_brief_if_exists(out, "out (after backward)");
        } catch (...) { /* ignore */ }

        // Still continue to numeric check so we get the numbers
    }

    // Read analytic weight gradient (if available) - wrap with try/catch
    Tensor analytic_grad_weight(std::vector<size_t>{(size_t)out_c, (size_t)in_c, (size_t)kh, (size_t)kw}, DType::Float32, false);
    bool analytic_ready = false;
    try {
        if (conv.weight.impl->storage->grad) {
            analytic_grad_weight = tensor_from_grad(conv.weight);
            print_tensor_brief(analytic_grad_weight, "analytic_grad_weight");
            analytic_ready = true;
        }
    } catch (const std::exception &e) {
        std::cerr << "tensor_from_grad failed: " << e.what() << "\n";
    }

    // ---------- Numeric gradients (finite diff) ----------
    double eps = 1e-3;
    Tensor numeric_grad(weight.shape(), weight._dtype(), false);

    size_t WN = weight.numel_();
    for (size_t i = 0; i < WN; ++i) {
        // get strided index for weight element
        size_t sidx = flat_to_strided_index(weight, i);

        // original
        double orig = read_data_strided(weight, sidx);

        // +eps
        write_data_strided(weight, sidx, orig + eps);
        Tensor out_p = conv.forward(input);
        double loss_p = tensor_sum_data(out_p);

        // -eps
        write_data_strided(weight, sidx, orig - eps);
        Tensor out_m = conv.forward(input);
        double loss_m = tensor_sum_data(out_m);

        // restore
        write_data_strided(weight, sidx, orig);

        double gnum = (loss_p - loss_m) / (2.0 * eps);
        write_scalar_at(numeric_grad.impl->storage->data.get(), flat_to_strided_index(numeric_grad, i), numeric_grad._dtype(), gnum);
    }

    print_tensor_brief(numeric_grad, "numeric_grad_weight");

    if (analytic_ready) {
        compare_tensors(analytic_grad_weight, numeric_grad);
    } else {
        std::cout << "Analytic grad not available to compare.\n";
    }

    std::cout << "Done.\n";
    return 0;
}
