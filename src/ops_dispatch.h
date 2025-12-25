#pragma once
#include "tensor.h"

// Check CPU features (can remain inline static or move to cpp)
static inline bool cpu_has_avx2();
static inline bool cpu_has_avx512f();

// Binary Ops
Tensor add(const Tensor &a, const Tensor &b);
Tensor sub(const Tensor &a, const Tensor &b);
Tensor mul(const Tensor &a, const Tensor &b);
Tensor div(const Tensor &a, const Tensor &b);
Tensor pow(const Tensor &a, const Tensor &b);
Tensor matmul(const Tensor &a, const Tensor &b);

// Comparisons
Tensor lt(const Tensor &a, const Tensor &b);
Tensor le(const Tensor &a, const Tensor &b);
Tensor gt(const Tensor &a, const Tensor &b);
Tensor ge(const Tensor &a, const Tensor &b);
Tensor eq(const Tensor &a, const Tensor &b);
Tensor ne(const Tensor &a, const Tensor &b);

// Unary Ops
Tensor abs(const Tensor &a);
Tensor log(const Tensor &a);
Tensor exp(const Tensor &a);
Tensor sqrt(const Tensor &a);
Tensor sin(const Tensor &a);
Tensor cos(const Tensor &a);
Tensor tan(const Tensor &a);
Tensor asin(const Tensor &a);
Tensor acos(const Tensor &a);
Tensor atan(const Tensor &a);
Tensor sinh(const Tensor &a);
Tensor cosh(const Tensor &a);
Tensor tanh(const Tensor &a);
Tensor sigmoid(const Tensor &a);
Tensor relu(const Tensor &a);
Tensor softplus(const Tensor &a);

// Reductions
Tensor sum(const Tensor &a, int dim = -1);
Tensor mean(const Tensor &a, int dim = -1);
Tensor max(const Tensor &a, int dim = -1);
Tensor min(const Tensor &a, int dim = -1);

// Utils
Tensor cat(const std::vector<Tensor>& tensors, size_t dim);