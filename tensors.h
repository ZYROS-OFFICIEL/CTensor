#include <iostream>
#include <cstring>
#include <cmath>

typedef struct {
    float *data;
    size_t ndim;
    size_t *shape;
    size_t *strides;
} Tensor;