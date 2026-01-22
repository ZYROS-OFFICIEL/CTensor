#include "batchnorm.h"
#include "ops_dispatch.h"
#include <stdexcept>
#include <cmath>
#include <numeric> 

// Constructor
BatchNorm::BatchNorm(int num_features, double eps, double momentum)
    : num_features(num_features), eps(eps), momentum(momentum), training(true)
{
    // Gamma initialized to 1, Beta to 0
    gamma = Tensor::full({(size_t)num_features}, 1.0, DType::Float32, true);
    beta = Tensor::zeros({(size_t)num_features}, DType::Float32, true);

    // Running stats initialized to 0 and 1 (mean=0, var=1)
    running_mean = Tensor::zeros({(size_t)num_features}, DType::Float32, false);
    running_var = Tensor::full({(size_t)num_features}, 1.0, DType::Float32, false);
}