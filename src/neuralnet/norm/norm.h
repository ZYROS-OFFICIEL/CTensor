#pragma once
#include "core.h"


Tensor norm(const Tensor& t, double p = 2.0, int dim = -1, bool keepdim = false);

Tensor infinity_norm(const Tensor& t, int dim = -1, bool keepdim = false);
Tensor zero_norm(const Tensor& t, int dim = -1, bool keepdim = false);


Tensor Lp_Norm(Tensor& t,int p = 2, int dim = -1, double eps = 1e-12);

