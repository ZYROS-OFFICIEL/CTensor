#include "core/tensor.h"
#include "core/ops_dispatch.h"
#include <iostream>
#include <fstream> 
#include <numeric>
#include <cmath>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <omp.h>

void kaiming_init(std::vector<Tensor*>& params) ;
void uniform_(Tensor& tensor, double a = 0.0, double b = 1.0);
void normal_(Tensor& tensor, double mean = 0.0, double std = 1.0);
void constant_(Tensor& tensor, double val);
void zeros_(Tensor& tensor);
void ones_(Tensor& tensor);
void xavier_uniform_(Tensor& tensor, double gain = 1.0);
void xavier_normal_(Tensor& tensor, double gain = 1.0);