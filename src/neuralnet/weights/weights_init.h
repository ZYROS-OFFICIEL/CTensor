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