#include "Relu.h"


Tensor LeakyRelu(const Tensor& a_,double negative_slope ){
    if (!a_.impl)
        throw std::runtime_error("Relu_: null tensor implementation");
        bool req = a_.requires_grad();

    Tensor result(a_.impl->shape,a_.impl->ndim, a_.impl->dtype, req);
    
    if (req) {
        result.impl->grad_fn = std::make_shared<GradLeakyRelu>(a_);
    }

    auto* a_data = a_.impl->storage->data.get();
    auto* r_data = result.impl->storage->data.get();

    for (size_t i = 0; i < n; ++i) {
        double val = read_scalar_at(a_data, i, a_.impl->dtype);
        write_scalar_at(r_data, i, result.impl->dtype, ((val >= 0.0 )? val : val*negative_slope));
    }

    return result;

}
Tensor PRelu(const Tensor& a_,DType init,int num_parameters,DType dtype){
    size_t in_c  = a_.impl->shape[1];
    if (!a_.impl)
        throw std::runtime_error("Relu_: null tensor implementation");

    bool req = a_.requires_grad();
    Tensor result(a_.impl->shape,a_.impl->ndim, a_.impl->dtype, req);
    if(num_parameters == in_c && num_parameters>1){

    }

}