#include "tensors.h"
#include "ops.h"
int main() {
    Tensor a = Tensor::empty({3,2,4}, DType::Float32);
    print_t(a); // prints [1,1,1]
    a.to_(DType::Double64); // convert in-place to double
    std::cout << "dtype now: " << a._dtype() << "\n"; // prints Double64

    Tensor b = Tensor::full({3,2,4}, 2.5, DType::Int32); // integer tensor
    print_t(b); // prints [2, 2, 2]  (written via rounding)
    Tensor c = a.astype(DType::Int32); // returns new tensor cast to Int32
    print_t(c);
    print_(c); // prints with newlines and braces
    // Indexing:
    Tensor x = Tensor::zeros({2,3}, DType::Float32);
    x[0][1] = 3.14;           // write (Proxy)
    double v = x[0][1];       // read (ConstProxy/Proxy -> double)
    std::cout << v << "\n";
    a.t_(); 
    print_(a); 
    Tensor d = matmul(a,b); // (3,2,4) @ (3,) -> (3,2)
    print_t(d);
    return 0;
}
