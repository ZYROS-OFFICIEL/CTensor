#include "view.h"
#include "tensors.h"
#include "ops.h"

int main() {
    Tensor a = Tensor::empty({3,2,4}, DType::Float32);
    print_t(a); // prints [1,1,1]
    a.to_(DType::Double64); // convert in-place to double
    std::cout << "dtype now: " << a._dtype() << "\n"; // prints Double64

    Tensor b = Tensor::full({3,2,4}, 2.5, DType::Int32); // integer tensor
    print_(b); // prints [2, 2, 2]  (written via rounding)
    std::cout << "b max\n"; // prints Int32
    print_(max(b,1));  
    a = Tensor::arange(0, 12, 1).reshape({3,4});
    print_t(a);
    // [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
    
    Tensor row1 = a.select(0, 1);
    print_t(row1);
    return 0;
}
