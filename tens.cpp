#include "tensors.h"
#include "ops.h"
#include "transforms.h"
int main() {
    Tensor a = Tensor::empty({3,2,4}, DType::Float32);
    print_t(a);
    a.to_(DType::Double64);
    std::cout << "dtype now: " << a._dtype() << "\n";

    Tensor b = Tensor::full({3,2,4}, 2.5, DType::Int32);
    print_t(b);

    Tensor c = a.astype(DType::Int32);
    print_t(c);

    Tensor x = Tensor::zeros({2,3}, DType::Float32);
    x[0][1] = 3.14;
    double v = x[0][1];
    std::cout << v << "\n";

    Tensor z = Tensor::arange(0, 12, 1).reshape({3,4});
    print_t(z);

    Tensor row1 = z.select(0, 1);
    print_t(row1);

    Tensor sub = z.slice(1, 1, 3);
    print_t(sub);

    return 0;
}

