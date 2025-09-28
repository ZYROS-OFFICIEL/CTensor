#include <iostream>
#include "ops.h"

int main() {
    Tensor t({2,3}, true);
    std::cout << "Number of elements: " << t.numel() << std::endl;

    t.data[0] = 1.0f;
    t.grad[0] = 0.5f;

    std::cout << "First element: " << t.data[0] << ", grad: " << t.grad[0] << std::endl;
    print_t(t);
    Tensor a({2,3});
    Tensor b({2,3});

    for(size_t i=0;i<a.numel();i++){
        a.data[i] = i+1;
        b.data[i] = i+10;
    }

    Tensor c = a + b;
    print_t(a);
    print_t(b);
    print_t(c);
}
