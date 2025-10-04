#include <iostream>
#include "ops.h"

using namespace std;

int main() {
    Tensor t({2,3}, true);
    std::cout << "Number of elements: " << t.numel() << std::endl;

    t.data[0] = 1.0f;
    t.grad[0] = 0.5f;

    std::cout << "First element: " << t.data[0] << ", grad: " << t.grad[0] << std::endl;
    print_t(t);
    Tensor a({5});
    Tensor b({2,3,5});

    for(size_t i=0;i<a.numel();i++){
        a.data[i] = i;
    }
    for (size_t x = 0; x < b.shape[0]; x++) {
    for (size_t y = 0; y < b.shape[1]; y++) {
        for (size_t z = 0; z < b.shape[2]; z++) {
            b[x][y][z] = 2; 
        }
    }
}

    Tensor c = a + b;
    c.print_shape(); // (2, 3, 4)
    auto s = c.shape_(); // returns std::vector<size_t>{2,3,4}
    Tensor d = a * b;
    Tensor e = a / b;
    Tensor f = a ^ b;
    print_t(a);
    cout << "b:\n";
    print_t(b);
    cout << "c:\n";
    print_t(c);
    cout << "d:\n";
    print_t(d);
    cout << "e:\n";
    print_t(e);
    cout << "f:\n";
    print_t(f);
}
