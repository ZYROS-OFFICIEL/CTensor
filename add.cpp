#include "ops1.h"

using namespace std;
int main() {
    // Example usage of add_ function
    Tensor a = Tensor::ones({3, 3}, DType::Float32);
    Tensor b = Tensor::full({2, 3}, 2.0f, DType::Float32);
    Tensor c = add_(a, b);
    Tensor d = matmul(b, a);

    cout<<("Tensor a:") <<endl;
    print_(a);
    cout <<("Tensor b:") << endl;
    print_(b);
    cout << ("Tensor c = a + b:") <<endl;
    print_(c);
    cout << ("Tensor c = a ** b:") <<endl;
    print_(d);

    return 0;

}