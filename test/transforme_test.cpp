#include <vector>
#include <functional>
#include <stdexcept>
#include "core/transforme.h"
#include "core/ops_dispatch.h"

using namespace std;

int main() {
    Transforme transforms;
    transforms.normalize_({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f});
    transforms.resize_(224, 224);
    transforms.to_(DType::Int32);
    Tensor img = Tensor::ones({3, 256, 256});  // input image
    cout << "Before transforms:\n" << img._dtype() << "\n";
    Tensor out = transforms(img); // applies normalize + resize
    cout << "After transforms:\n" << out._dtype() << "\n";
    print_t(out);
    out.print_shape(); // should be (3, 224, 224)
    Tensor a = Tensor::rand({5});
    Tensor b = Tensor::full({2,3,5},2.0f);
    print_t(a);
    print_t(b);
}