#include <vector>
#include <functional>
#include <stdexcept>
#include "transforms.h"
#include "ops.h"


int main() {
    Transforme transforms;
    transforms.normalize_({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f});
    transforms.resize_(224, 224);
    Tensor img = Tensor::ones({3, 256, 256});  // input image
    Tensor out = transforms(img); // applies normalize + resize
    out.print_shape(); // should be (3, 224, 224)
    Tensor a = Tensor::ones({5});
    Tensor b = Tensor::full({2,3,5},2.0f);
    print_t(a);
    print_t(b);
}