#include <iostream>
#include "neuralnet/dataset/mnist.h"

int main() {
    MNISTData data = load_mnist(
        "train-images.idx3-ubyte",
        "train-labels.idx1-ubyte"
    );

    const Tensor& images = data.images;
    const Tensor& labels = data.labels;

    std::cout << "Loaded images: " << images.numel() << "\n";
    std::cout << "Images shape: ";
    for (auto s : images.shape()) std::cout << s << " ";
    std::cout << "\n";

    std::cout << "Labels shape: ";
    for (auto s : labels.shape()) std::cout << s << " ";
    std::cout << "\n";

    // ---- Check first label ----
    int32_t first_label =
        *((int32_t*)labels.impl->data->data.get());
    std::cout << "First label = " << first_label << "\n";

    // ---- Check first image ----
    float* img0 = (float*)images.impl->data->data.get();

    float mn = 1e9, mx = -1e9;
    for (int i = 0; i < 28 * 28; i++) {
        float v = img0[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }

    std::cout << "First image min = " << mn << "\n";
    std::cout << "First image max = " << mx << "\n";

    std::cout << "First 20 pixels:\n";
    for (int i = 0; i < 20; i++)
        std::cout << img0[i] << " ";
    std::cout << "\n";
}
