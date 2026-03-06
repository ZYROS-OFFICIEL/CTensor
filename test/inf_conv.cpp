#include "core.h"
#include "neuralnet.h"
#include <omp.h>
#include <iomanip>

using namespace torch;

class ConvNet : public nn::Module {
public:
    // FIX: Added the 4th argument (kernel_size_w) to match your custom Conv2d constructor.
    // Signature: Conv2d(in_channels, out_channels, kernel_size_h, kernel_size_w)
    nn::Conv2d conv1{1, 16, 3, 3};  
    nn::Conv2d conv2{16, 32, 3, 3};
    nn::Conv2d conv3{32, 64, 3, 3};
    
    // MaxPool2d requires kernel dimensions (from pooling.h: MaxPool2d(kh, kw))
    ::MaxPool2d maxpool1{2, 2};  
    
    nn::Flatten flat;
    
    // MNIST input is 28x28.
    // After Conv1 (3x3, pad=0): 28 -> 26
    // After Conv2 (3x3, pad=0): 26 -> 24
    // After Conv3 (3x3, pad=0): 24 -> 22
    // After MaxPool1 (2x2): 22 / 2 = 11
    // Final feature map shape: [Batch, 64 channels, 11 height, 11 width]
    nn::Linear fc1{64 * 11 * 11, 10}; 

    Tensor forward(const Tensor& x) {
        Tensor out = nn::functional::relu(conv1(x));
        out = nn::functional::relu(conv2(out));
        out = nn::functional::relu(conv3(out));
        
        out = maxpool1(out);
        
        out = flat(out); 
        
        return fc1(out);
    }

    Tensor operator()(const Tensor& x) { 
        return forward(x); 
    }

    std::vector<Tensor*> parameters() override {
        return nn::combine_params(conv1, conv2, conv3, fc1);
    }
};


int main() {
    omp_set_num_threads(4);
    std::cout << "PyTorch-Style CTensor MNIST Training\n";

    auto dataset = vision::datasets::MNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    
    // Batch size 64 provides smoother gradients than 32
    torch::DataLoader train_loader(dataset, 32, /*shuffle=*/true);

    // 2. Model, Optimizer, Loss
    ConvNet model;
    
    // Cache parameters to local variable
    auto params = model.parameters();
    checkpoints::load_weights(params, "mnist_weights.bin");

    evaluate(model , train_loader);
    return 0;
}