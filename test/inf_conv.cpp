#include "core.h"
#include "neuralnet.h"
#include <omp.h>
#include <iomanip>

using namespace torch;

class ConvNet : public nn::Module {
public:
    // Block 1
    nn::Conv2d conv1{1, 16, 3, 3};  
    ::MaxPool2d maxpool1{2, 2};  

    // Block 2
    nn::Conv2d conv2{16, 32, 3, 3};
    ::MaxPool2d maxpool2{2, 2};  
    
    nn::Flatten flat;
    
    // New Dimension Math:
    // Input: 28x28
    // conv1 (3x3): 28 -> 26
    // maxpool1 (2x2): 26 / 2 = 13
    // conv2 (3x3): 13 -> 11
    // maxpool2 (2x2): 11 / 2 = 5  (integer division: 11/2 = 5)
    // Final feature map shape: [Batch, 32 channels, 5 height, 5 width]
    nn::Linear fc1{32 * 5 * 5, 128}; 
    nn::Linear fc2{128, 10}; 

    Tensor forward(const Tensor& x) {
        // Block 1
        Tensor out = conv1(x);
        out = nn::functional::relu(out);
        out = maxpool1(out);
        
        // Block 2
        out = conv2(out);
        out = nn::functional::relu(out);
        out = maxpool2(out);
        
        // Classifier
        out = flat(out); 
        out = nn::functional::relu(fc1(out));
        return fc2(out);
    }

    Tensor operator()(const Tensor& x) { 
        return forward(x); 
    }

    std::vector<Tensor*> parameters() override {
        return nn::combine_params(conv1, conv2, fc1, fc2);
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