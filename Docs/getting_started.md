# CTensor
CTensor is a tensor library for different tensor applications although engineered for deep learning.

## Installation 

To install CTensor to your machine you should follow these steps:

1. Clone the repo:
'git clone https://github.com/ZYROS-OFFICIEL/CTensor'
2. Indicate to cmake the build directory : 
'cmake -B build'
3. Build:
'cmake --build build'

And you are done .Great!


## Core Concepts

If you know PyTorch, you already know CTensor. The library is built around a few familiar namespaces:

torch::nn: Contains neural network layers (Linear, Flatten), activations (relu, sigmoid), and loss functions (CrossEntropyLoss).

torch::optim: Contains optimizers like SGD, Adam, AdamW, RMSprop, etc.

torch::DataLoader: A simple Pythonic iterator for batching datasets.

metrics: Easy-to-use metric calculators (e.g., accuracy, mse, f1_score).

1. Defining a Model

Creating a neural network is as simple as inheriting from torch::nn::Module. Define your layers as class members and implement the forward pass and operator().

#include "core.h"
#include "neuralnet.h"

using namespace torch;

class MLPNet : public nn::Module {
public:
    nn::Flatten flat;
    nn::Linear fc1{784, 128};
    nn::Linear fc2{128, 64};
    nn::Linear fc3{64, 10};

    Tensor forward(const Tensor& x) {
        Tensor out = flat(x);
        out = nn::functional::relu(fc1(out));
        out = nn::functional::relu(fc2(out));
        return fc3(out);
    }

    // Overload for elegant model(x) syntax
    Tensor operator()(const Tensor& x) { 
        return forward(x); 
    }

    // Combine parameters for the optimizer
    std::vector<Tensor*> parameters() override {
        return nn::combine_params(fc1, fc2, fc3);
    }
};


2. Loading Data

CTensor includes built-in vision dataset loaders and standard transforms.

// 1. Load MNIST from raw bytes
auto dataset = vision::datasets::MNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

// 2. Wrap it in a DataLoader (Batch Size 64, Shuffle = True)
torch::DataLoader train_loader(dataset, 64, true);


3. The Training Loop

The training loop looks virtually identical to PyTorch. We use range-based for loops to iterate through the dataloader cleanly.

int main() {
    // Instantiate Model
    MLPNet model;
    auto params = model.parameters();
    
    // Enable gradients and initialize weights (He/Kaiming)
    for (auto* p : params) if (p) p->requires_grad_(true);
    nn::init::kaiming_uniform_(params);
    
    // Setup Optimizer and Loss Function
    optim::AdamW optimizer(params, 0.001);
    auto criterion = nn::CrossEntropyLoss();

    model.train(); // Set model to training mode
    
    for (int epoch = 1; epoch <= 5; ++epoch) {
        // Iterate through batches natively!
        for (auto& batch : train_loader) {
            
            optimizer.zero_grad();                   // 1. Clear previous gradients
            
            Tensor output = model(batch.data);       // 2. Forward pass
            Tensor loss = criterion(output, batch.target); // 3. Compute loss
            
            loss.backward();                         // 4. Autograd backward pass
            nn::utils::clip_grad_norm_(params, 1.0); // 5. Clip gradients for stability
            optimizer.step();                        // 6. Update weights
            
        }
    }
    
    // Save your trained model
    checkpoints::save_weights(params, "mnist_weights.bin");
    return 0;
}

If you want to learn more you can folllow the sources below.
