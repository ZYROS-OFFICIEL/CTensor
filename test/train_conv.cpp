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

    // Initialize using proper PyTorch He/Kaiming initialization for ReLU
    nn::init::kaiming_uniform_(params);
    optim::AdamW optimizer(params, 0.001);
    auto criterion = nn::CrossEntropyLoss();

    // 3. Training Loop
    for (int epoch = 1; epoch <= 5; ++epoch) {
        model.train();
        
        double total_loss = 0.0;
        size_t correct = 0, total_samples = 0;
        int batch_idx = 0;

        for (auto& batch : train_loader) {
            
            optimizer.zero_grad();
            
            Tensor output = model(batch.data);
            Tensor loss = criterion(output, batch.target);
            
            loss.backward();
            
            optimizer.step();

            total_loss += loss.item<double>();
            correct += metrics::accuracy(output, batch.target);
            total_samples += batch.data.shape()[0];
            batch_idx++;

            // FIX 3: Print more frequently (every 10 batches instead of 50) 
            // and use \n instead of \r so the console history isn't overwritten.
            if (batch_idx % 10 == 0) {
                 std::cout << "Train Epoch: " << epoch 
                           << " [" << std::setw(5) << total_samples << "/" << train_loader.size() << "]\t"
                           << "Loss: " << std::fixed << std::setprecision(4) << (total_loss / batch_idx) << "\t"
                           << "Acc: " << std::setprecision(2) << (100.0 * correct / total_samples) << "%\n";
            }
        }
        
        std::cout << "\nEpoch " << epoch << " Finished. Avg Loss: " << (total_loss / batch_idx) 
                  << " Accuracy: " << (100.0 * correct / total_samples) << "%\n";
        
        checkpoints::save_weights(params, "mnist_weights.bin");
    }

    return 0;
}