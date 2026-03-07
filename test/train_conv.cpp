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
    
    torch::DataLoader train_loader(dataset, 64, /*shuffle=*/true);

    ConvNet model;
    auto params = model.parameters();

    nn::init::kaiming_uniform_(params);
    optim::AdamW optimizer(params, 0.001);
    auto criterion = nn::CrossEntropyLoss();

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