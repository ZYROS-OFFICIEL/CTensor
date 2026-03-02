#include "core.h"
#include "neuralnet.h"
#include <iomanip>
#include <omp.h>

using namespace torch;

//                              MODEL

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

    Tensor operator()(const Tensor& x) { 
        return forward(x); 
    }

    std::vector<Tensor*> parameters() override {
        // Magically combines parameters from all layers
        return nn::combine_params(fc1, fc2, fc3);
    }
};

//                              MAIN

int main() {
    omp_set_num_threads(4);
    std::cout << "PyTorch-Style CTensor MNIST Training\n";

    auto dataset = vision::datasets::MNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    
    // Batch size 64 provides smoother gradients than 32
    torch::DataLoader train_loader(dataset, 32, /*shuffle=*/true);

    // 2. Model, Optimizer, Loss
    MLPNet model;
    
    // Cache parameters to local variable
    auto params = model.parameters();
    
    // Initialize using proper PyTorch He/Kaiming initialization for ReLU
    nn::init::kaiming_uniform_(params);
    optim::AdamW optimizer(params, 0.01);
    
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