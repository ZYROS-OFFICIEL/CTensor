#include "core.h"
#include "neuralnet.h"
#include <iomanip>
#include <omp.h>

using namespace torch;

// =======================================================================
//                              MODEL
// =======================================================================

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

// =======================================================================
//                              MAIN
// =======================================================================

int main() {
    omp_set_num_threads(4);
    std::cout << "PyTorch-Style CTensor MNIST Training\n";

    // 1. Data Pipeline
    auto dataset = vision::datasets::MNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    
    // Explicitly use torch::DataLoader to completely bypass the old DataLoader conflict
    torch::DataLoader train_loader(dataset, /*batch_size=*/64, /*shuffle=*/true);

    // 2. Model, Optimizer, Loss
    MLPNet model;
    
    // Cache parameters to local variable to fix R-value reference compilation error
    auto params = model.parameters();
    
    nn::init::kaiming_uniform_(params);
    optim::AdamW optimizer(params, /*lr=*/0.001);
    
    auto criterion = nn::CrossEntropyLoss();

    // 3. Training Loop
    for (int epoch = 1; epoch <= 5; ++epoch) {
        model.train();
        
        double total_loss = 0.0;
        size_t correct = 0, total_samples = 0;
        int batch_idx = 0;

        // Elegant Pythonic Range-based Loop!
        for (auto& batch : train_loader) {
            
            // Forward & Backward Pass
            optimizer.zero_grad();
            
            // This now works because of the operator() overload
            Tensor output = model(batch.data);
            Tensor loss = criterion(output, batch.target);
            
            loss.backward();
            nn::utils::clip_grad_norm_(params, 1.0); // Use cached params!
            optimizer.step();

            // Track Metrics
            total_loss += loss.item<double>();
            correct += metrics::accuracy(output, batch.target);
            total_samples += batch.data.shape()[0];
            batch_idx++;

            // Console Progress
            if (batch_idx % 50 == 0) {
                 std::cout << "Train Epoch: " << epoch 
                           << " [" << std::setw(5) << total_samples << "/" << train_loader.size() << "]\t"
                           << "Loss: " << std::fixed << std::setprecision(4) << (total_loss / batch_idx) << "\t"
                           << "Acc: " << std::setprecision(2) << (100.0 * correct / total_samples) << "% \r" << std::flush;
            }
        }
        
        std::cout << "\nEpoch " << epoch << " Finished. Avg Loss: " << (total_loss / batch_idx) 
                  << " Accuracy: " << (100.0 * correct / total_samples) << "%\n";
        
        // Save Checkpoint
        checkpoints::save_weights(params, "mnist_weights.bin");
    }

    return 0;
}