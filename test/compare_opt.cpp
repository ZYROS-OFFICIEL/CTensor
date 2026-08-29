#include "core.h"
#include "neuralnet.h"
#include <iomanip>
#include <omp.h>
#include <chrono>
#include <vector>

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

    Tensor operator()(const Tensor& x) { 
        return forward(x); 
    }

    std::vector<Tensor*> parameters() override {
        return nn::combine_params(fc1, fc2, fc3);
    }
};

template <typename Model>
double evaluate_model(Model& model, SimpleDataLoader& test_loader) {
    model.eval();
    test_loader.reset();
    size_t correct = 0;
    size_t total = 0;

    while (test_loader.has_next()) {
        auto batch = test_loader.next();
        Tensor output = model(batch.first);
        
        correct += metrics::accuracy(output, batch.second);
        total += batch.first.shape()[0];
    }
    return (total > 0) ? (100.0 * correct / total) : 0.0;
}

int main() {
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    std::cout << "====================================================\n";
    std::cout << "    OPTIMIZER BENCHMARK: Pure AdamW vs. Hybrid GA   \n";
    std::cout << "    Running on OpenMP with " << num_threads << " threads           \n";
    std::cout << "====================================================\n";

    // Load MNIST Dataset
    auto train_dataset = vision::datasets::MNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    auto test_dataset  = vision::datasets::MNIST("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
    
    SimpleDataLoader train_loader(train_dataset, 32, true);
    SimpleDataLoader test_loader(test_dataset, 64, false);

    int total_epochs = 3;

    std::cout << "\n[1/2] Training baseline model with Pure AdamW...\n";
    MLPNet adam_model;
    auto adam_params = adam_model.parameters();
    nn::init::kaiming_uniform_(adam_params);
    optim::AdamW adam_optimizer(adam_params, 0.01);
    auto criterion = nn::CrossEntropyLoss();

    auto start_adam = std::chrono::high_resolution_clock::now();

    for (int epoch = 1; epoch <= total_epochs; ++epoch) {
        adam_model.train();
        train_loader.reset();
        double epoch_loss = 0.0;
        int batch_count = 0;

        while (train_loader.has_next()) {
            auto batch = train_loader.next();
            adam_optimizer.zero_grad();
            
            Tensor output = adam_model(batch.first);
            Tensor loss = criterion(output, batch.second);
            loss.backward();
            adam_optimizer.step();

            epoch_loss += loss.item<double>();
            batch_count++;
        }
        std::cout << "AdamW Epoch " << epoch << " | Avg Loss: " << (epoch_loss / batch_count) << "\n";
    }

    auto end_adam = std::chrono::high_resolution_clock::now();
    double adam_time = std::chrono::duration<double>(end_adam - start_adam).count();
    double adam_acc = evaluate_model(adam_model, test_loader);


    std::cout << "\n[2/2] Training model with Parallel Hybrid GA (Memetic)...\n";
    MLPNet hga_model;
    auto hga_params = hga_model.parameters();
    nn::init::kaiming_uniform_(hga_params);

    HybridGA ga_optimizer(hga_params, 10, 0.05f,0.1f);

    std::vector<std::unique_ptr<MLPNet>> thread_models;
    std::vector<std::unique_ptr<optim::AdamW>> thread_optimizers;
    for (int i = 0; i < num_threads; ++i) {
        thread_models.push_back(std::make_unique<MLPNet>());
        thread_optimizers.push_back(std::make_unique<optim::AdamW>(thread_models[i]->parameters(), 0.01));
    }

    auto start_hga = std::chrono::high_resolution_clock::now();
    
    train_loader.reset(); 

    for (int gen = 1; gen <= total_epochs; ++gen) {
        std::vector<std::pair<Tensor, Tensor>> gen_batches;
        for (int b = 0; b < 3; ++b) {
            if (!train_loader.has_next()) {
                train_loader.reset();
            }
            gen_batches.push_back(train_loader.next());
        }

        ga_optimizer.step([&](size_t ind_idx, const std::vector<float>& genome) -> std::pair<double, std::vector<float>> {
            int tid = omp_get_thread_num();
            MLPNet& local_model = *thread_models[tid];
            optim::AdamW& local_optim = *thread_optimizers[tid];
            auto local_params = local_model.parameters();

            local_model.train();
            inject_genome(local_params, genome); 
            double final_loss = 0.0;
            for (const auto& batch : gen_batches) {
                local_optim.zero_grad();
                Tensor output = local_model(batch.first);
                Tensor loss = criterion(output, batch.second);
                loss.backward();
                local_optim.step();
                final_loss = loss.item<double>();
            }

            return {final_loss, extract_genome(local_params)};
        });

        std::cout << "Hybrid GA Generation " << gen << " completed.\n";
    }

    auto end_hga = std::chrono::high_resolution_clock::now();
    double hga_time = std::chrono::duration<double>(end_hga - start_hga).count();
    
    inject_genome(hga_params, ga_optimizer.population[0].genome);
    double hga_acc = evaluate_model(hga_model, test_loader);


    std::cout << "\n====================================================\n";
    std::cout << "                 FINAL BENCHMARK RESULTS            \n";
    std::cout << "====================================================\n";
    std::cout << std::left << std::setw(20) << "Metric" 
              << std::setw(15) << "Pure AdamW" 
              << std::setw(15) << "Hybrid GA" << "\n";
    std::cout << "----------------------------------------------------\n";
    std::cout << std::left << std::setw(20) << "Test Accuracy (%)" 
              << std::setw(15) << std::fixed << std::setprecision(2) << adam_acc 
              << std::setw(15) << hga_acc << "\n";
    std::cout << std::left << std::setw(20) << "Training Time (s)" 
              << std::setw(15) << std::fixed << std::setprecision(2) << adam_time 
              << std::setw(15) << hga_time << "\n";
    std::cout << "====================================================\n";

    return 0;
}