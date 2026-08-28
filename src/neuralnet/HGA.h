#pragma once
#include "core.h"
#include "neuralnet.h"
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <omp.h>

inline std::vector<float> extract_genome(const std::vector<Tensor*>& params) {
    std::vector<float> genome;
    for (auto* p : params) {
        if (!p || !p->impl) continue;
        
        if (p->_dtype() == DType::Float32) {
            float* ptr = (float*)p->impl->data->data.get();
            size_t n = p->numel();
            genome.insert(genome.end(), ptr, ptr + n);
        }
    }
    return genome;
}

inline void inject_genome(const std::vector<Tensor*>& params, const std::vector<float>& genome) {
    size_t offset = 0;
    for (auto* p : params) {
        if (!p || !p->impl) continue;
        
        if (p->_dtype() == DType::Float32) {
            float* ptr = (float*)p->impl->data->data.get();
            size_t n = p->numel();
            std::memcpy(ptr, genome.data() + offset, n * sizeof(float));
            offset += n;
        }
    }
}

struct Individual {
    std::vector<float> genome;
    double loss;
};

class HybridGA {
public:
    std::vector<Tensor*> base_params;
    size_t pop_size;
    float mutation_rate;
    float mutation_scale;
    std::mt19937 gen;
    std::vector<Individual> population;

    HybridGA(std::vector<Tensor*> p, size_t population_size = 10, 
             float mut_rate = 0.05f, float mut_scale = 0.1f) 
        : base_params(p), pop_size(population_size), 
          mutation_rate(mut_rate), mutation_scale(mut_scale), 
          gen(std::random_device{}()) 
    {
        std::normal_distribution<float> noise(0.0f, mutation_scale);
        std::vector<float> base_genome = extract_genome(base_params);
        
        population.push_back({base_genome, 1e9}); 
        for (size_t i = 1; i < pop_size; ++i) {
            std::vector<float> mutant = base_genome;
            for (float& w : mutant) w += noise(gen); 
            population.push_back({mutant, 1e9});
        }
    }

    template <typename TrainClosure>
    void step(TrainClosure train_individual) {
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        std::normal_distribution<float> noise(0.0f, mutation_scale);

        std::cout << "\n Starting Parallel Memetic Evolution Step \n";

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)pop_size; ++i) {
            auto result = train_individual(i, population[i].genome);
            
            population[i].loss = result.first;    
            population[i].genome = result.second; 
        }

        std::sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
            return a.loss < b.loss;
        });
        std::cout << "Generation Best Loss: " << std::fixed << std::setprecision(4) << population[0].loss << "\n";

        std::vector<Individual> next_gen;
        size_t elite_count = std::max((size_t)1, pop_size / 5);
        for (size_t i = 0; i < elite_count; ++i) {
            next_gen.push_back(population[i]);
        }

        while (next_gen.size() < pop_size) {
            size_t p1 = std::uniform_int_distribution<size_t>(0, elite_count - 1)(gen);
            size_t p2 = std::uniform_int_distribution<size_t>(0, elite_count - 1)(gen);
            
            std::vector<float> child(population[p1].genome.size());
            for (size_t j = 0; j < child.size(); ++j) {
                child[j] = (prob(gen) > 0.5f) ? population[p1].genome[j] : population[p2].genome[j];
                if (prob(gen) < mutation_rate) child[j] += noise(gen);
            }
            next_gen.push_back({std::move(child), 1e9});
        }

        population = std::move(next_gen);
        
        inject_genome(base_params, population[0].genome);
    }
};