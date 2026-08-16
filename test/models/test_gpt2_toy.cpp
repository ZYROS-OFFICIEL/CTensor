#include <iostream>
#include <cassert>
#include <cmath>
#include "models/gpt2.h"
#include "neuralnet/training_utils.h"

void test_forward_shape() {
    GPT2Config cfg = GPT2Config::toy(50);
    GPT2Model model(cfg);

    Tensor ids = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3}, DType::Int32);
    Tensor logits = model(ids);

    assert(logits.shape().size() == 3);
    assert(logits.shape()[0] == 2);
    assert(logits.shape()[1] == 3);
    assert(logits.shape()[2] == 50);
}

void test_gradients_flow() {
    GPT2Config cfg = GPT2Config::toy(20);
    GPT2Model model(cfg);
    model.train();

    Tensor ids = Tensor::from_vector({1, 2, 3, 4, 5, 6, 7, 8}, {2, 4}, DType::Int32);
    Tensor logits = model(ids);

    Tensor loss = sum(logits.flatten(), 0);
    loss.backward();

    auto params = model.parameters();
    assert(params.size() == (size_t)(2 + 16 * cfg.n_layer + 2));

    for (auto* p : params) {
        Tensor g = p->grad();
        assert(g.impl != nullptr);
        bool any_nonzero = false;
        for (size_t i = 0; i < g.numel(); ++i) {
            double v = g.read_scalar(i);
            assert(std::isfinite(v));
            if (v != 0.0) any_nonzero = true;
        }
        assert(any_nonzero);
    }
}

void test_overfit_tiny_batch() {
    GPT2Config cfg = GPT2Config::toy(10);
    cfg.dropout = 0.0;
    GPT2Model model(cfg);
    model.train();

    // fixed tiny "next token = same token + 1 mod vocab" sequence, repeated
    Tensor input = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {1, 6}, DType::Int32);
    Tensor target = Tensor::from_vector({2, 3, 4, 5, 6, 7}, {1, 6}, DType::Int32);

    auto params = model.parameters();
    AdamW optim(params, 0.01);

    double first_loss = -1.0, last_loss = -1.0;
    for (int step = 0; step < 150; ++step) {
        optim.zero_grad();
        Tensor logits = model(input);           // [1,6,10]
        Tensor logits2d = logits.reshape({6, 10});
        Tensor loss = Loss::CrossEntropy(logits2d, target.reshape({6, 1}), "mean");
        loss.backward();
        optim.step();

        double l = loss.read_scalar(0);
        if (step == 0) first_loss = l;
        last_loss = l;
    }

    std::cout << "  overfit: first_loss=" << first_loss << " last_loss=" << last_loss << "\n";
    assert(last_loss < first_loss * 0.65);
}

int main() {
    test_forward_shape();
    std::cout << "test_forward_shape passed\n";
    test_gradients_flow();
    std::cout << "test_gradients_flow passed\n";
    test_overfit_tiny_batch();
    std::cout << "test_overfit_tiny_batch passed\n";
    std::cout << "test_gpt2_toy passed\n";
    return 0;
}
