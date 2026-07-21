#pragma once
#include "core.h"
#include "neuralnet.h"
#include "neuralnet/embedding/embedding.h"
#include "neuralnet/norm/layernorm.h"
#include <vector>
#include <memory>
#include <stdexcept>

struct GPT2Config {
    int n_layer;
    int n_head;
    int n_embd;
    int vocab_size;
    int n_positions;
    double dropout = 0.1;
    double layer_norm_eps = 1e-5;

    static GPT2Config toy(int vocab_size) {
        GPT2Config c;
        c.n_layer = 2; c.n_head = 2; c.n_embd = 32;
        c.vocab_size = vocab_size; c.n_positions = 64;
        return c;
    }

    static GPT2Config gpt2_small_124M() {
        GPT2Config c;
        c.n_layer = 12; c.n_head = 12; c.n_embd = 768;
        c.vocab_size = 50257; c.n_positions = 1024;
        return c;
    }
};

inline Tensor make_causal_mask(size_t S) {
    Tensor rows = Tensor::arange(0, (double)S, 1, DType::Float32).unsqueeze(1);
    Tensor cols = Tensor::arange(0, (double)S, 1, DType::Float32).unsqueeze(0);
    return (cols > rows) * (-1e9);
}

class GPT2Attention : public Module {
public:
    int n_embd, n_head, head_dim;
    Linear q_proj, k_proj, v_proj, out_proj;
    Dropout attn_dropout, resid_dropout;

    GPT2Attention(const GPT2Config& cfg);

    Tensor forward(const Tensor& x, const Tensor& causal_mask);
    Tensor operator()(const Tensor& x, const Tensor& causal_mask) { return forward(x, causal_mask); }

    void train() override { Module::train(); attn_dropout.train(); resid_dropout.train(); }
    void eval() override { Module::eval(); attn_dropout.eval(); resid_dropout.eval(); }

    std::vector<Tensor*> parameters() override {
        return torch::nn::combine_params(q_proj, k_proj, v_proj, out_proj);
    }
};

class GPT2MLP : public Module {
public:
    Linear c_fc, c_proj;
    Dropout resid_dropout;

    GPT2MLP(const GPT2Config& cfg)
        : c_fc(cfg.n_embd, 4 * cfg.n_embd),
          c_proj(4 * cfg.n_embd, cfg.n_embd),
          resid_dropout(cfg.dropout) {}

    Tensor forward(const Tensor& x) { return resid_dropout(c_proj(gelu(c_fc(x)))); }
    Tensor operator()(const Tensor& x) { return forward(x); }

    void train() override { Module::train(); resid_dropout.train(); }
    void eval() override { Module::eval(); resid_dropout.eval(); }

    std::vector<Tensor*> parameters() override {
        return torch::nn::combine_params(c_fc, c_proj);
    }
};

class GPT2Block : public Module {
public:
    LayerNorm ln_1, ln_2;
    GPT2Attention attn;
    GPT2MLP mlp;

    GPT2Block(const GPT2Config& cfg)
        : ln_1(cfg.n_embd, cfg.layer_norm_eps),
          ln_2(cfg.n_embd, cfg.layer_norm_eps),
          attn(cfg), mlp(cfg) {}

    Tensor forward(const Tensor& x, const Tensor& mask) {
        Tensor h = x + attn(ln_1(x), mask);
        return h + mlp(ln_2(h));
    }
    Tensor operator()(const Tensor& x, const Tensor& mask) { return forward(x, mask); }

    void train() override { Module::train(); attn.train(); mlp.train(); }
    void eval() override { Module::eval(); attn.eval(); mlp.eval(); }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> p;
        auto add = [&](std::vector<Tensor*> sub) { p.insert(p.end(), sub.begin(), sub.end()); };
        add(ln_1.parameters());
        add(attn.parameters());
        add(ln_2.parameters());
        add(mlp.parameters());
        return p;
    }
};

class GPT2Model : public Module {
public:
    GPT2Config config;
    nn::Embedding wte, wpe;
    Dropout drop;
    std::vector<std::shared_ptr<GPT2Block>> blocks;
    LayerNorm ln_f;

    GPT2Model(const GPT2Config& cfg);

    Tensor forward(const Tensor& token_ids);
    Tensor operator()(const Tensor& token_ids) { return forward(token_ids); }

    void train() override {
        Module::train();
        drop.train();
        for (auto& b : blocks) b->train();
    }
    void eval() override {
        Module::eval();
        drop.eval();
        for (auto& b : blocks) b->eval();
    }

    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> p;
        p.push_back(&wte.weight);
        p.push_back(&wpe.weight);
        for (auto& b : blocks) {
            auto sub = b->parameters();
            p.insert(p.end(), sub.begin(), sub.end());
        }
        p.push_back(&ln_f.weight);
        p.push_back(&ln_f.bias);
        return p;
    }

    std::vector<int> generate(const std::vector<int>& prompt_ids, int max_new_tokens, double temperature = 1.0);

private:
    void init_weights();
};
