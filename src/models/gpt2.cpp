#include "gpt2.h"
#include "neuralnet/weights/weights_init.h"
#include <cmath>
#include <random>
#include <algorithm>

GPT2Attention::GPT2Attention(const GPT2Config& cfg)
    : n_embd(cfg.n_embd), n_head(cfg.n_head), head_dim(cfg.n_embd / cfg.n_head),
      q_proj(cfg.n_embd, cfg.n_embd), k_proj(cfg.n_embd, cfg.n_embd),
      v_proj(cfg.n_embd, cfg.n_embd), out_proj(cfg.n_embd, cfg.n_embd),
      attn_dropout(cfg.dropout), resid_dropout(cfg.dropout)
{
    if (cfg.n_embd % cfg.n_head != 0)
        throw std::invalid_argument("GPT2Attention: n_embd must be divisible by n_head");
}

Tensor GPT2Attention::forward(const Tensor& x, const Tensor& causal_mask) {
    std::vector<size_t> xs = x.shape();
    size_t B = xs[0], S = xs[1];
    size_t H = (size_t)n_head, Dh = (size_t)head_dim;

    Tensor q = q_proj(x);
    Tensor k = k_proj(x);
    Tensor v = v_proj(x);

    auto to_heads = [&](const Tensor& t) {
        return t.reshape({B, S, H, Dh}).permute({0, 2, 1, 3}).contiguous().reshape({B * H, S, Dh});
    };

    Tensor q3 = to_heads(q);
    Tensor k3 = to_heads(k);
    Tensor v3 = to_heads(v);

    double scale = 1.0 / std::sqrt((double)head_dim);
    Tensor scores = bmm(q3, k3.permute({0, 2, 1})) * scale;
    scores = scores + causal_mask;

    Tensor attn_w = attn_dropout(functional::softmax(scores, -1));
    Tensor out3 = bmm(attn_w, v3);

    Tensor out = out3.reshape({B, H, S, Dh}).permute({0, 2, 1, 3}).contiguous().reshape({B, S, (size_t)n_embd});

    return resid_dropout(out_proj(out));
}

GPT2Model::GPT2Model(const GPT2Config& cfg)
    : config(cfg),
      wte(cfg.vocab_size, cfg.n_embd),
      wpe(cfg.n_positions, cfg.n_embd),
      drop(cfg.dropout),
      ln_f(cfg.n_embd, cfg.layer_norm_eps)
{
    for (int i = 0; i < cfg.n_layer; ++i) blocks.push_back(std::make_shared<GPT2Block>(cfg));
    init_weights();
}

void GPT2Model::init_weights() {
    normal_(wte.weight, 0.0, 0.02);
    normal_(wpe.weight, 0.0, 0.02);

    double resid_std = 0.02 / std::sqrt(2.0 * config.n_layer);

    for (auto& blk : blocks) {
        normal_(blk->attn.q_proj.weight, 0.0, 0.02);
        zeros_(blk->attn.q_proj.bias);
        normal_(blk->attn.k_proj.weight, 0.0, 0.02);
        zeros_(blk->attn.k_proj.bias);
        normal_(blk->attn.v_proj.weight, 0.0, 0.02);
        zeros_(blk->attn.v_proj.bias);
        normal_(blk->attn.out_proj.weight, 0.0, resid_std);
        zeros_(blk->attn.out_proj.bias);

        normal_(blk->mlp.c_fc.weight, 0.0, 0.02);
        zeros_(blk->mlp.c_fc.bias);
        normal_(blk->mlp.c_proj.weight, 0.0, resid_std);
        zeros_(blk->mlp.c_proj.bias);

        ones_(blk->ln_1.weight);
        zeros_(blk->ln_1.bias);
        ones_(blk->ln_2.weight);
        zeros_(blk->ln_2.bias);
    }

    ones_(ln_f.weight);
    zeros_(ln_f.bias);
}

Tensor GPT2Model::forward(const Tensor& token_ids) {
    std::vector<size_t> ts = token_ids.shape();
    size_t B = ts[0], S = ts[1];

    Tensor pos_ids = Tensor::arange(0, (double)S, 1, DType::Int32);
    Tensor x = drop(wte(token_ids) + wpe(pos_ids));

    Tensor mask = make_causal_mask(S);
    for (auto& blk : blocks) x = (*blk)(x, mask);
    x = ln_f(x);

    Tensor flat = x.contiguous().reshape({B * S, (size_t)config.n_embd});
    Tensor logits2d = matmul(flat, wte.weight.permute({1, 0}));
    return logits2d.reshape({B, S, (size_t)config.vocab_size});
}

std::vector<int> GPT2Model::generate(const std::vector<int>& prompt_ids, int max_new_tokens, double temperature) {
    eval();
    std::vector<int> ids = prompt_ids;
    static std::mt19937 gen(42);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    for (int step = 0; step < max_new_tokens; ++step) {
        std::vector<int> window = ids;
        if ((int)window.size() > config.n_positions) {
            window = std::vector<int>(window.end() - config.n_positions, window.end());
        }
        size_t S = window.size();

        std::vector<double> ids_d(window.begin(), window.end());
        Tensor token_ids = Tensor::from_vector(ids_d, {1, S}, DType::Int32);
        Tensor logits = forward(token_ids);

        size_t V = (size_t)config.vocab_size;
        size_t last_off = (S - 1) * V;

        int next_id;
        if (temperature <= 1e-6) {
            int best = 0;
            double best_val = -1e300;
            for (size_t v = 0; v < V; ++v) {
                double val = logits.read_scalar(last_off + v);
                if (val > best_val) { best_val = val; best = (int)v; }
            }
            next_id = best;
        } else {
            double max_val = -1e300;
            for (size_t v = 0; v < V; ++v)
                max_val = std::max(max_val, logits.read_scalar(last_off + v) / temperature);

            std::vector<double> probs(V);
            double total = 0.0;
            for (size_t v = 0; v < V; ++v) {
                probs[v] = std::exp(logits.read_scalar(last_off + v) / temperature - max_val);
                total += probs[v];
            }
            double r = uni(gen) * total;
            double cum = 0.0;
            next_id = (int)V - 1;
            for (size_t v = 0; v < V; ++v) {
                cum += probs[v];
                if (cum >= r) { next_id = (int)v; break; }
            }
        }
        ids.push_back(next_id);
    }
    return ids;
}
