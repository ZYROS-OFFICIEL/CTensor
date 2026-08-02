#include "gpt2.h"
#include <cstring>
#include <unordered_map>

namespace gpt2_io {
namespace {

Tensor transpose2d(const Tensor& t) {
    return t.permute({1, 0}).contiguous();
}

Tensor slice_dim0(const Tensor& t, size_t start, size_t count) {
    Tensor tc = t.contiguous();
    auto shape = tc.shape();
    size_t elems_per_row = 1;
    for (size_t i = 1; i < shape.size(); ++i) elems_per_row *= shape[i];

    std::vector<size_t> out_shape = shape;
    out_shape[0] = count;
    Tensor out = Tensor::empty(out_shape, tc._dtype());

    size_t esz = tc.dtype_bytes();
    const char* src = static_cast<const char*>(tc.impl->data->data.get())
                       + (tc.impl->offset + start * elems_per_row) * esz;
    std::memcpy(out.impl->data->data.get(), src, count * elems_per_row * esz);
    return out;
}

void copy_into(Tensor& dst, const Tensor& src, const std::string& debug_name) {
    if (dst.shape() != src.shape())
        throw std::runtime_error("gpt2_io: shape mismatch for '" + debug_name + "'");
    Tensor converted = (src._dtype() == dst._dtype()) ? src : src.astype(dst._dtype());
    size_t nbytes = converted.numel() * converted.dtype_bytes();
    char* d = static_cast<char*>(dst.impl->data->data.get()) + dst.impl->offset * dst.dtype_bytes();
    const char* s = static_cast<const char*>(converted.impl->data->data.get())
                     + converted.impl->offset * converted.dtype_bytes();
    std::memcpy(d, s, nbytes);
}

} // namespace

safetensors::LoadReport load_huggingface(const std::string& path, GPT2Model& model, bool strict) {
    auto raw = safetensors::load(path);
    safetensors::LoadReport report;

    std::unordered_map<std::string, bool> consumed;
    consumed.reserve(raw.size());
    for (auto& [k, v] : raw) consumed[k] = false;

    auto take = [&](const std::string& key) -> Tensor& {
        auto it = raw.find(key);
        if (it == raw.end())
            throw std::runtime_error("gpt2_io: missing key '" + key + "' in checkpoint '" + path + "'");
        consumed[key] = true;
        return it->second;
    };
    auto skip = [&](const std::string& key) {
        auto it = raw.find(key);
        if (it != raw.end()) consumed[key] = true;
    };

    copy_into(model.wte.weight, take("wte.weight"), "wte.weight");
    copy_into(model.wpe.weight, take("wpe.weight"), "wpe.weight");

    size_t n_embd = (size_t)model.config.n_embd;

    for (size_t i = 0; i < model.blocks.size(); ++i) {
        auto& blk = *model.blocks[i];
        std::string p = "h." + std::to_string(i) + ".";

        copy_into(blk.ln_1.weight, take(p + "ln_1.weight"), p + "ln_1.weight");
        copy_into(blk.ln_1.bias,   take(p + "ln_1.bias"),   p + "ln_1.bias");
        copy_into(blk.ln_2.weight, take(p + "ln_2.weight"), p + "ln_2.weight");
        copy_into(blk.ln_2.bias,   take(p + "ln_2.bias"),   p + "ln_2.bias");

        Tensor c_attn_w = transpose2d(take(p + "attn.c_attn.weight"));
        Tensor c_attn_b = take(p + "attn.c_attn.bias");
        copy_into(blk.attn.q_proj.weight, slice_dim0(c_attn_w, 0 * n_embd, n_embd), p + "attn.q_proj.weight");
        copy_into(blk.attn.k_proj.weight, slice_dim0(c_attn_w, 1 * n_embd, n_embd), p + "attn.k_proj.weight");
        copy_into(blk.attn.v_proj.weight, slice_dim0(c_attn_w, 2 * n_embd, n_embd), p + "attn.v_proj.weight");
        copy_into(blk.attn.q_proj.bias, slice_dim0(c_attn_b, 0 * n_embd, n_embd), p + "attn.q_proj.bias");
        copy_into(blk.attn.k_proj.bias, slice_dim0(c_attn_b, 1 * n_embd, n_embd), p + "attn.k_proj.bias");
        copy_into(blk.attn.v_proj.bias, slice_dim0(c_attn_b, 2 * n_embd, n_embd), p + "attn.v_proj.bias");

        copy_into(blk.attn.out_proj.weight, transpose2d(take(p + "attn.c_proj.weight")), p + "attn.out_proj.weight");
        copy_into(blk.attn.out_proj.bias, take(p + "attn.c_proj.bias"), p + "attn.out_proj.bias");

        copy_into(blk.mlp.c_fc.weight, transpose2d(take(p + "mlp.c_fc.weight")), p + "mlp.c_fc.weight");
        copy_into(blk.mlp.c_fc.bias,   take(p + "mlp.c_fc.bias"),   p + "mlp.c_fc.bias");
        copy_into(blk.mlp.c_proj.weight, transpose2d(take(p + "mlp.c_proj.weight")), p + "mlp.c_proj.weight");
        copy_into(blk.mlp.c_proj.bias,   take(p + "mlp.c_proj.bias"),   p + "mlp.c_proj.bias");

        skip(p + "attn.bias");
        skip(p + "attn.masked_bias");
    }

    copy_into(model.ln_f.weight, take("ln_f.weight"), "ln_f.weight");
    copy_into(model.ln_f.bias,   take("ln_f.bias"),   "ln_f.bias");

    skip("lm_head.weight");

    for (auto& [name, was_consumed] : consumed) {
        if (!was_consumed) report.unexpected.push_back(name);
    }

    if (strict && !report.unexpected.empty()) {
        std::string msg = "gpt2_io: unexpected checkpoint keys:";
        for (auto& n : report.unexpected) msg += " " + n;
        throw std::runtime_error(msg);
    }

    return report;
}

} 
