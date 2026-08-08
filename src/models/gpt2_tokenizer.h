#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class GPT2Tokenizer {
public:
    static GPT2Tokenizer from_files(const std::string& vocab_json_path, const std::string& merges_txt_path);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> bpe_ranks_;
    std::string byte_encoder_[256];
    std::unordered_map<uint32_t, uint8_t> byte_decoder_;
};
