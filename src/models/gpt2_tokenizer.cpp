#include "gpt2_tokenizer.h"
#include "core/json.h"
#include <algorithm>
#include <climits>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

struct Codepoint { uint32_t cp; size_t byte_start; size_t byte_len; };

std::string utf8_encode(uint32_t cp) {
    std::string out;
    if (cp <= 0x7F) {
        out += (char)cp;
    } else if (cp <= 0x7FF) {
        out += (char)(0xC0 | (cp >> 6));
        out += (char)(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        out += (char)(0xE0 | (cp >> 12));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    } else {
        out += (char)(0xF0 | (cp >> 18));
        out += (char)(0x80 | ((cp >> 12) & 0x3F));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    }
    return out;
}

std::vector<Codepoint> utf8_decode(const std::string& s) {
    std::vector<Codepoint> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c0 = (unsigned char)s[i];
        uint32_t cp;
        size_t len;
        if (c0 < 0x80)              { cp = c0;          len = 1; }
        else if ((c0 & 0xE0) == 0xC0) { cp = c0 & 0x1F;  len = 2; }
        else if ((c0 & 0xF0) == 0xE0) { cp = c0 & 0x0F;  len = 3; }
        else if ((c0 & 0xF8) == 0xF0) { cp = c0 & 0x07;  len = 4; }
        else                          { cp = c0;          len = 1; }

        if (i + len > s.size()) {
            cp = c0; len = 1;
        } else {
            bool valid = true;
            uint32_t acc = cp;
            for (size_t k = 1; k < len; ++k) {
                unsigned char ck = (unsigned char)s[i + k];
                if ((ck & 0xC0) != 0x80) { valid = false; break; }
                acc = (acc << 6) | (ck & 0x3F);
            }
            if (!valid) { cp = c0; len = 1; }
            else        { cp = acc; }
        }

        out.push_back({cp, i, len});
        i += len;
    }
    return out;
}

bool is_letter_cp(uint32_t cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    return cp >= 0x80;
}
bool is_digit_cp(uint32_t cp) { return cp >= '0' && cp <= '9'; }
bool is_space_cp(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v';
}
bool is_other_cp(uint32_t cp) { return !is_space_cp(cp) && !is_letter_cp(cp) && !is_digit_cp(cp); }

std::vector<std::string> pretokenize(const std::string& text) {
    std::vector<Codepoint> cps = utf8_decode(text);
    size_t n = cps.size();
    std::vector<std::string> tokens;

    auto substr_for = [&](size_t start_idx, size_t end_idx) {
        size_t byte_start = cps[start_idx].byte_start;
        size_t byte_end = (end_idx < n) ? cps[end_idx].byte_start : text.size();
        return text.substr(byte_start, byte_end - byte_start);
    };

    static const std::vector<std::string> contractions = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};

    size_t i = 0;
    while (i < n) {
        if (cps[i].cp == '\'') {
            bool matched = false;
            for (auto& c : contractions) {
                size_t byte_start = cps[i].byte_start;
                if (byte_start + c.size() <= text.size() && text.compare(byte_start, c.size(), c) == 0) {
                    size_t j = i, consumed = 0;
                    while (j < n && consumed < c.size()) { consumed += cps[j].byte_len; ++j; }
                    if (consumed == c.size()) {
                        tokens.push_back(substr_for(i, j));
                        i = j;
                        matched = true;
                        break;
                    }
                }
            }
            if (matched) continue;
        }

        if (cps[i].cp == ' ' && i + 1 < n && is_letter_cp(cps[i + 1].cp)) {
            size_t j = i + 1;
            while (j < n && is_letter_cp(cps[j].cp)) ++j;
            tokens.push_back(substr_for(i, j));
            i = j;
            continue;
        }
        if (is_letter_cp(cps[i].cp)) {
            size_t j = i;
            while (j < n && is_letter_cp(cps[j].cp)) ++j;
            tokens.push_back(substr_for(i, j));
            i = j;
            continue;
        }

        if (cps[i].cp == ' ' && i + 1 < n && is_digit_cp(cps[i + 1].cp)) {
            size_t j = i + 1;
            while (j < n && is_digit_cp(cps[j].cp)) ++j;
            tokens.push_back(substr_for(i, j));
            i = j;
            continue;
        }
        if (is_digit_cp(cps[i].cp)) {
            size_t j = i;
            while (j < n && is_digit_cp(cps[j].cp)) ++j;
            tokens.push_back(substr_for(i, j));
            i = j;
            continue;
        }

        if (cps[i].cp == ' ' && i + 1 < n && is_other_cp(cps[i + 1].cp)) {
            size_t j = i + 1;
            while (j < n && is_other_cp(cps[j].cp)) ++j;
            tokens.push_back(substr_for(i, j));
            i = j;
            continue;
        }
        if (is_other_cp(cps[i].cp)) {
            size_t j = i;
            while (j < n && is_other_cp(cps[j].cp)) ++j;
            tokens.push_back(substr_for(i, j));
            i = j;
            continue;
        }

        if (is_space_cp(cps[i].cp)) {
            size_t j = i;
            while (j < n && is_space_cp(cps[j].cp)) ++j;
            size_t end = (j == n || j - i < 2) ? j : j - 1;
            tokens.push_back(substr_for(i, end));
            i = end;
            continue;
        }

        tokens.push_back(substr_for(i, i + 1));
        i += 1;
    }

    return tokens;
}

std::vector<std::string> bpe_merge(std::vector<std::string> word,
                                    const std::unordered_map<std::string, int>& bpe_ranks) {
    if (word.size() < 2) return word;
    while (true) {
        int best_rank = INT_MAX;
        std::string best_left, best_right;
        bool found = false;
        for (size_t k = 0; k + 1 < word.size(); ++k) {
            auto it = bpe_ranks.find(word[k] + " " + word[k + 1]);
            if (it != bpe_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_left = word[k];
                best_right = word[k + 1];
                found = true;
            }
        }
        if (!found) break;

        std::vector<std::string> new_word;
        new_word.reserve(word.size());
        size_t k = 0;
        while (k < word.size()) {
            if (k + 1 < word.size() && word[k] == best_left && word[k + 1] == best_right) {
                new_word.push_back(best_left + best_right);
                k += 2;
            } else {
                new_word.push_back(word[k]);
                k += 1;
            }
        }
        word = std::move(new_word);
        if (word.size() == 1) break;
    }
    return word;
}

void build_byte_tables(std::string byte_encoder[256], std::unordered_map<uint32_t, uint8_t>& byte_decoder) {
    std::vector<int> bs;
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);

    std::vector<int> cs = bs;
    bool in_bs[256] = {};
    for (int b : bs) in_bs[b] = true;

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (!in_bs[b]) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    for (size_t k = 0; k < bs.size(); ++k) {
        uint32_t cp = (uint32_t)cs[k];
        std::string s = utf8_encode(cp);
        byte_encoder[(uint8_t)bs[k]] = s;
        byte_decoder[cp] = (uint8_t)bs[k];
    }
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("gpt2_tokenizer: cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace

GPT2Tokenizer GPT2Tokenizer::from_files(const std::string& vocab_json_path, const std::string& merges_txt_path) {
    GPT2Tokenizer tok;
    build_byte_tables(tok.byte_encoder_, tok.byte_decoder_);

    JsonValue root = parse_json(read_file(vocab_json_path));
    if (root.type != JsonValue::Type::Object)
        throw std::runtime_error("gpt2_tokenizer: vocab.json is not a JSON object");

    int max_id = -1;
    for (auto& [k, v] : root.obj) {
        int id = (int)v.num;
        tok.token_to_id_[k] = id;
        max_id = std::max(max_id, id);
    }
    tok.id_to_token_.resize((size_t)max_id + 1);
    for (auto& [k, id] : tok.token_to_id_) tok.id_to_token_[(size_t)id] = k;

    std::ifstream mf(merges_txt_path);
    if (!mf) throw std::runtime_error("gpt2_tokenizer: cannot open '" + merges_txt_path + "'");
    std::string line;
    std::getline(mf, line);
    int rank = 0;
    while (std::getline(mf, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty()) continue;
        tok.bpe_ranks_[line] = rank++;
    }

    return tok;
}

std::vector<int> GPT2Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    for (auto& pretoken : pretokenize(text)) {
        std::vector<std::string> symbols;
        symbols.reserve(pretoken.size());
        for (unsigned char b : pretoken) symbols.push_back(byte_encoder_[b]);

        for (auto& sym : bpe_merge(std::move(symbols), bpe_ranks_)) {
            auto it = token_to_id_.find(sym);
            if (it == token_to_id_.end())
                throw std::runtime_error("gpt2_tokenizer: unknown token '" + sym + "'");
            ids.push_back(it->second);
        }
    }
    return ids;
}

std::string GPT2Tokenizer::decode(const std::vector<int>& ids) const {
    std::string mapped;
    for (int id : ids) {
        if (id < 0 || (size_t)id >= id_to_token_.size())
            throw std::runtime_error("gpt2_tokenizer: id out of range: " + std::to_string(id));
        mapped += id_to_token_[(size_t)id];
    }

    std::string out;
    for (auto& c : utf8_decode(mapped)) {
        auto it = byte_decoder_.find(c.cp);
        if (it == byte_decoder_.end())
            throw std::runtime_error("gpt2_tokenizer: invalid byte-mapped codepoint in decode");
        out += (char)it->second;
    }
    return out;
}
