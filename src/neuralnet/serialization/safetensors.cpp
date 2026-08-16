#include "neuralnet/serialization/safetensors.h"
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace safetensors {
namespace {

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object } type = Type::Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    const JsonValue* find(const std::string& key) const {
        for (auto& kv : obj) if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& s) : s_(s) {}

    JsonValue parse() {
        skip_ws();
        JsonValue v = parse_value();
        skip_ws();
        return v;
    }

private:
    const std::string& s_;
    size_t i_ = 0;

    [[noreturn]] void fail(const std::string& msg) {
        throw std::runtime_error("safetensors: JSON parse error at byte " + std::to_string(i_) + ": " + msg);
    }

    char peek() { if (i_ >= s_.size()) fail("unexpected end of input"); return s_[i_]; }
    char advance() { if (i_ >= s_.size()) fail("unexpected end of input"); return s_[i_++]; }
    void expect(char c) { if (advance() != c) fail(std::string("expected '") + c + "'"); }

    void skip_ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) ++i_;
    }

    JsonValue parse_value() {
        skip_ws();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') { JsonValue v; v.type = JsonValue::Type::String; v.str = parse_string(); return v; }
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        fail("unexpected character");
    }

    JsonValue parse_object() {
        JsonValue v; v.type = JsonValue::Type::Object;
        expect('{');
        skip_ws();
        if (peek() == '}') { ++i_; return v; }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            v.obj.emplace_back(std::move(key), parse_value());
            skip_ws();
            char c = advance();
            if (c == ',') continue;
            if (c == '}') break;
            fail("expected ',' or '}'");
        }
        return v;
    }

    JsonValue parse_array() {
        JsonValue v; v.type = JsonValue::Type::Array;
        expect('[');
        skip_ws();
        if (peek() == ']') { ++i_; return v; }
        while (true) {
            v.arr.push_back(parse_value());
            skip_ws();
            char c = advance();
            if (c == ',') continue;
            if (c == ']') break;
            fail("expected ',' or ']'");
        }
        return v;
    }

    unsigned parse_hex4() {
        if (i_ + 4 > s_.size()) fail("truncated \\u escape");
        unsigned v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s_[i_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
            else fail("invalid hex digit");
        }
        return v;
    }

    static void append_utf8(std::string& out, unsigned cp) {
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
    }

    std::string parse_string() {
        skip_ws();
        expect('"');
        std::string out;
        while (true) {
            char c = advance();
            if (c == '"') break;
            if (c == '\\') {
                char e = advance();
                switch (e) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'b':  out += '\b'; break;
                    case 'f':  out += '\f'; break;
                    case 'n':  out += '\n'; break;
                    case 'r':  out += '\r'; break;
                    case 't':  out += '\t'; break;
                    case 'u': {
                        unsigned cp = parse_hex4();
                        if (cp >= 0xD800 && cp <= 0xDBFF && i_ + 1 < s_.size() &&
                            s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                            i_ += 2;
                            unsigned lo = parse_hex4();
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default: fail("invalid escape sequence");
                }
            } else {
                out += c;
            }
        }
        return out;
    }

    JsonValue parse_bool() {
        JsonValue v; v.type = JsonValue::Type::Bool;
        if (s_.compare(i_, 4, "true") == 0) { v.b = true; i_ += 4; }
        else if (s_.compare(i_, 5, "false") == 0) { v.b = false; i_ += 5; }
        else fail("invalid literal");
        return v;
    }

    JsonValue parse_null() {
        if (s_.compare(i_, 4, "null") != 0) fail("invalid literal");
        i_ += 4;
        return JsonValue{};
    }

    JsonValue parse_number() {
        size_t start = i_;
        if (peek() == '-') ++i_;
        while (i_ < s_.size() && isdigit((unsigned char)s_[i_])) ++i_;
        if (i_ < s_.size() && s_[i_] == '.') {
            ++i_;
            while (i_ < s_.size() && isdigit((unsigned char)s_[i_])) ++i_;
        }
        if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            ++i_;
            if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
            while (i_ < s_.size() && isdigit((unsigned char)s_[i_])) ++i_;
        }
        JsonValue v; v.type = JsonValue::Type::Number;
        v.num = std::stod(s_.substr(start, i_ - start));
        return v;
    }
};

uint64_t read_u64_le(const char* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | (uint8_t)p[i];
    return v;
}

void write_u64_le(std::ostream& os, uint64_t v) {
    char buf[8];
    for (int i = 0; i < 8; ++i) { buf[i] = (char)(v & 0xFF); v >>= 8; }
    os.write(buf, 8);
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

DType from_safetensors_dtype(const std::string& s) {
    if (s == "F32")  return DType::Float32;
    if (s == "F64")  return DType::Double64;
    if (s == "F16")  return DType::Float16;
    if (s == "BF16") return DType::BFloat16;
    if (s == "I64")  return DType::Int64;
    if (s == "I32")  return DType::Int32;
    if (s == "I16")  return DType::Int16;
    if (s == "I8")   return DType::Int8;
    if (s == "U64")  return DType::UInt64;
    if (s == "U32")  return DType::UInt32;
    if (s == "U16")  return DType::UInt16;
    if (s == "U8")   return DType::UInt8;
    if (s == "BOOL") return DType::Bool;
    throw std::runtime_error("safetensors: unsupported dtype '" + s + "'");
}

const char* to_safetensors_dtype(DType dt) {
    switch (dt) {
        case DType::Float32:  return "F32";
        case DType::Double64: return "F64";
        case DType::Float16:  return "F16";
        case DType::BFloat16: return "BF16";
        case DType::Int64:    return "I64";
        case DType::Int32:    return "I32";
        case DType::Int16:    return "I16";
        case DType::Int8:     return "I8";
        case DType::UInt64:   return "U64";
        case DType::UInt32:   return "U32";
        case DType::UInt16:   return "U16";
        case DType::UInt8:    return "U8";
        case DType::Bool:     return "BOOL";
        default: throw std::runtime_error("safetensors: dtype has no safetensors equivalent");
    }
}

} // namespace

std::unordered_map<std::string, Tensor> load(
    const std::string& path,
    std::unordered_map<std::string, std::string>* metadata_out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("safetensors: cannot open file '" + path + "'");

    f.seekg(0, std::ios::end);
    std::streamoff file_size_off = f.tellg();
    if (file_size_off < 8) throw std::runtime_error("safetensors: file too small: " + path);
    size_t file_size = (size_t)file_size_off;
    f.seekg(0, std::ios::beg);

    char header_len_buf[8];
    f.read(header_len_buf, 8);
    uint64_t header_len = read_u64_le(header_len_buf);
    if (header_len > file_size - 8)
        throw std::runtime_error("safetensors: header length exceeds file size: " + path);

    std::string header_json(header_len, '\0');
    f.read(header_json.data(), (std::streamsize)header_len);

    JsonValue root = JsonParser(header_json).parse();
    if (root.type != JsonValue::Type::Object)
        throw std::runtime_error("safetensors: header is not a JSON object: " + path);

    size_t blob_start = 8 + (size_t)header_len;
    size_t blob_size = file_size - blob_start;

    std::string blob(blob_size, '\0');
    if (blob_size > 0) f.read(blob.data(), (std::streamsize)blob_size);
    if (!f && blob_size > 0) throw std::runtime_error("safetensors: truncated data section: " + path);

    std::unordered_map<std::string, Tensor> result;
    result.reserve(root.obj.size());

    for (auto& [name, entry] : root.obj) {
        if (name == "__metadata__") {
            if (metadata_out) {
                for (auto& [k, v] : entry.obj) {
                    if (v.type == JsonValue::Type::String) (*metadata_out)[k] = v.str;
                }
            }
            continue;
        }
        if (entry.type != JsonValue::Type::Object)
            throw std::runtime_error("safetensors: tensor entry '" + name + "' is not an object");

        const JsonValue* dtype_v = entry.find("dtype");
        const JsonValue* shape_v = entry.find("shape");
        const JsonValue* offsets_v = entry.find("data_offsets");
        if (!dtype_v || dtype_v->type != JsonValue::Type::String)
            throw std::runtime_error("safetensors: tensor '" + name + "' missing 'dtype'");
        if (!shape_v || shape_v->type != JsonValue::Type::Array)
            throw std::runtime_error("safetensors: tensor '" + name + "' missing 'shape'");
        if (!offsets_v || offsets_v->type != JsonValue::Type::Array || offsets_v->arr.size() != 2)
            throw std::runtime_error("safetensors: tensor '" + name + "' missing 'data_offsets'");

        std::vector<size_t> shape;
        shape.reserve(shape_v->arr.size());
        for (auto& d : shape_v->arr) {
            if (d.type != JsonValue::Type::Number || d.num < 0)
                throw std::runtime_error("safetensors: tensor '" + name + "' has invalid shape entry");
            shape.push_back((size_t)d.num);
        }

        size_t begin = (size_t)offsets_v->arr[0].num;
        size_t end   = (size_t)offsets_v->arr[1].num;
        if (end < begin || end > blob_size)
            throw std::runtime_error("safetensors: tensor '" + name + "' has out-of-range data_offsets");

        DType out_dtype = from_safetensors_dtype(dtype_v->str);

        size_t numel = 1;
        for (size_t s : shape) numel *= s;
        size_t expected_bytes = numel * dtype_size(out_dtype);
        if (end - begin != expected_bytes)
            throw std::runtime_error("safetensors: tensor '" + name + "' byte length " +
                std::to_string(end - begin) + " does not match shape*dtype size " +
                std::to_string(expected_bytes));

        Tensor t = Tensor::empty(shape, out_dtype);
        if (numel > 0) std::memcpy(t.impl->data->data.get(), blob.data() + begin, expected_bytes);

        result.emplace(name, std::move(t));
    }

    return result;
}

void save(
    const std::string& path,
    const std::vector<std::pair<std::string, Tensor>>& tensors,
    const std::unordered_map<std::string, std::string>& metadata)
{
    std::vector<Tensor> contiguous_tensors;
    contiguous_tensors.reserve(tensors.size());
    std::vector<size_t> byte_lens;
    byte_lens.reserve(tensors.size());

    std::ostringstream json;
    json << '{';
    bool first = true;

    if (!metadata.empty()) {
        json << "\"__metadata__\":{";
        bool mfirst = true;
        for (auto& [k, v] : metadata) {
            if (!mfirst) json << ',';
            mfirst = false;
            json << '"' << json_escape(k) << "\":\"" << json_escape(v) << '"';
        }
        json << '}';
        first = false;
    }

    size_t cursor = 0;
    for (auto& [name, t] : tensors) {
        if (name == "__metadata__")
            throw std::runtime_error("safetensors: '__metadata__' is reserved and cannot be used as a tensor name");
        if (!t.impl)
            throw std::runtime_error("safetensors: tensor '" + name + "' is empty");

        Tensor cpu_t = t.device().is_cuda() ? t.to(Device(DeviceType::CPU)) : t;
        Tensor c = cpu_t.contiguous();
        size_t nbytes = c.numel() * c.dtype_bytes();

        if (!first) json << ',';
        first = false;
        json << '"' << json_escape(name) << "\":{\"dtype\":\"" << to_safetensors_dtype(c._dtype())
             << "\",\"shape\":[";
        auto shp = c.shape();
        for (size_t i = 0; i < shp.size(); ++i) {
            if (i) json << ',';
            json << shp[i];
        }
        json << "],\"data_offsets\":[" << cursor << ',' << (cursor + nbytes) << "]}";

        contiguous_tensors.push_back(std::move(c));
        byte_lens.push_back(nbytes);
        cursor += nbytes;
    }
    json << '}';

    std::string header = json.str();
    size_t pad = (8 - (8 + header.size()) % 8) % 8;
    header.append(pad, ' ');

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("safetensors: cannot open file for writing '" + path + "'");

    write_u64_le(f, header.size());
    f.write(header.data(), (std::streamsize)header.size());

    for (size_t k = 0; k < contiguous_tensors.size(); ++k) {
        Tensor& ct = contiguous_tensors[k];
        const char* raw = static_cast<const char*>(ct.impl->data->data.get())
                           + ct.impl->offset * ct.dtype_bytes();
        f.write(raw, (std::streamsize)byte_lens[k]);
    }

    if (!f) throw std::runtime_error("safetensors: write failure for '" + path + "'");
}

void save(
    const std::string& path,
    const Module::NamedParams& params,
    const std::unordered_map<std::string, std::string>& metadata)
{
    std::vector<std::pair<std::string, Tensor>> tensors;
    tensors.reserve(params.size());
    for (auto& [name, ptr] : params) {
        if (!ptr || !ptr->impl) continue;
        tensors.emplace_back(name, *ptr);
    }
    save(path, tensors, metadata);
}

LoadReport load_into(
    const std::string& path,
    const Module::NamedParams& params,
    bool strict)
{
    auto loaded = load(path);
    LoadReport report;

    std::unordered_map<std::string, bool> consumed;
    consumed.reserve(loaded.size());
    for (auto& [name, t] : loaded) consumed[name] = false;

    for (auto& [name, ptr] : params) {
        if (!ptr || !ptr->impl) continue;

        auto it = loaded.find(name);
        if (it == loaded.end()) {
            report.missing.push_back(name);
            continue;
        }
        consumed[name] = true;

        Tensor& src = it->second;
        if (src.shape() != ptr->shape()) {
            report.shape_mismatch.push_back(name);
            continue;
        }

        Tensor converted = (src._dtype() == ptr->_dtype()) ? src : src.astype(ptr->_dtype());
        size_t nbytes = converted.numel() * converted.dtype_bytes();
        char* dst = static_cast<char*>(ptr->impl->data->data.get())
                    + ptr->impl->offset * ptr->dtype_bytes();
        const char* src_ptr = static_cast<const char*>(converted.impl->data->data.get())
                              + converted.impl->offset * converted.dtype_bytes();
        std::memcpy(dst, src_ptr, nbytes);
    }

    for (auto& [name, was_consumed] : consumed) {
        if (!was_consumed) report.unexpected.push_back(name);
    }

    if (strict && !report.ok()) {
        std::string msg = "safetensors: load_into failed strict check for '" + path + "':";
        auto append_list = [&](const char* label, const std::vector<std::string>& v) {
            if (v.empty()) return;
            msg += std::string(" ") + label + "=[";
            for (size_t i = 0; i < v.size(); ++i) msg += (i ? "," : "") + v[i];
            msg += "]";
        };
        append_list("missing", report.missing);
        append_list("unexpected", report.unexpected);
        append_list("shape_mismatch", report.shape_mismatch);
        throw std::runtime_error(msg);
    }

    return report;
}

} // namespace safetensors
