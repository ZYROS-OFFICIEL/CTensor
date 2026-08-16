#include "core/json.h"
#include <cctype>
#include <stdexcept>

namespace {

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
        throw std::runtime_error("json: parse error at byte " + std::to_string(i_) + ": " + msg);
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

} // namespace

JsonValue parse_json(const std::string& text) {
    return JsonParser(text).parse();
}
