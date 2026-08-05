#pragma once
#include <string>
#include <utility>
#include <vector>

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

JsonValue parse_json(const std::string& text);
