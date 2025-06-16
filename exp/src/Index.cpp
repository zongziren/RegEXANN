// src/Index.cpp
#include "Index.h"
#include <algorithm>
#include <cctype>

std::string preprocess_string(const std::string& input) {
    std::string result;
    for (char c : input) {
        if (std::isalpha(c) || std::isspace(c)) result += std::tolower(c);
    }
    return result;
}

std::set<std::string> extract_3grams(const std::string& input) {
    std::string processed = preprocess_string(input);
    std::set<std::string> grams;
    if (processed.length() < 3) return grams;
    for (size_t i = 0; i + 2 < processed.length(); ++i)
        grams.insert(processed.substr(i, 3));
    return grams;
}

void build_3gram_index(const std::vector<std::string>& strings,
                       const std::vector<int>& cluster_ids,
                       std::unordered_map<std::string, std::set<int>>& gram_index) {
    for (size_t i = 0; i < strings.size(); ++i) {
        auto grams = extract_3grams(strings[i]);
        for (const auto& gram : grams)
            gram_index[gram].insert(cluster_ids[i]);
    }
}