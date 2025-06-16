#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <set>

std::string preprocess_string(const std::string& input);
std::set<std::string> extract_3grams(const std::string& input);
void build_3gram_index(const std::vector<std::string>& strings,
                       const std::vector<int>& cluster_ids,
                       std::unordered_map<std::string, std::set<int>>& gram_index);