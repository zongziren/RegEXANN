#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <set>

struct SearchResult {
    std::vector<int> top_ids;
    double setop_time_ms = 0.0;   // NEW
    double query_time_ms = 0.0;   // NEW

};

SearchResult perform_search(const std::string& regex,
                            const std::vector<float>& query_vector,
                            const std::unordered_map<std::string, std::set<int>>& gram_index,
                            const std::vector<std::vector<float>>& centroids,
                            const std::vector<std::vector<int>>& clusters,
                            const std::vector<std::vector<float>>& all_vectors,
                            const std::vector<std::string>& all_strings,
                            int K);