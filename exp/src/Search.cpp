// src/Search.cpp
#include "Search.h"
#include "Index.h"
#include <regex>
#include <cmath>
#include <queue>
#include <iostream>
#include <fstream>

float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(sum);
}

void dump_query_cluster_debug(const std::string& regex_str,
                              const std::unordered_map<std::string, std::set<int>>& gram_index,
                              const std::string& filename) {
    std::ofstream fout(filename);
    auto grams = extract_3grams(regex_str);

    fout << "[DEBUG] Extracted 3-grams from regex: ";
    for (const auto& g : grams) fout << g << " ";
    fout << "\n";

    std::set<int> candidate_clusters;
    for (const auto& gram : grams) {
        fout << "[DEBUG] gram: " << gram << " → ";
        if (gram_index.count(gram)) {
            for (int cid : gram_index.at(gram)) {
                fout << cid << " ";
                candidate_clusters.insert(cid);
            }
        } else {
            fout << "not found";
        }
        fout << "\n";
    }
    fout << "[DEBUG] Total candidate clusters = " << candidate_clusters.size() << ": ";
    for (int cid : candidate_clusters) fout << cid << " ";
    fout << "\n";
    fout.close();
}

SearchResult perform_search(const std::string& regex_str,
                            const std::vector<float>& query_vector,
                            const std::unordered_map<std::string, std::set<int>>& gram_index,
                            const std::vector<std::vector<float>>& centroids,
                            const std::vector<std::vector<int>>& clusters,
                            const std::vector<std::vector<float>>& all_vectors,
                            const std::vector<std::string>& all_strings,
                            int K) {
    std::set<int> candidate_clusters;
    auto grams = extract_3grams(regex_str);

    //dump_query_cluster_debug(regex_str, gram_index, "./debug/query_debug.txt");

    for (const auto& gram : grams) {
        if (gram_index.count(gram))
            candidate_clusters.insert(gram_index.at(gram).begin(), gram_index.at(gram).end());
    }

    std::vector<std::pair<float, int>> cluster_dists;
    for (int cid : candidate_clusters)
        cluster_dists.emplace_back(euclidean_distance(query_vector, centroids[cid]), cid);
    std::sort(cluster_dists.begin(), cluster_dists.end());

    std::priority_queue<std::pair<float, int>> pq;
    std::regex pattern(regex_str);

    for (const auto& [_, cid] : cluster_dists) {
        for (int vid : clusters[cid]) {
            if (!std::regex_search(all_strings[vid], pattern)) continue;
            float dist = euclidean_distance(query_vector, all_vectors[vid]);
            pq.emplace(dist, vid);
            if ((int)pq.size() > K) pq.pop();
        }
        if ((int)pq.size() >= K) break;
    }

    std::vector<int> result;
    while (!pq.empty()) {
        result.push_back(pq.top().second);
        pq.pop();
    }
    std::reverse(result.begin(), result.end());
    return {result};
}
