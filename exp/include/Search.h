// include/Search.h
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include "PQ.h"

struct SearchResult {
    std::vector<int> top_ids;
    double setop_time_ms  = 0.0; // trigram index look-up + set operations
    double query_time_ms  = 0.0; // cluster ranking + PQ scan + regex verify
};

// Perform a regex-filtered ANN search using the RegExANN pipeline:
//   1. Trigram-based candidate cluster selection (set intersection/union)
//   2. Sort surviving clusters by centroid distance to query_vector
//   3. Within each cluster, rank vectors by PQ approximate distance
//   4. Verify regex on candidates in PQ-distance order; collect into ef-pool
//   5. Return top-K from ef-pool by exact Euclidean distance
//
// Parameters:
//   regex         – regular expression filter (std::regex syntax, icase)
//   query_vector  – query embedding vector
//   gram_index    – trigram → set<cluster_id> inverted index
//   centroids     – cluster centroid vectors
//   clusters      – clusters[cid] = list of vector IDs in that cluster
//   all_vectors   – full dataset vectors (for exact distance at the end)
//   all_strings   – full dataset strings  (for regex verification)
//   pq            – trained & encoded Product Quantization index
//   K             – number of results to return
//   ef            – candidate pool size (ef >= K); algorithm stops when ef
//                   regex-matching candidates are collected, then re-ranks
//                   by exact distance and returns top-K. Larger ef → higher
//                   recall at the cost of more regex checks. Default = K.
SearchResult perform_search(
    const std::string& regex,
    const std::vector<float>& query_vector,
    const std::unordered_map<std::string, std::set<int>>& gram_index,
    const std::vector<std::vector<float>>& centroids,
    const std::vector<std::vector<int>>& clusters,
    const std::vector<std::vector<float>>& all_vectors,
    const std::vector<std::string>& all_strings,
    const PQIndex& pq,
    int K,
    int ef = 0);   // 0 → use K (original behaviour)
