// include/Search.h
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include "PQ.h"

struct SearchResult {
    std::vector<int> top_ids;
    double setop_time_ms  = 0.0; // trigram index look-up + set operations (legacy, == t1+t2)
    double query_time_ms  = 0.0; // cluster ranking + PQ scan + regex verify (legacy, == t3+t4 portion before rerank)

    // ── Fine-grained stage breakdown (added for bottleneck profiling) ──────
    // t1: regex string -> trigram predicate parsing (pure string/DNF work,
    //     no index lookups yet)
    // t2: trigram predicate -> candidate cluster ids, via gram_index lookups
    //     and set union/intersection, PLUS sorting candidate clusters by
    //     centroid distance (kept with t2 since both are "which clusters do
    //     we scan" bookkeeping, not regex matching)
    // t3: PQ distance computation + sort + early-exit bookkeeping while
    //     scanning candidate clusters (excludes regex_search calls)
    // t4: std::regex_search calls against candidate strings (regex
    //     verification), counted separately from t3
    // t5: final exact-distance rerank of the ef survivors -> top-K
    double t1_trigram_parse_ms   = 0.0;
    double t2_cluster_lookup_ms  = 0.0;
    double t3_pq_scan_ms         = 0.0;
    double t4_regex_verify_ms    = 0.0;
    double t5_rerank_ms          = 0.0;
    long   num_candidate_clusters = 0;
    long   num_regex_checks       = 0;
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
    int ef = 0,      // 0 → use K (original behaviour)
    int nprobe = 0); // 0 → scan all candidate clusters (original behaviour)

// Profiling variant of perform_search. Identical algorithm and results,
// but fills in SearchResult's t1..t5 fine-grained stage timings:
//   t1 — regex -> trigram literal parsing
//   t2 — gram_index lookups/set-ops (cluster candidate selection)
//        + sorting candidates by centroid distance
//   t3 — PQ distance computation/sort/early-exit bookkeeping per cluster
//   t4 — std::regex_search calls (regex verification of candidates)
//   t5 — final exact-distance rerank of ef survivors -> top-K
// Used only by the profiling driver; not on the hot path of normal runs.
SearchResult perform_search_profiled(
    const std::string& regex,
    const std::vector<float>& query_vector,
    const std::unordered_map<std::string, std::set<int>>& gram_index,
    const std::vector<std::vector<float>>& centroids,
    const std::vector<std::vector<int>>& clusters,
    const std::vector<std::vector<float>>& all_vectors,
    const std::vector<std::string>& all_strings,
    const PQIndex& pq,
    int K,
    int ef = 0,
    int nprobe = 0);
