// include/Baseline.h
// Pre-filtering and post-filtering baseline strategies for regex-filtered ANN.
//
// Pre-filtering  : apply regex first → run kNN only on matching vectors.
//                  With sample_ratio < 1.0, only a random fraction of
//                  matching vectors are searched, reducing recall and latency.
//                  sample_ratio = 1.0 → exact (100% recall), original behaviour.
//
// Post-filtering : run global ANN to get K' candidates → apply regex.
//                  Uses adaptive expansion: if fewer than K results match
//                  the regex in the first K' candidates, the search window
//                  is doubled until K matches are found or the full dataset
//                  is exhausted.  max_expansion_factor caps the expansion.
//
// Full-scan      : brute-force exact search (ground-truth generator).
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <set>

struct BaselineResult {
    std::vector<int> top_ids;
    double query_time_ms = 0.0;
};

// ── Pre-filtering baseline ────────────────────────────────────────────────────
// 1. Scan all strings and collect matching indices.
// 2. Run kNN search among a sampled subset of matching vectors.
//
// sample_ratio  – fraction of matching vectors to search (0 < ratio ≤ 1.0).
//                 1.0 (default) = search all matches → 100% recall (original).
//                 < 1.0 = randomly subsample → lower recall, faster query.
BaselineResult run_prefilter(
    const std::string& regex,
    const std::vector<float>& query_vec,
    const std::vector<std::vector<float>>& all_vectors,
    const std::vector<std::string>& all_strings,
    int K,
    float sample_ratio = 1.0f);

// ── Post-filtering baseline ───────────────────────────────────────────────────
// 1. Run exact kNN globally, retrieving top-(oversample_factor * K).
// 2. Apply regex filter on that candidate set.
// 3. If fewer than K matches found, adaptively double the search window
//    (up to max_expansion_factor × original window) until K are found.
//
// oversample_factor      – initial multiplier for K (default 10).
// max_expansion_factor   – maximum total expansion relative to K (default 0
//                          means unlimited, i.e. scan full dataset if needed).
BaselineResult run_postfilter(
    const std::string& regex,
    const std::vector<float>& query_vec,
    const std::vector<std::vector<float>>& all_vectors,
    const std::vector<std::string>& all_strings,
    int K,
    int oversample_factor = 10,
    int max_expansion_factor = 0);

// ── Full-scan exact search (ground-truth) ─────────────────────────────────────
// Scans every element; returns exact top-K results that match the regex.
BaselineResult run_fullscan(
    const std::string& regex,
    const std::vector<float>& query_vec,
    const std::vector<std::vector<float>>& all_vectors,
    const std::vector<std::string>& all_strings,
    int K);
