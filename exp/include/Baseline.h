// include/Baseline.h
// Pre-filtering and post-filtering baseline strategies for regex-filtered ANN.
//
// Pre-filtering  : apply regex first → run ANN only on matching vectors.
//                  Efficient when the filter is highly selective; degrades
//                  when many items match (large candidate pool).
//
// Post-filtering : run global ANN to get K' candidates → apply regex.
//                  Efficient when the filter is non-selective; risks recall
//                  loss when selectivity is high and K' is too small.
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
// 2. Run exact kNN search among matching vectors only.
BaselineResult run_prefilter(
    const std::string& regex,
    const std::vector<float>& query_vec,
    const std::vector<std::vector<float>>& all_vectors,
    const std::vector<std::string>& all_strings,
    int K);

// ── Post-filtering baseline ───────────────────────────────────────────────────
// 1. Run exact kNN search globally, retrieving top-(oversample_factor * K).
// 2. Apply regex filter on that candidate set.
// oversample_factor controls the recall vs latency trade-off.
BaselineResult run_postfilter(
    const std::string& regex,
    const std::vector<float>& query_vec,
    const std::vector<std::vector<float>>& all_vectors,
    const std::vector<std::string>& all_strings,
    int K,
    int oversample_factor = 10);

// ── Full-scan exact search (ground-truth) ─────────────────────────────────────
// Scans every element; returns exact top-K results that match the regex.
BaselineResult run_fullscan(
    const std::string& regex,
    const std::vector<float>& query_vec,
    const std::vector<std::vector<float>>& all_vectors,
    const std::vector<std::string>& all_strings,
    int K);
