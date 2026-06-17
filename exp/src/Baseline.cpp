// src/Baseline.cpp
#include "Baseline.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <queue>
#include <random>
#include <regex>
#include <stdexcept>

using Clock = std::chrono::high_resolution_clock;

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

static float l2_dist(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i) { float d = a[i]-b[i]; s += d*d; }
    return std::sqrt(s);
}

// Build a max-heap of (distance, id) keeping the K smallest distances.
using MaxHeap = std::priority_queue<std::pair<float,int>>;

static void heap_push(MaxHeap& h, float dist, int id, int K) {
    h.emplace(dist, id);
    if ((int)h.size() > K) h.pop();
}

static std::vector<int> heap_drain(MaxHeap& h) {
    std::vector<int> out;
    while (!h.empty()) { out.push_back(h.top().second); h.pop(); }
    std::reverse(out.begin(), out.end());
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pre-filtering
// ─────────────────────────────────────────────────────────────────────────────

BaselineResult run_prefilter(
        const std::string& regex_str,
        const std::vector<float>& query_vec,
        const std::vector<std::vector<float>>& all_vectors,
        const std::vector<std::string>& all_strings,
        int K,
        float sample_ratio) {
    auto t0 = Clock::now();

    std::regex pat;
    try { pat = std::regex(regex_str, std::regex::icase); }
    catch (...) { return {{}, 0.0}; }

    // Phase 1: regex filter → collect matching indices
    std::vector<int> matching;
    for (int i = 0; i < (int)all_strings.size(); ++i)
        if (std::regex_search(all_strings[i], pat))
            matching.push_back(i);

    // Phase 2: optionally subsample the matching set.
    // sample_ratio = 1.0 → search all (original 100% recall behaviour).
    // sample_ratio < 1.0 → randomly keep that fraction, reducing recall.
    if (sample_ratio < 1.0f && !matching.empty()) {
        // Deterministic shuffle keyed on query vec (reproducible per query).
        std::mt19937 rng(static_cast<uint32_t>(
            std::hash<float>{}(query_vec.empty() ? 0.f : query_vec[0]) ^
            (matching.size() * 2654435761u)));
        std::shuffle(matching.begin(), matching.end(), rng);
        int keep = std::max(K, static_cast<int>(matching.size() * sample_ratio));
        if (keep < (int)matching.size())
            matching.resize(keep);
    }

    // Phase 3: exact kNN on the (possibly subsampled) matching subset
    MaxHeap h;
    for (int id : matching)
        heap_push(h, l2_dist(query_vec, all_vectors[id]), id, K);

    auto t1 = Clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() / 1000.0;
    return {heap_drain(h), ms};
}

// ─────────────────────────────────────────────────────────────────────────────
// Post-filtering  (adaptive expansion)
// ─────────────────────────────────────────────────────────────────────────────
//
// Strategy:
//   1. Start with a candidate window of size K * oversample_factor.
//   2. Sort the whole dataset by distance lazily: we maintain all distances
//      and sort once, then walk from nearest outward.
//   3. Keep applying regex until we collect K matches or exhaust the dataset.
//   4. max_expansion_factor = 0 means no cap (can go up to N).
//
// This fixes the recall problem: when regex selectivity is high (few matches),
// the fixed-oversample approach misses them; adaptive expansion guarantees
// at least as many results as exist in the dataset.

BaselineResult run_postfilter(
        const std::string& regex_str,
        const std::vector<float>& query_vec,
        const std::vector<std::vector<float>>& all_vectors,
        const std::vector<std::string>& all_strings,
        int K,
        int oversample_factor,
        int max_expansion_factor) {
    auto t0 = Clock::now();

    std::regex pat;
    try { pat = std::regex(regex_str, std::regex::icase); }
    catch (...) { return {{}, 0.0}; }

    int N = (int)all_vectors.size();

    // Compute all distances once and sort ascending.
    // This is O(N log N) like before but now lets us walk incrementally.
    std::vector<std::pair<float,int>> sorted_all;
    sorted_all.reserve(N);
    for (int i = 0; i < N; ++i)
        sorted_all.emplace_back(l2_dist(query_vec, all_vectors[i]), i);
    std::sort(sorted_all.begin(), sorted_all.end());

    // Determine the maximum number of candidates we are willing to inspect.
    int max_scan;
    if (max_expansion_factor <= 0) {
        max_scan = N;  // unlimited → guarantee best possible recall
    } else {
        max_scan = std::min(N, K * max_expansion_factor);
    }

    // Walk sorted list; initial window = K * oversample_factor.
    // If fewer than K matches found in window, expand (double) and continue
    // until we have K results or hit max_scan.
    int window = std::min(N, K * oversample_factor);
    std::vector<int> result;
    result.reserve(K);

    int scanned = 0;
    while (scanned < max_scan) {
        int end = std::min(max_scan, window);
        // Scan from where we left off
        for (; scanned < end; ++scanned) {
            const auto& [dist, id] = sorted_all[scanned];
            if (std::regex_search(all_strings[id], pat)) {
                result.push_back(id);
                if ((int)result.size() == K) goto done;
            }
        }
        if (scanned >= max_scan) break;
        // Not enough matches — double the window and continue
        window = std::min(max_scan, window * 2);
    }
done:

    auto t1 = Clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() / 1000.0;
    return {result, ms};
}

// ─────────────────────────────────────────────────────────────────────────────
// Full-scan (exact ground truth)
// ─────────────────────────────────────────────────────────────────────────────

BaselineResult run_fullscan(
        const std::string& regex_str,
        const std::vector<float>& query_vec,
        const std::vector<std::vector<float>>& all_vectors,
        const std::vector<std::string>& all_strings,
        int K) {
    auto t0 = Clock::now();

    std::regex pat;
    try { pat = std::regex(regex_str, std::regex::icase); }
    catch (...) { return {{}, 0.0}; }

    MaxHeap h;
    for (int i = 0; i < (int)all_vectors.size(); ++i) {
        if (!std::regex_search(all_strings[i], pat)) continue;
        heap_push(h, l2_dist(query_vec, all_vectors[i]), i, K);
    }

    auto t1 = Clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() / 1000.0;
    return {heap_drain(h), ms};
}
