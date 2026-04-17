// src/Baseline.cpp
#include "Baseline.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <queue>
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
        int K) {
    auto t0 = Clock::now();

    std::regex pat;
    try { pat = std::regex(regex_str, std::regex::icase); }
    catch (...) { return {{}, 0.0}; }

    // Phase 1: filter
    std::vector<int> matching;
    for (int i = 0; i < (int)all_strings.size(); ++i)
        if (std::regex_search(all_strings[i], pat))
            matching.push_back(i);

    // Phase 2: exact kNN on matching subset
    MaxHeap h;
    for (int id : matching)
        heap_push(h, l2_dist(query_vec, all_vectors[id]), id, K);

    auto t1 = Clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() / 1000.0;
    return {heap_drain(h), ms};
}

// ─────────────────────────────────────────────────────────────────────────────
// Post-filtering
// ─────────────────────────────────────────────────────────────────────────────

BaselineResult run_postfilter(
        const std::string& regex_str,
        const std::vector<float>& query_vec,
        const std::vector<std::vector<float>>& all_vectors,
        const std::vector<std::string>& all_strings,
        int K,
        int oversample_factor) {
    auto t0 = Clock::now();

    std::regex pat;
    try { pat = std::regex(regex_str, std::regex::icase); }
    catch (...) { return {{}, 0.0}; }

    int K_prime = K * oversample_factor;

    // Phase 1: global ANN (exact here; in practice use an ANN index)
    MaxHeap h;
    for (int i = 0; i < (int)all_vectors.size(); ++i)
        heap_push(h, l2_dist(query_vec, all_vectors[i]), i, K_prime);

    // Drain into ascending order
    std::vector<std::pair<float,int>> candidates;
    candidates.reserve(h.size());
    while (!h.empty()) { candidates.push_back(h.top()); h.pop(); }
    std::reverse(candidates.begin(), candidates.end());

    // Phase 2: apply regex filter, keep top-K matching
    std::vector<int> result;
    result.reserve(K);
    for (const auto& [dist, id] : candidates) {
        if (std::regex_search(all_strings[id], pat)) {
            result.push_back(id);
            if ((int)result.size() == K) break;
        }
    }

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
