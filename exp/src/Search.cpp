// src/Search.cpp
// RegExANN query processing pipeline (Algorithm 2 in the paper):
//   Step 1  Trigram extraction & candidate cluster selection (set ops)
//   Step 2  Sort candidate clusters by centroid distance to query
//   Step 3  For each cluster (nearest first):
//              a. Rank its vectors by PQ approximate distance
//              b. Verify regex in that order; collect into result
//              c. Stop as soon as K valid results are found
//   Step 4  Re-rank collected results by exact Euclidean distance → top-K
#include "Search.h"
#include "Index.h"
#include "PQ.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <queue>
#include <regex>

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

static float euclidean_distance(const std::vector<float>& a,
                                const std::vector<float>& b) {
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main search function
// ─────────────────────────────────────────────────────────────────────────────

SearchResult perform_search(
        const std::string& regex_str,
        const std::vector<float>& query_vector,
        const std::unordered_map<std::string, std::set<int>>& gram_index,
        const std::vector<std::vector<float>>& centroids,
        const std::vector<std::vector<int>>& clusters,
        const std::vector<std::vector<float>>& all_vectors,
        const std::vector<std::string>& all_strings,
        const PQIndex& pq,
        int K) {

    using namespace std::chrono;
    using Clock = high_resolution_clock;

    // ── Step 1: Trigram-based candidate cluster selection ─────────────────
    auto t0 = Clock::now();

    int num_clusters = static_cast<int>(centroids.size());
    std::set<int> candidate_clusters =
        get_candidate_clusters(regex_str, gram_index, num_clusters);

    auto t1 = Clock::now();
    double setop_ms =
        duration_cast<microseconds>(t1 - t0).count() / 1000.0;

    // ── Step 2 & 3: Cluster ranking + PQ scan + regex verify ─────────────
    auto t2 = Clock::now();

    // Sort candidate clusters by distance from query to centroid (ascending).
    std::vector<std::pair<float, int>> cluster_dists;
    cluster_dists.reserve(candidate_clusters.size());
    for (int cid : candidate_clusters)
        cluster_dists.emplace_back(
            euclidean_distance(query_vector, centroids[cid]), cid);
    std::sort(cluster_dists.begin(), cluster_dists.end());

    // Compile regex once (icase: case-insensitive matching).
    std::regex pattern;
    try {
        pattern = std::regex(regex_str, std::regex::icase);
    } catch (const std::regex_error& e) {
        std::cerr << "[WARN] Invalid regex: " << e.what() << "\n";
        return SearchResult{{}, setop_ms, 0.0};
    }

    // Precompute PQ distance lookup table for the query.
    auto dist_table = compute_dist_table(pq, query_vector);

    // We use a max-heap (size K) keyed by exact Euclidean distance to keep
    // the running top-K.  We fill it via early stopping across clusters.
    // Because PQ gives only approximate distances, we cannot stop within a
    // cluster — we must scan every regex-matching element of every cluster
    // we visit and rely on cluster-level early stopping.
    //
    // Early-stop heuristic: once the heap has K elements AND the current
    // cluster's centroid distance minus the cluster's approximate radius is
    // already larger than our K-th best distance, further clusters cannot
    // improve the result.  Here we use a simpler but safe variant: stop
    // when we have accumulated at least K results from scanned clusters
    // (the heap ensures we keep the global top-K).
    using DistId = std::pair<float, int>; // (distance, vector_id)
    std::priority_queue<DistId> heap;     // max-heap, size ≤ K

    for (const auto& [centroid_dist, cid] : cluster_dists) {
        const auto& vids = clusters[cid];

        // ── PQ-sort within this cluster ────────────────────────────────
        // Build (pq_dist, vid) pairs and sort ascending by PQ distance.
        std::vector<std::pair<float, int>> pq_order;
        pq_order.reserve(vids.size());
        for (int vid : vids)
            pq_order.emplace_back(
                pq_approx_dist(dist_table, pq.codes[vid]), vid);
        std::sort(pq_order.begin(), pq_order.end());

        // ── Scan in PQ-distance order; verify regex; update heap ───────
        for (const auto& [pq_dist, vid] : pq_order) {
            // Optional early exit inside cluster: if the heap is full and
            // the PQ lower-bound already exceeds the worst in the heap,
            // remaining elements in this cluster cannot enter the top-K.
            if ((int)heap.size() == K &&
                std::sqrt(pq_dist) > heap.top().first)
                break;

            if (!std::regex_search(all_strings[vid], pattern))
                continue;

            float exact_dist = euclidean_distance(query_vector, all_vectors[vid]);
            heap.emplace(exact_dist, vid);
            if ((int)heap.size() > K) heap.pop();
        }

        // Cluster-level early stop: heap is full and the closest possible
        // point in upcoming clusters (lower-bounded by centroid distance)
        // cannot beat the current K-th best.
        if ((int)heap.size() == K &&
            centroid_dist > heap.top().first)
            break;
    }

    auto t3 = Clock::now();
    double query_ms =
        duration_cast<microseconds>(t3 - t2).count() / 1000.0;

    // ── Step 4: Extract results in ascending distance order ───────────────
    std::vector<int> result;
    result.reserve(heap.size());
    while (!heap.empty()) {
        result.push_back(heap.top().second);
        heap.pop();
    }
    std::reverse(result.begin(), result.end()); // nearest first

    return SearchResult{result, setop_ms, query_ms};
}
