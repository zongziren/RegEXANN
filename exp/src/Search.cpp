// src/Search.cpp
// RegExANN query processing pipeline (Algorithm 2 in the paper):
//   Step 1  Trigram extraction & candidate cluster selection (set ops)
//   Step 2  Sort candidate clusters by centroid distance to query
//   Step 3  For each cluster (nearest first):
//              a. Rank its vectors by PQ approximate distance
//              b. Verify regex in that order; collect into ef-pool
//              c. Stop as soon as ef valid results are collected
//   Step 4  Re-rank ef-pool by exact Euclidean distance → top-K
//
// The `ef` parameter (ef >= K) controls the recall/speed trade-off:
//   ef == K  → original behaviour, stops as soon as K results found
//   ef >  K  → collects more candidates before stopping, improving recall
//              at the cost of extra regex checks and distance computations.
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
        int K,
        int ef) {

    // ef is the candidate pool size; must be >= K.
    // ef == 0 (default) means use K (original behaviour).
    if (ef <= 0) ef = K;
    if (ef < K)  ef = K;

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

    // We use a max-heap keyed by exact Euclidean distance, capped at size ef.
    // This collects the ef nearest regex-matching candidates across all
    // visited clusters.  After scanning, we drain the top-K from it.
    //
    // Early-stop logic (per cluster-level):
    //   Once we have ef candidates AND the current cluster's centroid distance
    //   already exceeds the ef-th best exact distance, no future cluster can
    //   contribute to the top-K result, so we stop.
    //
    // Early-stop logic (within a cluster, PQ-level):
    //   If the heap has ef elements and the PQ lower-bound of the next
    //   candidate already exceeds the ef-th best exact distance, the rest
    //   of this cluster cannot enter the pool either.
    using DistId = std::pair<float, int>; // (distance, vector_id)
    std::priority_queue<DistId> heap;     // max-heap, size ≤ ef

    for (const auto& [centroid_dist, cid] : cluster_dists) {
        const auto& vids = clusters[cid];

        // ── PQ-sort within this cluster ────────────────────────────────
        std::vector<std::pair<float, int>> pq_order;
        pq_order.reserve(vids.size());
        for (int vid : vids)
            pq_order.emplace_back(
                pq_approx_dist(dist_table, pq.codes[vid]), vid);
        std::sort(pq_order.begin(), pq_order.end());

        // ── Scan in PQ-distance order; verify regex; update ef-heap ───
        for (const auto& [pq_dist, vid] : pq_order) {
            // Within-cluster PQ early exit: heap is full at ef and PQ
            // lower-bound already beats worst in heap.
            if ((int)heap.size() == ef &&
                std::sqrt(pq_dist) > heap.top().first)
                break;

            if (!std::regex_search(all_strings[vid], pattern))
                continue;

            float exact_dist = euclidean_distance(query_vector, all_vectors[vid]);
            heap.emplace(exact_dist, vid);
            if ((int)heap.size() > ef) heap.pop();
        }

        // Cluster-level early stop: ef candidates collected AND centroid of
        // next cluster is already farther than the ef-th best we have.
        if ((int)heap.size() == ef &&
            centroid_dist > heap.top().first)
            break;
    }

    auto t3 = Clock::now();
    double query_ms =
        duration_cast<microseconds>(t3 - t2).count() / 1000.0;

    // ── Step 4: Extract top-K from ef-pool in ascending distance order ────
    // Drain the max-heap; discard entries beyond K.
    std::vector<DistId> pool;
    pool.reserve(heap.size());
    while (!heap.empty()) {
        pool.push_back(heap.top());
        heap.pop();
    }
    // heap was max-heap, so pool is in descending order; reverse → ascending.
    std::reverse(pool.begin(), pool.end());

    // Keep only the K nearest.
    if ((int)pool.size() > K) pool.resize(K);

    std::vector<int> result;
    result.reserve(pool.size());
    for (const auto& [dist, id] : pool)
        result.push_back(id);

    return SearchResult{result, setop_ms, query_ms};
}
