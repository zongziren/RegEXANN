// src/Search.cpp
// RegExANN query processing pipeline:
//
//   Step 1  Trigram-based candidate cluster selection
//   Step 2  Sort candidate clusters by exact centroid distance (ascending)
//   Step 3  For each cluster (nearest-centroid first, up to nprobe clusters):
//             a. Compute PQ approximate distance for every vector in cluster
//             b. partial_sort to bring the ef-nearest (by PQ) to the front
//             c. Walk in PQ-distance order:
//                  - PQ within-cluster early-exit: if heap full and
//                    pq_dist > worst pq_dist in heap → skip rest of cluster
//                  - regex check: skip non-matching vectors
//                  - collect matching vectors into ef-pool (keyed by PQ dist)
//             d. Cluster-level early-exit (only when nprobe=0):
//                  heap full AND next centroid's exact distance >
//                  sqrt(worst PQ dist in heap) * expansion_factor
//                  This is a heuristic; nprobe is the reliable hard limit.
//   Step 4  Exact rerank: compute true L2 for ef survivors → return top-K
//
// Parameters:
//   ef      >= K: candidate pool size. Larger ef → higher recall, slower.
//           ef=K is the original behaviour (stop as soon as K matches found).
//   nprobe  > 0: hard limit on clusters to scan (fast, lower recall).
//           = 0: scan all candidate clusters, use heuristic early-exit.

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
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static float sq_l2(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i) { float d = a[i]-b[i]; s += d*d; }
    return s;
}

static float l2(const std::vector<float>& a, const std::vector<float>& b) {
    return std::sqrt(sq_l2(a, b));
}

// ─────────────────────────────────────────────────────────────────────────────
// Main search
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
        int ef,
        int nprobe) {

    if (ef <= 0) ef = K;
    if (ef < K)  ef = K;

    using Clock = std::chrono::high_resolution_clock;
    using us    = std::chrono::microseconds;

    // ── Step 1: Trigram candidate cluster selection ───────────────────────
    auto t0 = Clock::now();
    int num_clusters = static_cast<int>(centroids.size());
    std::set<int> candidate_clusters =
        get_candidate_clusters(regex_str, gram_index, num_clusters);
    double setop_ms = std::chrono::duration_cast<us>(Clock::now()-t0).count()/1000.0;

    // ── Step 2: Sort candidate clusters by exact centroid distance ────────
    auto t2 = Clock::now();

    std::vector<std::pair<float,int>> cluster_dists; // (centroid_dist, cid)
    cluster_dists.reserve(candidate_clusters.size());
    for (int cid : candidate_clusters)
        cluster_dists.emplace_back(l2(query_vector, centroids[cid]), cid);
    std::sort(cluster_dists.begin(), cluster_dists.end());

    // Compile regex once
    std::regex pattern;
    try {
        pattern = std::regex(regex_str, std::regex::icase);
    } catch (const std::regex_error& e) {
        std::cerr << "[WARN] Invalid regex: " << e.what() << "\n";
        return SearchResult{{}, setop_ms, 0.0};
    }

    // Precompute PQ lookup table for this query
    auto dist_table = compute_dist_table(pq, query_vector);

    // ── Step 3: Scan clusters nearest-first ──────────────────────────────
    //
    // ef-pool: max-heap of (pq_sq_dist, vid), size ≤ ef.
    // All distances inside this heap are PQ squared distances.
    // They are comparable with each other (same space).
    // The heap gives us the ef best-by-PQ regex-matching candidates seen so far.
    //
    // Within-cluster early-exit (correct):
    //   pq_dist (candidate) > heap.top().pq_dist (worst in pool)
    //   → candidate cannot enter pool → skip rest of cluster (PQ order)
    //
    // Cluster-level early-exit (heuristic, only when nprobe=0):
    //   next_centroid_dist² > heap.top().pq_dist * slack
    //   Rationale: PQ underestimates true distance. If even the centroid of
    //   the next cluster is already farther than our current worst PQ estimate
    //   (with a slack factor to account for PQ error), further clusters are
    //   unlikely to contribute. This is NOT a correctness guarantee; use
    //   nprobe for a hard limit.
    //   slack = 1.0 → aggressive (may miss some results, lower recall)
    //   slack = 2.0 → conservative (safer, closer to scanning all clusters)
    constexpr float EARLY_STOP_SLACK = 1.5f;

    using PqId = std::pair<float,int>; // (pq_sq_dist, vid)
    std::priority_queue<PqId> pq_heap; // max-heap, size ≤ ef

    int scanned = 0;
    for (int ci = 0; ci < (int)cluster_dists.size(); ++ci) {
        const auto [centroid_dist, cid] = cluster_dists[ci];

        // Hard cluster limit
        if (nprobe > 0 && scanned >= nprobe) break;
        ++scanned;

        const auto& vids = clusters[cid];

        // ── PQ distances for this cluster ──────────────────────────────
        std::vector<PqId> pq_order;
        pq_order.reserve(vids.size());
        for (int vid : vids)
            pq_order.emplace_back(pq_approx_dist(dist_table, pq.codes[vid]), vid);

        std::sort(pq_order.begin(), pq_order.end());

        // ── Scan in PQ order; regex check; update ef-pool ─────────────
        for (const auto& [pq_dist, vid] : pq_order) {
            // Within-cluster early-exit: all distances in same space (PQ sq)
            if ((int)pq_heap.size() == ef && pq_dist > pq_heap.top().first)
                break;

            if (!std::regex_search(all_strings[vid], pattern))
                continue;

            pq_heap.emplace(pq_dist, vid);
            if ((int)pq_heap.size() > ef) pq_heap.pop();
        }

        // ── Cluster-level heuristic early-exit (nprobe=0 only) ────────
        // Only trigger when pool is full (ef results collected).
        // Compare next centroid's exact squared distance against the worst
        // PQ squared distance in the pool (with slack for PQ error).
        if (nprobe == 0 &&
            (int)pq_heap.size() == ef &&
            ci + 1 < (int)cluster_dists.size()) {

            float next_cd    = cluster_dists[ci+1].first; // exact centroid dist
            float next_cd_sq = next_cd * next_cd;         // exact centroid dist²
            float worst_pq   = pq_heap.top().first;       // PQ sq dist (approx)

            // next centroid is farther than our worst candidate (with slack).
            // PQ underestimates, so worst_pq * slack compensates for PQ error.
            if (next_cd_sq > worst_pq * EARLY_STOP_SLACK)
                break;
        }
    }

    double query_ms = std::chrono::duration_cast<us>(Clock::now()-t2).count()/1000.0;

    // ── Step 4: Exact rerank ef survivors → top-K ────────────────────────
    // Drain pq_heap → compute exact L2 for each → sort → take top-K.
    using DistId = std::pair<float,int>;
    std::vector<DistId> pool;
    pool.reserve(pq_heap.size());
    while (!pq_heap.empty()) {
        int vid = pq_heap.top().second; pq_heap.pop();
        pool.emplace_back(l2(query_vector, all_vectors[vid]), vid);
    }
    std::sort(pool.begin(), pool.end()); // ascending exact dist
    if ((int)pool.size() > K) pool.resize(K);

    std::vector<int> result;
    result.reserve(pool.size());
    for (const auto& [dist, id] : pool) result.push_back(id);

    return SearchResult{result, setop_ms, query_ms};
}
