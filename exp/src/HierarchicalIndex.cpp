// src/HierarchicalIndex.cpp
// Two-level hierarchical RegExANN.
// The query pipeline mirrors the flat version (Algorithm 2) but operates on a
// two-level cluster hierarchy to reduce the number of centroid comparisons.
#include "HierarchicalIndex.h"
#include "Index.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <regex>

static float l2sq(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i) { float d=a[i]-b[i]; s+=d*d; }
    return s;
}
static float l2(const std::vector<float>& a, const std::vector<float>& b) {
    return std::sqrt(l2sq(a, b));
}

// ─────────────────────────────────────────────────────────────────────────────
// Build
// ─────────────────────────────────────────────────────────────────────────────

HierarchicalIndex build_hierarchical_index(
        const std::vector<std::vector<float>>& data,
        const std::vector<std::string>& strings,
        int k0, int k1, int pq_m, int pq_ksub, int max_iter) {

    int N = (int)data.size();
    HierarchicalIndex hidx;
    hidx.k0 = k0;
    hidx.k1 = k1;
    hidx.vec_to_fine.resize(N, 0);

    // Level-0 coarse k-means
    std::cout << "[HIdx] Level-0 k-means (k0=" << k0 << ") …\n";
    k0 = std::min(k0, N);
    auto km0 = run_kmeans(data, k0, max_iter);
    hidx.coarse_centroids = km0.centroids;
    hidx.coarse_to_fine.resize(k0);

    // Level-1 fine k-means within each coarse cluster
    int fine_id = 0;
    for (int c0 = 0; c0 < k0; ++c0) {
        const auto& members = km0.clusters[c0];
        int k1_eff = std::min(k1, (int)members.size());
        if (k1_eff == 0) {
            // Empty coarse cluster — insert one dummy fine cluster
            hidx.coarse_to_fine[c0].push_back(fine_id);
            hidx.fine_centroids.push_back(km0.centroids[c0]);
            hidx.fine_clusters.push_back({});
            ++fine_id;
            continue;
        }

        std::vector<std::vector<float>> sub(members.size());
        for (size_t i = 0; i < members.size(); ++i)
            sub[i] = data[members[i]];

        auto km1 = run_kmeans(sub, k1_eff, max_iter);

        for (int j = 0; j < k1_eff; ++j) {
            hidx.coarse_to_fine[c0].push_back(fine_id);
            hidx.fine_centroids.push_back(km1.centroids[j]);

            std::vector<int> fmembers;
            for (int vid : km1.clusters[j])
                fmembers.push_back(members[vid]);
            hidx.fine_clusters.push_back(fmembers);

            for (int vid : fmembers)
                hidx.vec_to_fine[vid] = fine_id;

            ++fine_id;
        }
    }
    hidx.num_fine = fine_id;

    // Trigram index (over fine cluster IDs)
    std::cout << "[HIdx] Building trigram index …\n";
    build_3gram_index(strings, hidx.vec_to_fine, hidx.gram_index);

    // Product Quantization
    std::cout << "[HIdx] Training PQ (m=" << pq_m << ") …\n";
    int dim   = (int)data[0].size();
    int eff_m = pq_m;
    while (eff_m > 1 && dim % eff_m != 0) --eff_m;
    int k_sub = std::min(pq_ksub, N);
    train_pq(hidx.pq, data, eff_m, k_sub, 25);
    encode_all(hidx.pq, data);

    std::cout << "[HIdx] Index ready. Fine clusters: " << hidx.num_fine << "\n";
    return hidx;
}

// ─────────────────────────────────────────────────────────────────────────────
// Query
// ─────────────────────────────────────────────────────────────────────────────

std::vector<int> query_hierarchical(
        const HierarchicalIndex& hidx,
        const std::string& regex_str,
        const std::vector<float>& query,
        const std::vector<std::vector<float>>& all_vectors,
        const std::vector<std::string>& all_strings,
        int K,
        int nprobe_coarse) {

    if (nprobe_coarse <= 0)
        nprobe_coarse = std::max(1, hidx.k0 / 2);

    // Step 1: trigram filtering → candidate fine clusters
    std::set<int> cand_fine =
        get_candidate_clusters(regex_str, hidx.gram_index, hidx.num_fine);

    // Step 2: rank coarse clusters by centroid distance
    std::vector<std::pair<float,int>> coarse_dists;
    coarse_dists.reserve(hidx.k0);
    for (int c = 0; c < hidx.k0; ++c)
        coarse_dists.emplace_back(l2(query, hidx.coarse_centroids[c]), c);
    std::sort(coarse_dists.begin(), coarse_dists.end());

    // Step 3: collect candidate fine clusters from top-nprobe coarse clusters
    std::vector<std::pair<float,int>> fine_dists; // (dist_to_centroid, fine_id)
    int probed = 0;
    for (const auto& [cd, c0] : coarse_dists) {
        if (probed >= nprobe_coarse) break;
        ++probed;
        for (int fid : hidx.coarse_to_fine[c0]) {
            if (cand_fine.count(fid))
                fine_dists.emplace_back(l2(query, hidx.fine_centroids[fid]), fid);
        }
    }
    std::sort(fine_dists.begin(), fine_dists.end());

    // Step 4: compile regex once
    std::regex pattern;
    try { pattern = std::regex(regex_str, std::regex::icase); }
    catch (...) { return {}; }

    // Step 5: PQ distance table
    auto dist_table = compute_dist_table(hidx.pq, query);

    // Step 6: scan clusters in distance order, verify regex, collect top-K
    using MaxH = std::priority_queue<std::pair<float,int>>;
    MaxH heap; // max-heap of (exact_dist, vid)

    for (const auto& [centroid_dist, fid] : fine_dists) {
        // PQ-sort vectors inside this fine cluster
        std::vector<std::pair<float,int>> pq_order;
        for (int vid : hidx.fine_clusters[fid])
            pq_order.emplace_back(
                pq_approx_dist(dist_table, hidx.pq.codes[vid]), vid);
        std::sort(pq_order.begin(), pq_order.end());

        for (const auto& [pqd, vid] : pq_order) {
            // PQ early exit within cluster
            if ((int)heap.size() == K && std::sqrt(pqd) > heap.top().first)
                break;

            if (!std::regex_search(all_strings[vid], pattern))
                continue;

            float exact = std::sqrt(l2sq(query, all_vectors[vid]));
            heap.emplace(exact, vid);
            if ((int)heap.size() > K) heap.pop();
        }

        // Cluster-level early stop
        if ((int)heap.size() == K && centroid_dist > heap.top().first)
            break;
    }

    std::vector<int> result;
    result.reserve(heap.size());
    while (!heap.empty()) { result.push_back(heap.top().second); heap.pop(); }
    std::reverse(result.begin(), result.end());
    return result;
}
