// include/HierarchicalIndex.h
// Two-level hierarchical k-means for RegExANN.
//
// Motivation (SR.3.6 in the paper):
//   A flat k-means with k clusters requires scanning up to k centroids per
//   query.  A two-level hierarchy reduces this to O(√k) centroid comparisons:
//   the top level has √k coarse clusters; each coarse cluster contains √k
//   fine clusters.  At query time we probe only the nearest coarse clusters
//   and then their fine children.
//
// Structure:
//   Level-0 (coarse): k0 clusters, each with a centroid μ₀ⱼ
//   Level-1 (fine):   each coarse cluster c split into k1 sub-clusters,
//                     total k0*k1 fine clusters
//   Trigram index:    maps trigrams → fine cluster IDs
//   PQ:               applied over fine cluster membership
//
// For flat k-means (k1==1) this degenerates to the original algorithm.
#pragma once
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "KMeans.h"
#include "PQ.h"

struct HierarchicalIndex {
    // Level-0 coarse clusters
    int k0 = 0;
    std::vector<std::vector<float>> coarse_centroids; // [k0][dim]
    std::vector<std::vector<int>>   coarse_to_fine;   // [k0] = list of fine IDs

    // Level-1 fine clusters
    int k1 = 0;
    int num_fine = 0;                                 // = k0 * k1
    std::vector<std::vector<float>> fine_centroids;   // [num_fine][dim]
    std::vector<std::vector<int>>   fine_clusters;    // [num_fine] = vector IDs
    std::vector<int>                vec_to_fine;      // [N] = fine cluster ID

    // Trigram index over fine clusters
    std::unordered_map<std::string, std::set<int>> gram_index;

    // PQ over all vectors
    PQIndex pq;
};

// Build a two-level hierarchical index.
//   data     : all dataset vectors
//   strings  : associated text strings (parallel to data)
//   k0       : number of coarse clusters  (e.g. 10)
//   k1       : fine clusters per coarse   (e.g. 10, giving 100 fine total)
//   pq_m     : PQ subspaces
//   pq_ksub  : PQ centroids per subspace
//   max_iter : k-means iterations for each level
HierarchicalIndex build_hierarchical_index(
    const std::vector<std::vector<float>>& data,
    const std::vector<std::string>& strings,
    int k0, int k1,
    int pq_m    = 8,
    int pq_ksub = 256,
    int max_iter = 30);

// Query the hierarchical index:
//   Returns top-K vector IDs matching `regex` and nearest to `query`.
//   nprobe_coarse: how many coarse clusters to probe (default = k0/2)
std::vector<int> query_hierarchical(
    const HierarchicalIndex& hidx,
    const std::string& regex,
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& all_vectors,
    const std::vector<std::string>& all_strings,
    int K,
    int nprobe_coarse = 0);
