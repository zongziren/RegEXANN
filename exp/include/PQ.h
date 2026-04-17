// include/PQ.h
#pragma once
#include <vector>
#include <cstdint>

// Product Quantization index structure.
// Each vector of dimension `dim` is split into `m` sub-vectors,
// each quantized to one of `k_sub` centroids learned per subspace.
struct PQIndex {
    int m       = 8;    // number of subspaces
    int k_sub   = 256;  // centroids per subspace (fit in uint8_t)
    int dim     = 0;    // total vector dimension
    int sub_dim = 0;    // dimension per subspace = dim / m

    // codebooks[subspace][centroid_id][coord_in_subspace]
    std::vector<std::vector<std::vector<float>>> codebooks;

    // codes[vector_id][subspace] = centroid_id  (uint8_t → 0..k_sub-1)
    std::vector<std::vector<uint8_t>> codes;
};

// Train PQ codebooks from data using k-means on each subspace.
// m        : number of subspaces (must evenly divide data[0].size())
// k_sub    : number of centroids per subspace (default 256)
// max_iters: k-means iterations per subspace
void train_pq(PQIndex& pq,
              const std::vector<std::vector<float>>& data,
              int m,
              int k_sub    = 256,
              int max_iters = 25);

// Encode all vectors in `data` using the trained codebooks.
// Must call train_pq first.
void encode_all(PQIndex& pq,
                const std::vector<std::vector<float>>& data);

// Precompute per-subspace distance tables for a query vector.
// dist_table[subspace][centroid_id] = squared distance from
//   query sub-vector to that centroid.
std::vector<std::vector<float>> compute_dist_table(
    const PQIndex& pq,
    const std::vector<float>& query);

// Approximate squared distance from query to vector vec_id,
// using precomputed dist_table (Asymmetric Distance Computation).
float pq_approx_dist(
    const std::vector<std::vector<float>>& dist_table,
    const std::vector<uint8_t>& code);
