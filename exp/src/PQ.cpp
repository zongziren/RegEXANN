// src/PQ.cpp
// Product Quantization implementation.
// Reference: Jégou et al., "Product Quantization for Nearest Neighbor Search", TPAMI 2011.
#include "PQ.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Extract the j-th sub-vector of a full vector.
static std::vector<float> subvec(const std::vector<float>& v, int subspace, int sub_dim) {
    int start = subspace * sub_dim;
    return std::vector<float>(v.begin() + start, v.begin() + start + sub_dim);
}

// Squared Euclidean distance between two equal-length vectors.
static float sq_dist(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// Run k-means on `points` (all of the same dimension) and return centroids.
// Returns k centroids; handles degenerate cases (empty clusters).
static std::vector<std::vector<float>> kmeans_subspace(
        const std::vector<std::vector<float>>& points,
        int k,
        int max_iters) {

    int n   = (int)points.size();
    int dim = (int)points[0].size();

    // Limit k to available points
    k = std::min(k, n);

    // Initialise centroids by sampling without replacement (reservoir-style)
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    for (int i = 0; i < k; ++i) {
        int j = i + rand() % (n - i);
        std::swap(perm[i], perm[j]);
    }
    std::vector<std::vector<float>> centroids(k);
    for (int i = 0; i < k; ++i)
        centroids[i] = points[perm[i]];

    std::vector<int> assign(n, 0);

    for (int iter = 0; iter < max_iters; ++iter) {
        // Assignment step
        bool changed = false;
        for (int i = 0; i < n; ++i) {
            float best = std::numeric_limits<float>::max();
            int   bi   = 0;
            for (int c = 0; c < k; ++c) {
                float d = sq_dist(points[i], centroids[c]);
                if (d < best) { best = d; bi = c; }
            }
            if (assign[i] != bi) { assign[i] = bi; changed = true; }
        }
        if (!changed) break;

        // Update step
        std::vector<std::vector<float>> new_c(k, std::vector<float>(dim, 0.f));
        std::vector<int> cnt(k, 0);
        for (int i = 0; i < n; ++i) {
            int c = assign[i];
            for (int d = 0; d < dim; ++d)
                new_c[c][d] += points[i][d];
            cnt[c]++;
        }
        for (int c = 0; c < k; ++c) {
            if (cnt[c] == 0) {
                // Reinitialise empty centroid to a random point
                new_c[c] = points[rand() % n];
            } else {
                for (int d = 0; d < dim; ++d)
                    new_c[c][d] /= cnt[c];
            }
        }
        centroids = new_c;
    }
    return centroids;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void train_pq(PQIndex& pq,
              const std::vector<std::vector<float>>& data,
              int m,
              int k_sub,
              int max_iters) {
    if (data.empty())
        throw std::runtime_error("train_pq: empty dataset");

    int dim = (int)data[0].size();
    if (dim % m != 0)
        throw std::runtime_error("train_pq: dim must be divisible by m");

    pq.m       = m;
    pq.k_sub   = k_sub;
    pq.dim     = dim;
    pq.sub_dim = dim / m;
    pq.codebooks.resize(m);

    // Train one codebook per subspace
    for (int s = 0; s < m; ++s) {
        // Collect sub-vectors for subspace s
        std::vector<std::vector<float>> subvecs(data.size());
        for (size_t i = 0; i < data.size(); ++i)
            subvecs[i] = subvec(data[i], s, pq.sub_dim);

        pq.codebooks[s] = kmeans_subspace(subvecs, k_sub, max_iters);
    }
}

void encode_all(PQIndex& pq, const std::vector<std::vector<float>>& data) {
    int n = (int)data.size();
    pq.codes.resize(n, std::vector<uint8_t>(pq.m));

    for (int i = 0; i < n; ++i) {
        for (int s = 0; s < pq.m; ++s) {
            std::vector<float> sv = subvec(data[i], s, pq.sub_dim);
            float best = std::numeric_limits<float>::max();
            int   bi   = 0;
            const auto& cb = pq.codebooks[s];
            for (int c = 0; c < (int)cb.size(); ++c) {
                float d = sq_dist(sv, cb[c]);
                if (d < best) { best = d; bi = c; }
            }
            pq.codes[i][s] = static_cast<uint8_t>(bi);
        }
    }
}

std::vector<std::vector<float>> compute_dist_table(
        const PQIndex& pq,
        const std::vector<float>& query) {
    // dist_table[subspace][centroid] = sq_dist(query_subvec, centroid)
    std::vector<std::vector<float>> table(pq.m);
    for (int s = 0; s < pq.m; ++s) {
        std::vector<float> qsub = subvec(query, s, pq.sub_dim);
        const auto& cb = pq.codebooks[s];
        table[s].resize(cb.size());
        for (int c = 0; c < (int)cb.size(); ++c)
            table[s][c] = sq_dist(qsub, cb[c]);
    }
    return table;
}

float pq_approx_dist(const std::vector<std::vector<float>>& dist_table,
                     const std::vector<uint8_t>& code) {
    float sum = 0.f;
    for (size_t s = 0; s < code.size(); ++s)
        sum += dist_table[s][code[s]];
    return sum; // squared approx distance; caller takes sqrt if needed
}
