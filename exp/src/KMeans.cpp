// src/KMeans.cpp
#include "KMeans.h"
#include <cstdlib>
#include <limits>
#include <cmath>
#include <fstream>
#include <iostream>

static float euclidean_dist2(const std::vector<float>& a,
                              const std::vector<float>& b) {
    float sum = 0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sum;
}

KMeansResult run_kmeans(const std::vector<std::vector<float>>& data,
                        int k, int max_iters) {
    int n   = (int)data.size();
    int dim = (int)data[0].size();

    // Initialise centroids from random distinct points
    std::vector<std::vector<float>> centroids;
    centroids.reserve(k);
    for (int i = 0; i < k; ++i)
        centroids.push_back(data[rand() % n]);

    std::vector<int> assignments(n, 0);

    for (int iter = 0; iter < max_iters; ++iter) {
        bool changed = false;
        for (int i = 0; i < n; ++i) {
            float best   = std::numeric_limits<float>::max();
            int   best_c = 0;
            for (int c = 0; c < k; ++c) {
                float d = euclidean_dist2(data[i], centroids[c]);
                if (d < best) { best = d; best_c = c; }
            }
            if (assignments[i] != best_c) { assignments[i] = best_c; changed = true; }
        }
        if (!changed) break;

        std::vector<std::vector<float>> new_c(k, std::vector<float>(dim, 0.f));
        std::vector<int> cnt(k, 0);
        for (int i = 0; i < n; ++i) {
            int c = assignments[i];
            for (int j = 0; j < dim; ++j)
                new_c[c][j] += data[i][j];
            cnt[c]++;
        }
        for (int c = 0; c < k; ++c)
            for (int j = 0; j < dim; ++j)
                if (cnt[c]) new_c[c][j] /= cnt[c];
        centroids = new_c;
    }

    std::vector<std::vector<int>> clusters(k);
    for (int i = 0; i < n; ++i)
        clusters[assignments[i]].push_back(i);

    return {assignments, centroids, clusters};
}

void dump_cluster_assignments(const std::vector<std::vector<int>>& clusters,
                              const std::string& filename) {
    std::ofstream fout(filename);
    for (size_t cid = 0; cid < clusters.size(); ++cid) {
        fout << "Cluster " << cid << " (" << clusters[cid].size() << " vectors): ";
        for (int vid : clusters[cid]) fout << vid << " ";
        fout << "\n";
    }
    fout.close();
    std::cout << "[DEBUG] Cluster assignments dumped to " << filename << "\n";
}
