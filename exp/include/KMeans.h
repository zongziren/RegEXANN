// include/KMeans.h
#pragma once
#include <vector>
#include <string>

struct KMeansResult {
    std::vector<int>               assignments; // cluster ID for each vector
    std::vector<std::vector<float>> centroids;  // centroid per cluster
    std::vector<std::vector<int>>   clusters;   // vector IDs per cluster
};

KMeansResult run_kmeans(const std::vector<std::vector<float>>& data,
                        int k, int max_iters);

void dump_cluster_assignments(const std::vector<std::vector<int>>& clusters,
                              const std::string& filename);
