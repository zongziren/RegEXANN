// include/KMeans.h
#pragma once
#include <vector>
#include <string>

struct KMeansResult {
    std::vector<int> assignments;        // 每个向量的聚类 ID
    std::vector<std::vector<float>> centroids; // 每个聚类质心
    std::vector<std::vector<int>> clusters;    // 每个聚类中的向量ID
};

KMeansResult run_kmeans(const std::vector<std::vector<float>>& data, int k, int max_iters);
void dump_cluster_assignments(const std::vector<std::vector<int>>& clusters, const std::string& filename);
