// src/Eval.cpp
#include "Eval.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

std::vector<std::vector<int>> load_results(const std::string& filename) {
    std::ifstream fin(filename);
    if (!fin) throw std::runtime_error("Cannot open results file: " + filename);
    std::vector<std::vector<int>> results;
    std::string line;
    while (std::getline(fin, line)) {
        std::istringstream iss(line);
        std::vector<int> ids;
        int id;
        while (iss >> id) ids.push_back(id);
        results.push_back(std::move(ids));
    }
    return results;
}

std::pair<double, std::vector<double>> compute_recall(
        const std::vector<std::vector<int>>& ground_truth,
        const std::vector<std::vector<int>>& predictions,
        int K) {
    if (ground_truth.size() != predictions.size())
        throw std::runtime_error("compute_recall: query count mismatch");

    std::vector<double> per_query;
    per_query.reserve(ground_truth.size());
    double total = 0.0;

    for (size_t i = 0; i < ground_truth.size(); ++i) {
        const auto& gt   = ground_truth[i];
        const auto& pred = predictions[i];

        if (gt.empty()) {
            // No ground-truth for this query (regex matched nothing): skip
            per_query.push_back(1.0);
            total += 1.0;
            continue;
        }

        // Build set of the true top-K IDs
        int effective_K = std::min(K, (int)gt.size());
        std::set<int> gt_set(gt.begin(),
                             gt.begin() + std::min(effective_K, (int)gt.size()));

        int hits = 0;
        for (int id : pred) {
            if (gt_set.count(id)) ++hits;
        }
        double recall = (double)hits / (double)gt_set.size();
        per_query.push_back(recall);
        total += recall;
    }

    double mean = ground_truth.empty() ? 0.0 : total / (double)ground_truth.size();
    return {mean, per_query};
}

void save_results(const std::string& filename,
                  const std::vector<std::vector<int>>& results) {
    std::ofstream fout(filename);
    if (!fout) throw std::runtime_error("Cannot write results file: " + filename);
    for (const auto& row : results) {
        for (int id : row) fout << id << " ";
        fout << "\n";
    }
}
