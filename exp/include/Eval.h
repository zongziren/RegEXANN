// include/Eval.h
// Evaluation utilities: recall@K computation and result I/O.
#pragma once
#include <string>
#include <vector>

// Load a result file produced by main.cpp (one space-separated ID list per line).
// Returns results[query_idx] = vector of returned IDs.
std::vector<std::vector<int>> load_results(const std::string& filename);

// Compute per-query Recall@K and mean recall.
// ground_truth[i] : the true top-K IDs for query i (order ignored)
// predictions[i]  : the IDs returned by the system for query i
// K               : number of neighbours to evaluate at
// Returns (mean_recall, per_query_recall[])
std::pair<double, std::vector<double>> compute_recall(
    const std::vector<std::vector<int>>& ground_truth,
    const std::vector<std::vector<int>>& predictions,
    int K);

// Write ground-truth or result IDs to a file (same format as main.cpp output).
void save_results(const std::string& filename,
                  const std::vector<std::vector<int>>& results);
