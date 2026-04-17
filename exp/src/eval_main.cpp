// src/eval_main.cpp
// Standalone Recall@K evaluation tool (C++ version of tools/eval_recall.py).
//
// Usage:
//   ./eval_recall --gt <gt_file> --pred <pred_file> --K <K> [--verbose]
//
// Both files: one space-separated list of integer IDs per line.
// Exit code 0 on success.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Eval.h"

static void usage() {
    std::cerr <<
"Usage: ./eval_recall --gt <file> --pred <file> --K <int> [--verbose]\n";
    std::exit(1);
}

int main(int argc, char* argv[]) {
    std::string gt_path, pred_path;
    int K = 10;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gt"      && i+1 < argc) gt_path   = argv[++i];
        else if (a == "--pred"    && i+1 < argc) pred_path = argv[++i];
        else if (a == "--K"       && i+1 < argc) K         = std::stoi(argv[++i]);
        else if (a == "--verbose") verbose = true;
        else { std::cerr << "Unknown: " << a << "\n"; usage(); }
    }
    if (gt_path.empty() || pred_path.empty()) usage();

    try {
        auto gt   = load_results(gt_path);
        auto pred = load_results(pred_path);

        if (gt.size() != pred.size())
            throw std::runtime_error("Query count mismatch: gt=" +
                std::to_string(gt.size()) + " pred=" +
                std::to_string(pred.size()));

        auto [mean, per_q] = compute_recall(gt, pred, K);

        if (verbose) {
            for (size_t i = 0; i < per_q.size(); ++i)
                std::cout << "Q" << i << "\t" << per_q[i] << "\n";
        }

        double mn = *std::min_element(per_q.begin(), per_q.end());
        double mx = *std::max_element(per_q.begin(), per_q.end());
        int perfect = 0;
        for (double r : per_q) if (r >= 1.0 - 1e-9) ++perfect;

        std::cout << "\n=== Recall@" << K << " ===\n";
        std::cout << "Queries    : " << per_q.size() << "\n";
        std::cout << "Mean       : " << mean * 100.0 << " %\n";
        std::cout << "Min        : " << mn   * 100.0 << " %\n";
        std::cout << "Max        : " << mx   * 100.0 << " %\n";
        std::cout << "Perfect    : " << perfect << " / " << per_q.size() << "\n";

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
    return 0;
}
