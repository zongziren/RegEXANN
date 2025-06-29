// src/main.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <regex>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <chrono>
#include <queue>

#include "KMeans.h"
#include "Index.h"
#include "Search.h"

std::vector<std::vector<float>> load_fvecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    std::vector<std::vector<float>> vectors;
    if (!file) throw std::runtime_error("Cannot open fvec file.");
    while (file.peek() != EOF) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), sizeof(float) * dim);
        vectors.push_back(vec);
    }
    return vectors;
}

std::vector<std::string> load_strings(const std::string& filename) {
    std::ifstream fin(filename);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(fin, line)) lines.push_back(line);
    return lines;
}

void parse_query_line(const std::string& line, std::string& regex_out, std::vector<float>& vector_out) {
    std::istringstream iss(line);
    iss >> regex_out;
    float val;
    while (iss >> val) vector_out.push_back(val);
}

void dump_gram_index(const std::unordered_map<std::string, std::set<int>>& gram_index, const std::string& filename) {
    std::ofstream fout(filename);
    for (const auto& [gram, clusters] : gram_index) {
        fout << gram << ": ";
        for (int cid : clusters)
            fout << cid << " ";
        fout << "\n";
    }
    fout.close();
    std::cerr << "[DEBUG] Dumped 3-gram index to " << filename << "\n";
}

std::vector<int> run_baseline(const std::string& regex_str,
                              const std::vector<float>& query_vec,
                              const std::vector<std::vector<float>>& all_vectors,
                              const std::vector<std::string>& all_strings,
                              int K) {
    std::regex pattern(regex_str, std::regex::icase);  // ← 不区分大小写
    std::priority_queue<std::pair<float, int>> pq;

    for (int i = 0; i < (int)all_vectors.size(); ++i) {
        if (!std::regex_search(all_strings[i], pattern)) continue;
        float dist = 0;
        for (size_t j = 0; j < query_vec.size(); ++j)
            dist += (query_vec[j] - all_vectors[i][j]) * (query_vec[j] - all_vectors[i][j]);
        pq.emplace(std::sqrt(dist), i);
        if ((int)pq.size() > K) pq.pop();
    }

    std::vector<int> result;
    while (!pq.empty()) {
        result.push_back(pq.top().second);
        pq.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 9) {
        std::cerr << "Usage: ./ann_search <vector.fvecs> <string.txt> <query.txt> <K> <clusters> <output.txt> <max_iter> <alg=ann|baseline>\n";
        return 1;
    }

    std::string vec_path = argv[1];
    std::string str_path = argv[2];
    std::string qry_path = argv[3];
    int K = std::stoi(argv[4]);
    int X = std::stoi(argv[5]);
    std::string out_path = argv[6];
    int max_iter = std::stoi(argv[7]);
    std::string algorithm = argv[8];

    auto vectors = load_fvecs(vec_path);
    auto strings = load_strings(str_path);

    using namespace std::chrono;
    std::ifstream qfin(qry_path);
    std::ofstream fout(out_path);
    std::string line;
    int query_count = 0;
    double total_query_time_ms = 0.0;

    if (algorithm == "ann") {
        std::cout << "[INFO] Start indexing...\n";
        auto t1 = high_resolution_clock::now();

        auto kmeans_result = run_kmeans(vectors, X, max_iter);
        dump_cluster_assignments(kmeans_result.clusters, "./debug/cluster_assignments.txt");

        std::unordered_map<std::string, std::set<int>> gram_index;
        build_3gram_index(strings, kmeans_result.assignments, gram_index);
        dump_gram_index(gram_index, "./debug/debug_gram_index.txt");

        auto t2 = high_resolution_clock::now();
        auto indexing_duration = duration_cast<milliseconds>(t2 - t1).count();
        std::cout << "[INFO] Indexing completed in " << indexing_duration << " ms.\n";

        while (std::getline(qfin, line)) {
            std::string regex_str;
            std::vector<float> query_vec;
            parse_query_line(line, regex_str, query_vec);

            auto q_start = high_resolution_clock::now();

            auto result = perform_search(regex_str, query_vec, gram_index,
                                         kmeans_result.centroids,
                                         kmeans_result.clusters,
                                         vectors, strings, K);

            auto q_end = high_resolution_clock::now();
            total_query_time_ms += duration_cast<microseconds>(q_end - q_start).count() / 1000.0;
            query_count++;

            for (int id : result.top_ids) fout << id << " ";
            fout << "\n";
        }
    } else if (algorithm == "baseline") {
        std::cout << "[INFO] Running baseline full scan...\n";
        while (std::getline(qfin, line)) {
            std::string regex_str;
            std::vector<float> query_vec;
            parse_query_line(line, regex_str, query_vec);

            auto q_start = high_resolution_clock::now();
            auto result = run_baseline(regex_str, query_vec, vectors, strings, K);
            auto q_end = high_resolution_clock::now();
            total_query_time_ms += duration_cast<microseconds>(q_end - q_start).count() / 1000.0;
            query_count++;

            for (int id : result) fout << id << " ";
            fout << "\n";
        }
    } else {
        std::cerr << "[ERROR] Unknown algorithm: " << algorithm << "\n";
        return 1;
    }

    std::cout << "[INFO] Average query time: "
              << (total_query_time_ms / query_count) << " ms over "
              << query_count << " queries.\n";

    return 0;
}
