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

int main(int argc, char* argv[]) {
    if (argc < 8) {
        std::cerr << "Usage: ./ann_search <vector.fvecs> <string.txt> <query.txt> <K> <clusters> <output.txt> <max_iter>\n";
        return 1;
    }

    std::string vec_path = argv[1];
    std::string str_path = argv[2];
    std::string qry_path = argv[3];
    int K = std::stoi(argv[4]);
    int X = std::stoi(argv[5]);
    std::string out_path = argv[6];
    int max_iter = std::stoi(argv[7]);

    auto vectors = load_fvecs(vec_path);
    auto strings = load_strings(str_path);
    auto kmeans_result = run_kmeans(vectors, X, max_iter);

    std::unordered_map<std::string, std::set<int>> gram_index;
    build_3gram_index(strings, kmeans_result.assignments, gram_index);
    dump_gram_index(gram_index, "./debug/debug_gram_index.txt");

    std::ifstream qfin(qry_path);
    std::ofstream fout(out_path);
    std::string line;
    while (std::getline(qfin, line)) {
        
        //debug

        std::string regex_str;
        std::vector<float> query_vec;
        parse_query_line(line, regex_str, query_vec);
        auto result = perform_search(regex_str, query_vec, gram_index,
                                     kmeans_result.centroids,
                                     kmeans_result.clusters,
                                     vectors, strings, K);

        for (int id : result.top_ids) fout << id << " ";
        fout << "\n";
    }

    return 0;
}