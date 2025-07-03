// src/Search.cpp
#include "Search.h"
#include "Index.h"
#include <regex>
#include <cmath>
#include <queue>
#include <iostream>
#include <fstream>
#include <chrono>

std::string to_lower(const std::string &s)
{
    std::string res = s;
    std::transform(res.begin(), res.end(), res.begin(),
                   [](unsigned char c)
                   { return std::tolower(c); });
    return res;
}

float euclidean_distance(const std::vector<float> &a, const std::vector<float> &b)
{
    float sum = 0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(sum);
}

void dump_query_cluster_debug(const std::string &regex_str,
                              const std::unordered_map<std::string, std::set<int>> &gram_index,
                              const std::vector<std::pair<float, int>> &cluster_dists,
                              const std::string &filename)
{
    std::ofstream fout(filename);
    auto grams = extract_3grams(regex_str);

    fout << "[DEBUG] Extracted 3-grams from regex: ";
    for (const auto &g : grams)
        fout << g << " ";
    fout << "\n";

    std::set<int> all_candidates;
    for (const auto &gram : grams)
    {
        fout << "[DEBUG] gram: " << gram << " → ";
        if (gram_index.count(gram))
        {
            for (int cid : gram_index.at(gram))
            {
                fout << cid << " ";
                all_candidates.insert(cid);
            }
        }
        else
        {
            fout << "not found";
        }
        fout << "\n";
    }

    fout << "[DEBUG] Total candidate clusters (unordered): ";
    for (int cid : all_candidates)
        fout << cid << " ";
    fout << "\n";

    fout << "[DEBUG] Candidate clusters sorted by distance:\n";
    for (const auto &[dist, cid] : cluster_dists)
        fout << "cid = " << cid << ", dist = " << dist << "\n";

    fout.close();
}

SearchResult perform_search(const std::string &regex_str,
                            const std::vector<float> &query_vector,
                            const std::unordered_map<std::string, std::set<int>> &gram_index,
                            const std::vector<std::vector<float>> &centroids,
                            const std::vector<std::vector<int>> &clusters,
                            const std::vector<std::vector<float>> &all_vectors,
                            const std::vector<std::string> &all_strings,
                            int K)
{
    using namespace std::chrono;
    auto start_setop = high_resolution_clock::now();
    std::set<int> candidate_clusters;
    auto grams = extract_3grams(regex_str);

    bool first = true;
    for (const auto &gram : grams)
    {
        if (!gram_index.count(gram))
        {
            candidate_clusters.clear();
            break;
        }

        const auto &gram_clusters = gram_index.at(gram);
        if (first)
        {
            candidate_clusters = std::set<int>(gram_clusters.begin(), gram_clusters.end());
            first = false;
        }
        else
        {
            std::set<int> temp;
            std::set_intersection(candidate_clusters.begin(), candidate_clusters.end(),
                                  gram_clusters.begin(), gram_clusters.end(),
                                  std::inserter(temp, temp.begin()));
            candidate_clusters = std::move(temp);
        }
    }
    auto end_setop = high_resolution_clock::now();
    double setop_ms = duration_cast<microseconds>(end_setop - start_setop).count() / 1000.0;

    auto start_query = high_resolution_clock::now();

    std::vector<std::pair<float, int>> cluster_dists;
    for (int cid : candidate_clusters)
        cluster_dists.emplace_back(euclidean_distance(query_vector, centroids[cid]), cid);
    std::sort(cluster_dists.begin(), cluster_dists.end());

    //dump_query_cluster_debug(regex_str, gram_index, cluster_dists, "./debug/query_debug.txt");

    std::priority_queue<std::pair<float, int>> pq;
    std::regex pattern(regex_str, std::regex::icase);

    for (const auto &[_, cid] : cluster_dists)
    {
        for (int vid : clusters[cid])
        {
            if (!std::regex_search(all_strings[vid], pattern))
                continue;
            float dist = euclidean_distance(query_vector, all_vectors[vid]);
            pq.emplace(dist, vid);
            if ((int)pq.size() > K)
                pq.pop();
        }
        if ((int)pq.size() >= K)
            break;
    }

    
    auto end_query = high_resolution_clock::now();
    double query_ms = duration_cast<microseconds>(end_query - start_query).count() / 1000.0;

    std::vector<int> result;
    while (!pq.empty())
    {
        result.push_back(pq.top().second);
        pq.pop();
    }
    std::reverse(result.begin(), result.end());
    
    return SearchResult{result, setop_ms, query_ms};
}
