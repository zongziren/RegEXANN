// include/Serialize.h
// Save and load the RegExANN index structures to/from binary files.
//
// File format (each structure written sequentially):
//
//  KMeans index file (.kmidx):
//    [int32] k                        number of clusters
//    [int32] dim                      vector dimension
//    [int32] N                        number of data vectors
//    [float32 * k * dim]              centroids
//    for each cluster c in 0..k-1:
//      [int32] size                   cluster size
//      [int32 * size]                 vector IDs
//    [int32 * N]                      per-vector cluster assignment
//
//  Trigram index file (.gramidx):
//    [int32] num_entries
//    for each entry:
//      [int32]  gram_len              always 3
//      [char*3]                       the trigram bytes
//      [int32]  num_clusters
//      [int32 * num_clusters]         cluster IDs
//
//  PQ index file (.pqidx):
//    [int32] m                        subspaces
//    [int32] k_sub                    centroids per subspace
//    [int32] dim
//    [int32] sub_dim
//    [float32 * m * k_sub * sub_dim]  codebooks
//    [int32] N                        number of encoded vectors
//    [uint8 * N * m]                  codes
#pragma once
#include <string>
#include <set>
#include <unordered_map>
#include <vector>

#include "KMeans.h"
#include "PQ.h"

// ── KMeans index ──────────────────────────────────────────────────────────────
void save_kmeans(const std::string& path, const KMeansResult& km, int dim);
KMeansResult load_kmeans(const std::string& path, int& dim_out);

// ── Trigram index ─────────────────────────────────────────────────────────────
void save_gram_index(const std::string& path,
                     const std::unordered_map<std::string, std::set<int>>& idx);
void load_gram_index(const std::string& path,
                     std::unordered_map<std::string, std::set<int>>& idx);

// ── PQ index ──────────────────────────────────────────────────────────────────
void save_pq(const std::string& path, const PQIndex& pq);
PQIndex load_pq(const std::string& path);

// ── Convenience: save/load the complete RegExANN flat index ──────────────────
// Writes three files: <prefix>.kmidx, <prefix>.gramidx, <prefix>.pqidx
void save_index(const std::string& prefix,
                const KMeansResult& km,
                const std::unordered_map<std::string, std::set<int>>& gram_index,
                const PQIndex& pq,
                int dim);

bool load_index(const std::string& prefix,
                KMeansResult& km,
                std::unordered_map<std::string, std::set<int>>& gram_index,
                PQIndex& pq);
