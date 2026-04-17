// src/Serialize.cpp
#include "Serialize.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Low-level binary I/O helpers
// ─────────────────────────────────────────────────────────────────────────────

static void write_i32(std::ostream& out, int32_t v) {
    out.write(reinterpret_cast<const char*>(&v), 4);
}
static void write_f32(std::ostream& out, float v) {
    out.write(reinterpret_cast<const char*>(&v), 4);
}
static void write_u8(std::ostream& out, uint8_t v) {
    out.write(reinterpret_cast<const char*>(&v), 1);
}

static int32_t read_i32(std::istream& in) {
    int32_t v; in.read(reinterpret_cast<char*>(&v), 4); return v;
}
static float read_f32(std::istream& in) {
    float v; in.read(reinterpret_cast<char*>(&v), 4); return v;
}
static uint8_t read_u8(std::istream& in) {
    uint8_t v; in.read(reinterpret_cast<char*>(&v), 1); return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// KMeans
// ─────────────────────────────────────────────────────────────────────────────

void save_kmeans(const std::string& path, const KMeansResult& km, int dim) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_kmeans: cannot open " + path);
    int k = (int)km.centroids.size();
    int N = (int)km.assignments.size();
    write_i32(f, k);
    write_i32(f, dim);
    write_i32(f, N);
    // Centroids
    for (const auto& c : km.centroids)
        for (float v : c) write_f32(f, v);
    // Clusters
    for (const auto& cl : km.clusters) {
        write_i32(f, (int)cl.size());
        for (int id : cl) write_i32(f, id);
    }
    // Assignments
    for (int a : km.assignments) write_i32(f, a);
    std::cout << "[Serialize] Saved KMeans index → " << path << "\n";
}

KMeansResult load_kmeans(const std::string& path, int& dim_out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_kmeans: cannot open " + path);
    int k   = read_i32(f);
    int dim = read_i32(f);
    int N   = read_i32(f);
    dim_out = dim;
    KMeansResult km;
    km.centroids.resize(k, std::vector<float>(dim));
    for (auto& c : km.centroids) for (float& v : c) v = read_f32(f);
    km.clusters.resize(k);
    for (auto& cl : km.clusters) {
        int sz = read_i32(f); cl.resize(sz);
        for (int& id : cl) id = read_i32(f);
    }
    km.assignments.resize(N);
    for (int& a : km.assignments) a = read_i32(f);
    std::cout << "[Serialize] Loaded KMeans (k=" << k << ", dim=" << dim
              << ", N=" << N << ") from " << path << "\n";
    return km;
}

// ─────────────────────────────────────────────────────────────────────────────
// Trigram index
// ─────────────────────────────────────────────────────────────────────────────

void save_gram_index(const std::string& path,
                     const std::unordered_map<std::string, std::set<int>>& idx) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_gram_index: cannot open " + path);
    write_i32(f, (int)idx.size());
    for (const auto& [gram, clusters] : idx) {
        // Always 3-char gram
        write_i32(f, 3);
        f.write(gram.data(), 3);
        write_i32(f, (int)clusters.size());
        for (int cid : clusters) write_i32(f, cid);
    }
    std::cout << "[Serialize] Saved trigram index (" << idx.size()
              << " entries) → " << path << "\n";
}

void load_gram_index(const std::string& path,
                     std::unordered_map<std::string, std::set<int>>& idx) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_gram_index: cannot open " + path);
    int n = read_i32(f);
    idx.reserve(n * 2);
    for (int i = 0; i < n; ++i) {
        int glen = read_i32(f);
        std::string gram(glen, '\0');
        f.read(gram.data(), glen);
        int nc = read_i32(f);
        auto& s = idx[gram];
        for (int j = 0; j < nc; ++j) s.insert(read_i32(f));
    }
    std::cout << "[Serialize] Loaded trigram index (" << idx.size()
              << " entries) from " << path << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// PQ index
// ─────────────────────────────────────────────────────────────────────────────

void save_pq(const std::string& path, const PQIndex& pq) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_pq: cannot open " + path);
    write_i32(f, pq.m);
    write_i32(f, pq.k_sub);
    write_i32(f, pq.dim);
    write_i32(f, pq.sub_dim);
    // Codebooks
    for (const auto& subspace : pq.codebooks)
        for (const auto& centroid : subspace)
            for (float v : centroid) write_f32(f, v);
    // Codes
    int N = (int)pq.codes.size();
    write_i32(f, N);
    for (const auto& code : pq.codes)
        for (uint8_t c : code) write_u8(f, c);
    std::cout << "[Serialize] Saved PQ index (m=" << pq.m
              << ", k_sub=" << pq.k_sub << ", N=" << N
              << ") → " << path << "\n";
}

PQIndex load_pq(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_pq: cannot open " + path);
    PQIndex pq;
    pq.m       = read_i32(f);
    pq.k_sub   = read_i32(f);
    pq.dim     = read_i32(f);
    pq.sub_dim = read_i32(f);
    // Codebooks
    pq.codebooks.resize(pq.m, std::vector<std::vector<float>>(
        pq.k_sub, std::vector<float>(pq.sub_dim)));
    for (auto& subspace : pq.codebooks)
        for (auto& centroid : subspace)
            for (float& v : centroid) v = read_f32(f);
    // Codes
    int N = read_i32(f);
    pq.codes.resize(N, std::vector<uint8_t>(pq.m));
    for (auto& code : pq.codes)
        for (uint8_t& c : code) c = read_u8(f);
    std::cout << "[Serialize] Loaded PQ (m=" << pq.m
              << ", k_sub=" << pq.k_sub << ", N=" << N
              << ") from " << path << "\n";
    return pq;
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience wrappers
// ─────────────────────────────────────────────────────────────────────────────

void save_index(const std::string& prefix,
                const KMeansResult& km,
                const std::unordered_map<std::string, std::set<int>>& gram_index,
                const PQIndex& pq,
                int dim) {
    save_kmeans   (prefix + ".kmidx",   km, dim);
    save_gram_index(prefix + ".gramidx", gram_index);
    save_pq        (prefix + ".pqidx",   pq);
}

bool load_index(const std::string& prefix,
                KMeansResult& km,
                std::unordered_map<std::string, std::set<int>>& gram_index,
                PQIndex& pq) {
    try {
        int dim;
        km         = load_kmeans   (prefix + ".kmidx",   dim);
        load_gram_index(prefix + ".gramidx", gram_index);
        pq         = load_pq       (prefix + ".pqidx");
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Serialize] Load failed: " << e.what() << "\n";
        return false;
    }
}
