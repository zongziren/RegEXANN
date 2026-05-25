// src/GraphIndex.cpp
// Trigram-aware NSW graph index.
//
// Build algorithm:
//   For each trigram t in the vocabulary:
//     Collect all nodes containing t → sub-graph nodes
//     Insert them one by one into a temporary NSW (greedy, no trigram filter)
//     For every edge (u,v) added, record t as a shared trigram on that edge
//   After processing all trigrams:
//     Multi-edges between the same pair → merge into one edge with union of
//     trigram labels, then prune each node's neighbour list to at most M entries
//     (keep the M neighbours with the smallest Euclidean distance).
//
// Query (one conjunction):
//   Entry: random node among those containing at least one query trigram.
//   Beam search with candidate set size ef_search:
//     - Only expand via edges whose trigram label ∩ query trigrams ≠ ∅
//     - Track visited nodes and a result priority queue
//     - Stop when the closest unvisited candidate is farther than the ef-th
//       result (standard NSW termination).
#include "GraphIndex.h"
#include "RegexParser.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <regex>
#include <unordered_set>

// ─────────────────────────────────────────────────────────────────────────────
// Distance helpers
// ─────────────────────────────────────────────────────────────────────────────

static float l2sq(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i) { float d=a[i]-b[i]; s+=d*d; }
    return s;
}
static float l2(const std::vector<float>& a, const std::vector<float>& b) {
    return std::sqrt(l2sq(a, b));
}

// ─────────────────────────────────────────────────────────────────────────────
// Adjacency helper: add or merge an edge (u ↔ v) with trigram `tgram_id`
// ─────────────────────────────────────────────────────────────────────────────

// Insert tgram_id into edge from `node` to `dst`, creating the edge if absent.
static void add_edge(Node& node, int dst, int tgram_id) {
    for (auto& e : node.neighbors) {
        if (e.dst == dst) {
            // Edge already exists: add trigram if not present
            auto it = std::lower_bound(e.trigrams.begin(),
                                       e.trigrams.end(), tgram_id);
            if (it == e.trigrams.end() || *it != tgram_id)
                e.trigrams.insert(it, tgram_id);
            return;
        }
    }
    // New edge
    node.neighbors.push_back({dst, {tgram_id}});
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-trigram NSW insertion
// For the sub-graph of nodes that share trigram `tgram_id`, insert node `u`
// using greedy nearest-neighbour search (ignoring trigram labels during
// build — we just need geometric proximity).  The best `M` neighbours found
// become bidirectional edges, all labelled with `tgram_id`.
// ─────────────────────────────────────────────────────────────────────────────

static void nsw_insert_node(
        GraphIndex& idx,
        int u,                      // node being inserted
        int tgram_id,               // trigram governing this sub-graph
        const std::vector<int>& sub_nodes,  // nodes already in sub-graph
        int M, int ef_build) {

    if (sub_nodes.empty()) return;

    const auto& qvec = idx.nodes[u].vec;

    // Use a max-heap of (dist, node_id) for candidates
    using DistId = std::pair<float, int>;
    // candidates: min-heap (closest first)
    std::priority_queue<DistId, std::vector<DistId>, std::greater<DistId>> cands;
    // results: max-heap (worst of best ef_build)
    std::priority_queue<DistId> results;
    std::unordered_set<int> visited;

    // Start from the last inserted node in this sub-graph
    int entry = sub_nodes.back();
    float d0 = l2(qvec, idx.nodes[entry].vec);
    cands.push({d0, entry});
    results.push({d0, entry});
    visited.insert(entry);

    while (!cands.empty()) {
        auto [cd, cur] = cands.top(); cands.pop();
        if ((int)results.size() >= ef_build &&
            cd > results.top().first) break;

        // Expand neighbours already in the sub-graph
        for (const auto& e : idx.nodes[cur].neighbors) {
            int nb = e.dst;
            if (visited.count(nb)) continue;
            // Only consider nodes in this sub-graph
            // (Check: nb contains tgram_id)
            const auto& nb_trigs = idx.nodes[nb].trigrams;
            if (!std::binary_search(nb_trigs.begin(), nb_trigs.end(), tgram_id))
                continue;
            visited.insert(nb);
            float nd = l2(qvec, idx.nodes[nb].vec);
            if ((int)results.size() < ef_build || nd < results.top().first) {
                cands.push({nd, nb});
                results.push({nd, nb});
                if ((int)results.size() > ef_build) results.pop();
            }
        }
    }

    // Collect best M neighbours
    std::vector<DistId> chosen;
    while (!results.empty()) {
        chosen.push_back(results.top());
        results.pop();
    }
    // Sort ascending by distance, keep at most M
    std::sort(chosen.begin(), chosen.end());
    if ((int)chosen.size() > M) chosen.resize(M);

    // Add bidirectional edges labelled with tgram_id
    for (auto& [d, v] : chosen) {
        add_edge(idx.nodes[u], v, tgram_id);
        add_edge(idx.nodes[v], u, tgram_id);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Prune neighbour list of every node to at most M entries
// Keep the M geometrically closest neighbours.
// ─────────────────────────────────────────────────────────────────────────────

static void prune_neighbors(GraphIndex& idx, int M) {
    for (auto& node : idx.nodes) {
        if ((int)node.neighbors.size() <= M) continue;
        // Sort by Euclidean distance to this node, keep M closest
        std::sort(node.neighbors.begin(), node.neighbors.end(),
                  [&](const Edge& a, const Edge& b) {
                      return l2sq(node.vec, idx.nodes[a.dst].vec) <
                             l2sq(node.vec, idx.nodes[b.dst].vec);
                  });
        node.neighbors.resize(M);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Build
// ─────────────────────────────────────────────────────────────────────────────

GraphIndex build_graph_index(
        const std::vector<std::vector<float>>& vectors,
        const std::vector<std::string>&        strings,
        int M,
        int ef_build) {

    assert(vectors.size() == strings.size());
    int N = (int)vectors.size();

    GraphIndex idx;
    idx.M        = M;
    idx.ef_build = ef_build;
    idx.nodes.resize(N);

    // ── Step 1: populate nodes and build trigram vocabulary ───────────────
    std::cout << "[Graph] Populating " << N << " nodes and trigram vocab…\n";
    for (int i = 0; i < N; ++i) {
        idx.nodes[i].id  = i;
        idx.nodes[i].vec = vectors[i];
        idx.nodes[i].str = strings[i];

        // Extract trigrams from the string (lower-case alpha only)
        std::string proc;
        for (unsigned char c : strings[i])
            if (std::isalpha(c)) proc += (char)std::tolower(c);

        std::set<int> tset;
        for (size_t j = 0; j + 2 < proc.size(); ++j) {
            std::string gram = proc.substr(j, 3);
            int tid = idx.vocab.intern(gram);
            tset.insert(tid);
        }
        idx.nodes[i].trigrams.assign(tset.begin(), tset.end()); // sorted
    }

    // ── Step 2: build trigram → node list ─────────────────────────────────
    for (int i = 0; i < N; ++i)
        for (int tid : idx.nodes[i].trigrams)
            idx.trigram_to_nodes[tid].push_back(i);

    // ── Step 3: for each trigram, run NSW insertion ───────────────────────
    int num_trigrams = idx.vocab.size();
    std::cout << "[Graph] Building NSW per trigram ("
              << num_trigrams << " trigrams, M=" << M
              << ", ef=" << ef_build << ")…\n";

    int progress_step = std::max(1, num_trigrams / 10);
    for (int tid = 0; tid < num_trigrams; ++tid) {
        if (tid % progress_step == 0)
            std::cout << "[Graph]   trigram " << tid << "/" << num_trigrams
                      << "  (" << idx.vocab.id_to_gram[tid] << ")\n";

        auto& sub = idx.trigram_to_nodes[tid];
        // Shuffle to randomise insertion order (affects graph quality)
        // Use a fixed seed for reproducibility
        std::shuffle(sub.begin(), sub.end(),
                     std::mt19937(tid * 1234567u));

        std::vector<int> inserted;
        inserted.reserve(sub.size());
        for (int u : sub) {
            nsw_insert_node(idx, u, tid, inserted, M, ef_build);
            inserted.push_back(u);
        }
    }

    // ── Step 4: prune to M neighbours per node ────────────────────────────
    std::cout << "[Graph] Pruning neighbour lists to M=" << M << "…\n";
    prune_neighbors(idx, M);

    // ── Stats ─────────────────────────────────────────────────────────────
    size_t total_edges = 0;
    for (const auto& n : idx.nodes) total_edges += n.neighbors.size();
    std::cout << "[Graph] Done. Nodes=" << N
              << "  Edges(directed)=" << total_edges
              << "  Trigrams=" << num_trigrams << "\n";

    return idx;
}

// ─────────────────────────────────────────────────────────────────────────────
// Query: one conjunction
// ─────────────────────────────────────────────────────────────────────────────

std::vector<int> search_conjunction(
        const GraphIndex&         idx,
        const std::vector<float>& query_vec,
        const std::vector<int>&   conj_trigrams,
        int                       ef_search,
        unsigned int              seed) {

    int N = (int)idx.nodes.size();
    if (N == 0) return {};

    // ── Choose entry node ─────────────────────────────────────────────────
    // Pick a random node that contains at least one query trigram.
    int entry = -1;
    if (conj_trigrams.empty()) {
        // Unconstrained: any node
        std::mt19937 rng(seed);
        entry = std::uniform_int_distribution<int>(0, N-1)(rng);
    } else {
        // Collect candidates across all conjunction trigrams, pick randomly
        std::vector<int> candidates;
        for (int tid : conj_trigrams) {
            auto it = idx.trigram_to_nodes.find(tid);
            if (it != idx.trigram_to_nodes.end()) {
                candidates.insert(candidates.end(),
                                  it->second.begin(), it->second.end());
            }
        }
        if (candidates.empty()) return {}; // trigram not in dataset

        std::mt19937 rng(seed);
        entry = candidates[std::uniform_int_distribution<int>(
                               0, (int)candidates.size()-1)(rng)];
    }

    // ── NSW beam search ───────────────────────────────────────────────────
    using DistId = std::pair<float, int>;
    // candidates (min-heap: pop closest first)
    std::priority_queue<DistId, std::vector<DistId>, std::greater<DistId>> cands;
    // results (max-heap: worst of best ef_search)
    std::priority_queue<DistId> results;
    std::unordered_set<int> visited;

    auto push = [&](int node_id) {
        if (visited.count(node_id)) return;
        visited.insert(node_id);
        float d = l2(query_vec, idx.nodes[node_id].vec);
        cands.push({d, node_id});
        results.push({d, node_id});
        if ((int)results.size() > ef_search) results.pop();
    };

    push(entry);

    while (!cands.empty()) {
        auto [cd, cur] = cands.top(); cands.pop();

        // NSW termination: current candidate is worse than worst in result set
        if ((int)results.size() >= ef_search && cd > results.top().first)
            break;

        // Expand neighbours — only via edges sharing at least one query trigram
        // (or all edges if conjunction is empty / unconstrained)
        for (const auto& edge : idx.nodes[cur].neighbors) {
            if (visited.count(edge.dst)) continue;

            // Edge filter: edge.trigrams ∩ conj_trigrams ≠ ∅
            if (!conj_trigrams.empty()) {
                bool shared = false;
                // Both sorted → linear intersection check
                size_t pi = 0, pj = 0;
                while (pi < edge.trigrams.size() &&
                       pj < conj_trigrams.size()) {
                    if (edge.trigrams[pi] == conj_trigrams[pj]) {
                        shared = true; break;
                    } else if (edge.trigrams[pi] < conj_trigrams[pj]) ++pi;
                    else ++pj;
                }
                if (!shared) continue;
            }

            push(edge.dst);
        }
    }

    // Drain result heap (ascending distance order)
    std::vector<int> out;
    out.reserve(results.size());
    while (!results.empty()) {
        out.push_back(results.top().second);
        results.pop();
    }
    std::reverse(out.begin(), out.end());
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Full query: DNF → merge → regex verify → top-K
// ─────────────────────────────────────────────────────────────────────────────

std::vector<int> search_graph(
        const GraphIndex&                    idx,
        const std::vector<float>&            query_vec,
        const std::string&                   regex_str,
        const std::vector<std::vector<int>>& dnf,
        int                                  K,
        int                                  ef_search) {

    // Compile regex once
    std::regex pattern;
    try { pattern = std::regex(regex_str, std::regex::icase); }
    catch (...) { return {}; }

    // Run one search per conjunction, union results
    std::unordered_set<int> seen;
    std::vector<int> all_candidates;

    for (size_t ci = 0; ci < dnf.size(); ++ci) {
        auto cands = search_conjunction(idx, query_vec, dnf[ci],
                                        ef_search,
                                        /*seed=*/(unsigned int)ci);
        for (int id : cands) {
            if (seen.insert(id).second)
                all_candidates.push_back(id);
        }
    }

    // Regex verify + exact distance re-ranking
    using DistId = std::pair<float, int>;
    std::priority_queue<DistId> heap; // max-heap, keep K smallest

    for (int id : all_candidates) {
        if (!std::regex_search(idx.nodes[id].str, pattern)) continue;
        float d = l2(query_vec, idx.nodes[id].vec);
        heap.push({d, id});
        if ((int)heap.size() > K) heap.pop();
    }

    std::vector<int> result;
    result.reserve(heap.size());
    while (!heap.empty()) { result.push_back(heap.top().second); heap.pop(); }
    std::reverse(result.begin(), result.end());
    return result;
}
