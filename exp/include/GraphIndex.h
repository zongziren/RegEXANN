// include/GraphIndex.h
// Hybrid Graph-Trigram index for regex-filtered ANN search.
//
// Design (solving the sparse-trigram recall problem):
//
//   The core insight: the two tasks are fundamentally different:
//     - String filtering: "does this string match the regex?"
//       → Solved by trigram inverted index at NODE level (not cluster level).
//         Any matching string MUST contain certain trigrams.
//         The index gives exact candidate node sets, not approximations.
//     - Vector ANN: "which candidate is closest to query vector?"
//       → Solved by NSW graph navigation within the candidate set.
//
//   Index structure:
//     1. NSW graph on all N nodes (backbone for global connectivity).
//        Edge label = sorted intersection of the two endpoints' trigram sets.
//     2. Node-level trigram inverted index: gram → sorted list of node IDs.
//        (This differs from ann's cluster-level index — here it maps to
//         individual nodes, so regex filtering is exact with no false positives
//         at the candidate selection stage.)
//
//   Query pipeline:
//     Step 1 — Regex → DNF (same recursive-descent parser as RegexParser.cpp)
//     Step 2 — DNF evaluated on node-level trigram index:
//                 ∩ per conjunction, ∪ across conjunctions
//               → C_regex = set of node IDs that MUST contain matching strings
//               (may still have false positives from trigram approximation,
//                but far fewer than cluster-level filtering)
//     Step 3 — If |C_regex| ≤ EXACT_THRESH: exact K-NN within C_regex
//              Else: graph-guided search restricted to C_regex
//     Step 4 — Regex verify remaining candidates, return top-K by exact dist.
//
//   Why this works on sparse-trigram datasets:
//     The node-level index guarantees we only look at nodes whose strings
//     contain the required trigrams — even if each trigram is rare (appears
//     in 1-2 nodes), the intersection of posting lists is still exact.
//     The graph then efficiently finds the nearest among those candidates.
#pragma once

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// ── Trigram vocabulary ────────────────────────────────────────────────────────
struct TrigramVocab {
    std::unordered_map<std::string, int> gram_to_id;
    std::vector<std::string>             id_to_gram;
    int intern(const std::string& gram);
    int lookup(const std::string& gram) const;
    int size() const { return (int)id_to_gram.size(); }
};

// ── Graph node / edge ─────────────────────────────────────────────────────────
struct Edge {
    int              dst;
    std::vector<int> trigrams; // sorted shared trigram IDs (may be empty = backbone)
};

struct Node {
    int                id;
    std::vector<float> vec;
    std::string        str;
    std::vector<int>   trigrams; // sorted trigram IDs of str
    std::vector<Edge>  neighbors;
};

// ── Hybrid index ──────────────────────────────────────────────────────────────
struct GraphIndex {
    TrigramVocab   vocab;
    std::vector<Node> nodes;

    // Node-level inverted index: trigram_id → sorted list of node IDs
    std::unordered_map<int, std::vector<int>> trigram_to_nodes;

    int M        = 16;
    int ef_build = 64;
};

// ── Build ─────────────────────────────────────────────────────────────────────
GraphIndex build_graph_index(
    const std::vector<std::vector<float>>& vectors,
    const std::vector<std::string>&        strings,
    int M        = 16,
    int ef_build = 64);

// ── Query ─────────────────────────────────────────────────────────────────────

// Step 2: evaluate DNF on node-level inverted index.
// Returns sorted list of candidate node IDs.
// Each element of dnf is an AND-group (conjunction of trigram IDs).
// Empty inner vector means unconstrained (return all nodes).
std::vector<int> trigram_filter_nodes(
    const GraphIndex&                    idx,
    const std::vector<std::vector<int>>& dnf);

// Step 3+4: find top-K nearest among candidates that match regex.
// Uses graph navigation if |candidates| > EXACT_THRESH, else exact scan.
std::vector<int> search_graph(
    const GraphIndex&        idx,
    const std::vector<float>& query_vec,
    const std::string&        regex_str,
    const std::vector<int>&   candidates, // from trigram_filter_nodes
    int K,
    int ef_search = 64);

// Convenience: full pipeline (trigram filter + graph search).
std::vector<int> search_graph_full(
    const GraphIndex&                    idx,
    const std::vector<float>&            query_vec,
    const std::string&                   regex_str,
    const std::vector<std::vector<int>>& dnf,
    int K,
    int ef_search = 64);

// Legacy signatures kept for RegexParser compatibility
std::vector<int> search_conjunction(
    const GraphIndex&        idx,
    const std::vector<float>& query_vec,
    const std::vector<int>&   conj_trigrams,
    int ef_search,
    unsigned int seed = 42);
