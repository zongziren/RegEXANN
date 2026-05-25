// include/GraphIndex.h
// Trigram-aware NSW graph index for regex-filtered ANN search.
//
// Core idea:
//   Each node stores a vector, a string, and its trigram set (as IDs).
//   Each edge stores the set of trigram IDs shared by its two endpoints.
//   The graph is built by merging per-trigram NSW sub-graphs:
//     for each trigram t, build an NSW among all nodes containing t,
//     then collapse multi-edges between the same pair of nodes into one
//     edge whose label = union of all shared trigrams.
//
// Query (for one conjunction group, i.e. one AND-of-trigrams):
//   1. Pick a random entry node that contains at least one query trigram.
//   2. Greedy NSW traversal: from the candidate set, expand only via
//      edges whose label intersects the query trigram set (at least one
//      common trigram).  Score = Euclidean distance to query vector.
//   3. Stop when no neighbor improves the best distance (local optimum).
//   4. Return the top-K nodes seen during traversal that satisfy the
//      full regex (verified by std::regex_search).
//
// For a full query (DNF = OR of conjunctions):
//   Run the above once per conjunction, merge results, deduplicate,
//   re-rank by exact distance, return top-K.
#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// ── Trigram vocabulary ────────────────────────────────────────────────────────
// Maps 3-char strings ↔ compact integer IDs.
struct TrigramVocab {
    std::unordered_map<std::string, int> gram_to_id;
    std::vector<std::string>             id_to_gram;

    // Insert gram if absent; return its ID.
    int intern(const std::string& gram);
    // Return ID or -1 if not present.
    int lookup(const std::string& gram) const;
    int size() const { return (int)id_to_gram.size(); }
};

// ── Graph structures ──────────────────────────────────────────────────────────

struct Edge {
    int  dst;                    // neighbour node ID
    std::vector<int> trigrams;   // shared trigram IDs (sorted)
};

struct Node {
    int               id;
    std::vector<float> vec;      // embedding vector
    std::string        str;      // original string
    std::vector<int>   trigrams; // trigram IDs of str (sorted)
    std::vector<Edge>  neighbors;
};

// ── Index ─────────────────────────────────────────────────────────────────────

struct GraphIndex {
    TrigramVocab              vocab;
    std::vector<Node>         nodes;

    // trigram_id → list of node IDs that contain it
    // Used to pick random entry points at query time.
    std::unordered_map<int, std::vector<int>> trigram_to_nodes;

    // Build parameters
    int M        = 16;  // max neighbours per node in NSW
    int ef_build = 64;  // candidate set size during construction
};

// ── Build ─────────────────────────────────────────────────────────────────────

// Construct the trigram-aware NSW graph from vectors + strings.
//   M        : max out-degree per node
//   ef_build : beam width during NSW insertion
GraphIndex build_graph_index(
    const std::vector<std::vector<float>>& vectors,
    const std::vector<std::string>&        strings,
    int M        = 16,
    int ef_build = 64);

// ── Query ─────────────────────────────────────────────────────────────────────

// One conjunction group: a set of trigram IDs that must ALL be present.
// Traverse the graph from a random entry node that contains at least one
// of these trigrams, following only edges labelled with at least one of them.
// Returns up to ef_search candidate node IDs (not yet regex-verified).
std::vector<int> search_conjunction(
    const GraphIndex&      idx,
    const std::vector<float>& query_vec,
    const std::vector<int>&   conj_trigrams,  // AND-group trigram IDs
    int                    ef_search,
    unsigned int           seed = 42);

// Full query: DNF = list of conjunctions (each a list of trigram IDs).
// For each conjunction, run search_conjunction, merge results,
// verify regex, re-rank by exact distance, return top-K node IDs.
std::vector<int> search_graph(
    const GraphIndex&                      idx,
    const std::vector<float>&              query_vec,
    const std::string&                     regex_str,
    const std::vector<std::vector<int>>&   dnf,   // OR of AND-groups
    int                                    K,
    int                                    ef_search = 64);
