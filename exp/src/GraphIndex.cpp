// src/GraphIndex.cpp
// Hybrid Graph-Trigram index.  See include/GraphIndex.h for design rationale.
#include "GraphIndex.h"
#include "RegexParser.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <regex>
#include <unordered_set>

// ── Distance ──────────────────────────────────────────────────────────────────
static float l2sq(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0;
    for (size_t i = 0; i < a.size(); ++i) { float d = a[i]-b[i]; s += d*d; }
    return s;
}
static float l2(const std::vector<float>& a, const std::vector<float>& b) {
    return std::sqrt(l2sq(a, b));
}

// ── Trigram vocab ─────────────────────────────────────────────────────────────
int TrigramVocab::intern(const std::string& g) {
    auto it = gram_to_id.find(g);
    if (it != gram_to_id.end()) return it->second;
    int id = (int)id_to_gram.size();
    gram_to_id[g] = id;
    id_to_gram.push_back(g);
    return id;
}
int TrigramVocab::lookup(const std::string& g) const {
    auto it = gram_to_id.find(g);
    return it == gram_to_id.end() ? -1 : it->second;
}

// ── Sorted-list helpers ───────────────────────────────────────────────────────
static std::vector<int> sl_intersect(const std::vector<int>& a,
                                      const std::vector<int>& b) {
    std::vector<int> o; size_t i=0,j=0;
    while (i<a.size()&&j<b.size()) {
        if      (a[i]==b[j]) { o.push_back(a[i]); ++i; ++j; }
        else if (a[i]< b[j]) ++i;
        else                  ++j;
    }
    return o;
}
static std::vector<int> sl_union(const std::vector<int>& a,
                                  const std::vector<int>& b) {
    std::vector<int> o;
    std::set_union(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(o));
    return o;
}
static bool has_common(const std::vector<int>& a, const std::vector<int>& b) {
    size_t i=0,j=0;
    while (i<a.size()&&j<b.size()) {
        if      (a[i]==b[j]) return true;
        else if (a[i]< b[j]) ++i;
        else                  ++j;
    }
    return false;
}

// ── Edge helpers ──────────────────────────────────────────────────────────────
static void merge_edge(Node& u, int dst, const std::vector<int>& shared) {
    for (auto& e : u.neighbors) {
        if (e.dst == dst) {
            std::vector<int> m;
            std::set_union(e.trigrams.begin(),e.trigrams.end(),
                           shared.begin(),shared.end(),std::back_inserter(m));
            e.trigrams = std::move(m);
            return;
        }
    }
    u.neighbors.push_back({dst, shared});
}

// ── Prune to M neighbours ─────────────────────────────────────────────────────
static void prune_neighbors(GraphIndex& idx, int M) {
    for (auto& nd : idx.nodes) {
        if ((int)nd.neighbors.size() <= M) continue;
        std::sort(nd.neighbors.begin(), nd.neighbors.end(),
                  [&](const Edge& a, const Edge& b){
                      return l2sq(nd.vec,idx.nodes[a.dst].vec) <
                             l2sq(nd.vec,idx.nodes[b.dst].vec); });
        nd.neighbors.resize(M);
    }
}

// ── NSW backbone search (no candidate restriction) ───────────────────────────
static std::vector<std::pair<float,int>> nsw_all(
        const GraphIndex& idx,
        const std::vector<float>& qvec,
        int entry, int ef)
{
    using DI = std::pair<float,int>;
    std::priority_queue<DI,std::vector<DI>,std::greater<DI>> cands;
    std::priority_queue<DI> res;
    std::unordered_set<int> vis;
    auto push = [&](int v) {
        if (vis.count(v)) return; vis.insert(v);
        float d = l2(qvec, idx.nodes[v].vec);
        cands.push({d,v}); res.push({d,v});
        if ((int)res.size()>ef) res.pop();
    };
    push(entry);
    while (!cands.empty()) {
        auto [cd,cur] = cands.top(); cands.pop();
        if ((int)res.size()>=ef && cd>res.top().first) break;
        for (const auto& e : idx.nodes[cur].neighbors) push(e.dst);
    }
    std::vector<DI> out;
    while (!res.empty()) { out.push_back(res.top()); res.pop(); }
    return out;
}

// ── NSW search restricted to a candidate set ─────────────────────────────────
// Only visits nodes in `allowed`; follows all edges (backbone ensures path).
static std::vector<std::pair<float,int>> nsw_restricted(
        const GraphIndex& idx,
        const std::vector<float>& qvec,
        const std::unordered_set<int>& allowed,
        int ef)
{
    if (allowed.empty()) return {};

    // Pick best of a few random entries from allowed set
    std::vector<int> av(allowed.begin(), allowed.end());
    std::mt19937 rng(42);
    int entry = av[0];
    float bd = l2sq(qvec, idx.nodes[entry].vec);
    for (int t = 0; t < std::min((int)av.size(), 8); ++t) {
        int c = av[std::uniform_int_distribution<int>(0,(int)av.size()-1)(rng)];
        float d = l2sq(qvec, idx.nodes[c].vec);
        if (d < bd) { bd=d; entry=c; }
    }

    using DI = std::pair<float,int>;
    std::priority_queue<DI,std::vector<DI>,std::greater<DI>> cands;
    std::priority_queue<DI> res;
    std::unordered_set<int> vis;

    auto push = [&](int v) {
        if (!allowed.count(v)) return; // restrict to candidate set
        if (vis.count(v)) return; vis.insert(v);
        float d = l2(qvec, idx.nodes[v].vec);
        cands.push({d,v}); res.push({d,v});
        if ((int)res.size()>ef) res.pop();
    };
    push(entry);

    while (!cands.empty()) {
        auto [cd,cur] = cands.top(); cands.pop();
        if ((int)res.size()>=ef && cd>res.top().first) break;
        // Follow ALL edges — even non-allowed neighbours help bridge gaps
        for (const auto& e : idx.nodes[cur].neighbors) push(e.dst);
    }

    std::vector<DI> out;
    while (!res.empty()) { out.push_back(res.top()); res.pop(); }
    return out;
}

// ── Build ─────────────────────────────────────────────────────────────────────
GraphIndex build_graph_index(
        const std::vector<std::vector<float>>& vectors,
        const std::vector<std::string>&        strings,
        int M, int ef_build)
{
    int N = (int)vectors.size();
    assert(N == (int)strings.size());

    GraphIndex idx;
    idx.M = M; idx.ef_build = ef_build;
    idx.nodes.resize(N);

    // ── Step 1: populate nodes + vocab ───────────────────────────────────
    std::cout << "[Graph] Populating " << N << " nodes…\n" << std::flush;
    for (int i = 0; i < N; ++i) {
        idx.nodes[i] = {i, vectors[i], strings[i], {}, {}};
        std::string proc;
        for (unsigned char c : strings[i])
            if (std::isalpha(c)) proc += (char)std::tolower(c);
        std::set<int> ts;
        for (size_t j = 0; j+2 < proc.size(); ++j)
            ts.insert(idx.vocab.intern(proc.substr(j,3)));
        idx.nodes[i].trigrams.assign(ts.begin(), ts.end());
    }

    // ── Step 2: build node-level inverted index ───────────────────────────
    for (int i = 0; i < N; ++i)
        for (int tid : idx.nodes[i].trigrams)
            idx.trigram_to_nodes[tid].push_back(i);
    // Sort posting lists for binary search later
    for (auto& [tid, lst] : idx.trigram_to_nodes)
        std::sort(lst.begin(), lst.end());

    int T = idx.vocab.size();
    float ap = T > 0 ? (float)N/T : 0;
    std::cout << "[Graph] " << T << " trigrams. Avg posting="
              << ap << " nodes/trigram.\n" << std::flush;
    std::cout << "[Graph] Building NSW backbone (M=" << M
              << ", ef=" << ef_build << ")…\n" << std::flush;

    // ── Step 3: NSW insertion (backbone — all nodes, no trigram filter) ───
    std::mt19937 rng(42);
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    int step = std::max(1, N/20);
    for (int i = 0; i < N; ++i) {
        int u = order[i];
        if (i % step == 0)
            std::cout << "[Graph]   " << i << "/" << N
                      << " (" << i*100/N << "%)\n" << std::flush;

        if (i == 0) continue; // first node has no neighbours yet

        const auto& uvec  = idx.nodes[u].vec;
        const auto& utrig = idx.nodes[u].trigrams;

        // Pick best entry from a few random already-inserted nodes
        int entry = order[0];
        float bd = l2sq(uvec, idx.nodes[entry].vec);
        for (int t = 0; t < std::min(i, 5); ++t) {
            int c = order[std::uniform_int_distribution<int>(0,i-1)(rng)];
            float d = l2sq(uvec, idx.nodes[c].vec);
            if (d < bd) { bd=d; entry=c; }
        }

        // NSW search among all inserted nodes
        auto res = nsw_all(idx, uvec, entry, M);
        std::sort(res.begin(), res.end());
        if ((int)res.size() > M) res.resize(M);

        for (auto& [d,v] : res) {
            auto shared = sl_intersect(utrig, idx.nodes[v].trigrams);
            // shared may be empty for pure backbone edges — that is fine
            merge_edge(idx.nodes[u], v, shared);
            merge_edge(idx.nodes[v], u, shared);
        }
    }
    std::cout << "[Graph]   " << N << "/" << N << " (100%)\n" << std::flush;

    std::cout << "[Graph] Pruning to M=" << M << "…\n" << std::flush;
    prune_neighbors(idx, M);

    size_t te = 0;
    for (const auto& n : idx.nodes) te += n.neighbors.size();
    std::cout << "[Graph] Done. Nodes=" << N << "  Edges=" << te
              << "  Trigrams=" << T << "\n" << std::flush;
    return idx;
}

// ── Step 2: evaluate DNF on node-level inverted index ────────────────────────
std::vector<int> trigram_filter_nodes(
        const GraphIndex&                    idx,
        const std::vector<std::vector<int>>& dnf)
{
    int N = (int)idx.nodes.size();
    // All-nodes sentinel
    auto all_nodes = [&]() -> std::vector<int> {
        std::vector<int> v(N); std::iota(v.begin(), v.end(), 0); return v;
    };

    if (dnf.empty()) return all_nodes();

    std::vector<int> result; // union across conjunctions
    bool result_init = false;

    for (const auto& conj : dnf) {
        if (conj.empty()) return all_nodes(); // unconstrained conjunction

        // Intersect posting lists for all trigrams in this conjunction
        std::vector<int> conj_result;
        bool first = true;
        bool impossible = false;

        for (int tid : conj) {
            auto it = idx.trigram_to_nodes.find(tid);
            if (it == idx.trigram_to_nodes.end()) {
                impossible = true; break; // trigram absent → no match
            }
            if (first) {
                conj_result = it->second; first = false;
            } else {
                conj_result = sl_intersect(conj_result, it->second);
            }
            if (conj_result.empty()) { impossible = true; break; }
        }

        if (impossible || conj_result.empty()) continue;

        // Union into overall result
        if (!result_init) {
            result = conj_result; result_init = true;
        } else {
            result = sl_union(result, conj_result);
        }
    }

    return result;
}

// ── Step 3+4: graph-guided K-NN within candidate set ─────────────────────────

// Threshold: if |candidates| ≤ this, do exhaustive exact scan (always optimal)
static const int EXACT_THRESH = 500;

std::vector<int> search_graph(
        const GraphIndex&        idx,
        const std::vector<float>& query_vec,
        const std::string&        regex_str,
        const std::vector<int>&   candidates,
        int K, int ef_search)
{
    std::regex pattern;
    try { pattern = std::regex(regex_str, std::regex::icase); }
    catch (...) { return {}; }

    using DI = std::pair<float,int>;
    std::priority_queue<DI> heap; // max-heap, keep K best

    if (candidates.empty()) return {};

    if ((int)candidates.size() <= EXACT_THRESH) {
        // ── Exact scan: small candidate set ──────────────────────────────
        for (int id : candidates) {
            if (!std::regex_search(idx.nodes[id].str, pattern)) continue;
            float d = l2(query_vec, idx.nodes[id].vec);
            heap.push({d, id});
            if ((int)heap.size() > K) heap.pop();
        }
    } else {
        // ── Graph-guided search within candidate set ──────────────────────
        std::unordered_set<int> allowed(candidates.begin(), candidates.end());
        auto found = nsw_restricted(idx, query_vec, allowed, ef_search);

        for (auto& [d, id] : found) {
            if (!std::regex_search(idx.nodes[id].str, pattern)) continue;
            heap.push({d, id});
            if ((int)heap.size() > K) heap.pop();
        }

        // If graph didn't find enough, fall back to exact scan on remainder
        if ((int)heap.size() < K) {
            std::unordered_set<int> already_seen;
            for (auto& [d,id] : found) already_seen.insert(id);
            for (int id : candidates) {
                if (already_seen.count(id)) continue;
                if (!std::regex_search(idx.nodes[id].str, pattern)) continue;
                float d = l2(query_vec, idx.nodes[id].vec);
                heap.push({d, id});
                if ((int)heap.size() > K) heap.pop();
            }
        }
    }

    std::vector<int> result;
    while (!heap.empty()) { result.push_back(heap.top().second); heap.pop(); }
    std::reverse(result.begin(), result.end());
    return result;
}

// ── Full pipeline convenience function ───────────────────────────────────────
std::vector<int> search_graph_full(
        const GraphIndex&                    idx,
        const std::vector<float>&            query_vec,
        const std::string&                   regex_str,
        const std::vector<std::vector<int>>& dnf,
        int K, int ef_search)
{
    auto candidates = trigram_filter_nodes(idx, dnf);
    return search_graph(idx, query_vec, regex_str, candidates, K, ef_search);
}

// ── Legacy search_conjunction (kept for compatibility) ────────────────────────
std::vector<int> search_conjunction(
        const GraphIndex&         idx,
        const std::vector<float>& query_vec,
        const std::vector<int>&   conj_trigrams,
        int ef_search, unsigned int /*seed*/)
{
    std::vector<std::vector<int>> dnf = {conj_trigrams};
    auto candidates = trigram_filter_nodes(idx, dnf);
    std::unordered_set<int> allowed(candidates.begin(), candidates.end());
    if (allowed.empty()) return {};
    auto res = nsw_restricted(idx, query_vec, allowed, ef_search);
    std::vector<int> out;
    for (auto& [d,v] : res) out.push_back(v);
    return out;
}
