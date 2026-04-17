// include/Index.h
#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <set>

// ── String preprocessing ─────────────────────────────────────────────────────

// Lower-case and keep only alpha + space characters.
std::string preprocess_string(const std::string& input);

// Extract all overlapping length-3 substrings from preprocessed `input`.
std::set<std::string> extract_3grams(const std::string& input);

// ── Index construction ────────────────────────────────────────────────────────

// Build a trigram → cluster-ID inverted index from the dataset strings.
// strings[i] is the text associated with vector i; cluster_ids[i] is its cluster.
// For each trigram g found in strings[i], gram_index[g] gets cluster_ids[i] added.
void build_3gram_index(const std::vector<std::string>& strings,
                       const std::vector<int>& cluster_ids,
                       std::unordered_map<std::string, std::set<int>>& gram_index);

// ── Regex-aware candidate cluster selection ───────────────────────────────────

// Parse a regex pattern and return the set of cluster IDs that *could* contain
// a matching string, using the trigram inverted index.
//
// The parser handles:
//   • Literals         → their trigrams are ANDed (intersection)
//   • Alternation  `|` → results are ORed  (union) at the top level and
//                        inside parenthesised groups
//   • Grouping     `(` `)` → recursive evaluation
//   • Wildcards/specials (`.` `*` `+` `?` `[`…`]` `{`…`}` `\`) →
//                        treated as unconstrained (contribute ALL clusters)
//
// If `num_clusters` == 0 the function returns an empty set with `all_flag`
// set to true (caller interprets as "all clusters").  If the regex produces
// no usable literals, the full cluster range [0, num_clusters) is returned.
//
// `num_clusters` is the total number of clusters (size of centroids vector).
std::set<int> get_candidate_clusters(
    const std::string& regex,
    const std::unordered_map<std::string, std::set<int>>& gram_index,
    int num_clusters);
