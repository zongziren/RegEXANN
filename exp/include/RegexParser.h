// include/RegexParser.h
// Convert a regular expression into a DNF (Disjunctive Normal Form) of
// trigram sets, suitable for driving the graph-index query.
//
// Output form:
//   DNF = [ conj_0, conj_1, … ]       (OR of conjunctions)
//   conj_i = [ trigram_id_a, trigram_id_b, … ]   (AND of trigrams)
//
// Derivation rules (same as Section 3.1.2 of the RegExANN paper, but now
// expressed in DNF so every conjunction can drive one graph traversal):
//
//   Q(literal)  = { all trigrams of literal }       → one conjunction
//   Q(r1 · r2)  = DNF_and(Q(r1), Q(r2))             → cross-product AND
//   Q(r1 | r2)  = Q(r1) ++ Q(r2)                    → concatenate conjunctions
//   Q(r*)       = [ [] ]   (empty conjunction = unconstrained)
//   Q([…])      = [ [] ]
//   Q(.)        = [ [] ]
//
// DNF_and(A, B):
//   for each (ca, cb) in A × B: emit ca ∪ cb  (merge the two AND-sets)
//   then deduplicate and prune dominated conjunctions
//   (a conjunction C dominates D if C ⊆ D, because C is stricter).
//
// Empty conjunction [] means "no trigram constraint" — the graph traversal
// starts from a random node and follows any edge.  If the entire DNF reduces
// to [[]],  fall back to plain ANN without trigram guidance.
#pragma once

#include <string>
#include <vector>
#include "GraphIndex.h"   // for TrigramVocab

// Convert regex to a DNF of trigram ID lists.
// `vocab` is used (and extended) to assign IDs to new trigrams.
// Returns the DNF; an empty inner vector means "unconstrained".
std::vector<std::vector<int>> regex_to_dnf(
    const std::string& regex,
    TrigramVocab&      vocab);

// Simplify a DNF:
//   - remove duplicate conjunctions
//   - remove conjunctions that are supersets of another (dominated)
//   After simplification, fewer graph traversals are needed.
std::vector<std::vector<int>> simplify_dnf(
    std::vector<std::vector<int>> dnf);
