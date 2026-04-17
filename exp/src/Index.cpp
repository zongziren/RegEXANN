// src/Index.cpp
// Trigram index construction and regex-aware cluster candidate selection.
//
// Regex Boolean Evaluation (Section 3.1.2 of the paper):
//   Q(r1 · r2) = Q(r1) ∩ Q(r2)   concatenation → intersection
//   Q(r1 | r2) = Q(r1) ∪ Q(r2)   alternation   → union
//   Q(r*)      = ALL              kleene star    → unconstrained
//   Q(r?)      = ALL              optional       → unconstrained
//   Q([…])     = ALL              char class     → unconstrained
//   Q(.)       = ALL              any char       → unconstrained
//
// Implementation: recursive-descent parser.
//   eval_expr   → alternation  (|)
//   eval_concat → concatenation (sequence of atoms, intersected)
//   eval_atom   → literal | (group) | wildcard | char-class | escape
#include "Index.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <numeric>

// ── Preprocessing (for index construction) ────────────────────────────────

std::string preprocess_string(const std::string& input) {
    std::string r;
    for (unsigned char c : input) {
        if (std::isalpha(c))  r += static_cast<char>(std::tolower(c));
        else if (std::isspace(c)) r += ' ';
    }
    return r;
}

std::set<std::string> extract_3grams(const std::string& input) {
    std::string p = preprocess_string(input);
    std::set<std::string> g;
    for (size_t i = 0; i + 2 < p.size(); ++i)
        g.insert(p.substr(i, 3));
    return g;
}

void build_3gram_index(const std::vector<std::string>& strings,
                       const std::vector<int>& cluster_ids,
                       std::unordered_map<std::string, std::set<int>>& gram_index) {
    for (size_t i = 0; i < strings.size(); ++i)
        for (const auto& g : extract_3grams(strings[i]))
            gram_index[g].insert(cluster_ids[i]);
}

// ── Set helpers ────────────────────────────────────────────────────────────

static std::set<int> make_all(int n) {
    std::set<int> s;
    for (int i = 0; i < n; ++i) s.insert(i);
    return s;
}

static std::set<int> set_isect(const std::set<int>& a, const std::set<int>& b) {
    std::set<int> r;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::inserter(r, r.begin()));
    return r;
}

static std::set<int> set_union2(const std::set<int>& a, const std::set<int>& b) {
    std::set<int> r;
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                   std::inserter(r, r.begin()));
    return r;
}

static std::set<int> clusters_for_literal(
        const std::string& lit,
        const std::unordered_map<std::string, std::set<int>>& idx,
        int N) {
    if (lit.size() < 3) return make_all(N);
    std::set<int> result = make_all(N);
    for (size_t i = 0; i + 2 < lit.size(); ++i) {
        std::string gram = lit.substr(i, 3);
        auto it = idx.find(gram);
        if (it == idx.end()) return std::set<int>(); // impossible trigram
        result = set_isect(result, it->second);
        if (result.empty()) return result;
    }
    return result;
}

// ── Recursive-descent parser ───────────────────────────────────────────────

struct Parser {
    const std::string& s;
    size_t pos;
    const std::unordered_map<std::string, std::set<int>>& idx;
    int N;

    Parser(const std::string& s_,
           const std::unordered_map<std::string, std::set<int>>& i_,
           int n) : s(s_), pos(0), idx(i_), N(n) {}

    char peek() const { return pos < s.size() ? s[pos] : '\0'; }
    char get()        { return pos < s.size() ? s[pos++] : '\0'; }
    bool done() const { return pos >= s.size(); }

    // expr = concat ('|' concat)*
    std::set<int> eval_expr() {
        std::set<int> result = eval_concat();
        while (peek() == '|') {
            get();
            result = set_union2(result, eval_concat());
            if ((int)result.size() == N) { skip_expr_tail(); return result; }
        }
        return result;
    }

    // Skip remaining concat arms after union already reached ALL
    void skip_expr_tail() {
        while (peek() == '|') { get(); skip_concat(); }
    }

    // concat = atom+
    std::set<int> eval_concat() {
        std::set<int> result = make_all(N);
        while (!done() && peek() != '|' && peek() != ')') {
            auto atom = eval_atom();
            result = set_isect(result, atom);
            if (result.empty()) { skip_concat(); return result; }
        }
        return result;
    }

    void skip_concat() {
        while (!done() && peek() != '|' && peek() != ')') skip_atom();
    }

    // atom = literal | '(' expr ')' quantifier | '[' class ']' | '.' | escape
    std::set<int> eval_atom() {
        char c = peek();

        // Character class
        if (c == '[') { skip_char_class(); consume_quantifier(); return make_all(N); }

        // Grouped subexpression
        if (c == '(') {
            get(); // consume '('
            // Lookahead for (?:…) non-capturing or (?=…) lookahead etc.
            if (peek() == '?') {
                get(); // consume '?'
                if (peek() == ':') {
                    get(); // non-capturing → evaluate normally
                } else {
                    // Other extensions → treat as unconstrained
                    skip_to_close_paren();
                    consume_quantifier();
                    return make_all(N);
                }
            }
            std::set<int> grp = eval_expr();
            if (peek() == ')') get();
            char q = consume_quantifier();
            if (q == '*' || q == '?') return make_all(N);
            return grp;
        }

        // Wildcard
        if (c == '.') { get(); consume_quantifier(); return make_all(N); }

        // Escape
        if (c == '\\') {
            get();
            if (!done()) get();
            consume_quantifier();
            return make_all(N);
        }

        // Anchors
        if (c == '^' || c == '$') { get(); return make_all(N); }

        // Literal run of alpha characters
        std::string lit;
        while (!done()) {
            char ch = peek();
            if (ch == '|' || ch == '(' || ch == ')' || ch == '.'  ||
                ch == '[' || ch == '\\' || ch == '^' || ch == '$'  ||
                ch == '*' || ch == '+' || ch == '?' || ch == '{')
                break;
            if (std::isalpha(static_cast<unsigned char>(ch)))
                lit += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            else
                lit += static_cast<char>(ch); // digits, punctuation
            get();
            // quantifier on last char → stop before it is consumed
            if (peek() == '*' || peek() == '+' || peek() == '?' || peek() == '{')
                break;
        }
        char q = consume_quantifier();
        if (q == '*' || q == '?') return make_all(N);
        return clusters_for_literal(lit, idx, N);
    }

    // Returns quantifier char or '\0'
    char consume_quantifier() {
        char c = peek();
        if (c == '*' || c == '+' || c == '?') { get(); return c; }
        if (c == '{') {
            while (!done() && get() != '}') {}
            return '{';
        }
        return '\0';
    }

    void skip_char_class() {
        assert(peek() == '['); get();
        if (peek() == '^') get();
        if (peek() == ']') get();
        while (!done() && peek() != ']') {
            if (peek() == '\\') { get(); if (!done()) get(); }
            else get();
        }
        if (!done()) get();
    }

    void skip_to_close_paren() {
        int depth = 1;
        while (!done() && depth > 0) {
            char c = get();
            if (c == '\\') { if (!done()) get(); continue; }
            if (c == '[')  { skip_char_class(); continue; }
            if (c == '(') ++depth;
            if (c == ')') --depth;
        }
    }

    void skip_atom() {
        char c = peek();
        if (c == '[')  { skip_char_class(); consume_quantifier(); return; }
        if (c == '(')  { get(); skip_to_close_paren(); consume_quantifier(); return; }
        if (c == '\\') { get(); if (!done()) get(); consume_quantifier(); return; }
        if (c == '.' || c == '^' || c == '$') { get(); return; }
        while (!done()) {
            char ch = peek();
            if (ch == '|' || ch == '(' || ch == ')' || ch == '.'  ||
                ch == '[' || ch == '\\' || ch == '^' || ch == '$'  ||
                ch == '*' || ch == '+' || ch == '?' || ch == '{')
                break;
            get();
            if (peek() == '*' || peek() == '+' || peek() == '?' || peek() == '{')
                break;
        }
        consume_quantifier();
    }
};

// ── Public API ─────────────────────────────────────────────────────────────

std::set<int> get_candidate_clusters(
        const std::string& regex,
        const std::unordered_map<std::string, std::set<int>>& gram_index,
        int num_clusters) {
    if (num_clusters <= 0 || gram_index.empty())
        return std::set<int>();
    Parser p(regex, gram_index, num_clusters);
    return p.eval_expr();
}
