// src/RegexParser.cpp
// Recursive-descent regex → DNF conversion.
// See include/RegexParser.h for the full derivation rules.
#include "RegexParser.h"
#include <algorithm>
#include <cassert>
#include <cctype>

// ─────────────────────────────────────────────────────────────────────────────
// DNF helpers
// ─────────────────────────────────────────────────────────────────────────────

using DNF  = std::vector<std::vector<int>>;
using Conj = std::vector<int>;

static const DNF UNCONSTRAINED = {{}};   // one empty conjunction = no filter

// AND of two DNFs: cross-product, merge conjunctions.
// DNF_and(A,B) = { sort(ca ∪ cb) | ca∈A, cb∈B }
static DNF dnf_and(const DNF& a, const DNF& b) {
    if (a == UNCONSTRAINED) return b;
    if (b == UNCONSTRAINED) return a;

    DNF result;
    result.reserve(a.size() * b.size());
    for (const auto& ca : a) {
        for (const auto& cb : b) {
            Conj merged;
            merged.reserve(ca.size() + cb.size());
            std::merge(ca.begin(), ca.end(),
                       cb.begin(), cb.end(),
                       std::back_inserter(merged));
            // deduplicate within the conjunction
            merged.erase(std::unique(merged.begin(), merged.end()),
                         merged.end());
            result.push_back(std::move(merged));
        }
    }
    return result;
}

// OR of two DNFs: simple concatenation.
static DNF dnf_or(const DNF& a, const DNF& b) {
    DNF result = a;
    result.insert(result.end(), b.begin(), b.end());
    return result;
}

// Build DNF from a single literal string (already lower-cased alpha only).
// Returns one conjunction containing all trigrams of the literal.
static DNF dnf_from_literal(const std::string& lit, TrigramVocab& vocab) {
    if (lit.size() < 3) return UNCONSTRAINED;
    Conj conj;
    for (size_t i = 0; i + 2 < lit.size(); ++i) {
        std::string gram = lit.substr(i, 3);
        conj.push_back(vocab.intern(gram));
    }
    std::sort(conj.begin(), conj.end());
    conj.erase(std::unique(conj.begin(), conj.end()), conj.end());
    return {conj};
}

// ─────────────────────────────────────────────────────────────────────────────
// Simplify DNF
// ─────────────────────────────────────────────────────────────────────────────

std::vector<std::vector<int>> simplify_dnf(std::vector<std::vector<int>> dnf) {
    // 1. Remove exact duplicates
    std::sort(dnf.begin(), dnf.end());
    dnf.erase(std::unique(dnf.begin(), dnf.end()), dnf.end());

    // 2. Remove dominated conjunctions:
    //    C dominates D if C ⊂ D (C has fewer constraints → C is more general
    //    and includes all results of D). Keep the more general ones.
    //    i.e. remove D if ∃ C such that C ⊆ D and C ≠ D.
    std::vector<bool> dominated(dnf.size(), false);
    for (size_t i = 0; i < dnf.size(); ++i) {
        if (dominated[i]) continue;
        for (size_t j = 0; j < dnf.size(); ++j) {
            if (i == j || dominated[j]) continue;
            // Check if dnf[i] ⊆ dnf[j]
            if (dnf[i].size() < dnf[j].size() &&
                std::includes(dnf[j].begin(), dnf[j].end(),
                              dnf[i].begin(), dnf[i].end())) {
                dominated[j] = true;
            }
        }
    }
    DNF out;
    for (size_t i = 0; i < dnf.size(); ++i)
        if (!dominated[i]) out.push_back(dnf[i]);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recursive-descent parser
// ─────────────────────────────────────────────────────────────────────────────

struct RParser {
    const std::string& s;
    size_t pos;
    TrigramVocab& vocab;

    RParser(const std::string& s_, TrigramVocab& v)
        : s(s_), pos(0), vocab(v) {}

    char peek() const { return pos < s.size() ? s[pos] : '\0'; }
    char get()        { return pos < s.size() ? s[pos++] : '\0'; }
    bool done() const { return pos >= s.size(); }

    // expr = concat ('|' concat)*
    DNF eval_expr() {
        DNF result = eval_concat();
        while (peek() == '|') {
            get();
            result = dnf_or(result, eval_concat());
        }
        return result;
    }

    // concat = atom+
    DNF eval_concat() {
        DNF result = UNCONSTRAINED;
        while (!done() && peek() != '|' && peek() != ')') {
            DNF atom = eval_atom();
            result = dnf_and(result, atom);
            // If result already unconstrained and growing large, short-circuit
        }
        return result;
    }

    // atom = literal | '(' expr ')' quant | '[' class ']' | '.' | escape
    DNF eval_atom() {
        char c = peek();

        // Character class [...] → unconstrained
        if (c == '[') {
            skip_char_class();
            consume_quantifier();
            return UNCONSTRAINED;
        }

        // Group (...)
        if (c == '(') {
            get(); // consume '('
            // Non-capturing (?:...) or other extensions
            if (peek() == '?') {
                get(); // consume '?'
                if (peek() == ':') {
                    get(); // (?:...) — evaluate normally
                } else {
                    // lookahead/lookbehind etc → unconstrained
                    skip_to_close_paren();
                    consume_quantifier();
                    return UNCONSTRAINED;
                }
            }
            DNF grp = eval_expr();
            if (peek() == ')') get();
            char q = consume_quantifier();
            // Optional group → could match zero times → unconstrained
            if (q == '*' || q == '?') return UNCONSTRAINED;
            return grp;
        }

        // Wildcard
        if (c == '.') { get(); consume_quantifier(); return UNCONSTRAINED; }

        // Escape
        if (c == '\\') {
            get();
            if (!done()) get();
            consume_quantifier();
            return UNCONSTRAINED;
        }

        // Anchors
        if (c == '^' || c == '$') { get(); return UNCONSTRAINED; }

        // Literal run of alpha characters
        std::string lit;
        while (!done()) {
            char ch = peek();
            if (ch == '|' || ch == '(' || ch == ')' || ch == '.'  ||
                ch == '[' || ch == '\\' || ch == '^' || ch == '$'  ||
                ch == '*' || ch == '+' || ch == '?' || ch == '{')
                break;
            if (std::isalpha((unsigned char)ch))
                lit += (char)std::tolower((unsigned char)ch);
            else
                lit += (char)ch; // digits, punctuation → break literal
            get();
            if (peek() == '*' || peek() == '+' ||
                peek() == '?' || peek() == '{') break;
        }
        char q = consume_quantifier();
        if (q == '*' || q == '?') {
            if (!lit.empty()) lit.pop_back();
        }
        return dnf_from_literal(lit, vocab);
    }

    char consume_quantifier() {
        char c = peek();
        if (c == '*' || c == '+' || c == '?') { get(); return c; }
        if (c == '{') { return consume_brace_quantifier(); }
        return '\0';
    }

    char consume_brace_quantifier() {
        get(); // consume '{'
        std::string lo;
        while (!done() && std::isdigit((unsigned char)peek())) lo += get();
        // Skip the rest of the quantifier (comma, upper bound) up to '}'.
        while (!done() && peek() != '}') get();
        if (!done()) get(); // consume '}'

        // Missing lower bound (malformed, e.g. stray "{,3}") or a lower
        // bound made entirely of zeros (e.g. "0", "00") both mean the
        // atom is not guaranteed to occur.
        bool lower_is_zero =
            lo.empty() || lo.find_first_not_of('0') == std::string::npos;
        return lower_is_zero ? '?' : '{';
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
};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

std::vector<std::vector<int>> regex_to_dnf(
        const std::string& regex,
        TrigramVocab& vocab) {
    RParser p(regex, vocab);
    DNF dnf = p.eval_expr();
    return simplify_dnf(dnf);
}
