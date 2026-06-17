// src/main.cpp – RegExANN unified driver.
//
// Usage:
//   ./regann <vectors.fvecs> <strings.txt> <queries.txt> <K>
//            <clusters> <output.txt> <max_iter> <algorithm> [options...]
//
// Algorithms:
//   ann          RegExANN (k-means + trigram index + PQ)
//   hier         Two-level hierarchical RegExANN
//   groundtruth  Full-scan exact search (generate ground truth)
//   baseline     Alias for groundtruth
//   prefilter    Pre-filtering baseline (regex first, then kNN)
//   postfilter   Post-filtering baseline (kNN first, then regex)
//
// Options (key=value):
//   pq_m=N          PQ subspaces                  [ann/hier, default 8]
//   pq_ksub=N       PQ centroids per subspace      [ann/hier, default 256]
//   k0=N            Coarse clusters               [hier, default sqrt(clusters)]
//   nprobe=N        Coarse clusters to probe       [hier, default k0/2]
//   oversample=N    Post-filter oversample factor  [postfilter, default 10]
//   gt=<file>       Ground-truth file for Recall@K
//   save=<prefix>   Save index after build         [ann only]
//   load=<prefix>   Load pre-built index           [ann only]
//   fmt=fvecs|bvecs Vector file format             [default fvecs]
//
// Postfilter shorthand — bare integer after algorithm name sets oversample:
//   ./regann ... postfilter 20           (oversample=20)
//   ./regann ... postfilter oversample=20 (same)
//
// Ignored options are reported as warnings so users can catch typos.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "Baseline.h"
#include "Eval.h"
#include "HierarchicalIndex.h"
#include "Index.h"
#include "KMeans.h"
#include "PQ.h"
#include "Search.h"
#include "Serialize.h"
#include "GraphIndex.h"
#include "RegexParser.h"

// ─────────────────────────────────────────────────────────────────────────────
// Option parsing
// ─────────────────────────────────────────────────────────────────────────────

struct Options {
    int   pq_m        = 8;
    int   pq_ksub     = 256;
    int   k0          = 0;
    int   nprobe      = 0;
    int   oversample  = 10;
    int   max_expansion = 0;    // postfilter: max expansion factor (0 = unlimited)
    int   ef          = 0;      // ann: candidate pool size (0 → use K)
    int   ann_nprobe  = 0;      // ann: max clusters to scan (0 → all candidate clusters)
    float sample_ratio = 1.0f;  // prefilter: fraction of matching set to search
    int  graph_M         = 16;
    int  graph_ef_build  = 64;
    int  graph_ef_search = 64;
    std::string gt_file;
    std::string save_prefix;
    std::string load_prefix;
    std::string fmt  = "fvecs";

    // Parse key=value options starting from argv[start].
    // `algorithm` is used to emit targeted warnings for options that are
    // accepted by the parser but have no effect for the chosen algorithm.
    void parse(int argc, char* argv[], int start,
               const std::string& algorithm) {

        // Which options are meaningful for each algorithm
        static const std::unordered_map<std::string,
               std::vector<std::string>> valid_opts = {
            {"ann",         {"pq_m","pq_ksub","ef","nprobe","gt","save","load","fmt"}},
            {"hier",        {"pq_m","pq_ksub","k0","nprobe","gt","fmt"}},
            {"groundtruth", {"gt","fmt"}},
            {"baseline",    {"gt","fmt"}},
            {"prefilter",   {"sample_ratio","gt","fmt"}},
            {"postfilter",  {"oversample","max_expansion","gt","fmt"}},
            {"graph",       {"graph_M","graph_ef_build","graph_ef_search","gt","fmt"}},
        };
        auto it = valid_opts.find(algorithm);
        const std::vector<std::string>* valid =
            (it != valid_opts.end()) ? &it->second : nullptr;

        auto is_valid = [&](const std::string& key) -> bool {
            if (!valid) return true;
            for (const auto& v : *valid)
                if (v == key) return true;
            return false;
        };

        for (int i = start; i < argc; ++i) {
            std::string a = argv[i];

            // ── Bare integer → oversample shorthand for postfilter ────────
            // e.g.  postfilter 20   means oversample=20
            bool all_digits = !a.empty() &&
                              a.find_first_not_of("0123456789") == std::string::npos;
            if (all_digits) {
                if (algorithm == "postfilter") {
                    oversample = std::stoi(a);
                    std::cout << "[INFO] Bare integer '" << a
                              << "' interpreted as oversample=" << oversample
                              << " for postfilter.\n";
                } else {
                    std::cerr << "[WARN] Bare integer '" << a
                              << "' ignored (not valid for algorithm '"
                              << algorithm << "').\n";
                }
                continue;
            }

            // ── key=value options ─────────────────────────────────────────
            auto eq = a.find('=');
            if (eq == std::string::npos) {
                std::cerr << "[WARN] Unrecognised argument '" << a
                          << "' (expected key=value or integer).\n";
                continue;
            }
            std::string key = a.substr(0, eq);
            std::string val = a.substr(eq + 1);

            // Parse
            bool known = true;
            if      (key == "pq_m")         pq_m          = std::stoi(val);
            else if (key == "pq_ksub")      pq_ksub       = std::stoi(val);
            else if (key == "k0")           k0            = std::stoi(val);
            else if (key == "nprobe")       nprobe        = std::stoi(val);
            else if (key == "oversample")   oversample    = std::stoi(val);
            else if (key == "max_expansion") max_expansion = std::stoi(val);
            else if (key == "ef")           ef            = std::stoi(val);
            else if (key == "nprobe") {
                // nprobe applies to both ann and hier; route by algorithm
                if (algorithm == "ann") ann_nprobe = std::stoi(val);
                else                    nprobe     = std::stoi(val);
            }
            else if (key == "sample_ratio") sample_ratio  = std::stof(val);
            else if (key == "gt")           gt_file       = val;
            else if (key == "save")         save_prefix   = val;
            else if (key == "load")         load_prefix   = val;
            else if (key == "fmt")          fmt           = val;
            else if (key == "graph_M")         graph_M         = std::stoi(val);
            else if (key == "graph_ef_build")  graph_ef_build  = std::stoi(val);
            else if (key == "graph_ef_search") graph_ef_search = std::stoi(val);
            else { known = false; }

            if (!known) {
                std::cerr << "[WARN] Unknown option '" << a << "' — ignored.\n";
            } else if (!is_valid(key)) {
                // Known option but irrelevant for this algorithm
                std::cerr << "[WARN] Option '" << key << "' has no effect"
                          << " for algorithm '" << algorithm << "' — ignored.\n";
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// I/O helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<std::vector<float>> load_fvecs(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + p);
    std::vector<std::vector<float>> v;
    while (f.peek() != EOF) {
        int dim; f.read(reinterpret_cast<char*>(&dim), 4);
        if (!f) break;
        std::vector<float> vec(dim);
        f.read(reinterpret_cast<char*>(vec.data()), 4 * dim);
        v.push_back(std::move(vec));
    }
    return v;
}

static std::vector<std::vector<float>> load_bvecs(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + p);
    std::vector<std::vector<float>> v;
    while (f.peek() != EOF) {
        int dim; f.read(reinterpret_cast<char*>(&dim), 4);
        if (!f) break;
        std::vector<uint8_t> raw(dim);
        f.read(reinterpret_cast<char*>(raw.data()), dim);
        std::vector<float> vec(dim);
        for (int i = 0; i < dim; ++i) vec[i] = static_cast<float>(raw[i]);
        v.push_back(std::move(vec));
    }
    return v;
}

static std::vector<std::vector<int>> load_ivecs(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + p);
    std::vector<std::vector<int>> v;
    while (f.peek() != EOF) {
        int dim; f.read(reinterpret_cast<char*>(&dim), 4);
        if (!f) break;
        std::vector<int> vec(dim);
        f.read(reinterpret_cast<char*>(vec.data()), 4 * dim);
        v.push_back(std::move(vec));
    }
    return v;
}

static std::vector<std::vector<float>> load_vectors(
        const std::string& p, const std::string& fmt) {
    if (fmt == "bvecs") return load_bvecs(p);
    return load_fvecs(p);
}

static std::vector<std::string> load_strings(const std::string& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("Cannot open: " + p);
    std::vector<std::string> v;
    std::string line;
    while (std::getline(f, line)) v.push_back(line);
    return v;
}

static bool parse_query(const std::string& line,
                         std::string& rx, std::vector<float>& qv) {
    std::istringstream iss(line);
    if (!(iss >> rx)) return false;
    float val;
    while (iss >> val) qv.push_back(val);
    return !qv.empty();
}

static size_t rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmRSS:", 0) == 0) {
            size_t kb; std::string k, u;
            std::istringstream(line) >> k >> kb >> u;
            return kb;
        }
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recall helper
// ─────────────────────────────────────────────────────────────────────────────

static void report_recall(const std::vector<std::vector<int>>& preds,
                           const std::string& gt_file, int K) {
    if (gt_file.empty()) return;
    try {
        auto gt = load_results(gt_file);
        auto [mean, _] = compute_recall(gt, preds, K);
        std::cout << "[EVAL] Recall@" << K << " = "
                  << mean * 100.0 << " %\n";
    } catch (const std::exception& e) {
        std::cerr << "[WARN] Recall skipped: " << e.what() << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RegExANN (flat)
// ─────────────────────────────────────────────────────────────────────────────

static void run_ann(
        const std::vector<std::vector<float>>& vectors,
        const std::vector<std::string>& strings,
        std::ifstream& qfin, std::ofstream& fout,
        int K, int X, int max_iter, const Options& opts) {

    using Clock = std::chrono::high_resolution_clock;
    using ms    = std::chrono::milliseconds;
    using us    = std::chrono::microseconds;

    KMeansResult km;
    std::unordered_map<std::string, std::set<int>> gram_index;
    PQIndex pq;
    bool loaded = false;

    if (!opts.load_prefix.empty())
        loaded = load_index(opts.load_prefix, km, gram_index, pq);

    if (!loaded) {
        std::cout << "[INFO] Building index (k=" << X
                  << ", pq_m=" << opts.pq_m
                  << ", pq_ksub=" << opts.pq_ksub << ") …\n";
        auto t0 = Clock::now();

        km = run_kmeans(vectors, X, max_iter);
        build_3gram_index(strings, km.assignments, gram_index);
        std::cout << "[INFO] Trigram index: "
                  << gram_index.size() << " trigrams.\n";

        int dim = (int)vectors[0].size();
        int eff_m = opts.pq_m;
        while (eff_m > 1 && dim % eff_m != 0) --eff_m;
        if (eff_m != opts.pq_m)
            std::cout << "[INFO] pq_m: " << opts.pq_m
                      << " → " << eff_m << "\n";
        int k_sub = std::min(opts.pq_ksub, (int)vectors.size());

        train_pq(pq, vectors, eff_m, k_sub, 25);
        encode_all(pq, vectors);

        auto t1 = Clock::now();
        std::cout << "[INFO] Index built in "
                  << std::chrono::duration_cast<ms>(t1-t0).count() << " ms.\n";

        if (!opts.save_prefix.empty())
            save_index(opts.save_prefix, km, gram_index, pq, dim);
    }

    std::cout << "[INFO] Memory: " << rss_kb() / 1024.0 << " MB\n";
    int eff_ef = (opts.ef > 0) ? opts.ef : K;
    std::cout << "[INFO] ef=" << eff_ef
              << (opts.ef <= 0 ? " (default=K, original behaviour)" : "") << "\n";
    std::cout << "[INFO] nprobe=" << (opts.ann_nprobe > 0 ? opts.ann_nprobe : -1)
              << (opts.ann_nprobe <= 0 ? " (scan all candidate clusters)" : "") << "\n";

    std::string line;
    int qcnt = 0;
    double t_all = 0, t_set = 0, t_clust = 0;
    std::vector<std::vector<int>> all_preds;

    while (std::getline(qfin, line)) {
        std::string rx; std::vector<float> qv;
        if (!parse_query(line, rx, qv)) continue;
        auto qa = Clock::now();
        auto r  = perform_search(rx, qv, gram_index,
                                 km.centroids, km.clusters,
                                 vectors, strings, pq, K, opts.ef, opts.ann_nprobe);
        auto qb = Clock::now();
        t_all   += std::chrono::duration_cast<us>(qb-qa).count() / 1000.0;
        t_set   += r.setop_time_ms;
        t_clust += r.query_time_ms;
        ++qcnt;
        for (int id : r.top_ids) fout << id << " ";
        fout << "\n";
        all_preds.push_back(r.top_ids);
    }

    if (qcnt > 0) {
        std::cout << "[INFO] Queries          : " << qcnt          << "\n"
                  << "[INFO] Avg total time   : " << t_all/qcnt    << " ms\n"
                  << "[INFO] Avg set-op time  : " << t_set/qcnt    << " ms\n"
                  << "[INFO] Avg cluster time : " << t_clust/qcnt  << " ms\n"
                  << "[INFO] QPS              : "
                  << 1000.0*qcnt/t_all << "\n";
    }
    report_recall(all_preds, opts.gt_file, K);
}

// ─────────────────────────────────────────────────────────────────────────────
// Hierarchical RegExANN
// ─────────────────────────────────────────────────────────────────────────────

static void run_hier(
        const std::vector<std::vector<float>>& vectors,
        const std::vector<std::string>& strings,
        std::ifstream& qfin, std::ofstream& fout,
        int K, int X, int max_iter, const Options& opts) {

    using Clock = std::chrono::high_resolution_clock;
    using us    = std::chrono::microseconds;

    int k0     = opts.k0 > 0 ? opts.k0 : std::max(2, (int)std::sqrt((double)X));
    int k1     = std::max(1, X / k0);
    int nprobe = opts.nprobe > 0 ? opts.nprobe : std::max(1, k0 / 2);

    std::cout << "[INFO] Building hierarchical index (k0=" << k0
              << ", k1=" << k1 << ", nprobe=" << nprobe << ") …\n";

    auto hidx = build_hierarchical_index(
        vectors, strings, k0, k1, opts.pq_m, opts.pq_ksub, max_iter);

    std::cout << "[INFO] Memory: " << rss_kb() / 1024.0 << " MB\n";

    std::string line;
    int qcnt = 0; double t_all = 0;
    std::vector<std::vector<int>> all_preds;

    while (std::getline(qfin, line)) {
        std::string rx; std::vector<float> qv;
        if (!parse_query(line, rx, qv)) continue;
        auto qa  = Clock::now();
        auto ids = query_hierarchical(hidx, rx, qv, vectors, strings, K, nprobe);
        auto qb  = Clock::now();
        t_all += std::chrono::duration_cast<us>(qb-qa).count() / 1000.0;
        ++qcnt;
        for (int id : ids) fout << id << " ";
        fout << "\n";
        all_preds.push_back(ids);
    }

    if (qcnt > 0)
        std::cout << "[INFO] Queries: " << qcnt
                  << "  Avg: " << t_all/qcnt << " ms"
                  << "  QPS: " << 1000.0*qcnt/t_all << "\n";
    report_recall(all_preds, opts.gt_file, K);
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic baseline loop
// ─────────────────────────────────────────────────────────────────────────────

using BaseFn = std::function<BaselineResult(
    const std::string&, const std::vector<float>&,
    const std::vector<std::vector<float>>&,
    const std::vector<std::string>&, int)>;

static void run_baseline_loop(
        BaseFn fn,
        const std::vector<std::vector<float>>& vecs,
        const std::vector<std::string>& strs,
        std::ifstream& qfin, std::ofstream& fout,
        int K, const Options& opts) {

    std::string line;
    int qcnt = 0; double t_all = 0;
    std::vector<std::vector<int>> all_preds;

    while (std::getline(qfin, line)) {
        std::string rx; std::vector<float> qv;
        if (!parse_query(line, rx, qv)) continue;
        auto r = fn(rx, qv, vecs, strs, K);
        t_all += r.query_time_ms;
        ++qcnt;
        for (int id : r.top_ids) fout << id << " ";
        fout << "\n";
        all_preds.push_back(r.top_ids);
    }

    std::cout << "[INFO] Memory: " << rss_kb() / 1024.0 << " MB\n";
    if (qcnt > 0)
        std::cout << "[INFO] Queries: " << qcnt
                  << "  Avg: " << t_all/qcnt << " ms"
                  << "  QPS: " << 1000.0*qcnt/t_all << "\n";
    report_recall(all_preds, opts.gt_file, K);
}


// ─────────────────────────────────────────────────────────────────────────────
// Graph-NSW algorithm
// ─────────────────────────────────────────────────────────────────────────────

static void run_graph(
        const std::vector<std::vector<float>>& vectors,
        const std::vector<std::string>& strings,
        std::ifstream& qfin, std::ofstream& fout,
        int K, const Options& opts) {

    using Clock = std::chrono::high_resolution_clock;
    using ms    = std::chrono::milliseconds;
    using us    = std::chrono::microseconds;

    // ── Build graph index ─────────────────────────────────────────────────
    std::cout << "[INFO] Building Graph-NSW index"
              << " (M=" << opts.graph_M
              << ", ef_build=" << opts.graph_ef_build << ") ...\n";
    auto t0 = Clock::now();
    GraphIndex idx = build_graph_index(vectors, strings,
                                       opts.graph_M, opts.graph_ef_build);
    auto t1 = Clock::now();
    std::cout << "[INFO] Graph built in "
              << std::chrono::duration_cast<ms>(t1-t0).count() << " ms.\n";
    std::cout << "[INFO] Memory: " << rss_kb() / 1024.0 << " MB\n";

    // ── Query loop ────────────────────────────────────────────────────────
    std::string line;
    int qcnt = 0;
    double t_all = 0, t_parse = 0, t_search = 0;
    std::vector<std::vector<int>> all_preds;

    while (std::getline(qfin, line)) {
        std::string rx; std::vector<float> qv;
        if (!parse_query(line, rx, qv)) continue;

        // Parse regex → DNF of trigram ID groups
        auto tp0 = Clock::now();
        std::vector<std::vector<int>> dnf = regex_to_dnf(rx, idx.vocab);
        auto tp1 = Clock::now();
        t_parse += std::chrono::duration_cast<us>(tp1-tp0).count() / 1000.0;

        // Step 2: trigram filter → candidate node IDs
        auto ts0 = Clock::now();
        auto candidates = trigram_filter_nodes(idx, dnf);
        // Step 3+4: graph-guided K-NN within candidates + regex verify
        std::vector<int> result = search_graph(
            idx, qv, rx, candidates, K, opts.graph_ef_search);
        auto ts1 = Clock::now();
        t_search += std::chrono::duration_cast<us>(ts1-ts0).count() / 1000.0;

        t_all += std::chrono::duration_cast<us>(ts1-tp0).count() / 1000.0;
        ++qcnt;

        for (int id : result) fout << id << " ";
        fout << "\n";
        all_preds.push_back(result);
    }

    if (qcnt > 0) {
        std::cout << "[INFO] Queries          : " << qcnt          << "\n"
                  << "[INFO] Avg total time   : " << t_all/qcnt    << " ms\n"
                  << "[INFO] Avg parse time   : " << t_parse/qcnt  << " ms\n"
                  << "[INFO] Avg search time  : " << t_search/qcnt << " ms\n"
                  << "[INFO] QPS              : "
                  << 1000.0*qcnt/t_all << "\n";
    }
    report_recall(all_preds, opts.gt_file, K);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 9) {
        std::cerr <<
"Usage: ./regann <vectors.fvecs> <strings.txt> <queries.txt> <K>\n"
"               <clusters> <output.txt> <max_iter> <algorithm> [opts...]\n"
"\n"
"Algorithms: ann | hier | graph | groundtruth | baseline | prefilter | postfilter\n"
"\n"
"Options (key=value):\n"
"  pq_m=N            PQ subspaces                [ann/hier, default 8]\n"
"  pq_ksub=N         PQ centroids/subspace        [ann/hier, default 256]\n"
"  ef=N              Candidate pool size (>=K)    [ann, default K]\n"
"                    Larger ef → higher recall, slower query.\n"
"  k0=N              Coarse clusters              [hier, default sqrt(clusters)]\n"
"  nprobe=N          Coarse probe count           [hier, default k0/2]\n"
"  oversample=N      Post-filter initial window   [postfilter, default 10]\n"
"  max_expansion=N   Post-filter max window cap   [postfilter, default 0=unlimited]\n"
"                    0 = expand until K found or full scan; N = cap at K*N.\n"
"  sample_ratio=F    Pre-filter match subset frac [prefilter, default 1.0]\n"
"                    1.0 = exact (100% recall); <1.0 = faster, lower recall.\n"
"  gt=<file>         Ground-truth for Recall@K\n"
"  save=<prefix>     Save index                   [ann only]\n"
"  load=<prefix>     Load saved index             [ann only]\n"
"  fmt=fvecs|bvecs   Vector file format\n"
"  graph_M=N         Graph max degree             [graph, default 16]\n"
"  graph_ef_build=N  Graph build beam width       [graph, default 64]\n"
"  graph_ef_search=N Graph search beam width      [graph, default 64]\n"
"\n"
"Postfilter shorthand: bare integer = oversample value\n"
"  ./regann ... postfilter 20          (same as oversample=20)\n"
"\n"
"Note: options irrelevant to the chosen algorithm produce a [WARN].\n";
        return 1;
    }

    const std::string vec_path  = argv[1];
    const std::string str_path  = argv[2];
    const std::string qry_path  = argv[3];
    const int K                 = std::stoi(argv[4]);
    const int X                 = std::stoi(argv[5]);
    const std::string out_path  = argv[6];
    const int max_iter          = std::stoi(argv[7]);
    const std::string algorithm = argv[8];

    Options opts;
    opts.parse(argc, argv, 9, algorithm);

    try {
        auto vectors = load_vectors(vec_path, opts.fmt);
        auto strings = load_strings(str_path);
        if (vectors.empty()) throw std::runtime_error("No vectors loaded.");
        if (vectors.size() != strings.size())
            std::cerr << "[WARN] vectors/strings mismatch: "
                      << vectors.size() << " / " << strings.size() << "\n";

        std::cout << "[INFO] Dataset: " << vectors.size()
                  << " vectors (dim=" << vectors[0].size()
                  << "), " << strings.size() << " strings.\n";

        std::ifstream qfin(qry_path);
        if (!qfin) throw std::runtime_error("Cannot open: " + qry_path);
        std::ofstream fout(out_path);
        if (!fout) throw std::runtime_error("Cannot write: " + out_path);

        if (algorithm == "ann") {
            run_ann(vectors, strings, qfin, fout, K, X, max_iter, opts);

        } else if (algorithm == "graph") {
            run_graph(vectors, strings, qfin, fout, K, opts);

        } else if (algorithm == "hier") {
            run_hier(vectors, strings, qfin, fout, K, X, max_iter, opts);

        } else if (algorithm == "groundtruth" || algorithm == "baseline") {
            std::cout << "[INFO] Full-scan exact search …\n";
            run_baseline_loop(
                [](const std::string& rx, const std::vector<float>& qv,
                   const std::vector<std::vector<float>>& v,
                   const std::vector<std::string>& s, int k){
                    return run_fullscan(rx, qv, v, s, k); },
                vectors, strings, qfin, fout, K, opts);

        } else if (algorithm == "prefilter") {
            std::cout << "[INFO] Pre-filter baseline (sample_ratio="
                      << opts.sample_ratio << ") …\n";
            float sr = opts.sample_ratio;
            run_baseline_loop(
                [sr](const std::string& rx, const std::vector<float>& qv,
                   const std::vector<std::vector<float>>& v,
                   const std::vector<std::string>& s, int k){
                    return run_prefilter(rx, qv, v, s, k, sr); },
                vectors, strings, qfin, fout, K, opts);

        } else if (algorithm == "postfilter") {
            int ov = opts.oversample;
            int me = opts.max_expansion;
            std::cout << "[INFO] Post-filter baseline (oversample="
                      << ov << ", max_expansion="
                      << (me > 0 ? std::to_string(me) : "unlimited") << ") …\n";
            run_baseline_loop(
                [ov, me](const std::string& rx, const std::vector<float>& qv,
                     const std::vector<std::vector<float>>& v,
                     const std::vector<std::string>& s, int k){
                    return run_postfilter(rx, qv, v, s, k, ov, me); },
                vectors, strings, qfin, fout, K, opts);

        } else {
            std::cerr << "[ERROR] Unknown algorithm: " << algorithm << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
    return 0;
}
