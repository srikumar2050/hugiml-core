/**
 * mining_l3.cpp — Hot path implementation for L=3 pattern mining.
 *
 * This path mirrors the generic THUIsl recursion for ordinary L=3 mining:
 * singleton, pair, and triplet candidates compete in one utility-ranked top-K
 * heap; pair IG is computed against the global class distribution; triplet IG
 * is computed against the singleton root's covered population, matching
 * THUIsl::child_ul(...)->compute_ig(prefix_ul=singleton_root).
 *
 * The implementation avoids recursive UL allocation for the depth-0/depth-1
 * tree, uses compact structs for terminal triplets, and parallelises root
 * branches with per-thread heaps that are merged serially. A memory-bounded
 * cache carries each root's Phase A pair utility-lists into Phase B so
 * qualifying roots don't pay the tid-intersection cost twice; see
 * kL3PairCacheMaxTidsDefault.
 */

#include "mining_l3.hpp"
#include "transaction.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace hugiml {
namespace {

constexpr size_t kL3EucsMaxCellsDefault = 6000000ULL;
constexpr int kL3EucsMinItemsDefault = 32;
constexpr double kL3EucsMaxDensityDefault = 0.10;

static std::string l3_eucs_env_lower(const char* value) {
    std::string out(value ? value : "");
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

static bool l3_eucs_env_enabled() {
    const char* raw = std::getenv("HUGIML_EUCS_ENABLE");
    if (!raw) raw = std::getenv("HUGIML_EUCS_ENABLED");
    if (!raw || !*raw) return true;
    const std::string value = l3_eucs_env_lower(raw);
    if (value == "0" || value == "false" || value == "no" || value == "off" ||
        value == "disable" || value == "disabled") {
        return false;
    }
    if (value == "1" || value == "true" || value == "yes" || value == "on" ||
        value == "enable" || value == "enabled") {
        return true;
    }
    return true;
}

static int l3_eucs_env_int(const char* name, int fallback, int min_value) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    try {
        size_t idx = 0;
        long value = std::stol(raw, &idx, 10);
        if (idx != std::string(raw).size() || value < min_value ||
            value > static_cast<long>(std::numeric_limits<int>::max())) {
            return fallback;
        }
        return static_cast<int>(value);
    } catch (...) {
        return fallback;
    }
}

static size_t l3_eucs_env_size(const char* name, size_t fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    try {
        size_t idx = 0;
        unsigned long long value = std::stoull(raw, &idx, 10);
        if (idx != std::string(raw).size()) return fallback;
        return static_cast<size_t>(value);
    } catch (...) {
        return fallback;
    }
}

static double l3_eucs_env_double(const char* name, double fallback, double min_value) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    try {
        size_t idx = 0;
        double value = std::stod(raw, &idx);
        if (idx != std::string(raw).size() || value < min_value) return fallback;
        return value;
    } catch (...) {
        return fallback;
    }
}

static int l3_eucs_min_items_config() {
    return l3_eucs_env_int("HUGIML_EUCS_MIN_ITEMS", kL3EucsMinItemsDefault, 0);
}

static size_t l3_eucs_max_cells_config() {
    return l3_eucs_env_size("HUGIML_EUCS_MAX_CELLS", kL3EucsMaxCellsDefault);
}

static double l3_eucs_max_density_config() {
    return l3_eucs_env_double("HUGIML_EUCS_MAX_DENSITY", kL3EucsMaxDensityDefault, 0.0);
}

// Phase A computes each surviving root's pair utility-lists (an O(pairs)
// tid-intersection cost) to save pair candidates and derive root_triplet_ub.
// Phase B previously reconstructed the identical pair utility-lists from
// scratch for every root that clears the root_triplet_ub gate, paying that
// intersection cost twice for exactly the roots doing the most pair work.
// This budget bounds how many total tid-list entries may be *retained*
// from Phase A for direct reuse in Phase B, so the common case (moderate
// item universes) skips the second reconstruction entirely, while runs
// with very large item universes / very permissive K & G fall back to the
// original recompute-in-Phase-B behaviour once the cap is hit -- capping
// worst-case extra memory rather than letting it grow with data size.
// Reusing a cached pair is provably safe regardless of budget outcome:
// build_pair_ul_l3 is a pure function of two fixed per-item utility-lists,
// so a Phase-A-computed pair is byte-identical to what Phase B would have
// computed itself; only the incidental eucs pre-filter (a skip-work
// heuristic, not a correctness gate) could differ between the two phases,
// and retaining a pair it would have skipped only costs extra downstream
// gate checks, never a wrong result.
constexpr size_t kL3PairCacheMaxTidsDefault = 8000000ULL;

static bool l3_pair_cache_enabled() {
    const char* raw = std::getenv("HUGIML_L3_PAIR_CACHE_ENABLE");
    if (!raw || !*raw) return true;
    const std::string value = l3_eucs_env_lower(raw);
    if (value == "0" || value == "false" || value == "no" || value == "off" ||
        value == "disable" || value == "disabled") {
        return false;
    }
    return true;
}

static size_t l3_pair_cache_max_tids_config() {
    return l3_eucs_env_size("HUGIML_L3_PAIR_CACHE_MAX_TIDS", kL3PairCacheMaxTidsDefault);
}

static double l3_entropy_from_counts(const std::vector<int>& counts, int total) {
    if (total <= 0) return 0.0;
    double h = 0.0;
    const double inv = 1.0 / static_cast<double>(total);
    for (int c : counts) {
        if (c > 0) {
            const double p = static_cast<double>(c) * inv;
            h -= p * std::log(p);
        }
    }
    return std::max(h, 0.0);
}

static double l3_ig_from_counts(const std::vector<int>& parent_counts,
                                int parent_total,
                                const std::vector<int>& child_counts,
                                int child_total) {
    if (child_total <= 0 || parent_total <= 0 || child_total > parent_total) {
        return 0.0;
    }
    const int n_out = parent_total - child_total;
    if (n_out == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // out_counts[k] = parent_counts[k] - child_counts[k], entropy computed
    // inline rather than materialised into a temporary vector: this
    // function runs once per candidate pair and once per candidate
    // triplet (the two hottest call sites in the miner), so avoiding a
    // per-call heap allocation here is a direct, broad win. Same formula
    // as l3_entropy_from_counts applied to (parent_counts - child_counts),
    // just fused into one pass with no intermediate storage.
    double h_out = 0.0;
    const double inv_out = 1.0 / static_cast<double>(n_out);
    for (size_t k = 0; k < parent_counts.size(); ++k) {
        const int child = (k < child_counts.size()) ? child_counts[k] : 0;
        const int out_c = parent_counts[k] - child;
        if (out_c > 0) {
            const double p = static_cast<double>(out_c) * inv_out;
            h_out -= p * std::log(p);
        }
    }
    h_out = std::max(h_out, 0.0);

    const double base = l3_entropy_from_counts(parent_counts, parent_total);
    const double cond = (static_cast<double>(child_total) / parent_total)
            * l3_entropy_from_counts(child_counts, child_total)
        + (static_cast<double>(n_out) / parent_total) * h_out;
    return base - cond;
}

struct L3MinHeapCmp {
    bool operator()(const PatternEntry& a, const PatternEntry& b) const {
        return a.utility > b.utility;
    }
};

static void l3_save(std::vector<PatternEntry>& heap,
                    double& minU,
                    int K,
                    const std::vector<int>& items,
                    double utility,
                    double ig) {
    PatternEntry pe{utility, items, ig};
    if (static_cast<int>(heap.size()) < K) {
        heap.push_back(pe);
        std::push_heap(heap.begin(), heap.end(), L3MinHeapCmp{});
        if (static_cast<int>(heap.size()) == K) minU = heap.front().utility;
    } else if (utility > minU) {
        std::pop_heap(heap.begin(), heap.end(), L3MinHeapCmp{});
        heap.back() = pe;
        std::push_heap(heap.begin(), heap.end(), L3MinHeapCmp{});
        minU = heap.front().utility;
    }
}

static size_t l3_fnv1a_tids(const std::vector<int32_t>& v) {
    size_t h = 14695981039346656037ULL;
    for (int32_t x : v) {
        const unsigned char* b = reinterpret_cast<const unsigned char*>(&x);
        for (int byte = 0; byte < 4; byte++) {
            h ^= static_cast<size_t>(b[byte]);
            h *= 1099511628211ULL;
        }
    }
    return h;
}

// Coverage-hash membership check for the small per-root/per-pair dedup
// buffers below (typically a handful to a few dozen live entries: bounded
// by how many candidate partners survive the IG/EUCS gates at that branch).
// A flat vector + linear scan outperforms std::unordered_set here in
// practice: unordered_set pays one heap allocation per *element* (a bucket
// node) in addition to its own bucket-array allocation, while a vector's
// entries live contiguously in one growable buffer -- and at these sizes,
// a cache-friendly linear scan is not meaningfully slower than a hash
// probe. Semantics are unchanged: this is still just an O(1)-amortized
// pre-filter ahead of the existing full tid-vector equality check.
static inline bool l3_vec_contains(const std::vector<size_t>& v, size_t h) {
    for (size_t existing : v)
        if (existing == h) return true;
    return false;
}

struct L3UL {
    int item = 0;
    double sI = 0.0;
    double sR = 0.0;
    double ig = 0.0;
    std::vector<int32_t> tid;
    std::vector<double> iu;
    std::vector<double> ru;
    std::vector<double> rmax;
    std::vector<int> class_counts;
};

struct L3PairUL {
    int item_x = 0;
    int item_y = 0;
    double sI = 0.0;
    double sR = 0.0;       // legacy all-suffix remaining utility bound
    double sRMax1 = 0.0;   // tight L3 bound: at most one suffix item can be added
    double ig = 0.0;
    std::vector<int32_t> tid;
    std::vector<double> iu;
    std::vector<int> class_counts;
};

struct L3Triplet {
    int item_x = 0;
    int item_y = 0;
    int item_z = 0;
    double sI = 0.0;
    double ig = 0.0;
};

static bool same_feature_l3(const std::vector<int>& item_col, int item_a, int item_b) {
    if (item_col.empty()) return false;
    if (item_a <= 0 || item_b <= 0) return false;
    const size_t ia = static_cast<size_t>(item_a - 1);
    const size_t ib = static_cast<size_t>(item_b - 1);
    if (ia >= item_col.size() || ib >= item_col.size()) return false;
    return item_col[ia] == item_col[ib];
}


static size_t l3_eucs_index(int left_pos, int right_pos, int m) {
    if (left_pos > right_pos) std::swap(left_pos, right_pos);
    return static_cast<size_t>(left_pos) * static_cast<size_t>(m)
        - (static_cast<size_t>(left_pos) * static_cast<size_t>(left_pos + 1)) / 2
        + static_cast<size_t>(right_pos - left_pos - 1);
}

static bool l3_configure_eucs(const TransList& transactions,
                              const std::vector<int>& ci_for_item,
                              int m,
                              std::vector<double>& eucs_twu) {
    eucs_twu.clear();
    if (m <= 1) return false;
    if (!l3_eucs_env_enabled()) return false;

    const int eucs_min_items = l3_eucs_min_items_config();
    const size_t eucs_max_cells = l3_eucs_max_cells_config();
    const double eucs_max_density = l3_eucs_max_density_config();

    if (m <= eucs_min_items) {
        // Small item universes are already cheap enough that the quadratic
        // cache can cost more than it saves.
        return false;
    }

    size_t active_total = 0;
    size_t active_rows = 0;
    for (const Trans& trans : transactions) {
        if (trans.size() == 1 && trans[0] == -1) continue;
        size_t active_count = 0;
        for (int it : trans) {
            if (it > 0 && static_cast<size_t>(it) < ci_for_item.size()
                && ci_for_item[static_cast<size_t>(it)] >= 0) {
                ++active_count;
            }
        }
        if (active_count > 0) {
            active_total += active_count;
            ++active_rows;
        }
    }

    const double active_density = (active_rows == 0)
        ? 1.0
        : (static_cast<double>(active_total) / static_cast<double>(active_rows))
            / static_cast<double>(m);
    const size_t cells = (static_cast<size_t>(m) * static_cast<size_t>(m - 1)) / 2;
    if (cells > eucs_max_cells || active_density > eucs_max_density) return false;

    eucs_twu.assign(cells, 0.0);
    return true;
}

// See l2_update_eucs_for_active in mining_l2.cpp for the rationale: this
// runs once per transaction and is O(active^2) internally, and the
// original re-derived item_a's source column from item_col on every one
// of the O(A^2) (ai, bi) pairs via same_feature_l3(...), even though it
// only depends on ai. pos_scratch / col_scratch are precomputed once per
// transaction (O(A)) and reused across the O(A^2) inner loop; both are
// owned by the caller and reused across the whole transaction scan, so
// this adds no additional per-transaction heap allocation over the
// original.
constexpr int kL3NoColumn = -1;

static void l3_update_eucs_for_active(const std::vector<std::pair<int, double>>& active,
                                      const std::vector<int>& ci_for_item,
                                      const std::vector<int>& item_col,
                                      int m,
                                      std::vector<double>& eucs_twu,
                                      std::vector<int>& pos_scratch,
                                      std::vector<int>& col_scratch) {
    if (eucs_twu.empty()) return;
    const int A = static_cast<int>(active.size());
    double transaction_utility = 0.0;
    for (const auto& item : active) transaction_utility += item.second;

    if (pos_scratch.size() < static_cast<size_t>(A)) pos_scratch.resize(static_cast<size_t>(A));
    if (col_scratch.size() < static_cast<size_t>(A)) col_scratch.resize(static_cast<size_t>(A));

    // Same fallback semantics as same_feature_l3's guards (item_col.empty(),
    // item <= 0, or an out-of-range index all map to kL3NoColumn, which
    // never equals a real column id), evaluated once per item instead of
    // once per pair.
    for (int k = 0; k < A; ++k) {
        const int item = active[static_cast<size_t>(k)].first;
        pos_scratch[static_cast<size_t>(k)] = ci_for_item[static_cast<size_t>(item)];
        const size_t idx = static_cast<size_t>(item - 1);
        col_scratch[static_cast<size_t>(k)] =
            (!item_col.empty() && item > 0 && idx < item_col.size())
            ? item_col[idx] : kL3NoColumn;
    }

    for (int ai = 0; ai < A; ++ai) {
        const int pos_a = pos_scratch[static_cast<size_t>(ai)];
        if (pos_a < 0) continue;
        const int col_a = col_scratch[static_cast<size_t>(ai)];
        const int item_a = active[static_cast<size_t>(ai)].first;

        for (int bi = ai + 1; bi < A; ++bi) {
            const int item_b = active[static_cast<size_t>(bi)].first;
            if (item_a == item_b) continue;
            if (col_a != kL3NoColumn && col_a == col_scratch[static_cast<size_t>(bi)]) continue;
            const int pos_b = pos_scratch[static_cast<size_t>(bi)];
            if (pos_b < 0 || pos_a == pos_b) continue;
            eucs_twu[l3_eucs_index(pos_a, pos_b, m)] += transaction_utility;
        }
    }
}

static bool conflicts_with_root_pair_l3(const std::vector<int>& item_col,
                                        int root_item,
                                        int pair_item,
                                        int candidate_item) {
    return same_feature_l3(item_col, root_item, candidate_item)
        || same_feature_l3(item_col, pair_item, candidate_item);
}

static L3PairUL build_pair_ul_l3(const L3UL& ux,
                                 const L3UL& uy,
                                 const std::vector<int>& ytrain,
                                 int n_cls) {
    L3PairUL out;
    out.item_x = ux.item;
    out.item_y = uy.item;
    if (ux.tid.empty() || uy.tid.empty()) return out;
    out.class_counts.assign(static_cast<size_t>(n_cls), 0);

    out.tid.reserve(std::min(ux.tid.size(), uy.tid.size()));
    out.iu.reserve(out.tid.capacity());
    size_t xi = 0, yi = 0;
    while (xi < ux.tid.size() && yi < uy.tid.size()) {
        const int32_t xt = ux.tid[xi];
        const int32_t yt = uy.tid[yi];
        if (xt < yt) { ++xi; continue; }
        if (yt < xt) { ++yi; continue; }

        const double iu = ux.iu[xi] + uy.iu[yi];
        const double ru = uy.ru[yi];
        const double rmax = (yi < uy.rmax.size()) ? uy.rmax[yi] : 0.0;
        out.tid.push_back(xt);
        out.iu.push_back(iu);
        out.sI += iu;
        out.sR += ru;
        out.sRMax1 += rmax;
        const int lbl = ytrain[static_cast<size_t>(xt)];
        if (lbl >= 0 && lbl < n_cls) out.class_counts[static_cast<size_t>(lbl)]++;
        ++xi; ++yi;
    }
    return out;
}

static L3Triplet build_triplet_l3(const L3UL& root,
                                  const L3PairUL& pxy,
                                  const L3PairUL& pxz,
                                  const std::vector<int>& ytrain,
                                  int n_cls) {
    L3Triplet out;
    out.item_x = pxy.item_x;
    out.item_y = pxy.item_y;
    out.item_z = pxz.item_y;
    if (pxy.tid.empty() || pxz.tid.empty() || root.tid.empty()) return out;

    std::vector<int> cnt_triplet(static_cast<size_t>(n_cls), 0);
    size_t yi = 0, zi = 0, ri = 0;
    while (yi < pxy.tid.size() && zi < pxz.tid.size()) {
        const int32_t yt = pxy.tid[yi];
        const int32_t zt = pxz.tid[zi];
        if (yt < zt) { ++yi; continue; }
        if (zt < yt) { ++zi; continue; }

        while (ri < root.tid.size() && root.tid[ri] < yt) ++ri;
        if (ri == root.tid.size() || root.tid[ri] != yt) {
            ++yi; ++zi;
            continue;
        }
        out.sI += pxy.iu[yi] + pxz.iu[zi] - root.iu[ri];
        const int lbl = ytrain[static_cast<size_t>(yt)];
        if (lbl >= 0 && lbl < n_cls) cnt_triplet[static_cast<size_t>(lbl)]++;
        ++yi; ++zi;
    }
    const int child_total = 0;
    (void)child_total;
    int support = 0;
    for (int c : cnt_triplet) support += c;
    out.ig = l3_ig_from_counts(root.class_counts,
                               static_cast<int>(root.tid.size()),
                               cnt_triplet,
                               support);
    return out;
}

static bool reconstruct_pair_tid_l3(const L3UL& ux,
                                    const L3UL& uy,
                                    std::vector<int32_t>& out) {
    out.clear();
    if (ux.tid.empty() || uy.tid.empty()) return false;
    out.reserve(std::min(ux.tid.size(), uy.tid.size()));
    size_t xi = 0, yi = 0;
    while (xi < ux.tid.size() && yi < uy.tid.size()) {
        if (ux.tid[xi] < uy.tid[yi]) { ++xi; continue; }
        if (uy.tid[yi] < ux.tid[xi]) { ++yi; continue; }
        out.push_back(ux.tid[xi]);
        ++xi; ++yi;
    }
    return !out.empty();
}

static bool reconstruct_triplet_tid_l3(const L3PairUL& pxy,
                                       const L3PairUL& pxz,
                                       std::vector<int32_t>& out) {
    out.clear();
    if (pxy.tid.empty() || pxz.tid.empty()) return false;
    out.reserve(std::min(pxy.tid.size(), pxz.tid.size()));
    size_t yi = 0, zi = 0;
    while (yi < pxy.tid.size() && zi < pxz.tid.size()) {
        if (pxy.tid[yi] < pxz.tid[zi]) { ++yi; continue; }
        if (pxz.tid[zi] < pxy.tid[yi]) { ++zi; continue; }
        out.push_back(pxy.tid[yi]);
        ++yi; ++zi;
    }
    return !out.empty();
}

}  // namespace

std::vector<PatternEntry> mine_patterns_l3_cpp(
    const TransactionDataCpp& td,
    const std::vector<int>&   ytrain,
    int n_cls, int K, double G,
    double timeout_s)
{
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    static constexpr int CHECK_INTERVAL = 256;

    bool has_deadline = (timeout_s > 0.0);
    TimePoint deadline_tp = {};
    if (has_deadline) {
        deadline_tp = Clock::now() +
            std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(timeout_s));
    }
    auto timed_out = [&]() -> bool { return has_deadline && Clock::now() >= deadline_tp; };

    double minU = td.riu_thresh(K);
    const int n_train = static_cast<int>(ytrain.size());
    const int n_items = static_cast<int>(td.item_twu.size());

    std::vector<int> sorted_items;
    sorted_items.reserve(static_cast<size_t>(n_items));
    for (int iid = 1; iid <= n_items; iid++) {
        if (td.item_twu[static_cast<size_t>(iid - 1)] >= minU)
            sorted_items.push_back(iid);
    }
    std::sort(sorted_items.begin(), sorted_items.end(),
              [&](int a, int b) {
                  const double ta = td.item_twu[static_cast<size_t>(a - 1)];
                  const double tb = td.item_twu[static_cast<size_t>(b - 1)];
                  return (ta < tb) || (ta == tb && a < b);
              });

    const int m = static_cast<int>(sorted_items.size());
    if (m == 0) return {};

    std::vector<int> ci_for_item(static_cast<size_t>(n_items) + 1, -1);
    for (int pos = 0; pos < m; pos++)
        ci_for_item[static_cast<size_t>(sorted_items[static_cast<size_t>(pos)])] = pos;

    std::vector<double> eucs_twu;
    const bool eucs_enabled = l3_configure_eucs(
        td.transactions, ci_for_item, m, eucs_twu);
    // Reused across the whole transaction scan by l3_update_eucs_for_active
    // (see its definition): avoids a fresh per-transaction allocation for
    // these O(active-size) buffers.
    std::vector<int> eucs_pos_scratch;
    std::vector<int> eucs_col_scratch;

    std::vector<L3UL> uls(static_cast<size_t>(m));
    for (int ci = 0; ci < m; ci++) {
        L3UL& ul = uls[static_cast<size_t>(ci)];
        ul.item = sorted_items[static_cast<size_t>(ci)];
        ul.class_counts.assign(static_cast<size_t>(n_cls), 0);
    }

    std::vector<int> cnt_global(static_cast<size_t>(n_cls), 0);
    for (int lbl : ytrain)
        if (lbl >= 0 && lbl < n_cls) cnt_global[static_cast<size_t>(lbl)]++;

    int loop_ctr = 0;
    const int n_tx = static_cast<int>(td.transactions.size());
    for (int tid = 0; tid < n_tx; tid++) {
        if (has_deadline && ((++loop_ctr & (CHECK_INTERVAL - 1)) == 0) && timed_out())
            return {};

        const Trans& trans = td.transactions[static_cast<size_t>(tid)];
        if (trans.size() == 1 && trans[0] == -1) continue;

        std::vector<std::pair<int, double>> active;
        active.reserve(trans.size());
        for (size_t pos = 0; pos < trans.size(); ++pos) {
            const int it = trans[pos];
            if (it <= 0 || it > n_items) continue;
            if (ci_for_item[static_cast<size_t>(it)] < 0) continue;
            double u = 0.0;
            if (static_cast<size_t>(tid) < td.transaction_utils.size() &&
                pos < td.transaction_utils[static_cast<size_t>(tid)].size()) {
                u = td.transaction_utils[static_cast<size_t>(tid)][pos];
            } else if (static_cast<size_t>(it - 1) < td.item_iu.size()) {
                u = td.item_iu[static_cast<size_t>(it - 1)];
            }
            active.push_back({it, u});
        }
        if (active.empty()) continue;

        std::sort(active.begin(), active.end(), [&](auto& a, auto& b) {
            return ci_for_item[static_cast<size_t>(a.first)] <
                   ci_for_item[static_cast<size_t>(b.first)];
        });

        if (eucs_enabled) {
            l3_update_eucs_for_active(
                active, ci_for_item, td.item_col, m, eucs_twu,
                eucs_pos_scratch, eucs_col_scratch);
        }

        double rem = 0.0;
        double rem_max = 0.0;
        const int lbl = (tid < static_cast<int>(ytrain.size())) ? ytrain[static_cast<size_t>(tid)] : -1;
        const bool lbl_valid = (lbl >= 0 && lbl < n_cls);
        for (int ai = static_cast<int>(active.size()) - 1; ai >= 0; --ai) {
            const int it = active[static_cast<size_t>(ai)].first;
            const double u = active[static_cast<size_t>(ai)].second;
            const int ci = ci_for_item[static_cast<size_t>(it)];
            L3UL& ul = uls[static_cast<size_t>(ci)];
            ul.sI += u;
            ul.sR += rem;
            ul.tid.push_back(static_cast<int32_t>(tid));
            ul.iu.push_back(u);
            ul.ru.push_back(rem);
            ul.rmax.push_back(rem_max);
            if (lbl_valid) ul.class_counts[static_cast<size_t>(lbl)]++;
            rem += u;
            if (u > rem_max) rem_max = u;
        }
    }

    for (int ci = 0; ci < m; ++ci) {
        L3UL& ul = uls[static_cast<size_t>(ci)];
        ul.ig = l3_ig_from_counts(cnt_global, n_train, ul.class_counts,
                                  static_cast<int>(ul.tid.size()));
    }

    std::vector<PatternEntry> heap;
    heap.reserve(static_cast<size_t>(K) + 1);

    // Depth 0 save loop — same reverse order as THUIsl::explore({}).
    for (int ci = m - 1; ci >= 0; --ci) {
        const L3UL& ux = uls[static_cast<size_t>(ci)];
        if (ux.sI >= minU && ux.sI > 0.0 && ux.ig > G)
            l3_save(heap, minU, K, std::vector<int>{ux.item}, ux.sI, ux.ig);
    }


    std::vector<double> root_triplet_ub(static_cast<size_t>(m), -std::numeric_limits<double>::infinity());

    // Bounded cache of each root's Phase-A pair utility-lists, reused by
    // Phase B instead of recomputing them (see kL3PairCacheMaxTidsDefault
    // above). pair_cache[i]/pair_cache_valid[i] are only ever written by
    // the single thread that owns root i within a given phase's parallel
    // region, and Phase A's implicit OpenMP barrier happens-before Phase B
    // reads them, so no synchronization beyond the atomic budget is needed.
    const bool pair_cache_on = l3_pair_cache_enabled();
    const long long pair_cache_budget_max = static_cast<long long>(l3_pair_cache_max_tids_config());
    std::vector<std::vector<L3PairUL>> pair_cache(pair_cache_on ? static_cast<size_t>(m) : size_t{0});
    std::vector<char> pair_cache_valid(pair_cache_on ? static_cast<size_t>(m) : size_t{0}, 0);
    std::atomic<long long> pair_cache_budget_used{0};

    // Phase A: mine singleton/pair candidates first.  This intentionally does
    // not enumerate triplets yet.  Once all pair candidates have been merged,
    // minU is the global top-K floor among all length <= 2 patterns; in common
    // L3 classifier runs that floor is much higher than the RIU seed and lets
    // Phase B reject most triplet branches before intersection.
#ifdef _OPENMP
    const int n_threads = omp_get_max_threads();
    std::vector<std::vector<PatternEntry>> pair_thread_heaps(static_cast<size_t>(n_threads));
    std::vector<double> pair_thread_minU(static_cast<size_t>(n_threads), minU);
    for (auto& th : pair_thread_heaps) th.reserve(static_cast<size_t>(K) + 1);
    std::atomic<bool> pair_parallel_timed_out{false};

    #pragma omp parallel for schedule(dynamic, 2) if(m > 32)
    for (int i = m - 2; i >= 0; --i) {
        if (has_deadline && timed_out()) pair_parallel_timed_out.store(true, std::memory_order_relaxed);
        if (pair_parallel_timed_out.load(std::memory_order_relaxed)) continue;

        const int tid_omp = omp_get_thread_num();
        auto& local_heap = pair_thread_heaps[static_cast<size_t>(tid_omp)];
        double& local_minU = pair_thread_minU[static_cast<size_t>(tid_omp)];
#else
    for (int i = m - 2; i >= 0; --i) {
        if (has_deadline && ((++loop_ctr & (CHECK_INTERVAL - 1)) == 0) && timed_out())
            return heap;
        auto& local_heap = heap;
        double& local_minU = minU;
#endif
        const L3UL& ux = uls[static_cast<size_t>(i)];
        if (ux.ig <= G) continue;
        if (ux.sI + ux.sR < local_minU || ux.sI <= 0.0) continue;

        std::vector<L3PairUL> pairs;
        pairs.reserve(static_cast<size_t>(m - i - 1));
        std::vector<size_t> pair_cov_hashes;
        long long pairs_tid_total = 0;

        for (int j = i + 1; j < m; ++j) {
            const L3UL& uy = uls[static_cast<size_t>(j)];
            if (same_feature_l3(td.item_col, ux.item, uy.item)) continue;
            if (uy.ig <= G) continue;
            if (eucs_enabled) {
                const size_t eidx = l3_eucs_index(i, j, m);
                if (eucs_twu[eidx] < local_minU) continue;
            }

            L3PairUL pair = build_pair_ul_l3(ux, uy, ytrain, n_cls);
            if (pair.tid.empty()) continue;
            pair.ig = l3_ig_from_counts(cnt_global, n_train, pair.class_counts,
                                        static_cast<int>(pair.tid.size()));
            if (pair.ig <= G) continue;

            const size_t h = l3_fnv1a_tids(pair.tid);
            bool duplicate_tids = false;
            if (l3_vec_contains(pair_cov_hashes, h)) {
                for (const L3PairUL& existing : pairs) {
                    if (existing.tid == pair.tid) { duplicate_tids = true; break; }
                }
            }
            if (duplicate_tids) continue;
            pair_cov_hashes.push_back(h);
            if (pair_cache_on) pairs_tid_total += static_cast<long long>(pair.tid.size());
            pairs.push_back(std::move(pair));
        }

        // Reverse save order mirrors THUIsl::explore at depth 1.
        for (int e = static_cast<int>(pairs.size()) - 1; e >= 0; --e) {
            const L3PairUL& p = pairs[static_cast<size_t>(e)];
            if (p.sI >= local_minU && p.sI > 0.0 && p.ig > G)
                l3_save(local_heap, local_minU, K,
                        std::vector<int>{p.item_x, p.item_y}, p.sI, p.ig);
        }

        double root_ub = -std::numeric_limits<double>::infinity();
        for (int e = 0; e + 1 < static_cast<int>(pairs.size()); ++e) {
            const L3PairUL& p = pairs[static_cast<size_t>(e)];
            if (p.ig > G && p.sI > 0.0)
                root_ub = std::max(root_ub, p.sI + p.sRMax1);
        }
        root_triplet_ub[static_cast<size_t>(i)] = root_ub;

        // Cache this root's pairs for Phase B reuse, but only when it can
        // possibly matter there. local_minU only rises over time, and Phase
        // B applies the exact same "root_triplet_ub[i] < local_minU" gate
        // (against a value that can only be >= this one) -- so a root that
        // already fails it here is guaranteed to fail it in Phase B too,
        // meaning its cached pairs would never be read back. Skipping the
        // cache for those roots avoids paying retention cost (extra live
        // memory competing for cache lines across the rest of Phase A) with
        // no matching benefit, which matters most exactly when *no* root
        // survives to Phase B and the whole phase is skipped by the
        // global_triplet_ub check below.
        if (pair_cache_on && root_ub >= local_minU) {
            const long long prev = pair_cache_budget_used.fetch_add(
                pairs_tid_total, std::memory_order_relaxed);
            if (prev + pairs_tid_total <= pair_cache_budget_max) {
                pair_cache[static_cast<size_t>(i)] = std::move(pairs);
                pair_cache_valid[static_cast<size_t>(i)] = 1;
            } else {
                pair_cache_budget_used.fetch_sub(pairs_tid_total, std::memory_order_relaxed);
            }
        }
    }

#ifdef _OPENMP
    if (pair_parallel_timed_out.load(std::memory_order_relaxed)) return heap;
    for (int t = 0; t < n_threads; ++t) {
        for (const PatternEntry& pe : pair_thread_heaps[static_cast<size_t>(t)]) {
            l3_save(heap, minU, K, pe.items, pe.utility, pe.ig);
        }
    }
#endif

    double global_triplet_ub = -std::numeric_limits<double>::infinity();
    for (double ub : root_triplet_ub) global_triplet_ub = std::max(global_triplet_ub, ub);
    if (static_cast<int>(heap.size()) == K && global_triplet_ub < minU) {
        return heap;
    }

    // Phase B: enumerate only terminal triplets using the pair-established
    // global minU.  Pair ULs are reused from the Phase A cache when available
    // (see kL3PairCacheMaxTidsDefault); once the cache's tid-count budget is
    // exhausted, remaining roots fall back to reconstructing their pair ULs
    // here, exactly as before this optimization -- so pathological inputs
    // degrade to the prior (bounded-memory, recompute-per-root) behaviour
    // rather than growing memory unboundedly.
#ifdef _OPENMP
    std::vector<std::vector<PatternEntry>> trip_thread_heaps(static_cast<size_t>(n_threads));
    std::vector<double> trip_thread_minU(static_cast<size_t>(n_threads), minU);
    for (auto& th : trip_thread_heaps) th.reserve(static_cast<size_t>(K) + 1);
    std::atomic<bool> trip_parallel_timed_out{false};

    #pragma omp parallel for schedule(dynamic, 2) if(m > 32)
    for (int i = m - 2; i >= 0; --i) {
        if (has_deadline && timed_out()) trip_parallel_timed_out.store(true, std::memory_order_relaxed);
        if (trip_parallel_timed_out.load(std::memory_order_relaxed)) continue;

        const int tid_omp = omp_get_thread_num();
        auto& local_heap = trip_thread_heaps[static_cast<size_t>(tid_omp)];
        double& local_minU = trip_thread_minU[static_cast<size_t>(tid_omp)];
#else
    for (int i = m - 2; i >= 0; --i) {
        if (has_deadline && ((++loop_ctr & (CHECK_INTERVAL - 1)) == 0) && timed_out())
            return heap;
        auto& local_heap = heap;
        double& local_minU = minU;
#endif
        const L3UL& ux = uls[static_cast<size_t>(i)];
        if (ux.ig <= G) continue;
        if (ux.sI + ux.sR < local_minU || ux.sI <= 0.0) continue;
        if (root_triplet_ub[static_cast<size_t>(i)] < local_minU) continue;

        std::vector<L3PairUL> pairs;
        if (pair_cache_on && pair_cache_valid[static_cast<size_t>(i)]) {
            pairs = std::move(pair_cache[static_cast<size_t>(i)]);
            pair_cache_valid[static_cast<size_t>(i)] = 0;
        } else {
            pairs.reserve(static_cast<size_t>(m - i - 1));
            std::vector<size_t> pair_cov_hashes;

            for (int j = i + 1; j < m; ++j) {
                const L3UL& uy = uls[static_cast<size_t>(j)];
                if (same_feature_l3(td.item_col, ux.item, uy.item)) continue;
                if (uy.ig <= G) continue;
                if (eucs_enabled) {
                    const size_t eidx = l3_eucs_index(i, j, m);
                    if (eucs_twu[eidx] < local_minU) continue;
                }

                L3PairUL pair = build_pair_ul_l3(ux, uy, ytrain, n_cls);
                if (pair.tid.empty()) continue;
                pair.ig = l3_ig_from_counts(cnt_global, n_train, pair.class_counts,
                                            static_cast<int>(pair.tid.size()));
                if (pair.ig <= G) continue;

                const size_t h = l3_fnv1a_tids(pair.tid);
                bool duplicate_tids = false;
                if (l3_vec_contains(pair_cov_hashes, h)) {
                    for (const L3PairUL& existing : pairs) {
                        if (existing.tid == pair.tid) { duplicate_tids = true; break; }
                    }
                }
                if (duplicate_tids) continue;
                pair_cov_hashes.push_back(h);
                pairs.push_back(std::move(pair));
            }
        }

        if (pairs.size() <= 1) continue;

        for (int pi = static_cast<int>(pairs.size()) - 2; pi >= 0; --pi) {
#ifndef _OPENMP
            if (has_deadline && ((++loop_ctr & (CHECK_INTERVAL - 1)) == 0) && timed_out())
                return heap;
#endif
            const L3PairUL& pxy = pairs[static_cast<size_t>(pi)];
            if (pxy.ig <= G) continue;
            if (pxy.sI + pxy.sRMax1 < local_minU || pxy.sI <= 0.0) continue;

            std::vector<L3Triplet> triplets;
            std::vector<size_t> trip_cov_hashes;
            triplets.reserve(pairs.size() - static_cast<size_t>(pi) - 1);

            for (int zi = pi + 1; zi < static_cast<int>(pairs.size()); ++zi) {
                const L3PairUL& pxz = pairs[static_cast<size_t>(zi)];
                if (conflicts_with_root_pair_l3(td.item_col, ux.item, pxy.item_y, pxz.item_y))
                    continue;
                if (pxz.ig <= G) continue;
                if (eucs_enabled) {
                    const int y_pos = ci_for_item[static_cast<size_t>(pxy.item_y)];
                    const int z_pos = ci_for_item[static_cast<size_t>(pxz.item_y)];
                    if (y_pos < 0 || z_pos < 0 || y_pos == z_pos) continue;
                    if (eucs_twu[l3_eucs_index(y_pos, z_pos, m)] < local_minU) continue;
                }

                L3Triplet tri = build_triplet_l3(ux, pxy, pxz, ytrain, n_cls);
                if (tri.sI < local_minU || tri.sI <= 0.0) continue;
                if (tri.ig <= G) continue;

                std::vector<int32_t> tri_tid;
                reconstruct_triplet_tid_l3(pxy, pxz, tri_tid);
                if (tri_tid.empty()) continue;
                const size_t h = l3_fnv1a_tids(tri_tid);
                bool duplicate_tids = false;
                if (l3_vec_contains(trip_cov_hashes, h)) {
                    for (const L3Triplet& existing : triplets) {
                        const L3PairUL* exz = nullptr;
                        for (const L3PairUL& cand : pairs) {
                            if (cand.item_y == existing.item_z) { exz = &cand; break; }
                        }
                        if (!exz) continue;
                        std::vector<int32_t> e_tid;
                        reconstruct_triplet_tid_l3(pxy, *exz, e_tid);
                        if (e_tid == tri_tid) { duplicate_tids = true; break; }
                    }
                }
                if (duplicate_tids) continue;
                trip_cov_hashes.push_back(h);
                triplets.push_back(tri);
            }

            for (int e = static_cast<int>(triplets.size()) - 1; e >= 0; --e) {
                const L3Triplet& tri = triplets[static_cast<size_t>(e)];
                if (tri.sI >= local_minU && tri.sI > 0.0 && tri.ig > G) {
                    l3_save(local_heap, local_minU, K,
                            std::vector<int>{tri.item_x, tri.item_y, tri.item_z},
                            tri.sI, tri.ig);
                }
            }
        }
    }

#ifdef _OPENMP
    if (trip_parallel_timed_out.load(std::memory_order_relaxed)) return heap;
    for (int t = 0; t < n_threads; ++t) {
        for (const PatternEntry& pe : trip_thread_heaps[static_cast<size_t>(t)]) {
            l3_save(heap, minU, K, pe.items, pe.utility, pe.ig);
        }
    }
#endif

    return heap;
}

}  // namespace hugiml
