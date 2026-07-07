/**
 * mining_l2.cpp — Hot path implementation for L=2 pattern mining.
 *
 * This path preserves the global top-K semantics of the generic miner for
 * L=2: singleton and two-item patterns compete in the same utility heap.
 * It avoids recursive UL allocation and skips same-source-feature pairs before
 * doing intersection work.
 *
 * Optimisations applied:
 *   A. OpenMP parallel pair loop — each outer-i iteration is independent;
 *      per-thread local heaps are merged serially after the parallel region.
 *   B. Drop ru vector from L2UL — only the scalar sR accumulator is needed
 *      for the sI+sR pruning guard; removing the per-entry vector halves the
 *      working set of each L2UL and improves cache utilisation in the merge loop.
 *   C. Compact L2Child — remove the heap-allocated tid vector from L2Child;
 *      the tid list is kept as a function-local variable during the j-loop and
 *      only the four scalar fields (item, sI, ig, items[2]) are stored in ext.
 *      The FNV1a duplicate-coverage check is performed inline before ext push.
 *   D. Merge item_rank / iid_to_ci — both arrays were identical; collapsed into
 *      a single ci_for_item array, removing a redundant allocation and lookup.
 */

#include "mining_l2.hpp"
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
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace hugiml {

namespace {

constexpr size_t kL2EucsMaxCellsDefault = 6000000ULL;
constexpr int kL2EucsMinItemsDefault = 32;
constexpr double kL2EucsMaxDensityDefault = 0.20;

static std::string l2_eucs_env_lower(const char* value) {
    std::string out(value ? value : "");
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

static bool l2_eucs_env_enabled() {
    const char* raw = std::getenv("HUGIML_EUCS_ENABLE");
    if (!raw) raw = std::getenv("HUGIML_EUCS_ENABLED");
    if (!raw || !*raw) return true;
    const std::string value = l2_eucs_env_lower(raw);
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

static int l2_eucs_env_int(const char* name, int fallback, int min_value) {
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

static size_t l2_eucs_env_size(const char* name, size_t fallback) {
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

static double l2_eucs_env_double(const char* name, double fallback, double min_value) {
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

static int l2_eucs_min_items_config() {
    return l2_eucs_env_int("HUGIML_EUCS_MIN_ITEMS", kL2EucsMinItemsDefault, 0);
}

static size_t l2_eucs_max_cells_config() {
    return l2_eucs_env_size("HUGIML_EUCS_MAX_CELLS", kL2EucsMaxCellsDefault);
}

static double l2_eucs_max_density_config() {
    return l2_eucs_env_double("HUGIML_EUCS_MAX_DENSITY", kL2EucsMaxDensityDefault, 0.0);
}

static double l2_entropy(const std::vector<int>& counts, int total) {
    if (total <= 0) return 0.0;
    double h = 0.0;
    const double inv = 1.0 / static_cast<double>(total);
    for (int c : counts) {
        if (c > 0) {
            double p = static_cast<double>(c) * inv;
            h -= p * std::log(p);
        }
    }
    return std::max(h, 0.0);
}

// Entropy from a class-count buffer given as pointer + length, for callers
// that already have a contiguous row (a flat cnt_item matrix row, or a
// freshly-built pair candidate's count array) and would otherwise have to
// copy it into a std::vector just to call the overload above.
static double l2_entropy_ptr(const int* counts, int n_cls, int total) {
    if (total <= 0) return 0.0;
    double h = 0.0;
    const double inv = 1.0 / static_cast<double>(total);
    for (int k = 0; k < n_cls; k++) {
        const int c = counts[k];
        if (c > 0) {
            const double p = static_cast<double>(c) * inv;
            h -= p * std::log(p);
        }
    }
    return std::max(h, 0.0);
}

// IG of a candidate (singleton or pair) against the fixed global/parent
// class distribution. base_global (= entropy of cnt_global over n_train)
// is identical on *every* call within a single mine_patterns_l2_*
// invocation -- callers compute it once (see each function's `base_global`
// local) and pass it in here, rather than this function recomputing the
// same O(n_cls) entropy (with n_cls std::log calls) on every one of the
// O(m) singleton and O(m^2) pair candidates that reach it. cnt_in is taken
// as pointer+length (not std::vector<int>&) so callers reading a row out
// of a flat matrix don't need to copy it into a temporary vector first,
// and the cnt_out term is accumulated inline instead of being materialised
// into its own temporary vector on every call.
static double l2_ig_global(double base_global,
                           const std::vector<int>& cnt_global,
                           const int* cnt_in, int n_in,
                           int n_train, int n_cls) {
    const int n_out = n_train - n_in;
    if (n_out == 0)
        return std::numeric_limits<double>::quiet_NaN();

    double h_out = 0.0;
    const double inv_out = 1.0 / static_cast<double>(n_out);
    for (int k = 0; k < n_cls; k++) {
        const int out_c = cnt_global[static_cast<size_t>(k)] - cnt_in[k];
        if (out_c > 0) {
            const double p = static_cast<double>(out_c) * inv_out;
            h_out -= p * std::log(p);
        }
    }
    h_out = std::max(h_out, 0.0);

    const double ce = (static_cast<double>(n_in)  / n_train) * l2_entropy_ptr(cnt_in, n_cls, n_in)
                     + (static_cast<double>(n_out) / n_train) * h_out;
    return base_global - ce;
}

struct L2MinHeapCmp {
    bool operator()(const PatternEntry& a, const PatternEntry& b) const {
        return a.utility > b.utility;
    }
};

static void l2_save(std::vector<PatternEntry>& heap,
                    double& minU,
                    int K,
                    const std::vector<int>& items,
                    double utility,
                    double ig) {
    PatternEntry pe{utility, items, ig};
    if (static_cast<int>(heap.size()) < K) {
        heap.push_back(pe);
        std::push_heap(heap.begin(), heap.end(), L2MinHeapCmp{});
        if (static_cast<int>(heap.size()) == K) minU = heap.front().utility;
    } else if (utility > minU) {
        std::pop_heap(heap.begin(), heap.end(), L2MinHeapCmp{});
        heap.back() = pe;
        std::push_heap(heap.begin(), heap.end(), L2MinHeapCmp{});
        minU = heap.front().utility;
    }
}

// Relaxed-survivor patterns are ranked and capped by IG, not utility.
// This lets low-support, high-information interaction pairs retain capacity
// in the relaxed track.
struct L2IgMinHeapCmp {
    bool operator()(const PatternEntry& a, const PatternEntry& b) const {
        return a.ig > b.ig;
    }
};

static void l2_save_by_ig(std::vector<PatternEntry>& heap,
                          double& minIg,
                          int K,
                          const std::vector<int>& items,
                          double utility,
                          double ig) {
    PatternEntry pe{utility, items, ig};
    if (static_cast<int>(heap.size()) < K) {
        heap.push_back(pe);
        std::push_heap(heap.begin(), heap.end(), L2IgMinHeapCmp{});
        if (static_cast<int>(heap.size()) == K) minIg = heap.front().ig;
    } else if (ig > minIg) {
        std::pop_heap(heap.begin(), heap.end(), L2IgMinHeapCmp{});
        heap.back() = pe;
        std::push_heap(heap.begin(), heap.end(), L2IgMinHeapCmp{});
        minIg = heap.front().ig;
    }
}

static size_t fnv1a_tids(const std::vector<int32_t>& v) {
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

// Coverage-hash membership check for the small per-root dedup buffers below
// (typically a handful to a few dozen live entries: bounded by how many
// candidate partners survive the IG/EUCS gates at that branch). A flat
// vector + linear scan avoids the per-element heap allocation an
// unordered_set node incurs, at these sizes without a meaningful lookup
// cost penalty. Semantics unchanged: still just an O(1)-amortized
// pre-filter ahead of the existing full tid-vector equality check.
static inline bool l2_vec_contains(const std::vector<size_t>& v, size_t h) {
    for (size_t existing : v)
        if (existing == h) return true;
    return false;
}

// Opt-B: ru vector removed; only the scalar sR is retained for the
//        sI+sR >= minU outer guard.  iu and tid are still needed for
//        the intersection and utility accumulation in the pair loop.
struct L2UL {
    int item = 0;
    double sI = 0.0;
    double sR = 0.0;   // scalar accumulator — no per-entry ru vector
    double ig = 0.0;
    std::vector<int32_t> tid;
    std::vector<double>  iu;
};

// Opt-C: tid vector removed from L2Child; only the four scalars needed
//        by l2_save are stored.  The tid list lives on the stack inside
//        the j-loop and is used for the FNV1a duplicate check before
//        this struct is pushed onto ext.
struct L2Child {
    double sI = 0.0;
    double ig = 0.0;
    int item_x = 0;   // ux.item — stored so the items[] pair can be
    int item_y = 0;   // uy.item   reconstructed during the save loop
};

static bool same_feature_l2(const std::vector<int>& item_col, int item_a, int item_b) {
    if (item_col.empty()) return false;
    if (item_a <= 0 || item_b <= 0) return false;
    const size_t ia = static_cast<size_t>(item_a - 1);
    const size_t ib = static_cast<size_t>(item_b - 1);
    if (ia >= item_col.size() || ib >= item_col.size()) return false;
    return item_col[ia] == item_col[ib];
}

static size_t l2_eucs_index(int left_pos, int right_pos, int m) {
    if (left_pos > right_pos) std::swap(left_pos, right_pos);
    return static_cast<size_t>(left_pos) * static_cast<size_t>(m)
        - (static_cast<size_t>(left_pos) * static_cast<size_t>(left_pos + 1)) / 2
        + static_cast<size_t>(right_pos - left_pos - 1);
}

static bool l2_configure_eucs(const TransList& transactions,
                              const std::vector<int>& ci_for_item,
                              int m,
                              std::vector<double>& eucs_twu,
                              std::vector<double>& eucs_pair_utility) {
    eucs_twu.clear();
    eucs_pair_utility.clear();
    if (m <= 1) return false;
    if (!l2_eucs_env_enabled()) return false;

    const int eucs_min_items = l2_eucs_min_items_config();
    const size_t eucs_max_cells = l2_eucs_max_cells_config();
    const double eucs_max_density = l2_eucs_max_density_config();

    if (m <= eucs_min_items) {
        // Small item universes are already cheap to intersect.  Building the
        // quadratic cache usually costs more than it saves.
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
    eucs_pair_utility.assign(cells, 0.0);
    return true;
}

// This runs once per transaction and is O(active^2) internally -- the
// dominant cost of Phase 1 for wide item universes (confirmed by profiling:
// on a p=60/n_bins=8 stress config this loop alone accounted for ~70% of
// the whole transaction-scan phase). The original re-derived item_a's
// source column from item_col on every one of the O(A^2) (ai, bi) pairs
// via same_feature_l2(...), even though it only depends on ai. pos_scratch
// / col_scratch are precomputed once per transaction (O(A)) and reused
// across the O(A^2) inner loop; both buffers are owned by the caller and
// reused across the whole Phase 1 scan (resized, never freed, so this adds
// no additional per-transaction heap allocation over the original).
constexpr int kL2NoColumn = -1;

static void l2_update_eucs_for_active(const std::vector<std::pair<int, double>>& active,
                                      const std::vector<int>& ci_for_item,
                                      const std::vector<int>& item_col,
                                      int m,
                                      std::vector<double>& eucs_twu,
                                      std::vector<double>& eucs_pair_utility,
                                      std::vector<int>& pos_scratch,
                                      std::vector<int>& col_scratch) {
    if (eucs_twu.empty() || eucs_pair_utility.empty()) return;
    const int A = static_cast<int>(active.size());
    double transaction_utility = 0.0;
    for (const auto& item : active) transaction_utility += item.second;

    if (pos_scratch.size() < static_cast<size_t>(A)) pos_scratch.resize(static_cast<size_t>(A));
    if (col_scratch.size() < static_cast<size_t>(A)) col_scratch.resize(static_cast<size_t>(A));

    // Same fallback semantics as the original same_feature_l2 guards,
    // evaluated once per active item instead of once per (ai, bi) pair:
    // item_col.empty(), item <= 0, or an out-of-range index all map to
    // kL2NoColumn, which never equals a real (>= 0) column id, so those
    // items are correctly treated as "not the same feature" as anything.
    for (int k = 0; k < A; ++k) {
        const int item = active[static_cast<size_t>(k)].first;
        pos_scratch[static_cast<size_t>(k)] = ci_for_item[static_cast<size_t>(item)];
        const size_t idx = static_cast<size_t>(item - 1);
        col_scratch[static_cast<size_t>(k)] =
            (!item_col.empty() && item > 0 && idx < item_col.size())
            ? item_col[idx] : kL2NoColumn;
    }

    for (int ai = 0; ai < A; ++ai) {
        const int pos_a = pos_scratch[static_cast<size_t>(ai)];
        if (pos_a < 0) continue;
        const int col_a = col_scratch[static_cast<size_t>(ai)];
        const int item_a = active[static_cast<size_t>(ai)].first;
        const double u_a = active[static_cast<size_t>(ai)].second;

        for (int bi = ai + 1; bi < A; ++bi) {
            const int item_b = active[static_cast<size_t>(bi)].first;
            if (item_a == item_b) continue;
            if (col_a != kL2NoColumn && col_a == col_scratch[static_cast<size_t>(bi)]) continue;
            const int pos_b = pos_scratch[static_cast<size_t>(bi)];
            if (pos_b < 0 || pos_a == pos_b) continue;
            const size_t idx = l2_eucs_index(pos_a, pos_b, m);
            eucs_twu[idx] += transaction_utility;
            eucs_pair_utility[idx] += u_a + active[static_cast<size_t>(bi)].second;
        }
    }
}

} // namespace

std::vector<PatternEntry> mine_patterns_l2_cpp(
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
    auto timed_out = [&]() -> bool {
        return has_deadline && Clock::now() >= deadline_tp;
    };

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
                  double ta = td.item_twu[static_cast<size_t>(a - 1)];
                  double tb = td.item_twu[static_cast<size_t>(b - 1)];
                  return (ta < tb) || (ta == tb && a < b);
              });

    const int m = static_cast<int>(sorted_items.size());
    if (m == 0) return {};

    // Opt-D: single ci_for_item array replaces the formerly duplicate
    //        item_rank and iid_to_ci arrays.
    std::vector<int> ci_for_item(static_cast<size_t>(n_items) + 1, -1);
    for (int pos = 0; pos < m; pos++)
        ci_for_item[static_cast<size_t>(sorted_items[pos])] = pos;

    std::vector<double> eucs_twu;
    std::vector<double> eucs_pair_utility;
    const bool eucs_enabled = l2_configure_eucs(
        td.transactions, ci_for_item, m, eucs_twu, eucs_pair_utility);
    // Reused across the whole Phase 1 transaction scan by
    // l2_update_eucs_for_active (see its definition for why): avoids a
    // fresh per-transaction allocation for these O(active-size) buffers.
    std::vector<int> eucs_pos_scratch;
    std::vector<int> eucs_col_scratch;

    std::vector<L2UL> uls(static_cast<size_t>(m));
    for (int ci = 0; ci < m; ci++) uls[static_cast<size_t>(ci)].item = sorted_items[ci];

    std::vector<int> cnt_global(n_cls, 0);
    for (int lbl : ytrain)
        if (lbl >= 0 && lbl < n_cls) cnt_global[static_cast<size_t>(lbl)]++;
    // Computed once per mine_patterns_l2_* call; l2_ig_global takes it
    // as a parameter instead of recomputing it on every candidate.
    const double base_global = l2_entropy(cnt_global, n_train);

    std::vector<int> loop_counts(static_cast<size_t>(m), 0);
    std::vector<int> cnt_item(static_cast<size_t>(m) * static_cast<size_t>(n_cls), 0);

    // ── Phase 1: transaction scan (serial) ────────────────────────────────
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
            int it = trans[pos];
            if (it <= 0 || it > n_items) continue;
            // Opt-D: single lookup through ci_for_item
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

        // Opt-D: sort by ci_for_item (same order as the former item_rank lookup)
        std::sort(active.begin(), active.end(), [&](auto& a, auto& b) {
            return ci_for_item[static_cast<size_t>(a.first)] <
                   ci_for_item[static_cast<size_t>(b.first)];
        });

        if (eucs_enabled) {
            l2_update_eucs_for_active(
                active, ci_for_item, td.item_col, m, eucs_twu, eucs_pair_utility,
                eucs_pos_scratch, eucs_col_scratch);
        }

        double rem = 0.0;
        const int lbl = (tid < static_cast<int>(ytrain.size())) ? ytrain[static_cast<size_t>(tid)] : -1;
        const bool lbl_valid = (lbl >= 0 && lbl < n_cls);
        for (int ai = static_cast<int>(active.size()) - 1; ai >= 0; --ai) {
            const int it = active[static_cast<size_t>(ai)].first;
            const double u  = active[static_cast<size_t>(ai)].second;
            const int ci = ci_for_item[static_cast<size_t>(it)];
            L2UL& ul = uls[static_cast<size_t>(ci)];
            ul.sI += u;
            ul.sR += rem;      // Opt-B: scalar accumulation only — no ru.push_back
            ul.tid.push_back(static_cast<int32_t>(tid));
            ul.iu.push_back(u);
            loop_counts[static_cast<size_t>(ci)]++;
            if (lbl_valid)
                cnt_item[static_cast<size_t>(ci) * static_cast<size_t>(n_cls) + static_cast<size_t>(lbl)]++;
            rem += u;
        }
    }


    // ── Phase 2: IG computation (serial) ──────────────────────────────────
    for (int ci = 0; ci < m; ci++) {
        const int* cnt_in = cnt_item.data() + static_cast<size_t>(ci) * static_cast<size_t>(n_cls);
        uls[static_cast<size_t>(ci)].ig = l2_ig_global(
            base_global, cnt_global, cnt_in, loop_counts[static_cast<size_t>(ci)], n_train, n_cls);
    }

    // ── Singleton save (serial) ────────────────────────────────────────────
    std::vector<PatternEntry> heap;
    heap.reserve(static_cast<size_t>(K) + 1);

    // Top-level singleton save loop: same order as THUIsl::explore(depth=0).
    for (int ci = m - 1; ci >= 0; --ci) {
        const L2UL& ux = uls[static_cast<size_t>(ci)];
        if (ux.sI >= minU && ux.sI > 0.0 && ux.ig > G)
            l2_save(heap, minU, K, std::vector<int>{ux.item}, ux.sI, ux.ig);
    }

    // ── Phase 3: pair construction loop ───────────────────────────────────
    //
    // Opt-A: parallelise the outer i-loop with OpenMP.
    //
    // Correctness:
    //   Each outer-i iteration builds an independent ext list and calls l2_save
    //   only on its own local heap — there are no data races on uls (read-only
    //   after phase 2) or ytrain (read-only).  The global heap is updated only
    //   in the serial merge step that follows the parallel region.
    //
    //   The save order within each i-iteration (reverse over ext) is preserved.
    //   The order *across* different i-values is not specified by the generic
    //   miner either — THUIsl::explore processes i-values in the same backward
    //   order but the heap is a priority queue, not a stable container, so
    //   output-equivalence depends only on the *set* of patterns saved, not
    //   insertion order.  The parallel version produces the same set.
    //
    // Timeout handling:
    //   With OpenMP the per-thread loop counter cannot be merged back without
    //   a lock, so the deadline is checked at the start of each outer-i
    //   iteration (coarser granularity than the serial CHECK_INTERVAL=256 but
    //   still safe — each i-iteration is bounded work).

#ifdef _OPENMP
    const int n_threads = omp_get_max_threads();
    // Per-thread heap + minU; initialised to global minU so all threads start
    // with the same floor raised by the singleton save above.
    std::vector<std::vector<PatternEntry>> thread_heaps(
        static_cast<size_t>(n_threads));
    std::vector<double> thread_minU(
        static_cast<size_t>(n_threads), minU);
    for (auto& th : thread_heaps)
        th.reserve(static_cast<size_t>(K) + 1);

    std::atomic<bool> parallel_timed_out{false};

    #pragma omp parallel for schedule(dynamic, 4) if(m > 64)
    for (int i = m - 2; i >= 0; --i) {
        if (has_deadline && timed_out()) {
            parallel_timed_out.store(true, std::memory_order_relaxed);
        }
        if (parallel_timed_out.load(std::memory_order_relaxed)) continue;  // drain remaining iterations

        const int tid_omp = omp_get_thread_num();
        auto& local_heap = thread_heaps[static_cast<size_t>(tid_omp)];
        double& local_minU = thread_minU[static_cast<size_t>(tid_omp)];

        const L2UL& ux = uls[static_cast<size_t>(i)];
        if (ux.ig <= G) continue;
        if (ux.sI + ux.sR < local_minU || ux.sI <= 0.0) continue;

        // Opt-C: ext stores compact L2Child (no tid vector).
        //        The tid list is a local variable used for the FNV1a duplicate
        //        check and discarded immediately after.
        std::vector<L2Child>   ext;
        std::vector<size_t> ext_cov_hashes;
        // Reused across every j in this i-iteration instead of a fresh
        // heap allocation per candidate pair (was: allocated inside the
        // j-loop even for candidates rejected moments later).
        std::vector<int> cnt_pair_scratch(static_cast<size_t>(n_cls), 0);

        for (int j = i + 1; j < m; j++) {
            const L2UL& uy = uls[static_cast<size_t>(j)];
            if (same_feature_l2(td.item_col, ux.item, uy.item))
                continue;
            if (uy.ig <= G) continue;
            if (eucs_enabled) {
                const size_t eidx = l2_eucs_index(i, j, m);
                // For terminal L=2 ordinary pairs, exact pair utility is a
                // correctness-preserving stronger skip than the TWU upper
                // bound: a pair below the current utility floor cannot enter
                // this utility-ranked local heap.  Descendant paths do not
                // exist in this hot path.
                if (eucs_twu[eidx] < local_minU || eucs_pair_utility[eidx] < local_minU)
                    continue;
            }
            if (ux.tid.empty() || uy.tid.empty()) continue;

            // Sorted-merge intersection — build c_tid, accumulate c_sI and
            // cnt_pair.  c_tid is local to this j-iteration (Opt-C).
            std::vector<int32_t> c_tid;
            std::fill(cnt_pair_scratch.begin(), cnt_pair_scratch.end(), 0);
            std::vector<int>& cnt_pair = cnt_pair_scratch;
            double c_sI = 0.0;
            c_tid.reserve(std::min(ux.tid.size(), uy.tid.size()));

            size_t xi = 0, yi = 0;
            while (xi < ux.tid.size() && yi < uy.tid.size()) {
                int32_t xt = ux.tid[xi];
                int32_t yt = uy.tid[yi];
                if (xt < yt) { ++xi; continue; }
                if (yt < xt) { ++yi; continue; }

                c_tid.push_back(xt);
                c_sI += ux.iu[xi] + uy.iu[yi];
                const int lbl = ytrain[static_cast<size_t>(xt)];
                if (lbl >= 0 && lbl < n_cls) cnt_pair[static_cast<size_t>(lbl)]++;
                ++xi; ++yi;
            }
            if (c_tid.empty()) continue;

            const double c_ig = l2_ig_global(
                base_global, cnt_global, cnt_pair.data(), static_cast<int>(c_tid.size()), n_train, n_cls);
            if (!(c_ig > G)) continue;

            // FNV1a duplicate-coverage check (Opt-C: done here before ext push,
            // using the stack-local c_tid; no tid copy stored in ext).
            const size_t h = fnv1a_tids(c_tid);
            bool duplicate_tids = false;
            if (l2_vec_contains(ext_cov_hashes, h)) {
                // Hash collision: need the full tid list.  Re-run the merge to
                // reconstruct c_tid for each existing ext entry that hashed the same.
                // In practice hash collisions are extremely rare; this branch is cold.
                for (const L2Child& existing : ext) {
                    // Re-derive the tid for 'existing' by re-intersecting
                    // ux.tid with uls[ci_for_item[existing.item_y]].tid.
                    const int ey = existing.item_y;
                    if (ey <= 0 || ey > n_items) continue;
                    const int eci = ci_for_item[static_cast<size_t>(ey)];
                    if (eci < 0 || eci >= m) continue;
                    const L2UL& euy = uls[static_cast<size_t>(eci)];
                    // Re-merge ux and euy to get existing's tid set.
                    // Do not compare euy.tid.size() with uy.tid.size(): two
                    // different suffix items can have different singleton
                    // coverage while still producing the same intersection
                    // with ux; that is exactly the duplicate-coverage case.
                    std::vector<int32_t> e_tid;
                    e_tid.reserve(std::min(ux.tid.size(), euy.tid.size()));
                    size_t exi = 0, eyi = 0;
                    while (exi < ux.tid.size() && eyi < euy.tid.size()) {
                        if (ux.tid[exi] < euy.tid[eyi]) { ++exi; continue; }
                        if (euy.tid[eyi] < ux.tid[exi]) { ++eyi; continue; }
                        e_tid.push_back(ux.tid[exi]);
                        ++exi; ++eyi;
                    }
                    if (e_tid == c_tid) { duplicate_tids = true; break; }
                }
            }
            if (duplicate_tids) continue;
            ext_cov_hashes.push_back(h);

            // Opt-C: push only the four scalar fields — no tid vector copy.
            ext.push_back({c_sI, c_ig, ux.item, uy.item});
        }

        // Depth=1 save loop: reverse order over ext (mirrors THUIsl::explore).
        for (int e = static_cast<int>(ext.size()) - 1; e >= 0; --e) {
            const L2Child& ch = ext[static_cast<size_t>(e)];
            if (ch.sI >= local_minU && ch.sI > 0.0 && ch.ig > G)
                l2_save(local_heap, local_minU, K,
                        std::vector<int>{ch.item_x, ch.item_y}, ch.sI, ch.ig);
        }
    }

    if (parallel_timed_out.load(std::memory_order_relaxed))
        return heap;   // return singletons only; consistent with serial timeout behaviour

    // Merge per-thread heaps into the global heap (serial).
    // Each thread heap is a valid top-K for the i-values it processed.
    // Inserting every entry into the global heap produces the correct global top-K.
    for (int t = 0; t < n_threads; t++) {
        for (const PatternEntry& pe : thread_heaps[static_cast<size_t>(t)]) {
            l2_save(heap, minU, K, pe.items, pe.utility, pe.ig);
        }
    }

#else
    // ── Serial fallback (no OpenMP) ────────────────────────────────────────
    for (int i = m - 2; i >= 0; --i) {
        if (has_deadline && ((++loop_ctr & (CHECK_INTERVAL - 1)) == 0) && timed_out())
            return heap;

        const L2UL& ux = uls[static_cast<size_t>(i)];
        if (ux.ig <= G) continue;
        if (ux.sI + ux.sR < minU || ux.sI <= 0.0) continue;

        // Opt-C: compact ext — no tid vector in L2Child.
        std::vector<L2Child>       ext;
        std::vector<size_t> ext_cov_hashes;
        // Reused across every j in this i-iteration instead of a fresh
        // heap allocation per candidate pair (was: allocated inside the
        // j-loop even for candidates rejected moments later).
        std::vector<int> cnt_pair_scratch(static_cast<size_t>(n_cls), 0);

        for (int j = i + 1; j < m; j++) {
            if (has_deadline && ((++loop_ctr & (CHECK_INTERVAL - 1)) == 0) && timed_out())
                return heap;

            const L2UL& uy = uls[static_cast<size_t>(j)];
            if (same_feature_l2(td.item_col, ux.item, uy.item))
                continue;
            if (uy.ig <= G) continue;
            if (eucs_enabled) {
                const size_t eidx = l2_eucs_index(i, j, m);
                // Same terminal-pair exact-utility skip as the OpenMP path.
                if (eucs_twu[eidx] < minU || eucs_pair_utility[eidx] < minU)
                    continue;
            }
            if (ux.tid.empty() || uy.tid.empty()) continue;

            std::vector<int32_t> c_tid;
            std::fill(cnt_pair_scratch.begin(), cnt_pair_scratch.end(), 0);
            std::vector<int>& cnt_pair = cnt_pair_scratch;
            double c_sI = 0.0;
            c_tid.reserve(std::min(ux.tid.size(), uy.tid.size()));

            size_t xi = 0, yi = 0;
            while (xi < ux.tid.size() && yi < uy.tid.size()) {
                int32_t xt = ux.tid[xi];
                int32_t yt = uy.tid[yi];
                if (xt < yt) { ++xi; continue; }
                if (yt < xt) { ++yi; continue; }

                c_tid.push_back(xt);
                c_sI += ux.iu[xi] + uy.iu[yi];
                const int lbl = ytrain[static_cast<size_t>(xt)];
                if (lbl >= 0 && lbl < n_cls) cnt_pair[static_cast<size_t>(lbl)]++;
                ++xi; ++yi;
            }
            if (c_tid.empty()) continue;

            const double c_ig = l2_ig_global(
                base_global, cnt_global, cnt_pair.data(), static_cast<int>(c_tid.size()), n_train, n_cls);
            if (!(c_ig > G)) continue;

            const size_t h = fnv1a_tids(c_tid);
            bool duplicate_tids = false;
            if (l2_vec_contains(ext_cov_hashes, h)) {
                // Re-derive tids for hash-collision check (cold path).
                for (const L2Child& existing : ext) {
                    const int ey = existing.item_y;
                    if (ey <= 0 || ey > n_items) continue;
                    const int eci = ci_for_item[static_cast<size_t>(ey)];
                    if (eci < 0 || eci >= m) continue;
                    const L2UL& euy = uls[static_cast<size_t>(eci)];
                    std::vector<int32_t> e_tid;
                    e_tid.reserve(std::min(ux.tid.size(), euy.tid.size()));
                    size_t exi = 0, eyi = 0;
                    while (exi < ux.tid.size() && eyi < euy.tid.size()) {
                        if (ux.tid[exi] < euy.tid[eyi]) { ++exi; continue; }
                        if (euy.tid[eyi] < ux.tid[exi]) { ++eyi; continue; }
                        e_tid.push_back(ux.tid[exi]);
                        ++exi; ++eyi;
                    }
                    if (e_tid == c_tid) { duplicate_tids = true; break; }
                }
            }
            if (duplicate_tids) continue;
            ext_cov_hashes.push_back(h);

            ext.push_back({c_sI, c_ig, ux.item, uy.item});
        }

        // Depth=1 save loop: reverse order over ext.
        for (int e = static_cast<int>(ext.size()) - 1; e >= 0; --e) {
            const L2Child& ch = ext[static_cast<size_t>(e)];
            if (ch.sI >= minU && ch.sI > 0.0 && ch.ig > G)
                l2_save(heap, minU, K,
                        std::vector<int>{ch.item_x, ch.item_y}, ch.sI, ch.ig);
        }
    }
#endif

    return heap;
}

// ── Extended: augmented_patterns relaxed-gate L2 hot path ─────────────
//
// Same exact-intersection L2 algorithm as mine_patterns_l2_cpp. The only
// difference: items whose source column is in relaxed_cols are exempt from
// every G-based gate (singleton save, extension guard, pair ig check) and
// from the RIU/TWU-based item-set seed filter that would otherwise drop a
// near-zero-utility item before it is ever scored. This mirrors, inside
// native mining, the same "let interaction-information survivors through
// regardless of marginal signal" idea already used to pick augmented-pair
// source columns — but without generating any sum/product/etc. operator
// features: relaxed columns are simply admitted as ordinary mining items
// and combine with everything else through the normal pairwise loop.
//
// This is a standalone function (not a parameter added to the existing
// mine_patterns_l2_cpp) so the well-tested default hot path is untouched;
// it is intended purely to validate the idea before deciding whether to
// fold it into the production entry points.
//
// Budget contract: K is the caller's topK budget (the same K everywhere
// else in this codebase governs downstream matrix width / memory). The
// relaxed-survivor track is a CARVE-OUT of K, not an addition to it:
//   ordinary_K = K - relaxed_quota
//   relaxed_quota slots are reserved for IG-ranked relaxed-survivor patterns
// so total output size is always <= K, matching topK semantics elsewhere
// in the pipeline (no silent 2x widening of the downstream feature matrix).
//
// Routing is mutually exclusive, not additive: any pattern touching at
// least one relaxed item is routed ONLY to the relaxed (IG-ranked) heap,
// even if it would also have independently cleared G as an ordinary
// pattern. Without this exclusivity a single pattern could occupy one
// slot in each heap, which silently re-introduces the K + relaxed_quota
// overshoot this carve-out exists to prevent.
AugmentedPatternsResult mine_patterns_l2_augmented_patterns_cpp(
    const TransactionDataCpp& td,
    const std::vector<int>&   ytrain,
    int n_cls, int K, double G,
    const std::vector<int>&   relaxed_cols,
    int relaxed_quota,
    double timeout_s)
{
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    static constexpr int CHECK_INTERVAL = 256;

    // relaxed_quota carves a slice out of K; it never adds to it.
    // relaxed_quota <= 0 means "no reservation" (relaxed_heap stays empty,
    // behaviourally identical to mine_patterns_l2_cpp once relaxed_cols is
    // also empty). relaxed_quota >= K is clamped so at least one ordinary
    // slot always remains.
    if (relaxed_quota < 0) relaxed_quota = 0;
    if (relaxed_quota > K - 1 && K > 1) relaxed_quota = K - 1;
    if (relaxed_quota > K) relaxed_quota = K;
    const int ordinary_K = K - relaxed_quota;

    bool has_deadline = (timeout_s > 0.0);
    TimePoint deadline_tp = {};

    if (has_deadline) {
        deadline_tp = Clock::now() +
            std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(timeout_s));
    }
    auto timed_out = [&]() -> bool {
        return has_deadline && Clock::now() >= deadline_tp;
    };

    const int n_train = static_cast<int>(ytrain.size());
    const int n_items = static_cast<int>(td.item_twu.size());

    // is_relaxed_item[item-1] precomputed once via item_col lookup.
    std::vector<uint8_t> is_relaxed_item(static_cast<size_t>(n_items), 0);
    if (!relaxed_cols.empty() && !td.item_col.empty()) {
        std::unordered_set<int> relaxed_set(relaxed_cols.begin(), relaxed_cols.end());
        for (int iid = 1; iid <= n_items; iid++) {
            const int col = td.item_col[static_cast<size_t>(iid - 1)];
            if (relaxed_set.count(col)) is_relaxed_item[static_cast<size_t>(iid - 1)] = 1;
        }
    }
    auto relaxed = [&](int item) -> bool {
        if (item <= 0 || item > n_items) return false;
        return is_relaxed_item[static_cast<size_t>(item - 1)] != 0;
    };

    double minU = td.riu_thresh(K);

    // Item-set seed filter: ordinarily td.item_twu[iid-1] >= minU. Relaxed
    // items bypass this so a near-zero-utility interaction survivor is not
    // dropped before it can ever be scored by the G-based checks below.
    std::vector<int> sorted_items;
    sorted_items.reserve(static_cast<size_t>(n_items));
    for (int iid = 1; iid <= n_items; iid++) {
        if (td.item_twu[static_cast<size_t>(iid - 1)] >= minU || relaxed(iid))
            sorted_items.push_back(iid);
    }
    std::sort(sorted_items.begin(), sorted_items.end(),
              [&](int a, int b) {
                  double ta = td.item_twu[static_cast<size_t>(a - 1)];
                  double tb = td.item_twu[static_cast<size_t>(b - 1)];
                  return (ta < tb) || (ta == tb && a < b);
              });

    const int m = static_cast<int>(sorted_items.size());
    if (m == 0) return AugmentedPatternsResult{};

    std::vector<int> ci_for_item(static_cast<size_t>(n_items) + 1, -1);
    for (int pos = 0; pos < m; pos++)
        ci_for_item[static_cast<size_t>(sorted_items[pos])] = pos;

    std::vector<double> eucs_twu;
    std::vector<double> eucs_pair_utility;
    const bool eucs_enabled = l2_configure_eucs(
        td.transactions, ci_for_item, m, eucs_twu, eucs_pair_utility);
    // Reused across the whole Phase 1 transaction scan by
    // l2_update_eucs_for_active (see its definition for why): avoids a
    // fresh per-transaction allocation for these O(active-size) buffers.
    std::vector<int> eucs_pos_scratch;
    std::vector<int> eucs_col_scratch;

    std::vector<L2UL> uls(static_cast<size_t>(m));
    for (int ci = 0; ci < m; ci++) uls[static_cast<size_t>(ci)].item = sorted_items[ci];

    std::vector<int> cnt_global(n_cls, 0);
    for (int lbl : ytrain)
        if (lbl >= 0 && lbl < n_cls) cnt_global[static_cast<size_t>(lbl)]++;
    // Computed once per mine_patterns_l2_* call; l2_ig_global takes it
    // as a parameter instead of recomputing it on every candidate.
    const double base_global = l2_entropy(cnt_global, n_train);

    std::vector<int> loop_counts(static_cast<size_t>(m), 0);
    std::vector<int> cnt_item(static_cast<size_t>(m) * static_cast<size_t>(n_cls), 0);

    int loop_ctr = 0;
    const int n_tx = static_cast<int>(td.transactions.size());
    for (int tid = 0; tid < n_tx; tid++) {
        if (has_deadline && ((++loop_ctr & (CHECK_INTERVAL - 1)) == 0) && timed_out())
            return AugmentedPatternsResult{};

        const Trans& trans = td.transactions[static_cast<size_t>(tid)];
        if (trans.size() == 1 && trans[0] == -1) continue;

        std::vector<std::pair<int, double>> active;
        active.reserve(trans.size());
        for (size_t pos = 0; pos < trans.size(); ++pos) {
            int it = trans[pos];
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
            l2_update_eucs_for_active(
                active, ci_for_item, td.item_col, m, eucs_twu, eucs_pair_utility,
                eucs_pos_scratch, eucs_col_scratch);
        }

        double rem = 0.0;
        const int lbl = (tid < static_cast<int>(ytrain.size())) ? ytrain[static_cast<size_t>(tid)] : -1;
        const bool lbl_valid = (lbl >= 0 && lbl < n_cls);
        for (int ai = static_cast<int>(active.size()) - 1; ai >= 0; --ai) {
            const int it = active[static_cast<size_t>(ai)].first;
            const double u  = active[static_cast<size_t>(ai)].second;
            const int ci = ci_for_item[static_cast<size_t>(it)];
            L2UL& ul = uls[static_cast<size_t>(ci)];
            ul.sI += u;
            ul.sR += rem;
            ul.tid.push_back(static_cast<int32_t>(tid));
            ul.iu.push_back(u);
            loop_counts[static_cast<size_t>(ci)]++;
            if (lbl_valid)
                cnt_item[static_cast<size_t>(ci) * static_cast<size_t>(n_cls) + static_cast<size_t>(lbl)]++;
            rem += u;
        }
    }

    for (int ci = 0; ci < m; ci++) {
        const int* cnt_in = cnt_item.data() + static_cast<size_t>(ci) * static_cast<size_t>(n_cls);
        uls[static_cast<size_t>(ci)].ig = l2_ig_global(
            base_global, cnt_global, cnt_in, loop_counts[static_cast<size_t>(ci)], n_train, n_cls);
    }

    std::vector<PatternEntry> heap;
    heap.reserve(static_cast<size_t>(ordinary_K) + 1);
    std::vector<PatternEntry> relaxed_heap;
    relaxed_heap.reserve(static_cast<size_t>(relaxed_quota) + 1);
    double minIg = -std::numeric_limits<double>::infinity();

    // Singleton save. Mutually exclusive by design so the combined output
    // never exceeds K (ordinary_K + relaxed_quota == K):
    //   - a relaxed item goes to relaxed_heap (IG-ranked), period.
    //   - a non-relaxed item that clears G goes to the ordinary heap
    //     (utility-ranked), exactly as in mine_patterns_l2_cpp.
    // A relaxed item that also independently clears G is NOT additionally
    // saved to the ordinary heap — it would otherwise occupy one slot in
    // each track for a single pattern, silently inflating the K budget.
    for (int ci = m - 1; ci >= 0; --ci) {
        const L2UL& ux = uls[static_cast<size_t>(ci)];
        const bool ux_relaxed = relaxed(ux.item);
        if (ux_relaxed) {
            if (ux.sI > 0.0 && relaxed_quota > 0)
                l2_save_by_ig(relaxed_heap, minIg, relaxed_quota,
                              std::vector<int>{ux.item}, ux.sI, ux.ig);
            continue;
        }
        const bool passes_g = ux.ig > G;
        if (ux.sI >= minU && ux.sI > 0.0 && passes_g && ordinary_K > 0)
            l2_save(heap, minU, ordinary_K, std::vector<int>{ux.item}, ux.sI, ux.ig);
    }

    // Serial pair construction (no OpenMP in this bounded path — kept
    // simple and easy to audit; revisit for performance once validated).
    for (int i = m - 2; i >= 0; --i) {
        if (has_deadline && ((++loop_ctr & (CHECK_INTERVAL - 1)) == 0) && timed_out())
            return AugmentedPatternsResult{heap, relaxed_heap};

        const L2UL& ux = uls[static_cast<size_t>(i)];
        const bool ux_relaxed = relaxed(ux.item);
        // Extension guard: explore from ux if it clears G OR is relaxed
        // (relaxed items must still reach the pair loop to be scored,
        // even though their own singleton ig may sit below G).
        if (!(ux.ig > G) && !ux_relaxed) continue;
        // Growth-bound exemption for relaxed items -- see the matching
        // comment in mine_patterns_l2_augmented_patterns_v2_cpp below for
        // why this is needed (without it, a relaxed item could pass the
        // gate above and still be pruned here before reaching the pair
        // loop). This function is superseded by the v2 (overgenerate then
        // IG-refilter) design and is not the one any current caller is
        // expected to use, but it remains directly callable, so the same
        // exemption is applied here for consistency.
        if (!ux_relaxed && (ux.sI + ux.sR < minU || ux.sI <= 0.0)) continue;
        if (ux_relaxed && ux.sI <= 0.0) continue;

        std::vector<L2Child>       ext;
        std::vector<size_t> ext_cov_hashes;
        // Reused across every j in this i-iteration instead of a fresh
        // heap allocation per candidate pair (was: allocated inside the
        // j-loop even for candidates rejected moments later).
        std::vector<int> cnt_pair_scratch(static_cast<size_t>(n_cls), 0);

        for (int j = i + 1; j < m; j++) {
            if (has_deadline && ((++loop_ctr & (CHECK_INTERVAL - 1)) == 0) && timed_out())
                return AugmentedPatternsResult{heap, relaxed_heap};

            const L2UL& uy = uls[static_cast<size_t>(j)];
            if (same_feature_l2(td.item_col, ux.item, uy.item))
                continue;
            const bool uy_relaxed = relaxed(uy.item);
            const bool pair_relaxed = ux_relaxed || uy_relaxed;
            if (!(uy.ig > G) && !uy_relaxed) continue;
            if (eucs_enabled && !pair_relaxed) {
                const size_t eidx = l2_eucs_index(i, j, m);
                // Safe only for ordinary terminal pairs: relaxed pairs are
                // IG-ranked and may be retained below the ordinary utility
                // floor, so they deliberately bypass this utility skip.
                if (eucs_twu[eidx] < minU || eucs_pair_utility[eidx] < minU)
                    continue;
            }
            if (ux.tid.empty() || uy.tid.empty()) continue;

            std::vector<int32_t> c_tid;
            std::fill(cnt_pair_scratch.begin(), cnt_pair_scratch.end(), 0);
            std::vector<int>& cnt_pair = cnt_pair_scratch;
            double c_sI = 0.0;
            c_tid.reserve(std::min(ux.tid.size(), uy.tid.size()));

            size_t xi = 0, yi = 0;
            while (xi < ux.tid.size() && yi < uy.tid.size()) {
                int32_t xt = ux.tid[xi];
                int32_t yt = uy.tid[yi];
                if (xt < yt) { ++xi; continue; }
                if (yt < xt) { ++yi; continue; }

                c_tid.push_back(xt);
                c_sI += ux.iu[xi] + uy.iu[yi];
                const int lbl = ytrain[static_cast<size_t>(xt)];
                if (lbl >= 0 && lbl < n_cls) cnt_pair[static_cast<size_t>(lbl)]++;
                ++xi; ++yi;
            }
            if (c_tid.empty()) continue;

            const double c_ig = l2_ig_global(
                base_global, cnt_global, cnt_pair.data(), static_cast<int>(c_tid.size()), n_train, n_cls);
            // Pair gate: relaxed if EITHER source item is a relaxed survivor.
            // This is the key relaxation that lets a near-zero-marginal-IG
            // interaction survivor combine with any partner and still be
            // scored. Pairs that clear neither G nor relaxation are dropped
            // entirely (same as the ordinary hot path).
            if (!(c_ig > G) && !pair_relaxed) continue;

            const size_t h = fnv1a_tids(c_tid);
            bool duplicate_tids = false;
            if (l2_vec_contains(ext_cov_hashes, h)) {
                for (const L2Child& existing : ext) {
                    const int ey = existing.item_y;
                    if (ey <= 0 || ey > n_items) continue;
                    const int eci = ci_for_item[static_cast<size_t>(ey)];
                    if (eci < 0 || eci >= m) continue;
                    const L2UL& euy = uls[static_cast<size_t>(eci)];
                    std::vector<int32_t> e_tid;
                    e_tid.reserve(std::min(ux.tid.size(), euy.tid.size()));
                    size_t exi = 0, eyi = 0;
                    while (exi < ux.tid.size() && eyi < euy.tid.size()) {
                        if (ux.tid[exi] < euy.tid[eyi]) { ++exi; continue; }
                        if (euy.tid[eyi] < ux.tid[exi]) { ++eyi; continue; }
                        e_tid.push_back(ux.tid[exi]);
                        ++exi; ++eyi;
                    }
                    if (e_tid == c_tid) { duplicate_tids = true; break; }
                }
            }
            if (duplicate_tids) continue;
            ext_cov_hashes.push_back(h);

            ext.push_back({c_sI, c_ig, ux.item, uy.item});
        }

        for (int e = static_cast<int>(ext.size()) - 1; e >= 0; --e) {
            const L2Child& ch = ext[static_cast<size_t>(e)];
            const bool pair_relaxed = ux_relaxed || relaxed(ch.item_y);
            if (pair_relaxed) {
                if (ch.sI > 0.0 && relaxed_quota > 0)
                    l2_save_by_ig(relaxed_heap, minIg, relaxed_quota,
                                 std::vector<int>{ch.item_x, ch.item_y}, ch.sI, ch.ig);
                continue;
            }
            const bool passes_g = ch.ig > G;
            if (ch.sI >= minU && ch.sI > 0.0 && passes_g && ordinary_K > 0)
                l2_save(heap, minU, ordinary_K,
                        std::vector<int>{ch.item_x, ch.item_y}, ch.sI, ch.ig);
        }
    }

    return AugmentedPatternsResult{heap, relaxed_heap};
}

// ── Extended v2: single-budget overgenerate-then-IG-refilter ──────────
//
// Same item-level relaxation as mine_patterns_l2_augmented_patterns_cpp
// (relaxed_cols items bypass G-based and RIU/TWU-seed gates). The
// difference is purely in how the K budget is spent: instead of carving a
// constant relaxed_quota out of K up front, this variant runs the ordinary
// (utility-ranked) heap and the relaxed-survivor (IG-ranked) heap each at
// FULL capacity K, producing up to 2K raw candidates total, then merges
// both pools and keeps only the global top-K by IG. No relaxed_quota
// parameter is needed -- the "how much budget should go to rescued
// survivors" question is answered empirically per-dataset by IG ranking
// rather than constant in advance.
//
// Routing into the two intermediate heaps is still mutually exclusive
// (a pattern touching a relaxed item only ever enters the relaxed pool),
// purely so a pattern cannot be double-generated; the final cap is a
// single IG-sorted truncation over the union, not per-pool.
std::vector<PatternEntry> mine_patterns_l2_augmented_patterns_v2_cpp(
    const TransactionDataCpp& td,
    const std::vector<int>&   ytrain,
    int n_cls, int K, double G,
    const std::vector<int>&   relaxed_cols,
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
    auto timed_out = [&]() -> bool {
        return has_deadline && Clock::now() >= deadline_tp;
    };

    const int n_train = static_cast<int>(ytrain.size());
    const int n_items = static_cast<int>(td.item_twu.size());

    std::vector<uint8_t> is_relaxed_item(static_cast<size_t>(n_items), 0);
    if (!relaxed_cols.empty() && !td.item_col.empty()) {
        std::unordered_set<int> relaxed_set(relaxed_cols.begin(), relaxed_cols.end());
        for (int iid = 1; iid <= n_items; iid++) {
            const int col = td.item_col[static_cast<size_t>(iid - 1)];
            if (relaxed_set.count(col)) is_relaxed_item[static_cast<size_t>(iid - 1)] = 1;
        }
    }
    auto relaxed = [&](int item) -> bool {
        if (item <= 0 || item > n_items) return false;
        return is_relaxed_item[static_cast<size_t>(item - 1)] != 0;
    };

    double minU = td.riu_thresh(K);

    std::vector<int> sorted_items;
    sorted_items.reserve(static_cast<size_t>(n_items));
    for (int iid = 1; iid <= n_items; iid++) {
        if (td.item_twu[static_cast<size_t>(iid - 1)] >= minU || relaxed(iid))
            sorted_items.push_back(iid);
    }
    std::sort(sorted_items.begin(), sorted_items.end(),
              [&](int a, int b) {
                  double ta = td.item_twu[static_cast<size_t>(a - 1)];
                  double tb = td.item_twu[static_cast<size_t>(b - 1)];
                  return (ta < tb) || (ta == tb && a < b);
              });

    const int m = static_cast<int>(sorted_items.size());
    if (m == 0) return {};

    std::vector<int> ci_for_item(static_cast<size_t>(n_items) + 1, -1);
    for (int pos = 0; pos < m; pos++)
        ci_for_item[static_cast<size_t>(sorted_items[pos])] = pos;

    std::vector<double> eucs_twu;
    std::vector<double> eucs_pair_utility;
    const bool eucs_enabled = l2_configure_eucs(
        td.transactions, ci_for_item, m, eucs_twu, eucs_pair_utility);
    // Reused across the whole Phase 1 transaction scan by
    // l2_update_eucs_for_active (see its definition for why): avoids a
    // fresh per-transaction allocation for these O(active-size) buffers.
    std::vector<int> eucs_pos_scratch;
    std::vector<int> eucs_col_scratch;

    std::vector<L2UL> uls(static_cast<size_t>(m));
    for (int ci = 0; ci < m; ci++) uls[static_cast<size_t>(ci)].item = sorted_items[ci];

    std::vector<int> cnt_global(n_cls, 0);
    for (int lbl : ytrain)
        if (lbl >= 0 && lbl < n_cls) cnt_global[static_cast<size_t>(lbl)]++;
    // Computed once per mine_patterns_l2_* call; l2_ig_global takes it
    // as a parameter instead of recomputing it on every candidate.
    const double base_global = l2_entropy(cnt_global, n_train);

    std::vector<int> loop_counts(static_cast<size_t>(m), 0);
    std::vector<int> cnt_item(static_cast<size_t>(m) * static_cast<size_t>(n_cls), 0);

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
            int it = trans[pos];
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
            l2_update_eucs_for_active(
                active, ci_for_item, td.item_col, m, eucs_twu, eucs_pair_utility,
                eucs_pos_scratch, eucs_col_scratch);
        }

        double rem = 0.0;
        const int lbl = (tid < static_cast<int>(ytrain.size())) ? ytrain[static_cast<size_t>(tid)] : -1;
        const bool lbl_valid = (lbl >= 0 && lbl < n_cls);
        for (int ai = static_cast<int>(active.size()) - 1; ai >= 0; --ai) {
            const int it = active[static_cast<size_t>(ai)].first;
            const double u  = active[static_cast<size_t>(ai)].second;
            const int ci = ci_for_item[static_cast<size_t>(it)];
            L2UL& ul = uls[static_cast<size_t>(ci)];
            ul.sI += u;
            ul.sR += rem;
            ul.tid.push_back(static_cast<int32_t>(tid));
            ul.iu.push_back(u);
            loop_counts[static_cast<size_t>(ci)]++;
            if (lbl_valid)
                cnt_item[static_cast<size_t>(ci) * static_cast<size_t>(n_cls) + static_cast<size_t>(lbl)]++;
            rem += u;
        }
    }

    for (int ci = 0; ci < m; ci++) {
        const int* cnt_in = cnt_item.data() + static_cast<size_t>(ci) * static_cast<size_t>(n_cls);
        uls[static_cast<size_t>(ci)].ig = l2_ig_global(
            base_global, cnt_global, cnt_in, loop_counts[static_cast<size_t>(ci)], n_train, n_cls);
    }

    // Both pools run at FULL capacity K -- this is the only budget
    // difference from the carve-out variant.
    std::vector<PatternEntry> heap;
    heap.reserve(static_cast<size_t>(K) + 1);
    std::vector<PatternEntry> relaxed_heap;
    relaxed_heap.reserve(static_cast<size_t>(K) + 1);
    double minIg = -std::numeric_limits<double>::infinity();

    for (int ci = m - 1; ci >= 0; --ci) {
        const L2UL& ux = uls[static_cast<size_t>(ci)];
        const bool ux_relaxed = relaxed(ux.item);
        if (ux_relaxed) {
            if (ux.sI > 0.0)
                l2_save_by_ig(relaxed_heap, minIg, K,
                              std::vector<int>{ux.item}, ux.sI, ux.ig);
            continue;
        }
        const bool passes_g = ux.ig > G;
        if (ux.sI >= minU && ux.sI > 0.0 && passes_g)
            l2_save(heap, minU, K, std::vector<int>{ux.item}, ux.sI, ux.ig);
    }

    for (int i = m - 2; i >= 0; --i) {
        if (has_deadline && ((++loop_ctr & (CHECK_INTERVAL - 1)) == 0) && timed_out()) {
            std::vector<PatternEntry> merged(heap);
            merged.insert(merged.end(), relaxed_heap.begin(), relaxed_heap.end());
            std::sort(merged.begin(), merged.end(),
                      [](const PatternEntry& a, const PatternEntry& b) { return a.ig > b.ig; });
            if (static_cast<int>(merged.size()) > K) merged.resize(static_cast<size_t>(K));
            return merged;
        }

        const L2UL& ux = uls[static_cast<size_t>(i)];
        const bool ux_relaxed = relaxed(ux.item);
        if (!(ux.ig > G) && !ux_relaxed) continue;
        // Growth-bound exemption for relaxed items: without this, a
        // relaxed survivor admitted past the singleton G/seed-filter gate
        // above could still be silently pruned from ever reaching the pair
        // loop below by the ordinary heap's rising competitive minU bound.
        // This mirrors the same exemption applied to the generic L path's
        // explore() (mining.cpp).
        if (!ux_relaxed && (ux.sI + ux.sR < minU || ux.sI <= 0.0)) continue;
        if (ux_relaxed && ux.sI <= 0.0) continue;

        std::vector<L2Child>       ext;
        std::vector<size_t> ext_cov_hashes;
        // Reused across every j in this i-iteration instead of a fresh
        // heap allocation per candidate pair (was: allocated inside the
        // j-loop even for candidates rejected moments later).
        std::vector<int> cnt_pair_scratch(static_cast<size_t>(n_cls), 0);

        for (int j = i + 1; j < m; j++) {
            if (has_deadline && ((++loop_ctr & (CHECK_INTERVAL - 1)) == 0) && timed_out()) {
                std::vector<PatternEntry> merged(heap);
                merged.insert(merged.end(), relaxed_heap.begin(), relaxed_heap.end());
                std::sort(merged.begin(), merged.end(),
                          [](const PatternEntry& a, const PatternEntry& b) { return a.ig > b.ig; });
                if (static_cast<int>(merged.size()) > K) merged.resize(static_cast<size_t>(K));
                return merged;
            }

            const L2UL& uy = uls[static_cast<size_t>(j)];
            if (same_feature_l2(td.item_col, ux.item, uy.item))
                continue;
            const bool uy_relaxed = relaxed(uy.item);
            const bool pair_relaxed = ux_relaxed || uy_relaxed;
            if (!(uy.ig > G) && !uy_relaxed) continue;
            if (eucs_enabled && !pair_relaxed) {
                const size_t eidx = l2_eucs_index(i, j, m);
                // Safe only for ordinary terminal pairs: relaxed pairs are
                // IG-ranked and may be retained below the ordinary utility
                // floor, so they deliberately bypass this utility skip.
                if (eucs_twu[eidx] < minU || eucs_pair_utility[eidx] < minU)
                    continue;
            }
            if (ux.tid.empty() || uy.tid.empty()) continue;

            std::vector<int32_t> c_tid;
            std::fill(cnt_pair_scratch.begin(), cnt_pair_scratch.end(), 0);
            std::vector<int>& cnt_pair = cnt_pair_scratch;
            double c_sI = 0.0;
            c_tid.reserve(std::min(ux.tid.size(), uy.tid.size()));

            size_t xi = 0, yi = 0;
            while (xi < ux.tid.size() && yi < uy.tid.size()) {
                int32_t xt = ux.tid[xi];
                int32_t yt = uy.tid[yi];
                if (xt < yt) { ++xi; continue; }
                if (yt < xt) { ++yi; continue; }

                c_tid.push_back(xt);
                c_sI += ux.iu[xi] + uy.iu[yi];
                const int lbl = ytrain[static_cast<size_t>(xt)];
                if (lbl >= 0 && lbl < n_cls) cnt_pair[static_cast<size_t>(lbl)]++;
                ++xi; ++yi;
            }
            if (c_tid.empty()) continue;

            const double c_ig = l2_ig_global(
                base_global, cnt_global, cnt_pair.data(), static_cast<int>(c_tid.size()), n_train, n_cls);
            if (!(c_ig > G) && !pair_relaxed) continue;

            const size_t h = fnv1a_tids(c_tid);
            bool duplicate_tids = false;
            if (l2_vec_contains(ext_cov_hashes, h)) {
                for (const L2Child& existing : ext) {
                    const int ey = existing.item_y;
                    if (ey <= 0 || ey > n_items) continue;
                    const int eci = ci_for_item[static_cast<size_t>(ey)];
                    if (eci < 0 || eci >= m) continue;
                    const L2UL& euy = uls[static_cast<size_t>(eci)];
                    std::vector<int32_t> e_tid;
                    e_tid.reserve(std::min(ux.tid.size(), euy.tid.size()));
                    size_t exi = 0, eyi = 0;
                    while (exi < ux.tid.size() && eyi < euy.tid.size()) {
                        if (ux.tid[exi] < euy.tid[eyi]) { ++exi; continue; }
                        if (euy.tid[eyi] < ux.tid[exi]) { ++eyi; continue; }
                        e_tid.push_back(ux.tid[exi]);
                        ++exi; ++eyi;
                    }
                    if (e_tid == c_tid) { duplicate_tids = true; break; }
                }
            }
            if (duplicate_tids) continue;
            ext_cov_hashes.push_back(h);

            ext.push_back({c_sI, c_ig, ux.item, uy.item});
        }

        for (int e = static_cast<int>(ext.size()) - 1; e >= 0; --e) {
            const L2Child& ch = ext[static_cast<size_t>(e)];
            const bool pair_relaxed = ux_relaxed || relaxed(ch.item_y);
            if (pair_relaxed) {
                if (ch.sI > 0.0)
                    l2_save_by_ig(relaxed_heap, minIg, K,
                                 std::vector<int>{ch.item_x, ch.item_y}, ch.sI, ch.ig);
                continue;
            }
            const bool passes_g = ch.ig > G;
            if (ch.sI >= minU && ch.sI > 0.0 && passes_g)
                l2_save(heap, minU, K,
                        std::vector<int>{ch.item_x, ch.item_y}, ch.sI, ch.ig);
        }
    }

    // Merge both pools (up to 2K candidates) and keep the global top-K by IG.
    std::vector<PatternEntry> merged(heap);
    merged.insert(merged.end(), relaxed_heap.begin(), relaxed_heap.end());
    std::sort(merged.begin(), merged.end(),
              [](const PatternEntry& a, const PatternEntry& b) { return a.ig > b.ig; });
    if (static_cast<int>(merged.size()) > K) merged.resize(static_cast<size_t>(K));
    return merged;
}

}  // namespace hugiml
