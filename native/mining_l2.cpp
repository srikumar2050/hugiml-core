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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace hugiml {

namespace {

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

static double l2_ig_global(const std::vector<int>& cnt_global,
                           const std::vector<int>& cnt_in,
                           int n_in, int n_train, int n_cls) {
    const int n_out = n_train - n_in;
    if (n_out == 0)
        return std::numeric_limits<double>::quiet_NaN();

    double base = l2_entropy(cnt_global, n_train);
    std::vector<int> cnt_out(n_cls);
    for (int k = 0; k < n_cls; k++)
        cnt_out[k] = cnt_global[k] - cnt_in[k];

    double ce = (static_cast<double>(n_in)  / n_train * l2_entropy(cnt_in,  n_in)
               + static_cast<double>(n_out) / n_train * l2_entropy(cnt_out, n_out));
    return base - ce;
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

    std::vector<L2UL> uls(static_cast<size_t>(m));
    for (int ci = 0; ci < m; ci++) uls[static_cast<size_t>(ci)].item = sorted_items[ci];

    std::vector<int> cnt_global(n_cls, 0);
    for (int lbl : ytrain)
        if (lbl >= 0 && lbl < n_cls) cnt_global[static_cast<size_t>(lbl)]++;

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
        std::vector<int> cnt_in(
            cnt_item.data() + static_cast<size_t>(ci) * static_cast<size_t>(n_cls),
            cnt_item.data() + static_cast<size_t>(ci + 1) * static_cast<size_t>(n_cls));
        uls[static_cast<size_t>(ci)].ig = l2_ig_global(
            cnt_global, cnt_in, loop_counts[static_cast<size_t>(ci)], n_train, n_cls);
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
        std::unordered_set<size_t> ext_cov_hashes;

        for (int j = i + 1; j < m; j++) {
            const L2UL& uy = uls[static_cast<size_t>(j)];
            if (same_feature_l2(td.item_col, ux.item, uy.item))
                continue;
            if (uy.ig <= G) continue;
            if (ux.tid.empty() || uy.tid.empty()) continue;

            // Sorted-merge intersection — build c_tid, accumulate c_sI and
            // cnt_pair.  c_tid is local to this j-iteration (Opt-C).
            std::vector<int32_t> c_tid;
            std::vector<int> cnt_pair(n_cls, 0);
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
                cnt_global, cnt_pair, static_cast<int>(c_tid.size()), n_train, n_cls);
            if (!(c_ig > G)) continue;

            // FNV1a duplicate-coverage check (Opt-C: done here before ext push,
            // using the stack-local c_tid; no tid copy stored in ext).
            const size_t h = fnv1a_tids(c_tid);
            bool duplicate_tids = false;
            if (ext_cov_hashes.count(h)) {
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
            ext_cov_hashes.insert(h);

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
        std::unordered_set<size_t> ext_cov_hashes;

        for (int j = i + 1; j < m; j++) {
            if (has_deadline && ((++loop_ctr & (CHECK_INTERVAL - 1)) == 0) && timed_out())
                return heap;

            const L2UL& uy = uls[static_cast<size_t>(j)];
            if (same_feature_l2(td.item_col, ux.item, uy.item))
                continue;
            if (uy.ig <= G) continue;
            if (ux.tid.empty() || uy.tid.empty()) continue;

            std::vector<int32_t> c_tid;
            std::vector<int> cnt_pair(n_cls, 0);
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
                cnt_global, cnt_pair, static_cast<int>(c_tid.size()), n_train, n_cls);
            if (!(c_ig > G)) continue;

            const size_t h = fnv1a_tids(c_tid);
            bool duplicate_tids = false;
            if (ext_cov_hashes.count(h)) {
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
            ext_cov_hashes.insert(h);

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

}  // namespace hugiml
