/**
 * rpte_tree.cpp — implementation of the native leaf-wise
 * bounded-lookahead tree grower. See rpte_tree.hpp for the public
 * contract and algorithm overview.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 * Apache-2.0 License
 */

#include "rpte_tree.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "rpte_core.hpp"
#include "rpte_pair_search_core.hpp"
#include "rpte_significance.hpp"

namespace hugiml::rpte {

namespace {

// One candidate root+child(+grandchild) split under consideration for
// a leaf, before it is either discarded or committed to the tree.
struct Candidate {
    std::string kind;  // "greedy" or "lookahead"
    int32_t leaf_id = -1;
    int32_t leaf_depth = 0;
    double grow_gain = 0.0, probe_gain = 0.0, weighted_probe_gain = 0.0;
    int32_t leaf_delta = 1;
    int32_t root_index = -1;  // -1 while an unresolved synthetic root
    double root_threshold = 0.0;
    std::vector<int32_t> source_features;  // raw columns the root depends on
    int32_t operation = -1;  // rpte_core::PairOp code; -1 for a catalog (non-synthetic) root
    std::optional<ChildDescriptor> left_child, right_child, grandchild;
    double root_probe_gain = 0.0;
    bool root_is_synthetic = false;
    int32_t root_source_a = -1, root_source_b = -1;
    double root_source_a_threshold = 0.0, root_source_b_threshold = 0.0;
    bool root_is_binarized = true;
    bool statistically_significant = true;
    std::optional<double> significance_p_value;
};

double median_sorted(std::vector<double> vals) {
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    return rpte_core::binary_or_median_threshold(vals);
}

// One grow-stage root+child pairing under consideration, before
// probe-set rescoring decides whether it clears acceptance.
struct GrowCandidateItem {
    double grow_gain = 0.0;
    int32_t root_index = -1;               // -1 if the root is synthesized
    double root_threshold = 0.0;
    std::vector<int32_t> source_features;  // raw columns the root depends on
    int32_t operation = -1;
    ChildDescriptor left_descriptor, right_descriptor;
    bool root_is_synthetic = false;
    int32_t root_source_a = -1, root_source_b = -1;
    double root_source_a_threshold = 0.0, root_source_b_threshold = 0.0;
    bool root_is_binarized = true;
};

// Grows and holds all mutable state for a single tree fit.
class TreeBuilder {
public:
    TreeBuilder(
        const double* X, int64_t n_rows, int64_t p_base,
        const std::vector<int8_t>& labels,
        const std::vector<int32_t>& eligible_columns,
        const std::vector<int32_t>& raw_indices,
        const RootColumns& roots,
        const std::vector<std::vector<int32_t>>& owner_by_column,
        const TreeHyperparams& hp
    )
        : base_(X), n_(n_rows), p_base_(p_base), labels_(labels),
          eligible_columns_(eligible_columns),
          eligible_set_(eligible_columns.begin(), eligible_columns.end()),
          raw_indices_(raw_indices), roots_(roots),
          owner_by_column_(owner_by_column), hp_(hp) {}

    FittedTree fit(const std::vector<int64_t>& grow_rows, const std::vector<int64_t>& probe_rows);

private:
    const double* base_;
    int64_t n_, p_base_;
    std::vector<std::vector<double>> synthetic_columns_;
    std::vector<ColumnSpec> synthetic_specs_;

    const std::vector<int8_t>& labels_;
    std::vector<int32_t> eligible_columns_;
    std::unordered_set<int32_t> eligible_set_;
    std::vector<int32_t> raw_indices_;
    RootColumns roots_;
    std::vector<std::vector<int32_t>> owner_by_column_;  // grows as synthetic columns are added
    TreeHyperparams hp_;

    std::deque<TreeNodeData> nodes_;
    std::set<int32_t> leaf_nodes_;
    std::vector<GrowthLogEntry> growth_log_;
    std::set<int32_t> used_columns_;
    std::set<int32_t> used_raw_features_;
    int32_t root_probe_size_ = 0;

    int32_t n_cols() const { return static_cast<int32_t>(p_base_ + synthetic_columns_.size()); }

    double at(int64_t row, int32_t col) const {
        if (col < p_base_) return base_[static_cast<size_t>(row) * static_cast<size_t>(p_base_) + static_cast<size_t>(col)];
        return synthetic_columns_[static_cast<size_t>(col - p_base_)][static_cast<size_t>(row)];
    }

    const std::vector<int32_t>& owners_of(int32_t col) const { return owner_by_column_[static_cast<size_t>(col)]; }

    std::vector<double> column_over_rows(int32_t col, const std::vector<int64_t>& rows) const {
        std::vector<double> out(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) out[i] = at(rows[i], col);
        return out;
    }

    static bool owners_disjoint(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
        for (int32_t x : a) if (std::find(b.begin(), b.end(), x) != b.end()) return false;
        return true;
    }

    static std::vector<int32_t> owners_union(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
        std::vector<int32_t> out(a);
        out.insert(out.end(), b.begin(), b.end());
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    int32_t new_node(int32_t depth, std::vector<int64_t> grow_rows, std::vector<int64_t> probe_rows,
                      std::vector<int32_t> path_columns) {
        TreeNodeData node;
        node.depth = depth;
        node.grow_rows = std::move(grow_rows);
        node.probe_rows = std::move(probe_rows);
        node.path_columns = std::move(path_columns);
        nodes_.push_back(std::move(node));
        return static_cast<int32_t>(nodes_.size()) - 1;
    }

    // --- greedy single-feature stump -------------------------------------
    std::optional<Candidate> greedy_candidate(int32_t leaf_idx);

    // --- bounded-lookahead microtree search -------------------------------
    std::optional<Candidate> lookahead_candidate(int32_t leaf_idx);
    std::vector<uint8_t> root_bit_values(const Candidate& c, const std::vector<int64_t>& rows) const;
    std::vector<uint8_t> child_bit_values(const ChildDescriptor& d, const std::vector<int64_t>& rows) const;
    std::optional<ChildDescriptor> grandchild_candidate(
        const TreeNodeData& leaf,
        const std::vector<int64_t>& codes4_grow, const std::vector<int64_t>& codes4_probe,
        const std::vector<int8_t>& yg, const std::vector<int8_t>& yp,
        const std::vector<ChildDescriptor>& child_pool,
        const std::vector<uint8_t>& Cg_pool_bits, size_t pool_width,
        const std::vector<int32_t>& excluded_owners,
        double* out_probe_gain8, int32_t* out_n_candidates
    );

    struct ChildPool {
        std::vector<ChildDescriptor> descriptors;
        std::vector<uint8_t> bits;  // row-major grow_rows.size() x descriptors.size()
    };
    ChildPool build_child_pool(
        const TreeNodeData& leaf,
        const std::vector<int32_t>& usable_raw,
        const std::unordered_map<int32_t, double>& raw_thresholds,
        const std::unordered_map<int32_t, uint8_t>& raw_is_binary,
        std::mt19937_64& rng
    ) const;

    std::optional<Candidate> candidate_for_leaf(int32_t leaf_idx);

    // --- commit / materialize ----------------------------------------------
    int32_t materialize_synthetic_column(const ColumnSpec& spec);
    void materialize_synthetic_root(Candidate& c);
    std::vector<int32_t> commit_candidate(int32_t leaf_idx, Candidate& c);

    static std::pair<std::vector<int64_t>, std::vector<int64_t>> split_rows(
        const std::vector<int64_t>& rows, const std::vector<double>& col_values, double threshold) {
        std::vector<int64_t> left, right;
        left.reserve(rows.size());
        right.reserve(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            if (col_values[i] > threshold) right.push_back(rows[i]); else left.push_back(rows[i]);
        }
        return {left, right};
    }

    std::pair<std::vector<int64_t>, std::vector<int64_t>> partition_rows(
        const std::vector<int64_t>& rows, int32_t feature, double threshold) const {
        return split_rows(rows, column_over_rows(feature, rows), threshold);
    }
};

// ---------------------------------------------------------------------
// Greedy candidate: the single best (feature, threshold) stump by
// grow-set information gain, over every eligible column not already
// on this leaf's path -- equivalent to an ordinary depth-1 CART split.
// ---------------------------------------------------------------------
std::optional<Candidate> TreeBuilder::greedy_candidate(int32_t leaf_idx) {
    const TreeNodeData& leaf = nodes_[static_cast<size_t>(leaf_idx)];
    std::vector<int32_t> columns;
    columns.reserve(eligible_columns_.size());
    for (int32_t c : eligible_columns_) {
        if (std::find(leaf.path_columns.begin(), leaf.path_columns.end(), c) == leaf.path_columns.end()) {
            columns.push_back(c);
        }
    }
    if (columns.empty()) return std::nullopt;

    std::vector<int8_t> yg(leaf.grow_rows.size());
    for (size_t i = 0; i < leaf.grow_rows.size(); ++i) yg[i] = labels_[static_cast<size_t>(leaf.grow_rows[i])];

    int32_t best_feature = -1;
    double best_threshold = 0.0;
    double best_gain = -1.0;

    for (int32_t col : columns) {
        std::vector<double> vals = column_over_rows(col, leaf.grow_rows);
        std::vector<double> thresholds = rpte_core::candidate_thresholds(vals, std::numeric_limits<int>::max());
        for (double thr : thresholds) {
            std::vector<int64_t> codes(vals.size());
            for (size_t i = 0; i < vals.size(); ++i) codes[i] = vals[i] > thr ? 1 : 0;
            const double gain = rpte_core::partition_ig_bits(codes, yg);
            if (gain > best_gain) {
                best_gain = gain;
                best_feature = col;
                best_threshold = thr;
            }
        }
    }
    if (best_feature < 0) return std::nullopt;

    std::vector<double> grow_vals = column_over_rows(best_feature, leaf.grow_rows);
    std::vector<int64_t> grow_codes(grow_vals.size());
    int64_t g0 = 0, g1 = 0;
    for (size_t i = 0; i < grow_vals.size(); ++i) {
        grow_codes[i] = grow_vals[i] > best_threshold ? 1 : 0;
        if (grow_codes[i]) ++g1; else ++g0;
    }
    if (std::min(g0, g1) < hp_.min_samples_leaf) return std::nullopt;

    std::vector<double> probe_vals = column_over_rows(best_feature, leaf.probe_rows);
    std::vector<int64_t> probe_codes(probe_vals.size());
    int64_t p0 = 0, p1 = 0;
    for (size_t i = 0; i < probe_vals.size(); ++i) {
        probe_codes[i] = probe_vals[i] > best_threshold ? 1 : 0;
        if (probe_codes[i]) ++p1; else ++p0;
    }
    if (std::min(p0, p1) < hp_.min_probe_leaf) return std::nullopt;

    std::vector<int8_t> yp(leaf.probe_rows.size());
    for (size_t i = 0; i < leaf.probe_rows.size(); ++i) yp[i] = labels_[static_cast<size_t>(leaf.probe_rows[i])];

    const double grow_gain = rpte_core::partition_ig_bits(grow_codes, yg);
    const double probe_gain = rpte_core::partition_ig_bits(probe_codes, yp);
    const double weight = static_cast<double>(leaf.probe_rows.size()) / std::max<double>(1.0, root_probe_size_);

    Candidate c;
    c.kind = "greedy";
    c.leaf_id = leaf_idx;
    c.leaf_depth = leaf.depth;
    c.grow_gain = grow_gain;
    c.probe_gain = probe_gain;
    c.weighted_probe_gain = weight * probe_gain;
    c.leaf_delta = 1;
    c.root_index = best_feature;
    c.root_threshold = best_threshold;
    if (hp_.use_statistical_acceptance) {
        const double critical_ig = bonferroni_critical_ig(
            hp_.significance_alpha, 1.0, static_cast<int64_t>(columns.size()),
            static_cast<int64_t>(leaf.probe_rows.size()));
        c.statistically_significant = probe_gain >= critical_ig;
        c.significance_p_value = g2_p_value(probe_gain, static_cast<int64_t>(leaf.probe_rows.size()), 1.0);
    }
    return c;
}

bool column_is_binary(const std::vector<double>& vals) {
    std::vector<double> finite;
    finite.reserve(vals.size());
    for (double v : vals) if (std::isfinite(v)) finite.push_back(v);
    std::sort(finite.begin(), finite.end());
    finite.erase(std::unique(finite.begin(), finite.end()), finite.end());
    return finite.size() <= 2;
}

std::vector<uint8_t> TreeBuilder::root_bit_values(const Candidate& c, const std::vector<int64_t>& rows) const {
    std::vector<uint8_t> out(rows.size());
    if (c.root_is_synthetic) {
        for (size_t i = 0; i < rows.size(); ++i) {
            const double xa = at(rows[i], c.root_source_a);
            const double xb = at(rows[i], c.root_source_b);
            double va = xa, vb = xb;
            if (c.root_is_binarized) {
                va = (xa > c.root_source_a_threshold) ? 1.0 : 0.0;
                vb = (xb > c.root_source_b_threshold) ? 1.0 : 0.0;
            }
            const double val = rpte_core::apply_pair_op(va, vb, c.operation);
            out[i] = (val > c.root_threshold) ? 1 : 0;
        }
    } else {
        for (size_t i = 0; i < rows.size(); ++i) out[i] = (at(rows[i], c.root_index) > c.root_threshold) ? 1 : 0;
    }
    return out;
}

std::vector<uint8_t> TreeBuilder::child_bit_values(const ChildDescriptor& d, const std::vector<int64_t>& rows) const {
    std::vector<uint8_t> out(rows.size());
    if (d.kind == ChildDescriptor::kSynth) {
        for (size_t i = 0; i < rows.size(); ++i) {
            const double xa = at(rows[i], d.spec.a_idx);
            const double xb = at(rows[i], d.spec.b_idx);
            double va = xa, vb = xb;
            if (d.spec.is_binarized) {
                va = (xa > d.spec.a_threshold) ? 1.0 : 0.0;
                vb = (xb > d.spec.b_threshold) ? 1.0 : 0.0;
            }
            const double val = rpte_core::apply_pair_op(va, vb, d.spec.op);
            out[i] = (val > d.threshold) ? 1 : 0;
        }
    } else {
        for (size_t i = 0; i < rows.size(); ++i) out[i] = (at(rows[i], d.index) > d.threshold) ? 1 : 0;
    }
    return out;
}

// Builds this leaf's candidate CHILD pool: one thresholded bit column
// per candidate, drawn from three sources -- eligible raw features,
// existing catalog columns (mined augmented pairs / patterns) not
// already on the leaf's path, and a bounded set of raw pairs
// synthesized on the fly. A microtree's child can therefore be a
// single raw feature (root pair + child = 3-way interaction) or
// itself a pair (root pair + child pair = 4-way interaction).
TreeBuilder::ChildPool TreeBuilder::build_child_pool(
    const TreeNodeData& leaf,
    const std::vector<int32_t>& usable_raw,
    const std::unordered_map<int32_t, double>& raw_thresholds,
    const std::unordered_map<int32_t, uint8_t>& raw_is_binary,
    std::mt19937_64& rng
) const {
    ChildPool pool;
    const size_t n_grow = leaf.grow_rows.size();

    for (int32_t idx : usable_raw) {
        ChildDescriptor d;
        d.kind = ChildDescriptor::kRaw;
        d.index = idx;
        d.threshold = raw_thresholds.at(idx);
        d.owners = owners_of(idx);
        pool.descriptors.push_back(std::move(d));
    }

    if (hp_.aug_child_enabled) {
        std::vector<int32_t> catalog_roots;
        for (int32_t root_idx : roots_) {
            if (std::find(leaf.path_columns.begin(), leaf.path_columns.end(), root_idx) != leaf.path_columns.end())
                continue;
            std::vector<double> vals = column_over_rows(root_idx, leaf.grow_rows);
            const double threshold = rpte_core::binary_or_median_threshold(vals);
            if (std::isnan(threshold)) continue;
            ChildDescriptor d;
            d.kind = ChildDescriptor::kCatalog;
            d.index = root_idx;
            d.threshold = threshold;
            d.owners = owners_of(root_idx);
            pool.descriptors.push_back(std::move(d));
        }

        if (hp_.raw_pair_fallback && usable_raw.size() >= 2) {
            std::vector<std::pair<int32_t, int32_t>> pairs;
            for (size_t i = 0; i < usable_raw.size(); ++i)
                for (size_t j = i + 1; j < usable_raw.size(); ++j)
                    pairs.emplace_back(usable_raw[i], usable_raw[j]);
            if (static_cast<int32_t>(pairs.size()) > hp_.aug_child_max_candidates) {
                std::shuffle(pairs.begin(), pairs.end(), rng);
                pairs.resize(static_cast<size_t>(hp_.aug_child_max_candidates));
            }
            std::vector<int32_t> ops = hp_.lookahead_ops;
            if (ops.empty()) ops = {0, 1, 2, 3};
            for (const auto& [a_idx, b_idx] : pairs) {
                const bool is_bin = raw_is_binary.at(a_idx) && raw_is_binary.at(b_idx);
                std::vector<double> a_vals = column_over_rows(a_idx, leaf.grow_rows);
                std::vector<double> b_vals = column_over_rows(b_idx, leaf.grow_rows);
                const double a_thr = raw_thresholds.at(a_idx), b_thr = raw_thresholds.at(b_idx);
                for (int32_t op : ops) {
                    std::vector<double> combined(n_grow);
                    for (size_t i = 0; i < n_grow; ++i) {
                        double va = a_vals[i], vb = b_vals[i];
                        if (is_bin) {
                            va = (a_vals[i] > a_thr) ? 1.0 : 0.0;
                            vb = (b_vals[i] > b_thr) ? 1.0 : 0.0;
                        }
                        combined[i] = rpte_core::apply_pair_op(va, vb, op);
                    }
                    const double threshold = rpte_core::binary_or_median_threshold(combined);
                    if (std::isnan(threshold)) continue;
                    ChildDescriptor d;
                    d.kind = ChildDescriptor::kSynth;
                    d.threshold = threshold;
                    d.owners = owners_union(owners_of(a_idx), owners_of(b_idx));
                    d.spec.a_idx = a_idx;
                    d.spec.b_idx = b_idx;
                    d.spec.op = op;
                    d.spec.a_threshold = a_thr;
                    d.spec.b_threshold = b_thr;
                    d.spec.is_binarized = is_bin;
                    d.spec.owners = d.owners;
                    pool.descriptors.push_back(std::move(d));
                }
            }
        }
    }

    const size_t width = pool.descriptors.size();
    pool.bits.assign(n_grow * width, 0);
    for (size_t j = 0; j < width; ++j) {
        std::vector<uint8_t> col = child_bit_values(pool.descriptors[j], leaf.grow_rows);
        for (size_t i = 0; i < n_grow; ++i) pool.bits[i * width + j] = col[i];
    }
    return pool;
}

// Searches for the single best-scoring 3rd split level, extending a
// 4-cell root+child microtree to 8 cells (a 5-way interaction if the
// grandchild is a single raw feature, 6-way if it is itself a pair).
// Draws from the same child_pool the child slot itself uses.
std::optional<ChildDescriptor> TreeBuilder::grandchild_candidate(
    const TreeNodeData& leaf,
    const std::vector<int64_t>& codes4_grow, const std::vector<int64_t>& codes4_probe,
    const std::vector<int8_t>& yg, const std::vector<int8_t>& yp,
    const std::vector<ChildDescriptor>& child_pool,
    const std::vector<uint8_t>& Cg_pool_bits, size_t pool_width,
    const std::vector<int32_t>& excluded_owners,
    double* out_probe_gain8, int32_t* out_n_candidates
) {
    std::vector<size_t> allowed_positions;
    for (size_t p = 0; p < child_pool.size(); ++p) {
        if (owners_disjoint(child_pool[p].owners, excluded_owners)) allowed_positions.push_back(p);
    }
    if (allowed_positions.empty()) return std::nullopt;

    const size_t n_grow = leaf.grow_rows.size();
    std::vector<uint8_t> Cg(n_grow * allowed_positions.size());
    for (size_t i = 0; i < n_grow; ++i)
        for (size_t k = 0; k < allowed_positions.size(); ++k)
            Cg[i * allowed_positions.size() + k] = Cg_pool_bits[i * pool_width + allowed_positions[k]];

    std::vector<double> scores = rpte_core::shared_extension_scores(
        codes4_grow, 4, Cg.data(), n_grow, allowed_positions.size(), yg, hp_.min_samples_leaf);
    int64_t local = -1;
    double best = -std::numeric_limits<double>::infinity();
    for (size_t k = 0; k < scores.size(); ++k) {
        if (std::isfinite(scores[k]) && scores[k] > best) { best = scores[k]; local = static_cast<int64_t>(k); }
    }
    if (local < 0) return std::nullopt;

    const ChildDescriptor& descriptor = child_pool[allowed_positions[static_cast<size_t>(local)]];
    std::vector<uint8_t> gc_bit_probe = child_bit_values(descriptor, leaf.probe_rows);
    std::vector<int64_t> codes8_probe(leaf.probe_rows.size());
    int64_t counts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < leaf.probe_rows.size(); ++i) {
        codes8_probe[i] = 2 * codes4_probe[i] + gc_bit_probe[i];
        ++counts[static_cast<size_t>(codes8_probe[i])];
    }
    if (*std::min_element(counts, counts + 8) < hp_.min_probe_leaf) return std::nullopt;

    *out_probe_gain8 = rpte_core::partition_ig_bits(codes8_probe, yp);
    *out_n_candidates = static_cast<int32_t>(allowed_positions.size());
    return descriptor;
}

// Searches this leaf for the best root+child (optionally +grandchild)
// microtree: catalog roots (mined augmented pairs) first, then raw
// feature pairs synthesized on the fly via the native fused search --
// see rpte_pair_search_core.hpp -- when none of the catalog roots
// yield a strong grow-set candidate pairing. Every candidate is
// re-scored on the held-out probe split and must clear both the fixed
// thresholds and a Bonferroni-corrected significance bar to be
// returned.
std::optional<Candidate> TreeBuilder::lookahead_candidate(int32_t leaf_idx) {
    const TreeNodeData& leaf = nodes_[static_cast<size_t>(leaf_idx)];
    if (raw_indices_.empty()) return std::nullopt;
    if (roots_.empty() && !hp_.raw_pair_fallback) return std::nullopt;

    std::unordered_map<int32_t, double> raw_thresholds;
    std::unordered_map<int32_t, uint8_t> raw_is_binary;
    std::vector<int32_t> usable_raw;
    for (int32_t idx : raw_indices_) {
        if (std::find(leaf.path_columns.begin(), leaf.path_columns.end(), idx) != leaf.path_columns.end()) continue;
        std::vector<double> vals = column_over_rows(idx, leaf.grow_rows);
        const double thr = rpte_core::binary_or_median_threshold(vals);
        if (std::isnan(thr)) continue;
        raw_thresholds[idx] = thr;
        raw_is_binary[idx] = column_is_binary(vals) ? 1 : 0;
        usable_raw.push_back(idx);
    }
    if (usable_raw.empty()) return std::nullopt;

    std::mt19937_64 rng(hp_.random_state + static_cast<uint64_t>(leaf_idx));
    ChildPool pool = build_child_pool(leaf, usable_raw, raw_thresholds, raw_is_binary, rng);
    const size_t pool_width = pool.descriptors.size();

    std::vector<int8_t> yg(leaf.grow_rows.size()), yp(leaf.probe_rows.size());
    for (size_t i = 0; i < leaf.grow_rows.size(); ++i) yg[i] = labels_[static_cast<size_t>(leaf.grow_rows[i])];
    for (size_t i = 0; i < leaf.probe_rows.size(); ++i) yp[i] = labels_[static_cast<size_t>(leaf.probe_rows[i])];

    std::vector<GrowCandidateItem> grow_candidates;

    // Catalog roots: existing mined augmented-pair / pattern columns,
    // already filtered by the caller's allowed-operations rule.
    for (int32_t root_idx : roots_) {
        if (std::find(leaf.path_columns.begin(), leaf.path_columns.end(), root_idx) != leaf.path_columns.end())
            continue;
        const std::vector<int32_t>& excluded = owners_of(root_idx);
        std::vector<size_t> allowed_positions;
        for (size_t p = 0; p < pool_width; ++p)
            if (owners_disjoint(pool.descriptors[p].owners, excluded)) allowed_positions.push_back(p);
        if (allowed_positions.empty()) continue;

        std::vector<double> root_vals = column_over_rows(root_idx, leaf.grow_rows);
        std::vector<double> thresholds = rpte_core::candidate_thresholds(root_vals, hp_.max_root_thresholds);
        for (double root_threshold : thresholds) {
            std::vector<uint8_t> root_grow(root_vals.size());
            int64_t r0 = 0, r1 = 0;
            for (size_t i = 0; i < root_vals.size(); ++i) {
                root_grow[i] = root_vals[i] > root_threshold ? 1 : 0;
                if (root_grow[i]) ++r1; else ++r0;
            }
            if (std::min(r0, r1) < 2 * hp_.min_samples_leaf) continue;

            std::vector<uint8_t> Cg(root_vals.size() * allowed_positions.size());
            for (size_t i = 0; i < root_vals.size(); ++i)
                for (size_t k = 0; k < allowed_positions.size(); ++k)
                    Cg[i * allowed_positions.size() + k] = pool.bits[i * pool_width + allowed_positions[k]];

            size_t left_pos, right_pos;
            double grow_gain;
            if (hp_.lookahead_child_mode_shared) {
                std::vector<double> scores = rpte_core::shared_child_scores(
                    root_grow, Cg.data(), root_vals.size(), allowed_positions.size(), yg, hp_.min_samples_leaf);
                int64_t local = -1;
                double best = -std::numeric_limits<double>::infinity();
                for (size_t k = 0; k < scores.size(); ++k)
                    if (std::isfinite(scores[k]) && scores[k] > best) { best = scores[k]; local = static_cast<int64_t>(k); }
                if (local < 0) continue;
                grow_gain = best;
                left_pos = right_pos = allowed_positions[static_cast<size_t>(local)];
            } else {
                auto [left_local, ls] = rpte_core::best_branch_child(
                    root_grow, Cg.data(), root_vals.size(), allowed_positions.size(), yg, false, hp_.min_samples_leaf);
                auto [right_local, rs] = rpte_core::best_branch_child(
                    root_grow, Cg.data(), root_vals.size(), allowed_positions.size(), yg, true, hp_.min_samples_leaf);
                (void)ls; (void)rs;
                if (left_local < 0 || right_local < 0) continue;
                left_pos = allowed_positions[static_cast<size_t>(left_local)];
                right_pos = allowed_positions[static_cast<size_t>(right_local)];
                std::vector<int64_t> codes(root_vals.size());
                int64_t counts4[4] = {0, 0, 0, 0};
                for (size_t i = 0; i < root_vals.size(); ++i) {
                    const int64_t code = root_grow[i]
                        ? 2 + Cg[i * allowed_positions.size() + static_cast<size_t>(right_local)]
                        : Cg[i * allowed_positions.size() + static_cast<size_t>(left_local)];
                    codes[i] = code;
                    ++counts4[code];
                }
                if (*std::min_element(counts4, counts4 + 4) < hp_.min_samples_leaf) continue;
                grow_gain = rpte_core::partition_ig_bits(codes, yg);
            }

            GrowCandidateItem item;
            item.grow_gain = grow_gain;
            item.root_index = root_idx;
            item.root_threshold = root_threshold;
            item.source_features = excluded;
            item.left_descriptor = pool.descriptors[left_pos];
            item.right_descriptor = pool.descriptors[right_pos];
            item.root_is_synthetic = false;
            grow_candidates.push_back(std::move(item));
        }
    }

    // Raw-pair fallback: synthesize root candidates directly from
    // pairs of eligible raw features via the fused native search, for
    // when none of the catalog roots yield a viable candidate.
    if (hp_.raw_pair_fallback && usable_raw.size() >= 2) {
        const int32_t n_cols_now = n_cols();
        std::vector<double> Xg_dense(leaf.grow_rows.size() * static_cast<size_t>(n_cols_now));
        for (size_t i = 0; i < leaf.grow_rows.size(); ++i)
            for (int32_t c = 0; c < n_cols_now; ++c)
                Xg_dense[i * static_cast<size_t>(n_cols_now) + static_cast<size_t>(c)] = at(leaf.grow_rows[i], c);

        std::vector<double> raw_thr_arr(usable_raw.size());
        std::vector<uint8_t> raw_bin_arr(usable_raw.size());
        for (size_t i = 0; i < usable_raw.size(); ++i) {
            raw_thr_arr[i] = raw_thresholds.at(usable_raw[i]);
            raw_bin_arr[i] = raw_is_binary.at(usable_raw[i]);
        }
        std::vector<int32_t> ops = hp_.lookahead_ops;
        if (ops.empty()) ops = {0, 1, 2, 3};

        rpte_core::PairSearchResult result = rpte_core::raw_pair_root_search_core(
            Xg_dense.data(), leaf.grow_rows.size(), static_cast<size_t>(n_cols_now), yg.data(),
            usable_raw, raw_thr_arr, raw_bin_arr,
            pool.bits.data(), pool_width, ops,
            hp_.max_root_thresholds, hp_.min_samples_leaf, hp_.raw_pair_max_candidates,
            /*top_children_per_candidate=*/5, hp_.random_state + static_cast<uint64_t>(leaf_idx)
        );
        for (size_t k = 0; k < result.a_idx.size(); ++k) {
            const int32_t a_idx = result.a_idx[k], b_idx = result.b_idx[k];
            const std::vector<int32_t> owners_ab = owners_union(owners_of(a_idx), owners_of(b_idx));
            int64_t chosen_pos = -1;
            double chosen_score = 0.0;
            for (size_t w = 0; w < result.child_positions[k].size(); ++w) {
                const int32_t pos = result.child_positions[k][w];
                if (pos < 0) break;
                if (!owners_disjoint(pool.descriptors[static_cast<size_t>(pos)].owners, owners_ab)) continue;
                chosen_pos = pos;
                chosen_score = result.child_scores[k][w];
                break;
            }
            if (chosen_pos < 0) continue;

            GrowCandidateItem item;
            item.grow_gain = chosen_score;
            item.root_index = -1;
            item.root_threshold = result.threshold[k];
            item.source_features = owners_ab;
            item.operation = result.op[k];
            item.left_descriptor = item.right_descriptor = pool.descriptors[static_cast<size_t>(chosen_pos)];
            item.root_is_synthetic = true;
            item.root_source_a = a_idx;
            item.root_source_b = b_idx;
            item.root_source_a_threshold = raw_thresholds.at(a_idx);
            item.root_source_b_threshold = raw_thresholds.at(b_idx);
            item.root_is_binarized = result.is_binarized[k] != 0;
            grow_candidates.push_back(std::move(item));
        }
    }

    if (grow_candidates.empty()) return std::nullopt;
    std::stable_sort(grow_candidates.begin(), grow_candidates.end(),
                      [](const GrowCandidateItem& a, const GrowCandidateItem& b) { return a.grow_gain > b.grow_gain; });
    const size_t beam_size = std::min(grow_candidates.size(),
                                       static_cast<size_t>(std::max(1, hp_.lookahead_beam_width)));

    const int32_t n_tests = static_cast<int32_t>(grow_candidates.size());
    const int64_t n_probe = static_cast<int64_t>(leaf.probe_rows.size());
    double critical_ig_viable = 0.0, critical_ig_increment = 0.0;
    if (hp_.use_statistical_acceptance) {
        critical_ig_viable = bonferroni_critical_ig(hp_.significance_alpha, 3.0, n_tests, n_probe);
        critical_ig_increment = bonferroni_critical_ig(hp_.significance_alpha, 2.0, n_tests, n_probe);
    }

    const double weight = static_cast<double>(leaf.probe_rows.size()) / std::max<double>(1.0, root_probe_size_);
    std::optional<Candidate> best;

    for (size_t bi = 0; bi < beam_size; ++bi) {
        const GrowCandidateItem& item = grow_candidates[bi];
        Candidate item_as_candidate;  // partial, just to reuse root_bit_values
        item_as_candidate.root_is_synthetic = item.root_is_synthetic;
        item_as_candidate.root_index = item.root_index;
        item_as_candidate.root_threshold = item.root_threshold;
        item_as_candidate.root_source_a = item.root_source_a;
        item_as_candidate.root_source_b = item.root_source_b;
        item_as_candidate.root_source_a_threshold = item.root_source_a_threshold;
        item_as_candidate.root_source_b_threshold = item.root_source_b_threshold;
        item_as_candidate.root_is_binarized = item.root_is_binarized;
        item_as_candidate.operation = item.operation;

        std::vector<uint8_t> root_probe = root_bit_values(item_as_candidate, leaf.probe_rows);
        std::vector<uint8_t> left_probe = child_bit_values(item.left_descriptor, leaf.probe_rows);
        std::vector<uint8_t> right_probe = child_bit_values(item.right_descriptor, leaf.probe_rows);
        std::vector<int64_t> codes(leaf.probe_rows.size());
        int64_t counts4[4] = {0, 0, 0, 0};
        for (size_t i = 0; i < leaf.probe_rows.size(); ++i) {
            codes[i] = root_probe[i] ? 2 + right_probe[i] : left_probe[i];
            ++counts4[codes[i]];
        }
        if (*std::min_element(counts4, counts4 + 4) < hp_.min_probe_leaf) continue;

        const double probe_gain = rpte_core::partition_ig_bits(codes, yp);
        std::vector<int64_t> root_probe_codes(root_probe.begin(), root_probe.end());
        const double root_probe_gain = rpte_core::partition_ig_bits(root_probe_codes, yp);
        const double increment = probe_gain - root_probe_gain;
        bool four_cell_significant = probe_gain >= hp_.lookahead_min_probe_ig && increment >= hp_.lookahead_min_increment;
        if (hp_.use_statistical_acceptance) {
            four_cell_significant = four_cell_significant && probe_gain >= critical_ig_viable
                                     && increment >= critical_ig_increment;
        }

        std::optional<ChildDescriptor> gc_descriptor;
        double probe_gain8 = 0.0;
        int32_t gc_n_tests = 0;
        if (hp_.aug_child_enabled && leaf.depth + 3 <= hp_.max_depth) {
            std::vector<uint8_t> root_grow = root_bit_values(item_as_candidate, leaf.grow_rows);
            std::vector<uint8_t> left_grow = child_bit_values(item.left_descriptor, leaf.grow_rows);
            std::vector<uint8_t> right_grow = child_bit_values(item.right_descriptor, leaf.grow_rows);
            std::vector<int64_t> codes4_grow(leaf.grow_rows.size());
            for (size_t i = 0; i < leaf.grow_rows.size(); ++i)
                codes4_grow[i] = root_grow[i] ? 2 + right_grow[i] : left_grow[i];

            std::vector<int32_t> excluded_gc = owners_union(
                owners_union(item.source_features, item.left_descriptor.owners), item.right_descriptor.owners);
            gc_descriptor = grandchild_candidate(
                leaf, codes4_grow, codes, yg, yp, pool.descriptors, pool.bits, pool_width, excluded_gc,
                &probe_gain8, &gc_n_tests);
        }

        bool use_grandchild = false;
        double final_probe_gain = probe_gain;
        double final_df = 3.0;
        if (gc_descriptor.has_value()) {
            bool gc_ok;
            if (four_cell_significant) {
                const double gc_increment = probe_gain8 - probe_gain;
                gc_ok = gc_increment >= hp_.lookahead_min_increment;
                if (hp_.use_statistical_acceptance) {
                    const double critical_gc = bonferroni_critical_ig(hp_.significance_alpha, 4.0, gc_n_tests, n_probe);
                    gc_ok = gc_ok && gc_increment >= critical_gc;
                }
            } else {
                gc_ok = probe_gain8 >= hp_.lookahead_min_probe_ig;
                if (hp_.use_statistical_acceptance) {
                    const double critical_direct = bonferroni_critical_ig(hp_.significance_alpha, 7.0, gc_n_tests, n_probe);
                    gc_ok = gc_ok && probe_gain8 >= critical_direct;
                }
            }
            if (gc_ok) { use_grandchild = true; final_probe_gain = probe_gain8; final_df = 7.0; }
        }
        if (!four_cell_significant && !use_grandchild) continue;

        Candidate candidate;
        candidate.kind = "lookahead";
        candidate.leaf_id = leaf_idx;
        candidate.leaf_depth = leaf.depth;
        candidate.grow_gain = item.grow_gain;
        candidate.probe_gain = final_probe_gain;
        candidate.weighted_probe_gain = weight * final_probe_gain;
        candidate.leaf_delta = gc_descriptor.has_value() && use_grandchild ? 7 : 3;
        candidate.root_index = item.root_index;
        candidate.root_threshold = item.root_threshold;
        candidate.source_features = item.source_features;
        candidate.operation = item.operation;
        candidate.left_child = item.left_descriptor;
        candidate.right_child = item.right_descriptor;
        candidate.root_probe_gain = root_probe_gain;
        candidate.statistically_significant = true;
        if (hp_.use_statistical_acceptance)
            candidate.significance_p_value = g2_p_value(final_probe_gain, n_probe, final_df);
        candidate.root_is_synthetic = item.root_is_synthetic;
        candidate.root_source_a = item.root_source_a;
        candidate.root_source_b = item.root_source_b;
        candidate.root_source_a_threshold = item.root_source_a_threshold;
        candidate.root_source_b_threshold = item.root_source_b_threshold;
        candidate.root_is_binarized = item.root_is_binarized;
        if (use_grandchild) candidate.grandchild = gc_descriptor;

        if (!best.has_value() ||
            std::make_tuple(candidate.probe_gain, candidate.grow_gain, -candidate.root_index) >
                std::make_tuple(best->probe_gain, best->grow_gain, -best->root_index)) {
            best = candidate;
        }
    }
    return best;
}

// Picks between a greedy stump and a lookahead microtree for one open
// leaf: greedy first, falling through to a lookahead search only when
// greedy's held-out gain stalls below `greedy_stall_probe_ig`. A
// stalled-but-still-viable greedy split is kept if lookahead finds
// nothing better, rather than freezing the leaf outright.
std::optional<Candidate> TreeBuilder::candidate_for_leaf(int32_t leaf_idx) {
    const TreeNodeData& leaf = nodes_[static_cast<size_t>(leaf_idx)];
    if (leaf.depth >= hp_.max_depth) return std::nullopt;
    if (static_cast<int32_t>(leaf.grow_rows.size()) < 2 * hp_.min_samples_leaf) return std::nullopt;
    if (static_cast<int32_t>(leaf.probe_rows.size()) < 2 * hp_.min_probe_leaf) return std::nullopt;
    bool yg0 = false, yg1 = false;
    for (int64_t r : leaf.grow_rows) { if (labels_[static_cast<size_t>(r)] == 0) yg0 = true; else yg1 = true; }
    if (!(yg0 && yg1)) return std::nullopt;
    bool yp0 = false, yp1 = false;
    for (int64_t r : leaf.probe_rows) { if (labels_[static_cast<size_t>(r)] == 0) yp0 = true; else yp1 = true; }
    if (!(yp0 && yp1)) return std::nullopt;

    std::optional<Candidate> greedy = greedy_candidate(leaf_idx);
    const bool greedy_viable = greedy.has_value() && greedy->weighted_probe_gain >= hp_.min_weighted_probe_gain
                                && greedy->statistically_significant;
    if (greedy.has_value() && greedy->probe_gain >= hp_.greedy_stall_probe_ig) {
        return greedy_viable ? greedy : std::nullopt;
    }

    std::optional<Candidate> lookahead;
    if (leaf.depth + 2 <= hp_.max_depth) {
        lookahead = lookahead_candidate(leaf_idx);
        if (lookahead.has_value() &&
            (lookahead->weighted_probe_gain < hp_.min_weighted_probe_gain || !lookahead->statistically_significant)) {
            lookahead.reset();
        }
    }
    if (lookahead.has_value() && greedy_viable) {
        return (lookahead->weighted_probe_gain > greedy->weighted_probe_gain) ? lookahead : greedy;
    }
    if (lookahead.has_value()) return lookahead;
    if (greedy_viable) return greedy;
    return std::nullopt;
}

// Appends one synthesized-pair column (over all n rows, not just the
// leaf currently being grown) to this tree's working matrix, and
// records its provenance and reconstruction recipe so apply() can
// rebuild the same column later from fresh data.
int32_t TreeBuilder::materialize_synthetic_column(const ColumnSpec& spec) {
    std::vector<double> col(static_cast<size_t>(n_));
    for (int64_t i = 0; i < n_; ++i) {
        const double xa = at(i, spec.a_idx), xb = at(i, spec.b_idx);
        double va = xa, vb = xb;
        if (spec.is_binarized) {
            va = (xa > spec.a_threshold) ? 1.0 : 0.0;
            vb = (xb > spec.b_threshold) ? 1.0 : 0.0;
        }
        col[static_cast<size_t>(i)] = rpte_core::apply_pair_op(va, vb, spec.op);
    }
    synthetic_columns_.push_back(std::move(col));
    synthetic_specs_.push_back(spec);
    owner_by_column_.push_back(spec.owners);
    return n_cols() - 1;
}

void TreeBuilder::materialize_synthetic_root(Candidate& c) {
    ColumnSpec spec;
    spec.a_idx = c.root_source_a;
    spec.b_idx = c.root_source_b;
    spec.op = c.operation;
    spec.a_threshold = c.root_source_a_threshold;
    spec.b_threshold = c.root_source_b_threshold;
    spec.is_binarized = c.root_is_binarized;
    spec.owners = c.source_features;
    c.root_index = materialize_synthetic_column(spec);
}

// Applies an accepted candidate to the tree: materializes any
// still-synthetic root/child/grandchild columns, partitions the leaf
// into 2 children (greedy), 4 children (lookahead root+child), or 8
// children (lookahead + grandchild extension), and records raw
// feature reservation and a growth-log entry.
std::vector<int32_t> TreeBuilder::commit_candidate(int32_t leaf_idx, Candidate& c) {
    if (c.root_is_synthetic) materialize_synthetic_root(c);
    if (c.left_child.has_value() && c.left_child->kind == ChildDescriptor::kSynth) {
        const bool shared_with_right = c.right_child.has_value() &&
            c.left_child->spec.a_idx == c.right_child->spec.a_idx &&
            c.left_child->spec.b_idx == c.right_child->spec.b_idx &&
            c.left_child->spec.op == c.right_child->spec.op &&
            c.right_child->kind == ChildDescriptor::kSynth;
        const int32_t new_index = materialize_synthetic_column(c.left_child->spec);
        c.left_child->kind = ChildDescriptor::kCatalog;  // now a real, materialized column
        c.left_child->index = new_index;
        if (shared_with_right) {
            c.right_child->kind = ChildDescriptor::kCatalog;
            c.right_child->index = new_index;
            c.right_child->threshold = c.left_child->threshold;
        }
    }
    if (c.right_child.has_value() && c.right_child->kind == ChildDescriptor::kSynth) {
        const int32_t new_index = materialize_synthetic_column(c.right_child->spec);
        c.right_child->kind = ChildDescriptor::kCatalog;
        c.right_child->index = new_index;
    }
    if (c.grandchild.has_value() && c.grandchild->kind == ChildDescriptor::kSynth) {
        const int32_t new_index = materialize_synthetic_column(c.grandchild->spec);
        c.grandchild->kind = ChildDescriptor::kCatalog;
        c.grandchild->index = new_index;
    }

    const int32_t before = static_cast<int32_t>(leaf_nodes_.size());
    leaf_nodes_.erase(leaf_idx);
    nodes_[static_cast<size_t>(leaf_idx)].is_leaf = false;
    nodes_[static_cast<size_t>(leaf_idx)].feature_index = c.root_index;
    nodes_[static_cast<size_t>(leaf_idx)].threshold = c.root_threshold;
    auto [gl, gr] = partition_rows(nodes_[static_cast<size_t>(leaf_idx)].grow_rows, c.root_index, c.root_threshold);
    auto [pl, pr] = partition_rows(nodes_[static_cast<size_t>(leaf_idx)].probe_rows, c.root_index, c.root_threshold);
    std::vector<int32_t> root_path = nodes_[static_cast<size_t>(leaf_idx)].path_columns;
    root_path.push_back(c.root_index);
    std::sort(root_path.begin(), root_path.end());
    root_path.erase(std::unique(root_path.begin(), root_path.end()), root_path.end());
    const int32_t left_idx = new_node(nodes_[static_cast<size_t>(leaf_idx)].depth + 1, gl, pl, root_path);
    const int32_t right_idx = new_node(nodes_[static_cast<size_t>(leaf_idx)].depth + 1, gr, pr, root_path);
    nodes_[static_cast<size_t>(leaf_idx)].left = left_idx;
    nodes_[static_cast<size_t>(leaf_idx)].right = right_idx;

    std::vector<int32_t> new_leaves;
    if (c.kind == "greedy") {
        new_leaves = {left_idx, right_idx};
    } else {
        std::vector<std::tuple<int32_t, int32_t, double>> branches = {
            {left_idx, c.left_child->index, c.left_child->threshold},
            {right_idx, c.right_child->index, c.right_child->threshold},
        };
        for (auto& [branch_idx, child_index, child_threshold] : branches) {
            nodes_[static_cast<size_t>(branch_idx)].is_leaf = false;
            nodes_[static_cast<size_t>(branch_idx)].feature_index = child_index;
            nodes_[static_cast<size_t>(branch_idx)].threshold = child_threshold;
            auto [cgl, cgr] = partition_rows(nodes_[static_cast<size_t>(branch_idx)].grow_rows, child_index, child_threshold);
            auto [cpl, cpr] = partition_rows(nodes_[static_cast<size_t>(branch_idx)].probe_rows, child_index, child_threshold);
            std::vector<int32_t> child_path = nodes_[static_cast<size_t>(branch_idx)].path_columns;
            child_path.push_back(child_index);
            std::sort(child_path.begin(), child_path.end());
            child_path.erase(std::unique(child_path.begin(), child_path.end()), child_path.end());
            const int32_t cl = new_node(nodes_[static_cast<size_t>(branch_idx)].depth + 1, cgl, cpl, child_path);
            const int32_t cr = new_node(nodes_[static_cast<size_t>(branch_idx)].depth + 1, cgr, cpr, child_path);
            nodes_[static_cast<size_t>(branch_idx)].left = cl;
            nodes_[static_cast<size_t>(branch_idx)].right = cr;
            new_leaves.push_back(cl);
            new_leaves.push_back(cr);
        }
        if (c.grandchild.has_value()) {
            std::vector<int32_t> grandchild_leaves;
            for (int32_t branch_idx : new_leaves) {
                const int32_t gc_index = c.grandchild->index;
                const double gc_threshold = c.grandchild->threshold;
                nodes_[static_cast<size_t>(branch_idx)].is_leaf = false;
                nodes_[static_cast<size_t>(branch_idx)].feature_index = gc_index;
                nodes_[static_cast<size_t>(branch_idx)].threshold = gc_threshold;
                auto [ggl, ggr] = partition_rows(nodes_[static_cast<size_t>(branch_idx)].grow_rows, gc_index, gc_threshold);
                auto [gpl, gpr] = partition_rows(nodes_[static_cast<size_t>(branch_idx)].probe_rows, gc_index, gc_threshold);
                std::vector<int32_t> gc_path = nodes_[static_cast<size_t>(branch_idx)].path_columns;
                gc_path.push_back(gc_index);
                std::sort(gc_path.begin(), gc_path.end());
                gc_path.erase(std::unique(gc_path.begin(), gc_path.end()), gc_path.end());
                const int32_t gl2 = new_node(nodes_[static_cast<size_t>(branch_idx)].depth + 1, ggl, gpl, gc_path);
                const int32_t gr2 = new_node(nodes_[static_cast<size_t>(branch_idx)].depth + 1, ggr, gpr, gc_path);
                nodes_[static_cast<size_t>(branch_idx)].left = gl2;
                nodes_[static_cast<size_t>(branch_idx)].right = gr2;
                grandchild_leaves.push_back(gl2);
                grandchild_leaves.push_back(gr2);
            }
            new_leaves = grandchild_leaves;
        }
    }

    for (int32_t idx : new_leaves) leaf_nodes_.insert(idx);

    std::vector<int32_t> used = {c.root_index};
    if (c.left_child.has_value()) used.push_back(c.left_child->index);
    if (c.right_child.has_value()) used.push_back(c.right_child->index);
    if (c.grandchild.has_value()) used.push_back(c.grandchild->index);
    for (int32_t idx : used) {
        used_columns_.insert(idx);
        for (int32_t raw : owners_of(idx)) used_raw_features_.insert(raw);
    }

    GrowthLogEntry entry;
    entry.step = static_cast<int32_t>(growth_log_.size()) + 1;
    entry.leaf_id = leaf_idx;
    entry.leaf_depth = c.leaf_depth;
    entry.kind = c.kind;
    entry.root_index = c.root_index;
    entry.operation = c.operation;
    entry.source_features = c.source_features;
    if (c.left_child.has_value()) entry.left_child_index = c.left_child->index;
    if (c.right_child.has_value()) entry.right_child_index = c.right_child->index;
    if (c.grandchild.has_value()) entry.grandchild_index = c.grandchild->index;
    entry.grow_gain = c.grow_gain;
    entry.probe_gain = c.probe_gain;
    entry.root_probe_gain = c.root_probe_gain;
    entry.weighted_probe_gain = c.weighted_probe_gain;
    entry.statistically_significant = c.statistically_significant;
    entry.significance_p_value = c.significance_p_value;
    entry.leaf_count_before = before;
    entry.leaf_count_after = static_cast<int32_t>(leaf_nodes_.size());
    growth_log_.push_back(std::move(entry));

    return new_leaves;
}

// Best-first leaf-wise growth: repeatedly commits the open leaf's
// candidate with the largest probe-set weighted gain, re-evaluating
// each newly created leaf, until no leaf has a viable candidate or
// the leaf budget is exhausted.
FittedTree TreeBuilder::fit(const std::vector<int64_t>& grow_rows, const std::vector<int64_t>& probe_rows) {
    root_probe_size_ = static_cast<int32_t>(probe_rows.size());
    const int32_t root_idx = new_node(0, grow_rows, probe_rows, {});
    leaf_nodes_.insert(root_idx);

    struct Entry { double weighted_probe_gain; int64_t serial; Candidate candidate; };
    struct Cmp {
        bool operator()(const Entry& a, const Entry& b) const {
            if (a.weighted_probe_gain != b.weighted_probe_gain) return a.weighted_probe_gain < b.weighted_probe_gain;
            return a.serial > b.serial;  // earlier-pushed candidate wins exact ties
        }
    };
    std::priority_queue<Entry, std::vector<Entry>, Cmp> heap;
    int64_t serial = 0;
    auto push_if_valid = [&](int32_t leaf_idx) {
        std::optional<Candidate> candidate = candidate_for_leaf(leaf_idx);
        if (candidate.has_value()) heap.push(Entry{candidate->weighted_probe_gain, serial++, std::move(*candidate)});
    };
    push_if_valid(root_idx);

    while (!heap.empty() && static_cast<int32_t>(leaf_nodes_.size()) < hp_.max_leaf_nodes) {
        Entry top = heap.top();
        heap.pop();
        if (leaf_nodes_.find(top.candidate.leaf_id) == leaf_nodes_.end()) continue;
        if (static_cast<int32_t>(leaf_nodes_.size()) + top.candidate.leaf_delta > hp_.max_leaf_nodes) continue;
        std::vector<int32_t> new_leaves = commit_candidate(top.candidate.leaf_id, top.candidate);
        for (int32_t child : new_leaves) push_if_valid(child);
    }

    FittedTree result;
    result.is_fitted = !growth_log_.empty();
    if (!result.is_fitted) return result;
    result.nodes.assign(nodes_.begin(), nodes_.end());
    result.leaf_node_indices.assign(leaf_nodes_.begin(), leaf_nodes_.end());
    result.synthetic_specs = synthetic_specs_;
    result.used_columns.assign(used_columns_.begin(), used_columns_.end());
    result.used_raw_features.assign(used_raw_features_.begin(), used_raw_features_.end());
    result.growth_log = std::move(growth_log_);
    return result;
}

}  // namespace

FittedTree grow_tree(
    const double* X, int64_t n_rows, int64_t p_base,
    const std::vector<int8_t>& labels,
    const std::vector<int64_t>& grow_rows,
    const std::vector<int64_t>& probe_rows,
    const std::vector<int32_t>& eligible_columns,
    const std::vector<int32_t>& raw_indices,
    const RootColumns& roots,
    const std::vector<std::vector<int32_t>>& owner_by_column,
    const TreeHyperparams& hp
) {
    TreeBuilder builder(X, n_rows, p_base, labels, eligible_columns, raw_indices, roots, owner_by_column, hp);
    return builder.fit(grow_rows, probe_rows);
}

std::vector<int64_t> apply_tree(
    const double* X_new, int64_t n_rows, int64_t p_base,
    const FittedTree& tree
) {
    if (!tree.is_fitted) throw std::invalid_argument("apply_tree: tree has no accepted split.");

    std::vector<std::vector<double>> synthetic_columns(tree.synthetic_specs.size());
    for (size_t s = 0; s < tree.synthetic_specs.size(); ++s) {
        const ColumnSpec& spec = tree.synthetic_specs[s];
        std::vector<double> col(static_cast<size_t>(n_rows));
        for (int64_t i = 0; i < n_rows; ++i) {
            const double xa = X_new[static_cast<size_t>(i) * static_cast<size_t>(p_base) + static_cast<size_t>(spec.a_idx)];
            const double xb = X_new[static_cast<size_t>(i) * static_cast<size_t>(p_base) + static_cast<size_t>(spec.b_idx)];
            double va = xa, vb = xb;
            if (spec.is_binarized) {
                va = (xa > spec.a_threshold) ? 1.0 : 0.0;
                vb = (xb > spec.b_threshold) ? 1.0 : 0.0;
            }
            col[static_cast<size_t>(i)] = rpte_core::apply_pair_op(va, vb, spec.op);
        }
        synthetic_columns[s] = std::move(col);
    }
    auto at = [&](int64_t row, int32_t col) -> double {
        if (col < p_base) return X_new[static_cast<size_t>(row) * static_cast<size_t>(p_base) + static_cast<size_t>(col)];
        return synthetic_columns[static_cast<size_t>(col - p_base)][static_cast<size_t>(row)];
    };

    std::vector<int64_t> out(static_cast<size_t>(n_rows), -1);
    std::vector<int64_t> all_rows(static_cast<size_t>(n_rows));
    for (int64_t i = 0; i < n_rows; ++i) all_rows[static_cast<size_t>(i)] = i;

    std::vector<std::pair<int32_t, std::vector<int64_t>>> stack;
    stack.emplace_back(0, std::move(all_rows));
    while (!stack.empty()) {
        auto [node_idx, rows] = std::move(stack.back());
        stack.pop_back();
        if (rows.empty()) continue;
        const TreeNodeData& node = tree.nodes[static_cast<size_t>(node_idx)];
        if (node.is_leaf) {
            for (int64_t r : rows) out[static_cast<size_t>(r)] = node_idx;
            continue;
        }
        std::vector<int64_t> left_rows, right_rows;
        for (int64_t r : rows) {
            if (at(r, node.feature_index) > node.threshold) right_rows.push_back(r);
            else left_rows.push_back(r);
        }
        stack.emplace_back(node.left, std::move(left_rows));
        stack.emplace_back(node.right, std::move(right_rows));
    }
    return out;
}

}  // namespace hugiml::rpte
