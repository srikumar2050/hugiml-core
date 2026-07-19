/**
 * rpte_tree.hpp — native leaf-wise bounded-lookahead tree grower.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 * Apache-2.0 License
 *
 * One best-first tree in the RPTE ensemble. At each open leaf, tries
 * an ordinary greedy single-feature stump first; if its held-out gain
 * is too small, searches a bounded depth-two microtree whose root is
 * a raw-feature pair (mined or synthesized on the fly) and whose
 * child is a single raw feature or another pair, optionally extended
 * one more level (a "grandchild") for 5-way/6-way interactions.
 * Candidates are scored on a held-out probe split and accepted only
 * when they clear both fixed thresholds and a Bonferroni-corrected
 * statistical-significance bar (see rpte_significance.hpp). Raw
 * features used by an accepted split are reserved for the rest of
 * the ensemble's fit.
 *
 * All decision-making runs natively; Python calls `grow_tree` once
 * per boosting round and `apply_tree` to turn a fitted tree into a
 * leaf-id column for the downstream feature matrix.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace hugiml::rpte {

// A raw-feature pair mined ahead of time by HUGIML (an "augmented
// pair" column), usable as a lookahead root. Callers pass only
// columns already filtered by their own allowed-operations rule and
// by eligibility; ownership for reservation purposes comes from
// `owner_by_column`, not from this list.
using RootColumns = std::vector<int32_t>;

// Reconstruction recipe for one column this tree synthesized on the
// fly (a root, child, or grandchild built from a raw-feature pair
// rather than an existing column). Needed to rebuild the same column,
// bit-for-bit, from fresh data at predict time.
struct ColumnSpec {
    int32_t a_idx = -1, b_idx = -1;
    int32_t op = -1;
    double a_threshold = 0.0, b_threshold = 0.0;
    bool is_binarized = false;
    std::vector<int32_t> owners;  // raw feature column indices
};

// One candidate child (or grandchild) column in a leaf's search pool:
// an existing raw/catalog column referenced by index, or a pair
// synthesized on the fly (kind == kSynth, described by `spec`).
struct ChildDescriptor {
    enum Kind : int8_t { kRaw, kCatalog, kSynth } kind = kRaw;
    int32_t index = -1;   // for kRaw/kCatalog: column index
    double threshold = 0.0;
    std::vector<int32_t> owners;
    ColumnSpec spec;       // for kSynth
};

struct TreeNodeData {
    int32_t depth = 0;
    std::vector<int64_t> grow_rows;
    std::vector<int64_t> probe_rows;
    std::vector<int32_t> path_columns;  // sorted unique
    bool is_leaf = true;
    int32_t feature_index = -1;
    double threshold = 0.0;
    int32_t left = -1, right = -1;      // indices into FittedTree::nodes
};

// One accepted split, recorded for interpretability (mirrors Python's
// growth_log_ entries). Columns are identified by index; Python
// resolves display names and renders rule text.
struct GrowthLogEntry {
    int32_t step = 0;
    int32_t leaf_id = 0;
    int32_t leaf_depth = 0;
    std::string kind;  // "greedy" or "lookahead"
    int32_t root_index = -1;
    int32_t operation = -1;  // rpte_core::PairOp code; -1 when the root is not synthesized
    std::vector<int32_t> source_features;
    std::optional<int32_t> left_child_index;
    std::optional<int32_t> right_child_index;
    std::optional<int32_t> grandchild_index;
    double grow_gain = 0.0, probe_gain = 0.0, root_probe_gain = 0.0, weighted_probe_gain = 0.0;
    bool statistically_significant = true;
    std::optional<double> significance_p_value;
    int32_t leaf_count_before = 0, leaf_count_after = 0;
};

struct FittedTree {
    bool is_fitted = false;
    std::vector<TreeNodeData> nodes;             // nodes[0] is the root
    std::vector<int32_t> leaf_node_indices;      // sorted; leaf id == node index
    std::vector<ColumnSpec> synthetic_specs;     // materialization order
    std::vector<int32_t> used_columns;
    std::vector<int32_t> used_raw_features;
    std::vector<GrowthLogEntry> growth_log;
};

struct TreeHyperparams {
    int32_t max_depth = 8;
    int32_t max_leaf_nodes = 12;
    int32_t min_samples_leaf = 5;
    int32_t min_probe_leaf = 2;
    uint64_t random_state = 0;
    double greedy_stall_probe_ig = 0.02;
    double lookahead_min_probe_ig = 0.05;
    double lookahead_min_increment = 0.03;
    int32_t lookahead_beam_width = 64;
    bool lookahead_child_mode_shared = true;  // true: "shared", false: "branch_specific"
    std::vector<int32_t> lookahead_ops;       // empty == all ops allowed
    int32_t max_root_thresholds = 7;
    double min_weighted_probe_gain = 0.01;
    bool raw_pair_fallback = true;
    int32_t raw_pair_max_candidates = 400;
    bool aug_child_enabled = true;
    int32_t aug_child_max_candidates = 100;
    bool use_statistical_acceptance = true;
    double significance_alpha = 0.05;
};

// Grows one tree. `X` is the base design matrix (row-major, n x
// p_base); `labels` is this round's classification target (sign of
// the boosting residual). `owner_by_column[col]` lists the raw
// feature column indices `col` depends on (a raw column owns itself).
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
);

// Assigns each row of `X_new` (row-major, n x p_base, same base
// schema `grow_tree` was called with) to a leaf, reconstructing any
// synthetic columns the tree committed to along the way.
std::vector<int64_t> apply_tree(
    const double* X_new, int64_t n_rows, int64_t p_base,
    const FittedTree& tree
);

}  // namespace hugiml::rpte
