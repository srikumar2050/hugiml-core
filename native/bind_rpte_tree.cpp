/**
 * bind_rpte_tree.cpp — pybind11 bindings for the native RPTE
 * tree-growth engine (rpte_tree.cpp).
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 * Apache-2.0 License
 *
 * A fitted tree is exchanged with Python as a plain dict of arrays
 * and lists, not an opaque C++ object -- so it pickles for free
 * through the ordinary Python pickle protocol, with no custom
 * __getstate__/__setstate__ glue needed on either side.
 */

#include "pybind_common.hpp"
#include "rpte_tree.hpp"

#include <cstring>

namespace py = pybind11;
using namespace hugiml::rpte;

namespace {

py::dict column_spec_to_dict(const ColumnSpec& spec) {
    py::dict d;
    d["a_idx"] = spec.a_idx;
    d["b_idx"] = spec.b_idx;
    d["op"] = spec.op;
    d["a_threshold"] = spec.a_threshold;
    d["b_threshold"] = spec.b_threshold;
    d["is_binarized"] = spec.is_binarized;
    d["owners"] = spec.owners;
    return d;
}

ColumnSpec column_spec_from_dict(const py::dict& d) {
    ColumnSpec spec;
    spec.a_idx = d["a_idx"].cast<int32_t>();
    spec.b_idx = d["b_idx"].cast<int32_t>();
    spec.op = d["op"].cast<int32_t>();
    spec.a_threshold = d["a_threshold"].cast<double>();
    spec.b_threshold = d["b_threshold"].cast<double>();
    spec.is_binarized = d["is_binarized"].cast<bool>();
    spec.owners = d["owners"].cast<std::vector<int32_t>>();
    return spec;
}

py::dict fitted_tree_to_dict(const FittedTree& tree) {
    py::dict out;
    out["is_fitted"] = tree.is_fitted;
    if (!tree.is_fitted) return out;

    const size_t n_nodes = tree.nodes.size();
    std::vector<uint8_t> is_leaf(n_nodes);
    std::vector<int32_t> feature_index(n_nodes), left(n_nodes), right(n_nodes);
    std::vector<double> threshold(n_nodes);
    for (size_t i = 0; i < n_nodes; ++i) {
        is_leaf[i] = tree.nodes[i].is_leaf ? 1 : 0;
        feature_index[i] = tree.nodes[i].feature_index;
        threshold[i] = tree.nodes[i].threshold;
        left[i] = tree.nodes[i].left;
        right[i] = tree.nodes[i].right;
    }
    out["node_is_leaf"] = is_leaf;
    out["node_feature_index"] = feature_index;
    out["node_threshold"] = threshold;
    out["node_left"] = left;
    out["node_right"] = right;

    py::list specs;
    for (const auto& spec : tree.synthetic_specs) specs.append(column_spec_to_dict(spec));
    out["synthetic_specs"] = specs;
    out["used_columns"] = tree.used_columns;
    out["used_raw_features"] = tree.used_raw_features;

    py::list log;
    for (const auto& entry : tree.growth_log) {
        py::dict e;
        e["step"] = entry.step;
        e["leaf_id"] = entry.leaf_id;
        e["leaf_depth"] = entry.leaf_depth;
        e["kind"] = entry.kind;
        e["root_index"] = entry.root_index;
        e["operation"] = entry.operation;
        e["source_features"] = entry.source_features;
        e["left_child_index"] = entry.left_child_index ? py::cast(*entry.left_child_index) : py::none();
        e["right_child_index"] = entry.right_child_index ? py::cast(*entry.right_child_index) : py::none();
        e["grandchild_index"] = entry.grandchild_index ? py::cast(*entry.grandchild_index) : py::none();
        e["grow_gain"] = entry.grow_gain;
        e["probe_gain"] = entry.probe_gain;
        e["root_probe_gain"] = entry.root_probe_gain;
        e["weighted_probe_gain"] = entry.weighted_probe_gain;
        e["statistically_significant"] = entry.statistically_significant;
        e["significance_p_value"] = entry.significance_p_value ? py::cast(*entry.significance_p_value) : py::none();
        e["leaf_count_before"] = entry.leaf_count_before;
        e["leaf_count_after"] = entry.leaf_count_after;
        log.append(e);
    }
    out["growth_log"] = log;
    return out;
}

FittedTree fitted_tree_from_dict(const py::dict& d) {
    FittedTree tree;
    tree.is_fitted = d["is_fitted"].cast<bool>();
    if (!tree.is_fitted) return tree;

    auto is_leaf = d["node_is_leaf"].cast<std::vector<uint8_t>>();
    auto feature_index = d["node_feature_index"].cast<std::vector<int32_t>>();
    auto threshold = d["node_threshold"].cast<std::vector<double>>();
    auto left = d["node_left"].cast<std::vector<int32_t>>();
    auto right = d["node_right"].cast<std::vector<int32_t>>();
    tree.nodes.resize(is_leaf.size());
    for (size_t i = 0; i < is_leaf.size(); ++i) {
        tree.nodes[i].is_leaf = is_leaf[i] != 0;
        tree.nodes[i].feature_index = feature_index[i];
        tree.nodes[i].threshold = threshold[i];
        tree.nodes[i].left = left[i];
        tree.nodes[i].right = right[i];
        if (tree.nodes[i].is_leaf) tree.leaf_node_indices.push_back(static_cast<int32_t>(i));
    }

    for (auto item : d["synthetic_specs"].cast<py::list>()) {
        tree.synthetic_specs.push_back(column_spec_from_dict(item.cast<py::dict>()));
    }
    tree.used_columns = d["used_columns"].cast<std::vector<int32_t>>();
    tree.used_raw_features = d["used_raw_features"].cast<std::vector<int32_t>>();
    return tree;
}

}  // namespace

static py::dict py_rpte_grow_tree(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> labels,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> grow_rows,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> probe_rows,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> eligible_columns,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> raw_indices,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> roots,
    const std::vector<std::vector<int32_t>>& owner_by_column,
    int32_t max_depth, int32_t max_leaf_nodes, int32_t min_samples_leaf, int32_t min_probe_leaf,
    uint64_t random_state, double greedy_stall_probe_ig, double lookahead_min_probe_ig,
    double lookahead_min_increment, int32_t lookahead_beam_width, bool lookahead_child_mode_shared,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> lookahead_ops,
    int32_t max_root_thresholds, double min_weighted_probe_gain, bool raw_pair_fallback,
    int32_t raw_pair_max_candidates, bool aug_child_enabled, int32_t aug_child_max_candidates,
    bool use_statistical_acceptance, double significance_alpha
) {
    auto X_info = X.request();
    if (X_info.ndim != 2) throw std::invalid_argument("X must be 2D.");
    const int64_t n_rows = X_info.shape[0];
    const int64_t p_base = X_info.shape[1];
    if (static_cast<int64_t>(owner_by_column.size()) != p_base) {
        throw std::invalid_argument("owner_by_column must have one entry per base column.");
    }

    TreeHyperparams hp;
    hp.max_depth = max_depth;
    hp.max_leaf_nodes = max_leaf_nodes;
    hp.min_samples_leaf = min_samples_leaf;
    hp.min_probe_leaf = min_probe_leaf;
    hp.random_state = random_state;
    hp.greedy_stall_probe_ig = greedy_stall_probe_ig;
    hp.lookahead_min_probe_ig = lookahead_min_probe_ig;
    hp.lookahead_min_increment = lookahead_min_increment;
    hp.lookahead_beam_width = lookahead_beam_width;
    hp.lookahead_child_mode_shared = lookahead_child_mode_shared;
    {
        auto ops_info = lookahead_ops.request();
        const int32_t* ops_ptr = static_cast<const int32_t*>(ops_info.ptr);
        hp.lookahead_ops.assign(ops_ptr, ops_ptr + ops_info.shape[0]);
    }
    hp.max_root_thresholds = max_root_thresholds;
    hp.min_weighted_probe_gain = min_weighted_probe_gain;
    hp.raw_pair_fallback = raw_pair_fallback;
    hp.raw_pair_max_candidates = raw_pair_max_candidates;
    hp.aug_child_enabled = aug_child_enabled;
    hp.aug_child_max_candidates = aug_child_max_candidates;
    hp.use_statistical_acceptance = use_statistical_acceptance;
    hp.significance_alpha = significance_alpha;

    auto labels_info = labels.request();
    std::vector<int8_t> labels_vec(
        static_cast<int8_t*>(labels_info.ptr), static_cast<int8_t*>(labels_info.ptr) + labels_info.shape[0]);
    auto grow_info = grow_rows.request();
    std::vector<int64_t> grow_vec(
        static_cast<int64_t*>(grow_info.ptr), static_cast<int64_t*>(grow_info.ptr) + grow_info.shape[0]);
    auto probe_info = probe_rows.request();
    std::vector<int64_t> probe_vec(
        static_cast<int64_t*>(probe_info.ptr), static_cast<int64_t*>(probe_info.ptr) + probe_info.shape[0]);
    auto elig_info = eligible_columns.request();
    std::vector<int32_t> elig_vec(
        static_cast<int32_t*>(elig_info.ptr), static_cast<int32_t*>(elig_info.ptr) + elig_info.shape[0]);
    auto raw_info = raw_indices.request();
    std::vector<int32_t> raw_vec(
        static_cast<int32_t*>(raw_info.ptr), static_cast<int32_t*>(raw_info.ptr) + raw_info.shape[0]);
    auto roots_info = roots.request();
    RootColumns roots_vec(
        static_cast<int32_t*>(roots_info.ptr), static_cast<int32_t*>(roots_info.ptr) + roots_info.shape[0]);

    const double* X_ptr = static_cast<const double*>(X_info.ptr);

    FittedTree tree;
    {
        py::gil_scoped_release release;
        tree = grow_tree(X_ptr, n_rows, p_base, labels_vec, grow_vec, probe_vec, elig_vec, raw_vec,
                          roots_vec, owner_by_column, hp);
    }

    py::dict out = fitted_tree_to_dict(tree);
    if (tree.is_fitted) {
        std::vector<int64_t> train_leaf_ids;
        {
            py::gil_scoped_release release;
            train_leaf_ids = apply_tree(X_ptr, n_rows, p_base, tree);
        }
        out["train_leaf_ids"] = train_leaf_ids;
    }
    return out;
}

static py::array_t<int64_t> py_rpte_apply_tree(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::dict tree_dict
) {
    auto X_info = X.request();
    if (X_info.ndim != 2) throw std::invalid_argument("X must be 2D.");
    const int64_t n_rows = X_info.shape[0];
    const int64_t p_base = X_info.shape[1];
    const double* X_ptr = static_cast<const double*>(X_info.ptr);

    FittedTree tree = fitted_tree_from_dict(tree_dict);
    std::vector<int64_t> leaf_ids;
    {
        py::gil_scoped_release release;
        leaf_ids = apply_tree(X_ptr, n_rows, p_base, tree);
    }
    py::array_t<int64_t> out(static_cast<py::ssize_t>(leaf_ids.size()));
    std::memcpy(out.mutable_data(), leaf_ids.data(), leaf_ids.size() * sizeof(int64_t));
    return out;
}

void bind_rpte_tree(py::module_& m) {
    m.def(
        "rpte_grow_tree", &py_rpte_grow_tree,
        py::arg("X"), py::arg("labels"), py::arg("grow_rows"), py::arg("probe_rows"),
        py::arg("eligible_columns"), py::arg("raw_indices"), py::arg("roots"), py::arg("owner_by_column"),
        py::arg("max_depth"), py::arg("max_leaf_nodes"), py::arg("min_samples_leaf"), py::arg("min_probe_leaf"),
        py::arg("random_state"), py::arg("greedy_stall_probe_ig"), py::arg("lookahead_min_probe_ig"),
        py::arg("lookahead_min_increment"), py::arg("lookahead_beam_width"), py::arg("lookahead_child_mode_shared"),
        py::arg("lookahead_ops"), py::arg("max_root_thresholds"), py::arg("min_weighted_probe_gain"),
        py::arg("raw_pair_fallback"), py::arg("raw_pair_max_candidates"), py::arg("aug_child_enabled"),
        py::arg("aug_child_max_candidates"), py::arg("use_statistical_acceptance"), py::arg("significance_alpha"),
        "Grows one leaf-wise bounded-lookahead tree natively. Returns a plain "
        "dict describing the fitted tree structure (picklable as-is), including "
        "this round's leaf assignment over X."
    );
    m.def(
        "rpte_apply_tree", &py_rpte_apply_tree,
        py::arg("X"), py::arg("tree"),
        "Assigns each row of X to a leaf of a previously fitted tree (as "
        "returned by rpte_grow_tree), reconstructing any synthesized columns "
        "the tree committed to."
    );
}
