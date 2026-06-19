/**
 * bind_mine_patterns.cpp — Python binding for mine_patterns.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 * Apache-2.0 License
 */

#include "bind_helpers.hpp"
#include "mining.hpp"
#include "mining_l2.hpp"
#include "resource_guard.hpp"

namespace py = pybind11;
using namespace hugiml;

void bind_mine_patterns(py::module_& m)
{
    py::class_<PatternEntry>(m, "PatternEntry",
        "One mined HUG pattern: utility, item IDs, and information gain.")
        .def_readonly("utility", &PatternEntry::utility)
        .def_readonly("items",   &PatternEntry::items)
        .def_readonly("ig",      &PatternEntry::ig)
        // ── Pickle / deepcopy support ──────────────────────────────────────
        // pybind11 objects are not picklable by default; without __reduce__
        // both pickle.dumps and copy.deepcopy fail with
        // "cannot pickle '_hugiml_core.PatternEntry' object".
        // We expose the three fields as a plain tuple and reconstruct via a
        // module-level factory so that pickle, deepcopy, joblib, and any
        // other serialisation that goes through the C-pickle protocol works.
        .def(py::pickle(
            // __getstate__: return a (utility, items, ig) tuple
            [](const PatternEntry& pe) {
                return py::make_tuple(pe.utility, pe.items, pe.ig);
            },
            // __setstate__: reconstruct from the same tuple
            [](py::tuple t) {
                if (t.size() != 3)
                    throw std::runtime_error(
                        "PatternEntry.__setstate__: expected 3-tuple, "
                        "got " + std::to_string(t.size()));
                PatternEntry pe;
                pe.utility = t[0].cast<double>();
                pe.items   = t[1].cast<std::vector<int>>();
                pe.ig      = t[2].cast<double>();
                return pe;
            }
        ));

    m.def("mine_patterns",
        [](const TransactionDataCpp& td,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> ytrain_arr,
           int n_cls, int K, int L, double G,
           double timeout_s)
        {
            validate_1d_array(ytrain_arr, "ytrain");
            validate_mining_params(K, L, G);
            if (n_cls < 2)
                throw std::invalid_argument(
                    "n_cls must be >= 2, got " + std::to_string(n_cls));

            auto yb = ytrain_arr.unchecked<1>();
            std::vector<int> ytrain(static_cast<size_t>(yb.shape(0)));
            for (py::ssize_t i = 0; i < yb.shape(0); i++)
                ytrain[i] = static_cast<int>(yb(i));

            try {
                py::gil_scoped_release release;
                return mine_patterns_cpp(td, ytrain, n_cls, K, L, G, timeout_s);
            } catch (const NativeMemoryError& e) {
                PyErr_SetString(PyExc_MemoryError, e.what());
                throw py::error_already_set();
            } catch (const std::bad_alloc&) {
                PyErr_SetString(PyExc_MemoryError,
                    "HUGIML native OOM during pattern mining");
                throw py::error_already_set();
            }
        },
        py::arg("td"), py::arg("ytrain"), py::arg("n_cls"),
        py::arg("K"), py::arg("L"), py::arg("G"),
        py::arg("timeout_s") = 0.0,
        "Run top-K HUI mining; returns list of PatternEntry.\n\n"
        "timeout_s: wall-clock seconds before mining is aborted gracefully\n"
        "(0 = no limit; partial results are returned on timeout).");

    m.def("mine_patterns_generic",
        [](const TransactionDataCpp& td,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> ytrain_arr,
           int n_cls, int K, int L, double G,
           double timeout_s)
        {
            validate_1d_array(ytrain_arr, "ytrain");
            validate_mining_params(K, L, G);
            if (n_cls < 2)
                throw std::invalid_argument(
                    "n_cls must be >= 2, got " + std::to_string(n_cls));

            auto yb = ytrain_arr.unchecked<1>();
            std::vector<int> ytrain(static_cast<size_t>(yb.shape(0)));
            for (py::ssize_t i = 0; i < yb.shape(0); i++)
                ytrain[i] = static_cast<int>(yb(i));

            try {
                py::gil_scoped_release release;
                return mine_patterns_generic_cpp(td, ytrain, n_cls, K, L, G, timeout_s);
            } catch (const NativeMemoryError& e) {
                PyErr_SetString(PyExc_MemoryError, e.what());
                throw py::error_already_set();
            } catch (const std::bad_alloc&) {
                PyErr_SetString(PyExc_MemoryError,
                    "HUGIML native OOM during generic pattern mining");
                throw py::error_already_set();
            }
        },
        py::arg("td"), py::arg("ytrain"), py::arg("n_cls"),
        py::arg("K"), py::arg("L"), py::arg("G"),
        py::arg("timeout_s") = 0.0,
        "Run the generic recursive top-K HUI miner; intended for regression tests/benchmarks.");

    m.def("mine_patterns_l2",
        [](const TransactionDataCpp& td,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> ytrain_arr,
           int n_cls, int K, double G,
           double timeout_s)
        {
            validate_1d_array(ytrain_arr, "ytrain");
            if (K <= 0)
                throw std::invalid_argument("K must be > 0, got " + std::to_string(K));
            if (n_cls < 2)
                throw std::invalid_argument(
                    "n_cls must be >= 2, got " + std::to_string(n_cls));

            auto yb = ytrain_arr.unchecked<1>();
            std::vector<int> ytrain(static_cast<size_t>(yb.shape(0)));
            for (py::ssize_t i = 0; i < yb.shape(0); i++)
                ytrain[i] = static_cast<int>(yb(i));

            try {
                py::gil_scoped_release release;
                return mine_patterns_l2_cpp(td, ytrain, n_cls, K, G, timeout_s);
            } catch (const NativeMemoryError& e) {
                PyErr_SetString(PyExc_MemoryError, e.what());
                throw py::error_already_set();
            } catch (const std::bad_alloc&) {
                PyErr_SetString(PyExc_MemoryError,
                    "HUGIML native OOM during L=2 pattern mining hotpath");
                throw py::error_already_set();
            }
        },
        py::arg("td"), py::arg("ytrain"), py::arg("n_cls"),
        py::arg("K"), py::arg("G"), py::arg("timeout_s") = 0.0,
        "Run the exact L=2 top-K HUI hot path; intended for regression tests/benchmarks.");

    py::class_<AugmentedPatternsResult>(m, "AugmentedPatternsResult",
        "Result of mine_patterns_l2_augmented_patterns: ordinary (utility-"
        "ranked) patterns plus a separate IG-ranked list of patterns that "
        "touch at least one relaxed interaction-information survivor item. "
        "len(patterns) + len(relaxed_patterns) <= K always; relaxed_quota "
        "is carved out of K, not added to it.")
        .def_readonly("patterns", &AugmentedPatternsResult::patterns)
        .def_readonly("relaxed_patterns", &AugmentedPatternsResult::relaxed_patterns);

    m.def("mine_patterns_l2_augmented_patterns",
        [](const TransactionDataCpp& td,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> ytrain_arr,
           int n_cls, int K, double G,
           const std::vector<int>& relaxed_cols,
           int relaxed_quota,
           double timeout_s)
        {
            validate_1d_array(ytrain_arr, "ytrain");
            if (K <= 0)
                throw std::invalid_argument("K must be > 0, got " + std::to_string(K));
            if (n_cls < 2)
                throw std::invalid_argument(
                    "n_cls must be >= 2, got " + std::to_string(n_cls));
            if (relaxed_quota < 0)
                throw std::invalid_argument(
                    "relaxed_quota must be >= 0, got " + std::to_string(relaxed_quota));

            auto yb = ytrain_arr.unchecked<1>();
            std::vector<int> ytrain(static_cast<size_t>(yb.shape(0)));
            for (py::ssize_t i = 0; i < yb.shape(0); i++)
                ytrain[i] = static_cast<int>(yb(i));

            try {
                py::gil_scoped_release release;
                return mine_patterns_l2_augmented_patterns_cpp(
                    td, ytrain, n_cls, K, G, relaxed_cols, relaxed_quota, timeout_s);
            } catch (const NativeMemoryError& e) {
                PyErr_SetString(PyExc_MemoryError, e.what());
                throw py::error_already_set();
            } catch (const std::bad_alloc&) {
                PyErr_SetString(PyExc_MemoryError,
                    "HUGIML native OOM during L=2 augmented_patterns hot path");
                throw py::error_already_set();
            }
        },
        py::arg("td"), py::arg("ytrain"), py::arg("n_cls"),
        py::arg("K"), py::arg("G"), py::arg("relaxed_cols"),
        py::arg("relaxed_quota") = 0,
        py::arg("timeout_s") = 0.0,
        "Extended: L=2 hot path where items whose source column is in "
        "relaxed_cols bypass G-based and RIU/TWU-seed gates so interaction-"
        "information survivors are mined even below G, without generating "
        "any augmented-pair operator features.\n\n"
        "relaxed_quota is a CARVE-OUT of K, not an addition: ordinary "
        "patterns are capped at K - relaxed_quota (utility-ranked, same as "
        "mine_patterns_l2) and relaxed-survivor patterns are capped at "
        "relaxed_quota (IG-ranked, since utility-based ranking would crowd "
        "them out). len(patterns) + len(relaxed_patterns) <= K always. "
        "A pattern touching a relaxed item is routed only to "
        "relaxed_patterns, never both. relaxed_quota=0 with empty "
        "relaxed_cols is exactly output-equivalent to mine_patterns_l2. "
        "Returns an AugmentedPatternsResult with .patterns and "
        ".relaxed_patterns.");

    m.def("mine_patterns_l2_augmented_patterns_v2",
        [](const TransactionDataCpp& td,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> ytrain_arr,
           int n_cls, int K, double G,
           const std::vector<int>& relaxed_cols,
           double timeout_s)
        {
            validate_1d_array(ytrain_arr, "ytrain");
            if (K <= 0)
                throw std::invalid_argument("K must be > 0, got " + std::to_string(K));
            if (n_cls < 2)
                throw std::invalid_argument(
                    "n_cls must be >= 2, got " + std::to_string(n_cls));

            auto yb = ytrain_arr.unchecked<1>();
            std::vector<int> ytrain(static_cast<size_t>(yb.shape(0)));
            for (py::ssize_t i = 0; i < yb.shape(0); i++)
                ytrain[i] = static_cast<int>(yb(i));

            try {
                py::gil_scoped_release release;
                return mine_patterns_l2_augmented_patterns_v2_cpp(
                    td, ytrain, n_cls, K, G, relaxed_cols, timeout_s);
            } catch (const NativeMemoryError& e) {
                PyErr_SetString(PyExc_MemoryError, e.what());
                throw py::error_already_set();
            } catch (const std::bad_alloc&) {
                PyErr_SetString(PyExc_MemoryError,
                    "HUGIML native OOM during L=2 augmented_patterns_v2 hot path");
                throw py::error_already_set();
            }
        },
        py::arg("td"), py::arg("ytrain"), py::arg("n_cls"),
        py::arg("K"), py::arg("G"), py::arg("relaxed_cols"),
        py::arg("timeout_s") = 0.0,
        "Extended v2: like mine_patterns_l2_augmented_patterns but with "
        "no relaxed_quota parameter. The ordinary (utility-ranked) heap and "
        "the relaxed-survivor (IG-ranked) heap each run at full capacity K, "
        "producing up to 2K raw candidates; the union is then re-ranked "
        "globally by IG and truncated to the final top-K. Returns a plain "
        "list[PatternEntry] of length <= K (no .patterns/.relaxed_patterns "
        "split, since the final selection is a single IG-sorted list).");

    m.def("mine_patterns_relaxed",
        [](const TransactionDataCpp& td,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> ytrain_arr,
           int n_cls, int K, int L, double G,
           const std::vector<int>& relaxed_cols,
           double timeout_s)
        {
            validate_1d_array(ytrain_arr, "ytrain");
            validate_mining_params(K, L, G);
            if (n_cls < 2)
                throw std::invalid_argument(
                    "n_cls must be >= 2, got " + std::to_string(n_cls));

            auto yb = ytrain_arr.unchecked<1>();
            std::vector<int> ytrain(static_cast<size_t>(yb.shape(0)));
            for (py::ssize_t i = 0; i < yb.shape(0); i++)
                ytrain[i] = static_cast<int>(yb(i));

            try {
                py::gil_scoped_release release;
                return mine_patterns_relaxed_cpp(
                    td, ytrain, n_cls, K, L, G, relaxed_cols, timeout_s);
            } catch (const NativeMemoryError& e) {
                PyErr_SetString(PyExc_MemoryError, e.what());
                throw py::error_already_set();
            } catch (const std::bad_alloc&) {
                PyErr_SetString(PyExc_MemoryError,
                    "HUGIML native OOM during relaxed pattern mining");
                throw py::error_already_set();
            }
        },
        py::arg("td"), py::arg("ytrain"), py::arg("n_cls"),
        py::arg("K"), py::arg("L"), py::arg("G"), py::arg("relaxed_cols"),
        py::arg("timeout_s") = 0.0,
        "interaction_relaxed_mining for supported L values (dispatches to "
        "the L2 hotpath's v2 relaxed variant when L==2, otherwise to the "
        "generic recursive path). Items whose source column is in "
        "relaxed_cols are admitted as depth-0 (root) candidates even if "
        "they fail the ordinary TWU/RIU seed filter -- this is the ONLY "
        "relaxation applied; every gate at depth 1+ (G checks, pair "
        "construction) is unmodified, so a relaxed item can only ever "
        "anchor a pattern as its root, never appear spliced in deeper. "
        "Ordinary and relaxed-root patterns are tracked in independent "
        "K-sized pools (utility-ranked and IG-ranked respectively), then "
        "merged and globally re-ranked by IG, truncated to the final "
        "top-K. With relaxed_cols empty this is output-equivalent to "
        "mine_patterns. Returns a plain list[PatternEntry] of length <= K.");

}
