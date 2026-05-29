/**
 * bind_mine_patterns.cpp — Python binding for mine_patterns.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 * Apache-2.0 License
 */

#include "bind_helpers.hpp"
#include "mining.hpp"

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

            py::gil_scoped_release release;
            return mine_patterns_cpp(td, ytrain, n_cls, K, L, G, timeout_s);
        },
        py::arg("td"), py::arg("ytrain"), py::arg("n_cls"),
        py::arg("K"), py::arg("L"), py::arg("G"),
        py::arg("timeout_s") = 0.0,
        "Run top-K HUI mining; returns list of PatternEntry.\n\n"
        "timeout_s: wall-clock seconds before mining is aborted gracefully\n"
        "(0 = no limit; partial results are returned on timeout).");
}
