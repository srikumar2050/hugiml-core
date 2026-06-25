/**
 * bind_prepare_mine_l1.cpp — Python bindings for the fused L=1 hot path
 *                             and C++ adaptive B selection.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 * Apache-2.0 License
 */

#include "bind_helpers.hpp"
#include "prepare_mine_l1.hpp"
#include "resource_guard.hpp"

namespace py = pybind11;
using namespace hugiml;


namespace {
inline void validate_2d_shape(const py::array& arr, const char* name) {
    if (arr.ndim() != 2)
        throw std::invalid_argument(
            std::string(name) + " must be 2-D, got " +
            std::to_string(arr.ndim()) + "-D");
    if (arr.shape(0) == 0)
        throw std::invalid_argument(std::string(name) + " has 0 rows");
}

inline bool is_float32_array(const py::array& arr) {
    return py::dtype::of<float>().is(arr.dtype());
}

inline auto ensure_float32_c(const py::array& arr) {
    auto out = py::array_t<float, py::array::c_style>::ensure(arr);
    if (!out) throw std::invalid_argument("X_num float32 input must be C-contiguous");
    return out;
}

inline auto ensure_float64_c(const py::array& arr) {
    auto out = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(arr);
    if (!out) throw std::invalid_argument("X_num must be convertible to a C-contiguous float64 array");
    return out;
}
} // namespace

void bind_prepare_mine_l1(py::module_& m)
{
    // ── AdaptiveBinResult (returned by select_adaptive_bins) ──────────────────
    py::class_<ColAdaptResult>(m, "ColAdaptResult",
        "Per-column result from C++ adaptive B selection.")
        .def_readonly("chosen_b",  &ColAdaptResult::chosen_b)
        .def_readonly("edges",     &ColAdaptResult::edges)
        .def_readonly("ig_scores", &ColAdaptResult::ig_scores);

    py::class_<AdaptiveBinResult>(m, "AdaptiveBinResult",
        "Result of select_adaptive_bins(): per-column metadata + pre-coded X.")
        .def_readonly("cols",             &AdaptiveBinResult::cols)
        .def_readonly("n_rows",           &AdaptiveBinResult::n_rows)
        .def_readonly("n_num_cols",       &AdaptiveBinResult::n_num_cols)
        .def_readonly("num_col_indices",  &AdaptiveBinResult::num_col_indices)
        .def("get_X_codes",
            [](const AdaptiveBinResult& r) {
                // Return X_codes as a (n_rows × n_num_cols) numpy int32 array.
                // -1 is the missing/non-finite sentinel.  The classifier casts
                // one column at a time only when it must materialize a legacy
                // pre-binned pandas/NumPy object, avoiding a second full
                // float64 n×p allocation.
                py::array_t<int32_t> arr({r.n_rows, r.n_num_cols});
                auto buf = arr.mutable_unchecked<2>();
                for (int row = 0; row < r.n_rows; row++)
                    for (int ci = 0; ci < r.n_num_cols; ci++)
                        buf(row, ci) = r.X_codes_flat[
                            static_cast<size_t>(row) * r.n_num_cols + ci];
                return arr;
            },
            "Return pre-coded X as a (n_rows, n_num_cols) int32 array; -1 means missing.")
        .def("get_X_codes_col",
            [](const AdaptiveBinResult& r, int ci) {
                if (ci < 0 || ci >= r.n_num_cols)
                    throw std::out_of_range("ci out of range in get_X_codes_col");
                py::array_t<int32_t> arr({r.n_rows});
                auto buf = arr.mutable_unchecked<1>();
                for (int row = 0; row < r.n_rows; row++)
                    buf(row) = r.X_codes_flat[
                        static_cast<size_t>(row) * r.n_num_cols + static_cast<size_t>(ci)];
                return arr;
            },
            py::arg("ci"),
            "Return one pre-coded int32 column; -1 means missing.");

    // ── L1FitResult (returned by prepare_and_mine_l1) ────────────────────────
    py::class_<L1FitResult>(m, "L1FitResult",
        "Result of the fused L=1 hot path: td, patterns, and COO arrays.")
        .def_readonly("td",       &L1FitResult::td)
        .def_readonly("patterns", &L1FitResult::patterns)
        .def_readonly("adaptive_cols", &L1FitResult::adaptive_cols)
        .def_readonly("adaptive_num_col_indices", &L1FitResult::adaptive_num_col_indices)
        .def_readonly("original_feature_names", &L1FitResult::original_feature_names)
        .def_readonly("original_feature_scores", &L1FitResult::original_feature_scores)
        .def("get_coo",
            [](const L1FitResult& r) {
                auto rows_arr = py::array_t<int32_t>(r.coo_rows.size());
                auto cols_arr = py::array_t<int32_t>(r.coo_cols.size());
                auto ra = rows_arr.mutable_unchecked<1>();
                auto ca = cols_arr.mutable_unchecked<1>();
                for (size_t k = 0; k < r.coo_rows.size(); k++) {
                    ra(static_cast<py::ssize_t>(k)) = r.coo_rows[k];
                    ca(static_cast<py::ssize_t>(k)) = r.coo_cols[k];
                }
                return py::make_tuple(rows_arr, cols_arr);
            },
            "Return (rows, cols) COO arrays for the sparse training matrix.")
        .def("get_csr",
            [](const L1FitResult& r, int n_rows, int n_cols) {
                if (n_rows < 0 || n_cols < 0)
                    throw std::invalid_argument("n_rows and n_cols must be non-negative");
                py::array_t<int32_t> indptr_arr({n_rows + 1});
                py::array_t<int32_t> indices_arr({static_cast<py::ssize_t>(r.coo_cols.size())});
                auto indptr = indptr_arr.mutable_unchecked<1>();
                auto indices = indices_arr.mutable_unchecked<1>();
                for (int i = 0; i <= n_rows; ++i) indptr(i) = 0;
                for (size_t k = 0; k < r.coo_rows.size(); ++k) {
                    const int rr = r.coo_rows[k];
                    if (rr < 0 || rr >= n_rows) throw std::out_of_range("COO row out of CSR bounds");
                    indptr(rr + 1) += 1;
                }
                for (int i = 0; i < n_rows; ++i) indptr(i + 1) += indptr(i);

                std::vector<int32_t> cursor(static_cast<size_t>(n_rows));
                for (int i = 0; i < n_rows; ++i) cursor[static_cast<size_t>(i)] = indptr(i);
                for (size_t k = 0; k < r.coo_cols.size(); ++k) {
                    const int rr = r.coo_rows[k];
                    const int cc = r.coo_cols[k];
                    if (cc < 0 || cc >= n_cols) throw std::out_of_range("COO col out of CSR bounds");
                    const int pos = cursor[static_cast<size_t>(rr)]++;
                    indices(pos) = cc;
                }
                int32_t* idx_ptr = static_cast<int32_t*>(indices_arr.mutable_data());
                for (int i = 0; i < n_rows; ++i) {
                    std::sort(idx_ptr + indptr(i), idx_ptr + indptr(i + 1));
                }
                return py::make_tuple(indptr_arr, indices_arr);
            },
            py::arg("n_rows"), py::arg("n_cols"),
            "Return (indptr, indices) CSR structure directly, avoiding Python COO arrays.");

    // ── select_adaptive_bins ─────────────────────────────────────────────────
    m.def("select_adaptive_bins",
        [](py::array_t<double,  py::array::c_style | py::array::forcecast> X_num,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> y,
           int n_cls,
           py::object col_names_py,
           py::array_t<uint8_t, py::array::forcecast> is_cat,
           py::list candidates_py,
           double ratio,
           double adaptive_binning_sample_frac,
           uint64_t adaptive_binning_sample_random_state)
        {
            validate_2d_shape(X_num, "X_num");

            std::vector<std::string> col_names_cpp;
            if (!col_names_py.is_none()) {
                py::list lst = col_names_py.cast<py::list>();
                col_names_cpp.reserve(lst.size());
                for (auto& it : lst) col_names_cpp.push_back(it.cast<std::string>());
            }

            std::vector<int> candidates;
            for (auto& it : candidates_py) candidates.push_back(it.cast<int>());
            if (candidates.empty())
                throw std::invalid_argument("candidates must be non-empty");

            try {
                py::gil_scoped_release release;
                return select_adaptive_bins_cpp(
                    X_num, y, n_cls,
                    col_names_cpp, is_cat,
                    candidates, ratio,
                    adaptive_binning_sample_frac,
                    adaptive_binning_sample_random_state);
            } catch (const NativeMemoryError& e) {
                PyErr_SetString(PyExc_MemoryError, e.what());
                throw py::error_already_set();
            } catch (const std::bad_alloc&) {
                PyErr_SetString(PyExc_MemoryError,
                    "HUGIML native OOM during adaptive bin selection");
                throw py::error_already_set();
            }
        },
        py::arg("X_num"), py::arg("y"), py::arg("n_cls"),
        py::arg("col_names"), py::arg("is_cat"),
        py::arg("candidates"), py::arg("ratio"),
        py::arg("adaptive_binning_sample_frac") = 1.0,
        py::arg("adaptive_binning_sample_random_state") = 42,
        "C++ adaptive B selection.  Returns AdaptiveBinResult with per-column\n"
        "metadata (chosen_b, edges, ig_scores) and the pre-coded X matrix.");

    // ── prepare_and_mine_l1 ──────────────────────────────────────────────────
    m.def("prepare_and_mine_l1",
        [](py::array X_num,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> y,
           int B,
           py::object col_names_py,
           py::array_t<uint8_t, py::array::forcecast> is_cat,
           py::array_t<uint8_t, py::array::forcecast> is_int,
           py::object X_cat_raw,
           py::object is_precoded_py,
           int K, double G, double timeout_s, bool compute_original_scores)
        {
            validate_2d_shape(X_num, "X_num");
            validate_mining_params(K, 1, G);   // L=1 always

            int n = static_cast<int>(X_num.shape(0));
            int p = static_cast<int>(X_num.shape(1));

            std::vector<std::string> col_names_cpp;
            if (!col_names_py.is_none()) {
                py::list lst = col_names_py.cast<py::list>();
                col_names_cpp.reserve(lst.size());
                for (auto& it : lst) col_names_cpp.push_back(it.cast<std::string>());
            }

            // Categorical raw strings
            std::vector<std::vector<std::string>> cat_strs;
            std::vector<std::vector<bool>>        cat_valid;
            extract_cat_data(X_cat_raw, is_cat, n, p, cat_strs, cat_valid);

            // is_precoded mask
            std::vector<bool> ipc_vec(p, false);
            if (!is_precoded_py.is_none()) {
                auto ipc = is_precoded_py.cast<
                    py::array_t<uint8_t, py::array::forcecast>>();
                auto ipcb = ipc.unchecked<1>();
                for (int j = 0; j < p; j++)
                    ipc_vec[j] = static_cast<bool>(ipcb(j));
            }

            try {
                const bool x_is_float32 = is_float32_array(X_num);
                if (x_is_float32) {
                    auto Xf = ensure_float32_c(X_num);
                    py::gil_scoped_release release;
                    return prepare_and_mine_l1_cpp(
                        Xf, y, B,
                        std::move(col_names_cpp),
                        is_cat, is_int,
                        std::move(ipc_vec),
                        std::move(cat_strs),
                        std::move(cat_valid),
                        K, G, timeout_s, compute_original_scores);
                }
                auto Xd = ensure_float64_c(X_num);
                py::gil_scoped_release release;
                return prepare_and_mine_l1_cpp(
                    Xd, y, B,
                    std::move(col_names_cpp),
                    is_cat, is_int,
                    std::move(ipc_vec),
                    std::move(cat_strs),
                    std::move(cat_valid),
                    K, G, timeout_s, compute_original_scores);
            } catch (const NativeMemoryError& e) {
                PyErr_SetString(PyExc_MemoryError, e.what());
                throw py::error_already_set();
            } catch (const std::bad_alloc&) {
                PyErr_SetString(PyExc_MemoryError,
                    "HUGIML native OOM during fused prepare_and_mine_l1");
                throw py::error_already_set();
            }
        },
        py::arg("X_num"), py::arg("y"), py::arg("B"),
        py::arg("col_names"), py::arg("is_cat"), py::arg("is_int"),
        py::arg("X_cat_raw"),
        py::arg("is_precoded") = py::none(),
        py::arg("K") = 200, py::arg("G") = 0.0, py::arg("timeout_s") = 0.0,
        py::arg("compute_original_scores") = false,
        "Fused L=1 hot path: Phase-1 column stats + single fused scan\n"
        "(no TransList) + IG + top-K heap + direct COO build.\n\n"
        "Returns L1FitResult with td (Phase-1 artefacts; td.transactions empty),\n"
        "patterns (PatternEntry list), and coo rows/cols accessible via get_coo().");

    // ── prepare_and_mine_l1_adaptive ─────────────────────────────────────────
    m.def("prepare_and_mine_l1_adaptive",
        [](py::array X_num,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> y,
           py::object col_names_py,
           py::array_t<uint8_t, py::array::forcecast> is_cat,
           py::array_t<uint8_t, py::array::forcecast> is_int,
           py::object X_cat_raw,
           py::list candidates_py,
           double ratio,
           int K, double G, double timeout_s, bool compute_original_scores,
           double adaptive_binning_sample_frac,
           uint64_t adaptive_binning_sample_random_state)
        {
            validate_2d_shape(X_num, "X_num");
            validate_mining_params(K, 1, G);

            int n = static_cast<int>(X_num.shape(0));
            int p = static_cast<int>(X_num.shape(1));

            std::vector<std::string> col_names_cpp;
            if (!col_names_py.is_none()) {
                py::list lst = col_names_py.cast<py::list>();
                col_names_cpp.reserve(lst.size());
                for (auto& it : lst) col_names_cpp.push_back(it.cast<std::string>());
            }

            std::vector<std::vector<std::string>> cat_strs;
            std::vector<std::vector<bool>>        cat_valid;
            extract_cat_data(X_cat_raw, is_cat, n, p, cat_strs, cat_valid);

            std::vector<int> candidates;
            for (auto& it : candidates_py) candidates.push_back(it.cast<int>());
            if (candidates.empty())
                throw std::invalid_argument("candidates must be non-empty");

            try {
                const bool x_is_float32 = is_float32_array(X_num);
                if (x_is_float32) {
                    auto Xf = ensure_float32_c(X_num);
                    py::gil_scoped_release release;
                    return prepare_and_mine_l1_adaptive_cpp(
                        Xf, y,
                        std::move(col_names_cpp),
                        is_cat, is_int,
                        std::move(cat_strs),
                        std::move(cat_valid),
                        candidates, ratio,
                        K, G, timeout_s, compute_original_scores,
                        adaptive_binning_sample_frac,
                        adaptive_binning_sample_random_state);
                }
                auto Xd = ensure_float64_c(X_num);
                py::gil_scoped_release release;
                return prepare_and_mine_l1_adaptive_cpp(
                    Xd, y,
                    std::move(col_names_cpp),
                    is_cat, is_int,
                    std::move(cat_strs),
                    std::move(cat_valid),
                    candidates, ratio,
                    K, G, timeout_s, compute_original_scores,
                    adaptive_binning_sample_frac,
                    adaptive_binning_sample_random_state);
            } catch (const NativeMemoryError& e) {
                PyErr_SetString(PyExc_MemoryError, e.what());
                throw py::error_already_set();
            } catch (const std::bad_alloc&) {
                PyErr_SetString(PyExc_MemoryError,
                    "HUGIML native OOM during fused adaptive prepare_and_mine_l1");
                throw py::error_already_set();
            }
        },
        py::arg("X_num"), py::arg("y"),
        py::arg("col_names"), py::arg("is_cat"), py::arg("is_int"),
        py::arg("X_cat_raw"), py::arg("candidates"), py::arg("ratio"),
        py::arg("K") = 200, py::arg("G") = 0.0, py::arg("timeout_s") = 0.0,
        py::arg("compute_original_scores") = false,
        py::arg("adaptive_binning_sample_frac") = 1.0,
        py::arg("adaptive_binning_sample_random_state") = 42,
        "Fused adaptive-B + L=1 hot path. Selects adaptive edges and mines L1\n"
        "without materialising X_codes_flat or a Python pre-binned DataFrame.");


    // ── prepare_and_mine_l1_fixed_numeric ───────────────────────────────────
    m.def("prepare_and_mine_l1_fixed_numeric",
        [](py::array X_num,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> y,
           int B,
           py::object col_names_py,
           py::array_t<uint8_t, py::array::forcecast> is_int,
           int K, double G, double timeout_s, bool compute_original_scores)
        {
            validate_2d_shape(X_num, "X_num");
            validate_mining_params(K, 1, G);

            std::vector<std::string> col_names_cpp;
            if (!col_names_py.is_none()) {
                py::list lst = col_names_py.cast<py::list>();
                col_names_cpp.reserve(lst.size());
                for (auto& it : lst) col_names_cpp.push_back(it.cast<std::string>());
            }
            try {
                const bool x_is_float32 = is_float32_array(X_num);
                if (x_is_float32) {
                    auto Xf = ensure_float32_c(X_num);
                    py::gil_scoped_release release;
                    return prepare_and_mine_l1_fixed_numeric_cpp(
                        Xf, y, B, std::move(col_names_cpp), is_int, K, G, timeout_s, compute_original_scores);
                }
                auto Xd = ensure_float64_c(X_num);
                py::gil_scoped_release release;
                return prepare_and_mine_l1_fixed_numeric_cpp(
                    Xd, y, B, std::move(col_names_cpp), is_int, K, G, timeout_s, compute_original_scores);
            } catch (const NativeMemoryError& e) {
                PyErr_SetString(PyExc_MemoryError, e.what());
                throw py::error_already_set();
            } catch (const std::bad_alloc&) {
                PyErr_SetString(PyExc_MemoryError,
                    "HUGIML native OOM during fixed numeric prepare_and_mine_l1");
                throw py::error_already_set();
            }
        },
        py::arg("X_num"), py::arg("y"), py::arg("B"),
        py::arg("col_names"), py::arg("is_int"),
        py::arg("K") = 200, py::arg("G") = 0.0, py::arg("timeout_s") = 0.0,
        py::arg("compute_original_scores") = false,
        "Fast fixed-B dense numeric L=1 hot path.");

}
