# Native sparse index widths

Native pattern-matrix conversion selects signed 32-bit or 64-bit CSR indices and offsets from the matrix dimensions and actual number of nonzero entries. Structures fitting the signed 32-bit range retain 32-bit storage. The 64-bit path is used only when needed and requires a 64-bit process. Unsupported process address ranges raise `OverflowError`; values are never narrowed to fit.

Dense downstream blocks select their CSR index width before allocating index arrays. Strict-topK CSC scoring supports both widths with the same scoring algorithm. Its native entry points require the corresponding index-array dtype, preventing implicit narrowing. Python dispatch chooses the entry point from actual matrix dimensions and nonzero count. A loaded extension without the required capability raises an explicit error for large structures.

This is sparse-offset support, not unrestricted 64-bit mining. Transaction IDs, utility-list row IDs, and native mining dimensions retain their 32-bit representation. Classifier input shapes and principal native preparation/matrix entry points reject dimensions at or above 2,147,483,647. Maximum accepted dimension is 2,147,483,646; this is an indexing bound, not a guarantee that a dataset fits available memory. Further growth requires migration of mining data structures and their consumers.

The extension exposes `csr_index_bits_max` and `mining_row_count_max` for capacity planning. On a 32-bit process, small inputs retain 32-bit behavior and oversized structures raise an exception. Actual 32-bit operating-system builds require platform testing in addition to simulated address-width boundary tests.

## Validation

`tests/native/csr_validation.cpp` uses the production CSR conversion header. Its module exercises both index widths on small arrays and tests capacity arithmetic near 32-bit and 64-bit boundaries without large allocations.

Build into a separate directory using the configured platform compiler:

```text
python setup.py build_ext --build-lib build/index64_validation/lib --build-temp build/index64_validation/temp
python tests/native/setup_csr_validation.py build_ext --build-lib build/index64_validation/lib --build-temp build/index64_validation/test_temp
```

Run tests with `build/index64_validation/lib` first on `PYTHONPATH`, followed by `src` and the repository root. Confirm `_hugiml_core.__file__` points to that directory. Test files include `test_native_index_width.py`, `test_csr_sorted_contract.py`, `test_native_engine_internals.py`, `test_adaptive_native.py`, `test_augmented_pair_native.py`, `test_adaptive_sample_and_float32.py`, and `test_augmented_pair_cache_scoring_serialization.py`.

