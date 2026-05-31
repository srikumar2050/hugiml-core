Changelog
=========

v1.1.4
------

* Added the native L1 hot path for ``L=1`` fits. The C++ path fuses transaction preparation, single-item pattern mining, information-gain filtering, top-K retention, and sparse matrix construction to reduce Python/C++ overhead for the common L1 workflow.
* Moved adaptive binning selection into the C++ backend. Per-feature bin counts are now selected using supervised information-gain scoring and elbow-style stopping, while keeping Python metadata such as ``per_feature_b_``, ``_bin_edges_``, and ``ig_scores_`` available for inspection and serialization.
