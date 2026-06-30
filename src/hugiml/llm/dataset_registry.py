"""Dataset discovery and loading for the optional HUGIML NLP interface.

The registry keeps dataset origins logically separate while presenting a merged
view to the user:

* ``llm_builtin``: curated small first-run datasets under ``LLM/datasets/builtin``
* ``user``: files accepted into ``LLM/datasets/user`` with explicit target metadata
* ``benchmark``: public/package-backed benchmark datasets, loaded through the
  existing ``experiments/benchmark/benchmark_dashboard.py`` interface when the
  source checkout is available
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schemas import DatasetInfo

_DATA_EXTENSIONS = {".csv", ".tsv", ".txt", ".parquet", ".pq", ".xlsx", ".xls"}
_TARGET_CANDIDATES = (
    "target",
    "label",
    "class",
    "y",
    "outcome",
    "response",
    "default",
    "churn",
    "risk",
)


class DatasetRegistry:
    """Merged dataset registry for LLM-driven HUGIML workflows."""

    def __init__(self, repo_root: str | Path | None = None) -> None:
        self.repo_root = self._find_repo_root(repo_root)
        self.package_root = Path(__file__).resolve().parent
        self.package_assets = self.package_root / "assets"
        self.llm_root = self.repo_root / "LLM" if self.repo_root else None
        source_builtin = self.llm_root / "datasets" / "builtin" if self.llm_root else None
        package_builtin = self.package_assets / "datasets" / "builtin"
        self.builtin_dir = source_builtin if source_builtin is not None and source_builtin.exists() else package_builtin
        if self.llm_root is not None and self.llm_root.exists():
            self.user_dir = self.llm_root / "datasets" / "user"
        else:
            user_base = Path(os.environ.get("HUGIML_LLM_HOME", Path.home() / ".hugiml" / "llm"))
            self.user_dir = user_base / "datasets" / "user"
        self._benchmark_module: Any | None = None

    @staticmethod
    def _find_repo_root(repo_root: str | Path | None = None) -> Path | None:
        if repo_root:
            return Path(repo_root).expanduser().resolve()
        here = Path(__file__).resolve()
        for base in [here, *here.parents]:
            if (base / "src" / "hugiml").exists() and (base / "LLM").exists():
                return base
            if (base / "pyproject.toml").exists() and (base / "src" / "hugiml").exists():
                return base
        cwd = Path.cwd().resolve()
        for base in [cwd, *cwd.parents]:
            if (base / "src" / "hugiml").exists():
                return base
        return None

    def list_datasets(
        self,
        include_profiles: bool = True,
        *,
        include_benchmarks: bool = True,
    ) -> list[DatasetInfo]:
        """Return discovered datasets.

        ``include_benchmarks`` defaults to ``True`` for backward compatibility and
        for command-line discovery.  UI surfaces can set it to ``False`` for a
        calmer first-run experience that shows only curated built-in datasets and
        registered user uploads.
        """

        infos: list[DatasetInfo] = []
        infos.extend(self._list_file_datasets(self.builtin_dir, "llm_builtin", include_profiles))
        infos.extend(self._list_file_datasets(self.user_dir, "user", include_profiles))
        if include_benchmarks:
            infos.extend(self._list_benchmark_datasets(include_profiles))
        return sorted(infos, key=lambda d: (d.source, d.name.lower()))

    def describe_dataset(self, name: str) -> dict[str, Any]:
        X, y, info = self.load_dataset(name)
        y_arr = np.asarray(y)
        target = info.target or "target"
        class_counts = pd.Series(y_arr).value_counts(dropna=False).sort_index().to_dict()
        missing = X.isna().mean().sort_values(ascending=False).head(10)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]
        return {
            **info.to_dict(),
            "rows": int(X.shape[0]),
            "features": int(X.shape[1]),
            "target": target,
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "numeric_features": numeric_cols[:30],
            "categorical_features": categorical_cols[:30],
            "missing_top": {str(k): float(v) for k, v in missing.items() if float(v) > 0},
            "columns": [str(c) for c in X.columns[:100]],
        }

    def load_dataset(self, name: str) -> tuple[pd.DataFrame, np.ndarray, DatasetInfo]:
        for source, base in (("llm_builtin", self.builtin_dir), ("user", self.user_dir)):
            path = self._resolve_file_dataset(base, name)
            if path is not None:
                df = self._read_table(path)
                target = self._target_for_file(path, df)
                if target is None:
                    raise ValueError(
                        f"Could not infer target column for {name!r}. For user uploads, register the "
                        "dataset through the UI or create a .target.json sidecar with a target_column key."
                    )
                y = df[target].to_numpy()
                X = df.drop(columns=[target]).copy()
                X = self._prepare_frame(X)
                info = DatasetInfo(
                    name=path.stem,
                    source=source,
                    path=str(path),
                    target=target,
                    rows=int(X.shape[0]),
                    features=int(X.shape[1]),
                    description=f"File dataset from {self._display_path(path)}",
                )
                return X, np.asarray(y), info

        bench = self._load_benchmark_module()
        if bench is not None and name in getattr(bench, "DATASET_NAMES", []):
            X, y, group = bench.load_dataset(name)
            info = DatasetInfo(
                name=name,
                source="benchmark",
                target="target",
                rows=int(X.shape[0]),
                features=int(X.shape[1]),
                description=f"{group} benchmark dataset exposed by experiments/benchmark.",
                origin_detail="Loaded via existing benchmark_dashboard.load_dataset interface.",
            )
            return self._prepare_frame(pd.DataFrame(X).copy()), np.asarray(y), info
        raise KeyError(f"Unknown dataset {name!r}.")

    def register_user_dataset(
        self,
        source_path: str | Path,
        *,
        target_column: str,
        dataset_name: str | None = None,
        overwrite: bool = False,
    ) -> DatasetInfo:
        """Accept a user dataset after an explicit target-column selection.

        This method is intended for the optional chat UI.  The uploaded file is
        not exposed through ``list_datasets`` until a target has been selected
        and persisted in the ``.target.json`` sidecar.
        """

        source = Path(source_path).expanduser().resolve()
        if not source.exists() or source.suffix.lower() not in _DATA_EXTENSIONS:
            raise ValueError(f"Unsupported or missing dataset file: {source}")
        df = self._read_table(source, nrows=25)
        if target_column not in df.columns:
            raise ValueError(f"Target column {target_column!r} was not found in {source.name}.")
        safe_name = _safe_dataset_name(dataset_name or source.stem)
        self.user_dir.mkdir(parents=True, exist_ok=True)
        dest = self.user_dir / f"{safe_name}{source.suffix.lower()}"
        if dest.exists() and not overwrite:
            raise FileExistsError(f"User dataset {dest.name!r} already exists. Choose another name.")
        if source != dest:
            shutil.copyfile(source, dest)
        self._write_target_sidecar(dest, target_column)
        full_df = self._read_table(dest)
        return DatasetInfo(
            name=dest.stem,
            source="user",
            path=str(dest),
            target=target_column,
            rows=int(full_df.shape[0]),
            features=int(full_df.shape[1] - 1),
            description="User-provided dataset registered after explicit target-column selection.",
        )


    def _display_path(self, path: Path) -> str:
        """Return a compact path label that works for source checkouts and wheels."""

        try:
            if self.repo_root is not None:
                return str(path.relative_to(self.repo_root))
        except Exception:
            pass
        try:
            return str(path.relative_to(self.package_root))
        except Exception:
            return str(path)

    def _list_file_datasets(
        self, base: Path, source: str, include_profiles: bool = True
    ) -> list[DatasetInfo]:
        if not base.exists():
            return []
        out: list[DatasetInfo] = []
        for path in sorted(base.iterdir()):
            if not path.is_file() or path.suffix.lower() not in _DATA_EXTENSIONS:
                continue
            target = None
            rows = None
            features = None
            if include_profiles:
                try:
                    df = self._read_table(path, nrows=2000)
                    target = self._target_for_file(path, df)
                    rows = self._count_rows(path, df)
                    features = int(df.shape[1] - (1 if target else 0))
                except Exception:
                    pass
            out.append(
                DatasetInfo(
                    name=path.stem,
                    source=source,
                    path=str(path),
                    target=target,
                    rows=rows,
                    features=features,
                    description=(
                        "Curated zero-config NLP sample dataset."
                        if source == "llm_builtin"
                        else "User-provided dataset registered in LLM/datasets/user."
                    ),
                )
            )
        return out

    def _list_benchmark_datasets(self, include_profiles: bool = True) -> list[DatasetInfo]:
        bench = self._load_benchmark_module()
        if bench is None:
            return []
        names = list(getattr(bench, "DATASET_NAMES", []))
        infos: list[DatasetInfo] = []
        for name in names:
            rows = None
            features = None
            group = "benchmark"
            if include_profiles:
                try:
                    X, _y, group = bench.load_dataset(name)
                    rows = int(X.shape[0])
                    features = int(X.shape[1])
                except Exception:
                    pass
            infos.append(
                DatasetInfo(
                    name=name,
                    source="benchmark",
                    target="target",
                    rows=rows,
                    features=features,
                    description=f"{group} dataset from the existing benchmark catalog.",
                    origin_detail="experiments/benchmark/benchmark_dashboard.py",
                )
            )
        return infos

    def _load_benchmark_module(self) -> Any | None:
        if self._benchmark_module is not None:
            return self._benchmark_module
        if self.repo_root is None:
            return None
        path = self.repo_root / "experiments" / "benchmark" / "benchmark_dashboard.py"
        if not path.exists():
            return None
        spec = importlib.util.spec_from_file_location("hugiml_llm_benchmark_dashboard", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            return None
        self._benchmark_module = module
        return module

    @staticmethod
    def _read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path, nrows=nrows)
        if suffix in {".tsv", ".txt"}:
            return pd.read_csv(path, sep="\t", nrows=nrows)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path, nrows=nrows)
        if suffix in {".parquet", ".pq"}:
            if nrows is not None:
                return pd.read_parquet(path).head(nrows)
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported dataset file extension: {path.suffix}")

    @staticmethod
    def _count_rows(path: Path, loaded_sample: pd.DataFrame) -> int:
        if path.suffix.lower() == ".csv":
            try:
                with open(path, encoding="utf-8") as fh:
                    return max(sum(1 for _ in fh) - 1, 0)
            except Exception:
                pass
        return int(loaded_sample.shape[0])

    def _target_for_file(self, path: Path, df: pd.DataFrame) -> str | None:
        sidecar = self._target_sidecar(path)
        if sidecar.exists():
            try:
                target = json.loads(sidecar.read_text(encoding="utf-8")).get("target_column")
                if target in df.columns:
                    return str(target)
            except Exception:
                pass
        return self._infer_target_column(df)

    @staticmethod
    def _target_sidecar(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".target.json")

    def _write_target_sidecar(self, path: Path, target_column: str) -> None:
        payload = {"target_column": target_column, "dataset_name": path.stem, "source": "hugiml-llm-ui"}
        self._target_sidecar(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _infer_target_column(df: pd.DataFrame) -> str | None:
        lowered = {str(c).lower(): str(c) for c in df.columns}
        for candidate in _TARGET_CANDIDATES:
            if candidate in lowered:
                return lowered[candidate]
        if len(df.columns) >= 2:
            last = str(df.columns[-1])
            nunique = df[last].nunique(dropna=True)
            if 1 < nunique <= max(20, int(0.1 * len(df))):
                return last
        return None

    @staticmethod
    def _prepare_frame(X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            if pd.api.types.is_bool_dtype(X[col]) or pd.api.types.is_object_dtype(X[col]):
                X[col] = X[col].astype("category")
            elif pd.api.types.is_string_dtype(X[col]):
                X[col] = X[col].astype("category")
            elif pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors="coerce")
            else:
                X[col] = X[col].astype("category")
        return X

    @staticmethod
    def _resolve_file_dataset(base: Path, name: str) -> Path | None:
        if not base.exists():
            return None
        requested = Path(name)
        if requested.is_absolute() or len(requested.parts) > 1:
            # A path-like name is only honored if it resolves *inside* this
            # registry's known base directory. Without this check, a dataset
            # name that happens to match a file anywhere on disk (e.g. an
            # LLM-proposed or attacker-influenced path) would bypass the
            # builtin/user sandbox and the documented target-column
            # registration requirement entirely.
            try:
                resolved = requested.expanduser().resolve()
                resolved.relative_to(base.resolve())
            except (OSError, ValueError):
                return None
            if resolved.is_file() and resolved.suffix.lower() in _DATA_EXTENSIONS:
                return resolved
            return None
        normalized = name.lower()
        for path in base.iterdir():
            if path.is_file() and path.suffix.lower() in _DATA_EXTENSIONS:
                if path.stem.lower() == normalized or path.name.lower() == normalized:
                    return path.resolve()
        return None


def _safe_dataset_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip()).strip("._")
    return clean or "user_dataset"
