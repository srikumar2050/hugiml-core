from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_DATASETS_DIR = PROJECT_ROOT / "datasetsTabzilla"


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return value or "dataset"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def local_entries(root: Path) -> dict[int, dict[str, object]]:
    """Discover reusable task folders, including work from interrupted downloads."""
    entries: dict[int, dict[str, object]] = {}
    for metadata_path in sorted(root.glob("task_*/metadata.json")):
        try:
            entry = json.loads(metadata_path.read_text(encoding="utf-8"))
            entry["folder"] = metadata_path.parent.name
            entries[int(entry["task_id"])] = entry
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return entries


def reusable_entry(root: Path, entry: dict[str, object]) -> bool:
    """Return true only when dataset files and every split pair are present."""
    try:
        folder = root / str(entry["folder"])
        features = folder / str(entry["features_file"])
        target = folder / str(entry["target_file"])
        split_manifest = folder / str(entry["split_manifest_file"])
        if not features.is_file() or not target.is_file() or not split_manifest.is_file():
            return False
        expected_hashes = entry.get("sha256", {})
        if isinstance(expected_hashes, dict):
            if expected_hashes.get("features.parquet") != sha256(features):
                return False
            if expected_hashes.get("target.parquet") != sha256(target):
                return False
        splits = pd.read_csv(split_manifest)
        if len(splits) != int(entry["official_split_count"]):
            return False
        for split in splits.itertuples(index=False):
            if not (folder / str(split.train_file)).is_file():
                return False
            if not (folder / str(split.test_file)).is_file():
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError):
        return False


def task_rows(suite: object) -> list[dict[str, object]]:
    task_ids = [int(task_id) for task_id in suite.tasks]
    listing = openml.tasks.list_tasks(task_id=task_ids, output_format="dataframe")
    rows: list[dict[str, object]] = []
    for _, row in listing.iterrows():
        task_id = int(row.get("tid", row.get("task_id")))
        if task_id not in task_ids:
            continue
        rows.append(
            {
                "task_id": task_id,
                "dataset_id": int(row.get("did", row.get("data_id", -1))),
                "dataset_name": str(row.get("name", row.get("data_name", f"task-{task_id}"))),
                "n_rows": int(row.get("NumberOfInstances", -1)),
            }
        )
    found = {int(row["task_id"]) for row in rows}
    if found != set(task_ids):
        missing = sorted(set(task_ids) - found)
        for task_id in missing:
            task = openml.tasks.get_task(task_id, download_splits=False)
            dataset = task.get_dataset(download_data=False, download_qualities=True)
            qualities = dict(getattr(dataset, "qualities", {}) or {})
            rows.append(
                {
                    "task_id": task_id,
                    "dataset_id": int(getattr(task, "dataset_id", dataset.dataset_id)),
                    "dataset_name": str(dataset.name),
                    "n_rows": int(qualities.get("NumberOfInstances", -1)),
                }
            )
    rows.sort(key=lambda row: (int(row["n_rows"]) < 0, int(row["n_rows"]), int(row["task_id"])))
    return rows


def unique_columns(columns: object) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for raw in columns:
        name = str(raw)
        count = seen.get(name, 0)
        result.append(name if count == 0 else f"{name}__{count}")
        seen[name] = count + 1
    return result


def save_task(root: Path, selected: dict[str, object]) -> dict[str, object]:
    task_id = int(selected["task_id"])
    task = openml.tasks.get_task(task_id, download_splits=True)
    dataset = task.get_dataset(download_qualities=True)
    X, y, categorical, attributes = dataset.get_data(
        target=task.target_name, dataset_format="dataframe"
    )
    X = pd.DataFrame(X).reset_index(drop=True)
    X.columns = unique_columns(attributes if attributes and len(attributes) == X.shape[1] else X.columns)
    y_original = pd.Series(y, name=str(task.target_name)).reset_index(drop=True)
    if y_original.isna().any():
        raise ValueError(f"Task {task_id} contains missing target values")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_original.astype(str)).astype(np.int64)

    folder = root / f"task_{task_id}_{slugify(str(dataset.name))}"
    folder.mkdir(parents=True, exist_ok=True)
    split_dir = folder / "splits"
    split_dir.mkdir(exist_ok=True)

    features_path = folder / "features.parquet"
    target_path = folder / "target.parquet"
    X.to_parquet(features_path, index=False)
    pd.DataFrame({"target_original": y_original.astype(str), "target_encoded": y_encoded}).to_parquet(
        target_path, index=False
    )

    repeats, folds, samples = (int(value) for value in task.get_split_dimensions())
    split_rows: list[dict[str, object]] = []
    split_id = 0
    for repeat in range(repeats):
        for fold in range(folds):
            for sample in range(samples):
                train, test = task.get_train_test_split_indices(
                    repeat=repeat, fold=fold, sample=sample
                )
                train = np.asarray(train, dtype=np.int64)
                test = np.asarray(test, dtype=np.int64)
                if not len(train) or not len(test) or np.intersect1d(train, test).size:
                    raise ValueError(f"Invalid official split {split_id} for task {task_id}")
                if train.min() < 0 or test.min() < 0 or train.max() >= len(X) or test.max() >= len(X):
                    raise IndexError(f"Out-of-range official split {split_id} for task {task_id}")
                train_name = f"split_{split_id:03d}_train.npy"
                test_name = f"split_{split_id:03d}_test.npy"
                np.save(split_dir / train_name, train, allow_pickle=False)
                np.save(split_dir / test_name, test, allow_pickle=False)
                split_rows.append(
                    {
                        "split_id": split_id,
                        "repeat": repeat,
                        "fold": fold,
                        "sample": sample,
                        "train_file": f"splits/{train_name}",
                        "test_file": f"splits/{test_name}",
                        "n_train": len(train),
                        "n_test": len(test),
                    }
                )
                split_id += 1
    pd.DataFrame(split_rows).to_csv(folder / "split_manifest.csv", index=False)

    metadata = {
        "task_id": task_id,
        "dataset_id": int(getattr(task, "dataset_id", dataset.dataset_id)),
        "dataset_name": str(dataset.name),
        "target_name": str(task.target_name),
        "n_rows": len(X),
        "n_features": X.shape[1],
        "categorical_features": [str(X.columns[i]) for i, flag in enumerate(categorical or []) if flag],
        "class_labels": [str(value) for value in encoder.classes_],
        "n_classes": len(encoder.classes_),
        "official_repeats": repeats,
        "official_folds": folds,
        "official_samples": samples,
        "official_split_count": len(split_rows),
        "features_file": "features.parquet",
        "target_file": "target.parquet",
        "split_manifest_file": "split_manifest.csv",
        "sha256": {"features.parquet": sha256(features_path), "target.parquet": sha256(target_path)},
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {**metadata, "folder": folder.name}


def select_download_tasks(
    ordered_tasks: list[dict[str, object]],
    cached: dict[int, dict[str, object]],
    root: Path,
    *,
    smallest: int | None,
    add: int | None,
) -> tuple[list[dict[str, object]], str]:
    if add is not None:
        complete_ids = {
            int(task_id)
            for task_id, entry in cached.items()
            if reusable_entry(root, entry)
        }
        suite_ids = {int(row["task_id"]) for row in ordered_tasks}
        complete_ids.intersection_update(suite_ids)
        additions = [
            row for row in ordered_tasks if int(row["task_id"]) not in complete_ids
        ][: int(add)]
        selected_ids = complete_ids.union(int(row["task_id"]) for row in additions)
        selected = [row for row in ordered_tasks if int(row["task_id"]) in selected_ids]
        return selected, f"complete_cache_plus_{len(additions)}_next_smallest_missing_tasks"
    target_count = 36 if smallest is None else int(smallest)
    target_count = min(target_count, len(ordered_tasks))
    return ordered_tasks[:target_count], f"smallest_{target_count}_by_NumberOfInstances"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TabZilla datasets and official splits for offline benchmarking"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASETS_DIR)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--smallest",
        type=int,
        default=None,
        help="Target total number of TabZilla tasks to cache (default: all 36)",
    )
    selection.add_argument(
        "--add",
        type=int,
        default=None,
        help="Add this many next-smallest missing TabZilla tasks while retaining complete cached tasks",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download selected tasks even when a complete local copy exists",
    )
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(root / "openml_cache"))

    cached = local_entries(root)
    suite = openml.study.get_suite(379)
    ordered_tasks = task_rows(suite)
    if args.add is not None and args.add <= 0:
        parser.error("--add must be greater than zero")
    if args.smallest is not None and args.smallest <= 0:
        parser.error("--smallest must be greater than zero")
    selected, selection_label = select_download_tasks(
        ordered_tasks,
        cached,
        root,
        smallest=args.smallest,
        add=args.add,
    )
    datasets: list[dict[str, object]] = []

    def write_progress(*, complete: bool) -> None:
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "suite": "TabZilla",
            "suite_id": getattr(suite, "id", None),
            "selection": selection_label,
            "requested_dataset_count": len(selected),
            "dataset_count": len(datasets),
            "complete": complete,
            "datasets": datasets,
        }
        atomic_write_json(root / "manifest.json", manifest)

    for index, row in enumerate(selected, 1):
        task_id = int(row["task_id"])
        existing = cached.get(task_id)
        if not args.refresh and existing is not None and reusable_entry(root, existing):
            print(
                f"[{index}/{len(selected)}] cached task {task_id} {row['dataset_name']}",
                flush=True,
            )
            datasets.append(existing)
        else:
            print(
                f"[{index}/{len(selected)}] download task {task_id} {row['dataset_name']}",
                flush=True,
            )
            datasets.append(save_task(root, row))
        write_progress(complete=False)
    write_progress(complete=True)
    print(f"Saved {len(datasets)} datasets to {root}")


if __name__ == "__main__":
    main()


