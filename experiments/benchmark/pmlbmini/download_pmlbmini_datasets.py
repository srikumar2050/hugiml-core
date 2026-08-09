from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_DATASETS_DIR = PROJECT_ROOT / "datasetsPmlbmini"
# Pin the last repository commit before the 2025 duplicate-dataset cleanup,
# which removed several datasets included in the published PMLBmini-44 suite.
PMLB_COMMIT = "c936c9e8ca0e5995816a42e1e8c829b8bcd9a8c9"
PMLB_RAW = f"https://media.githubusercontent.com/media/EpistasisLab/pmlb/{PMLB_COMMIT}/datasets"
OUTER_FOLDS = 3
REPEAT_SEEDS = (42, 43, 44)
SPLIT_PROTOCOL = "repeated_stratified_3_fold_shuffle_seeds_42_43_44"

# PMLBmini, Table 3 order (ascending sample size).
DATASETS = [
    "parity5", "analcatdata_fraud", "analcatdata_aids",
    "analcatdata_bankruptcy", "analcatdata_japansolvent", "labor",
    "analcatdata_asbestos", "lupus", "postoperative_patient_data",
    "analcatdata_cyyoung9302", "analcatdata_cyyoung8092",
    "analcatdata_creditscore", "appendicitis", "molecular_biology_promoters",
    "analcatdata_boxing1", "mux6", "analcatdata_boxing2", "hepatitis",
    "corral", "glass2", "backache", "prnn_crabs", "sonar", "biomed",
    "prnn_synth", "analcatdata_lawsuit", "spect", "heart_statlog",
    "breast_cancer", "heart_h", "hungarian", "cleve", "heart_c",
    "haberman", "bupa", "spectf", "ionosphere", "colic", "horse_colic",
    "house_votes_84", "vote", "saheart", "clean1", "irish",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")


def complete_entry(root: Path, entry: dict) -> bool:
    try:
        folder = root / entry["folder"]
        features = folder / entry["features_file"]
        target = folder / entry["target_file"]
        splits = pd.read_csv(folder / entry["split_manifest_file"])
        if sha256(features) != entry["sha256"]["features.parquet"]:
            return False
        if sha256(target) != entry["sha256"]["target.parquet"]:
            return False
        return len(splits) == OUTER_FOLDS * len(REPEAT_SEEDS) and all(
            (folder / row.train_file).is_file() and (folder / row.test_file).is_file()
            for row in splits.itertuples(index=False)
        )
    except (OSError, KeyError, ValueError, pd.errors.ParserError):
        return False


def save_dataset(root: Path, order: int, name: str) -> dict:
    source_url = f"{PMLB_RAW}/{name}/{name}.tsv.gz"
    frame = pd.read_csv(source_url, sep="\t", compression="gzip")
    if "target" not in frame.columns:
        raise ValueError(f"{name} has no PMLB target column")
    y_original = frame.pop("target").reset_index(drop=True)
    X = frame.reset_index(drop=True)
    if X.isna().any().any() or y_original.isna().any():
        raise ValueError(f"{name} unexpectedly contains missing values")
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original.astype(str)).astype(np.int64)
    if len(encoder.classes_) != 2 or len(X) > 500:
        raise ValueError(f"{name} violates the PMLBmini binary/500-row definition")

    folder = root / f"task_{order:03d}_{slugify(name)}"
    split_dir = folder / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    features_path = folder / "features.parquet"
    target_path = folder / "target.parquet"
    X.to_parquet(features_path, index=False)
    pd.DataFrame(
        {"target_original": y_original.astype(str), "target_encoded": y}
    ).to_parquet(target_path, index=False)

    split_rows = []
    split_id = 0
    for repeat, seed in enumerate(REPEAT_SEEDS):
        splitter = StratifiedKFold(
            n_splits=OUTER_FOLDS, shuffle=True, random_state=seed
        )
        for fold, (train, test) in enumerate(splitter.split(X, y)):
            train_name = f"split_{split_id:03d}_train.npy"
            test_name = f"split_{split_id:03d}_test.npy"
            np.save(split_dir / train_name, np.asarray(train, dtype=np.int64), allow_pickle=False)
            np.save(split_dir / test_name, np.asarray(test, dtype=np.int64), allow_pickle=False)
            split_rows.append(
                {
                    "split_id": split_id, "repeat": repeat, "fold": fold, "sample": 0,
                    "train_file": f"splits/{train_name}",
                    "test_file": f"splits/{test_name}",
                    "n_train": len(train), "n_test": len(test), "split_seed": seed,
                }
            )
            split_id += 1
    pd.DataFrame(split_rows).to_csv(folder / "split_manifest.csv", index=False)
    entry = {
        "task_id": order,
        "dataset_id": order,
        "dataset_name": name,
        "target_name": "target",
        "n_rows": len(X),
        "n_features": X.shape[1],
        "categorical_features": [],
        "class_labels": [str(value) for value in encoder.classes_],
        "n_classes": 2,
        "official_repeats": len(REPEAT_SEEDS),
        "official_folds": OUTER_FOLDS,
        "official_samples": 1,
        "official_split_count": OUTER_FOLDS * len(REPEAT_SEEDS),
        "features_file": "features.parquet",
        "target_file": "target.parquet",
        "split_manifest_file": "split_manifest.csv",
        "split_protocol": SPLIT_PROTOCOL,
        "split_seeds": list(REPEAT_SEEDS),
        "source_url": source_url,
        "sha256": {
            "features.parquet": sha256(features_path),
            "target.parquet": sha256(target_path),
        },
        "folder": folder.name,
    }
    atomic_json(folder / "metadata.json", entry)
    return entry


def select_dataset_names(
    root: Path,
    cached: dict[str, dict],
    *,
    smallest: int | None,
    add: int | None,
) -> tuple[list[str], str]:
    if add is not None:
        complete = {
            name for name, entry in cached.items()
            if name in DATASETS and complete_entry(root, entry)
        }
        additions = [name for name in DATASETS if name not in complete][:add]
        selected = [name for name in DATASETS if name in complete or name in additions]
        return selected, f"complete_cache_plus_{len(additions)}_next_missing_datasets"
    target = len(DATASETS) if smallest is None else min(smallest, len(DATASETS))
    return DATASETS[:target], f"first_{target}_datasets_in_canonical_order"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PMLBmini-44 and persist three repeated 3-fold partitions"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASETS_DIR)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--smallest",
        type=int,
        default=None,
        help="Target total number of datasets in canonical PMLBmini order (default: all 44)",
    )
    selection.add_argument(
        "--add",
        type=int,
        default=None,
        help="Add this many next missing datasets while retaining complete cached datasets",
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.smallest is not None and args.smallest <= 0:
        parser.error("--smallest must be greater than zero")
    if args.add is not None and args.add <= 0:
        parser.error("--add must be greater than zero")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cached = {}
    for path in root.glob("task_*/metadata.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            cached[str(value["dataset_name"])] = value
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            pass

    selected_names, selection_label = select_dataset_names(
        root,
        cached,
        smallest=args.smallest,
        add=args.add,
    )
    entries = []
    for index, name in enumerate(selected_names, 1):
        order = DATASETS.index(name) + 1
        existing = cached.get(name)
        if not args.refresh and existing and complete_entry(root, existing):
            print(f"[{index}/{len(selected_names)}] cached {name}", flush=True)
            entry = existing
        else:
            print(f"[{index}/{len(selected_names)}] download {name}", flush=True)
            entry = save_dataset(root, order, name)
        entries.append(entry)
        atomic_json(
            root / "manifest.json",
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "suite": "PMLBmini",
                "suite_id": "PMLBmini-44",
                "source_revision": PMLB_COMMIT,
                "selection": selection_label,
                "requested_dataset_count": len(selected_names),
                "dataset_count": len(entries),
                "complete": False,
                "split_protocol": SPLIT_PROTOCOL,
        "split_seeds": list(REPEAT_SEEDS),
                "datasets": entries,
            },
        )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["complete"] = True
    atomic_json(root / "manifest.json", manifest)
    print(f"Saved {len(entries)} datasets to {root}")


if __name__ == "__main__":
    main()
