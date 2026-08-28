from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, sha256_file, utc_timestamp


LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "log_random_4_22_to_5_08_pure.csv",
)
FEATURE_FILES = (
    "user_features_pure.csv",
    "video_features_basic_pure.csv",
    "video_features_statistic_pure.csv",
)
REQUIRED_LOG_COLUMNS = {"user_id", "video_id", "date", "is_click"}


def summarize_log(path: Path) -> dict[str, Any]:
    users: set[str] = set()
    items: set[str] = set()
    rows = positives = 0
    minimum_date: int | None = None
    maximum_date: int | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = REQUIRED_LOG_COLUMNS - set(columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            label = row["is_click"]
            if label not in {"0", "1"}:
                raise ValueError(f"{path.name}:{row_number} has invalid is_click={label!r}")
            try:
                date = int(row["date"])
            except ValueError as exc:
                raise ValueError(f"{path.name}:{row_number} has invalid date={row['date']!r}") from exc
            rows += 1
            positives += int(label)
            users.add(row["user_id"])
            items.add(row["video_id"])
            minimum_date = date if minimum_date is None else min(minimum_date, date)
            maximum_date = date if maximum_date is None else max(maximum_date, date)
    if rows == 0:
        raise ValueError(f"{path.name} contains no interactions")
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "columns": columns,
        "rows": rows,
        "unique_users": len(users),
        "unique_items": len(items),
        "date_min": minimum_date,
        "date_max": maximum_date,
        "click_positives": positives,
        "click_rate": positives / rows,
    }


def summarize_table(path: Path) -> dict[str, Any]:
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        columns = next(reader, None)
        if not columns:
            raise ValueError(f"{path.name} has no header")
        for _ in reader:
            rows += 1
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "columns": columns,
        "rows": rows,
    }


def build_manifest(data_dir: Path, *, archive_md5: str | None = None) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    missing = [name for name in (*LOG_FILES, *FEATURE_FILES) if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing KuaiRand-Pure files: {', '.join(missing)}")
    return {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "dataset": "KuaiRand-Pure",
        "role": "public_source_data",
        "competition_split_status": "unverified_not_for_competition_selection",
        "positive_label_available": "is_click",
        "archive_md5": archive_md5,
        "data_layout": "KuaiRand-Pure/data",
        "logs": {name: summarize_log(data_dir / name) for name in LOG_FILES},
        "features": {name: summarize_table(data_dir / name) for name in FEATURE_FILES},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and manifest public KuaiRand-Pure data")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--archive-md5")
    args = parser.parse_args(argv)
    manifest = build_manifest(Path(args.data_dir), archive_md5=args.archive_md5)
    atomic_write_json(Path(args.output), manifest)
    print(
        f"manifested {sum(item['rows'] for item in manifest['logs'].values()):,} interactions "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
