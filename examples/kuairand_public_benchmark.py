from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def rate(positive: int, total: int, prior: float, global_rate: float) -> float:
    return (positive + prior * global_rate) / (total + prior)


def load_authors(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"video_id", "author_id"}.issubset(reader.fieldnames):
            raise ValueError("Video feature file must contain video_id and author_id")
        return {row["video_id"]: row["author_id"] for row in reader}


def train_statistics(train_path: Path, authors: dict[str, str]) -> dict[str, Any]:
    item: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    author: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    user_tab: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    user_duration: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    positives = rows = 0
    with train_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"user_id", "video_id", "is_click", "tab", "duration_ms"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Training log is missing columns: {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            label_text = row["is_click"]
            if label_text not in {"0", "1"}:
                raise ValueError(f"Invalid click label at training row {row_number}")
            label = int(label_text)
            video = row["video_id"]
            author_id = authors.get(video, "UNK")
            try:
                duration_bucket = min(9, max(0, int(float(row["duration_ms"])) // 30000))
            except ValueError:
                duration_bucket = -1
            keys = (
                (item, video),
                (author, author_id),
                (user_tab, (row["user_id"], row["tab"])),
                (user_duration, (row["user_id"], duration_bucket)),
            )
            for mapping, key in keys:
                mapping[key][0] += label
                mapping[key][1] += 1
            positives += label
            rows += 1
    if rows == 0:
        raise ValueError("Training log is empty")
    return {
        "global_rate": positives / rows,
        "rows": rows,
        "item": item,
        "author": author,
        "user_tab": user_tab,
        "user_duration": user_duration,
    }


def component(mapping: dict[Any, list[int]], key: Any, prior: float, global_rate: float) -> float:
    positive, total = mapping.get(key, (0, 0))
    return rate(positive, total, prior, global_rate)


def score(strategy: str, row: dict[str, str], stats: dict[str, Any], authors: dict[str, str]) -> float:
    global_rate = float(stats["global_rate"])
    video = row["video_id"]
    author_id = authors.get(video, "UNK")
    try:
        duration_bucket = min(9, max(0, int(float(row["duration_ms"])) // 30000))
    except ValueError:
        duration_bucket = -1
    item_score = component(stats["item"], video, 20.0, global_rate)
    author_score = component(stats["author"], author_id, 50.0, global_rate)
    tab_score = component(stats["user_tab"], (row["user_id"], row["tab"]), 20.0, global_rate)
    duration_score = component(
        stats["user_duration"], (row["user_id"], duration_bucket), 20.0, global_rate
    )
    if strategy == "official_baseline":
        return item_score
    if strategy == "author_context":
        return 0.72 * item_score + 0.28 * author_score
    if strategy == "user_tab_affinity":
        return 0.72 * item_score + 0.28 * tab_score
    if strategy == "hybrid_context":
        return 0.55 * item_score + 0.20 * author_score + 0.15 * tab_score + 0.10 * duration_score
    raise ValueError(f"Unknown strategy: {strategy}")


def evaluate(grouped: dict[str, list[tuple[float, int, str]]]) -> tuple[float, float, int]:
    ndcg_total = recall_total = 0.0
    evaluated_users = 0
    for candidates in grouped.values():
        ranked = sorted(candidates, key=lambda value: (-value[0], value[2]))
        positives = sum(label for _, label, _ in candidates)
        ideal_count = min(10, positives)
        ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
        dcg = sum(
            label / math.log2(index + 2)
            for index, (_, label, _) in enumerate(ranked[:10])
        )
        ndcg_total += dcg / ideal if ideal else 0.0
        recall_total += sum(label for _, label, _ in ranked[:50]) / positives if positives else 0.0
        evaluated_users += 1
    if evaluated_users == 0:
        raise ValueError("Validation log has no users")
    return ndcg_total / evaluated_users, recall_total / evaluated_users, evaluated_users


def run(args: argparse.Namespace) -> int:
    train_path = Path(args.train)
    validation_path = Path(args.validation)
    authors = load_authors(Path(args.video_features))
    stats = train_statistics(train_path, authors)
    submission_path = Path(args.submission)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
    validation_rows = positives = 0
    with validation_path.open("r", encoding="utf-8-sig", newline="") as source, submission_path.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        reader = csv.DictReader(source)
        required = {"user_id", "video_id", "is_click", "tab", "duration_ms"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Validation log is missing columns: {sorted(required)}")
        writer = csv.writer(destination)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, row in enumerate(reader):
            label_text = row["is_click"]
            if label_text not in {"0", "1"}:
                raise ValueError(f"Invalid click label at validation row {row_id + 2}")
            label = int(label_text)
            prediction = score(args.strategy, row, stats, authors)
            writer.writerow([row_id, row["user_id"], row["video_id"], f"{prediction:.12g}"])
            grouped[row["user_id"]].append((prediction, label, row["video_id"]))
            validation_rows += 1
            positives += label
    ndcg, recall, users = evaluate(grouped)
    result = {
        "status": "succeeded",
        "strategy": args.strategy,
        "metrics": {"NDCG@10": ndcg, "Recall@50": recall},
        "official_selection_score": (ndcg + recall) / 2,
        "resources": {"gpu_hours": 0.0, "llm_tokens": 0},
        "diagnostics": {
            "train_rows": stats["rows"],
            "validation_rows": validation_rows,
            "validation_users": users,
            "validation_clicks": positives,
            "evaluation_protocol": "rank_observed_standard_log_impressions_per_user",
            "competition_valid": False,
        },
    }
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result["metrics"], sort_keys=True))
    return 0


def validate(args: argparse.Namespace) -> int:
    rows = 0
    with Path(args.validation).open("r", encoding="utf-8-sig", newline="") as source, Path(
        args.submission
    ).open("r", encoding="utf-8", newline="") as submitted:
        expected = csv.DictReader(source)
        actual = csv.DictReader(submitted)
        if actual.fieldnames != ["row_id", "user_id", "video_id", "score"]:
            raise ValueError(f"Invalid submission columns: {actual.fieldnames}")
        for row_id, (expected_row, actual_row) in enumerate(zip(expected, actual, strict=True)):
            if int(actual_row["row_id"]) != row_id:
                raise ValueError(f"row_id mismatch at {row_id}")
            if actual_row["user_id"] != expected_row["user_id"] or actual_row["video_id"] != expected_row["video_id"]:
                raise ValueError(f"row alignment mismatch at {row_id}")
            if not math.isfinite(float(actual_row["score"])):
                raise ValueError(f"non-finite score at {row_id}")
            rows += 1
    print(f"submission valid: {rows:,} rows")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--strategy", required=True)
    run_parser.add_argument("--train", required=True)
    run_parser.add_argument("--validation", required=True)
    run_parser.add_argument("--video-features", required=True)
    run_parser.add_argument("--result", required=True)
    run_parser.add_argument("--submission", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--validation", required=True)
    validate_parser.add_argument("--submission", required=True)
    args = parser.parse_args()
    return run(args) if args.command == "run" else validate(args)


if __name__ == "__main__":
    raise SystemExit(main())

