from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def load_spec(path: str | Path) -> dict[str, int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    required = {"users", "items", "categories", "seed"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"Missing spec keys: {sorted(missing)}")
    return {key: int(value[key]) for key in required}


def probability(user: int, item: int, categories: int) -> float:
    preferred = user % categories
    category = item % categories
    quality = ((item * 17 + 11) % 19) / 190.0
    if category == preferred:
        return min(0.92, 0.72 + quality)
    if category == (preferred + 1) % categories:
        return 0.20 + quality / 2
    return 0.03 + quality / 4


def interactions(spec: dict[str, int], *, repeats: int) -> list[tuple[int, int, int]]:
    rng = random.Random(spec["seed"])
    rows: list[tuple[int, int, int]] = []
    for user in range(spec["users"]):
        for item in range(spec["items"]):
            for _ in range(repeats):
                label = int(rng.random() < probability(user, item, spec["categories"]))
                rows.append((user, item, label))
    return rows


def train_statistics(
    rows: Iterable[tuple[int, int, int]], categories: int
) -> tuple[dict[int, float], dict[tuple[int, int], float], float]:
    item_sum: dict[int, int] = defaultdict(int)
    item_count: dict[int, int] = defaultdict(int)
    user_category_sum: dict[tuple[int, int], int] = defaultdict(int)
    user_category_count: dict[tuple[int, int], int] = defaultdict(int)
    total_sum = total_count = 0
    for user, item, label in rows:
        category = item % categories
        item_sum[item] += label
        item_count[item] += 1
        user_category_sum[(user, category)] += label
        user_category_count[(user, category)] += 1
        total_sum += label
        total_count += 1
    global_rate = total_sum / total_count
    item_rate = {
        item: (item_sum[item] + 10 * global_rate) / (item_count[item] + 10)
        for item in item_count
    }
    affinity = {
        key: (user_category_sum[key] + 5 * global_rate) / (user_category_count[key] + 5)
        for key in user_category_count
    }
    return item_rate, affinity, global_rate


def score_rows(
    strategy: str,
    train_rows: list[tuple[int, int, int]],
    valid_rows: list[tuple[int, int, int]],
    categories: int,
) -> list[float]:
    item_rate, affinity, global_rate = train_statistics(train_rows, categories)
    scores: list[float] = []
    for user, item, _ in valid_rows:
        popularity = item_rate.get(item, global_rate)
        contextual = affinity.get((user, item % categories), global_rate)
        if strategy == "official_baseline":
            score = popularity
        elif strategy == "category_affinity":
            score = contextual
        elif strategy == "hybrid_affinity_popularity":
            score = 0.85 * contextual + 0.15 * popularity
        elif strategy == "shrinkage_popularity":
            score = 0.9 * popularity + 0.1 * global_rate
        elif strategy == "noise_control":
            score = ((user * 15485863 + item * 32452843) % 1009) / 1009.0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        scores.append(score)
    return scores


def evaluate(
    rows: list[tuple[int, int, int]], scores: list[float]
) -> tuple[float, float]:
    grouped: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for (user, item, label), score in zip(rows, scores, strict=True):
        grouped[user].append((item, label, score))
    ndcgs: list[float] = []
    recalls: list[float] = []
    for candidates in grouped.values():
        ranked = sorted(candidates, key=lambda row: (-row[2], row[0]))
        positives = sum(label for _, label, _ in candidates)
        ideal_count = min(10, positives)
        ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
        dcg = sum(
            label / math.log2(rank + 2)
            for rank, (_, label, _) in enumerate(ranked[:10])
        )
        ndcgs.append(dcg / ideal if ideal else 0.0)
        hits = sum(label for _, label, _ in ranked[:50])
        recalls.append(hits / positives if positives else 0.0)
    return sum(ndcgs) / len(ndcgs), sum(recalls) / len(recalls)


def write_submission(
    path: Path, rows: list[tuple[int, int, int]], scores: list[float]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "item_id", "score"])
        for row_id, ((user, item, _), score) in enumerate(zip(rows, scores, strict=True)):
            writer.writerow([row_id, user, item, f"{score:.12g}"])


def run(args: argparse.Namespace) -> int:
    train_spec = load_spec(args.train_spec)
    valid_spec = load_spec(args.validation_spec)
    if train_spec["users"] != valid_spec["users"] or train_spec["items"] != valid_spec["items"]:
        raise ValueError("Demo train/validation universes must match")
    train_rows = interactions(train_spec, repeats=3)
    valid_rows = interactions(valid_spec, repeats=1)
    scores = score_rows(
        args.strategy,
        train_rows,
        valid_rows,
        train_spec["categories"],
    )
    ndcg, recall = evaluate(valid_rows, scores)
    submission_path = Path(args.submission)
    result_path = Path(args.result)
    write_submission(submission_path, valid_rows, scores)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "status": "succeeded",
                "strategy": args.strategy,
                "metrics": {"NDCG@10": ndcg, "Recall@50": recall},
                "official_selection_score": (ndcg + recall) / 2,
                "resources": {"gpu_hours": 0.0, "llm_tokens": 0},
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(json.dumps({"strategy": args.strategy, "NDCG@10": ndcg, "Recall@50": recall}))
    return 0


def validate(args: argparse.Namespace) -> int:
    valid_spec = load_spec(args.validation_spec)
    expected_rows = valid_spec["users"] * valid_spec["items"]
    with Path(args.submission).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["row_id", "user_id", "item_id", "score"]:
            raise ValueError(f"Invalid submission schema: {reader.fieldnames}")
        count = 0
        for expected_id, row in enumerate(reader):
            if int(row["row_id"]) != expected_id:
                raise ValueError(f"Row alignment error at {expected_id}")
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError(f"Non-finite score at {expected_id}")
            count += 1
    if count != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {count}")
    print(f"submission valid: {count} rows")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--strategy", required=True)
    run_parser.add_argument("--train-spec", required=True)
    run_parser.add_argument("--validation-spec", required=True)
    run_parser.add_argument("--result", required=True)
    run_parser.add_argument("--submission", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--validation-spec", required=True)
    validate_parser.add_argument("--submission", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    return run(args) if args.command == "run" else validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
