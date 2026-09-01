"""Agent-contract adapter for the official KuaiRand-Pure competition protocol.

This is the only benchmark adapter that speaks the real task definition:
`long_view` labels, GAUC and nDCG@5, and the official date splits from
``data.SPLITS``. It wraps the fixed pipeline in ``data.py`` / ``baseline.py`` /
``evaluate.py`` without modifying it, and emits the result JSON and submission
CSV that ``automl_agent`` expects.

Selection happens on the **valid** split only. The test split is never loaded
for scoring here, so an agent run cannot consume hidden-test evaluations; test
is reserved for a single manual check once a candidate is chosen.

    run       --strategy S --result R --submission C   train, score valid, emit both
    validate  --submission C                           format and row alignment only
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import baseline as B  # noqa: E402
from data import FIELDS, SPLITS, load  # noqa: E402
from evaluate import evaluate  # noqa: E402
from submit import HEADER  # noqa: E402

SELECTION_SPLIT = "valid"
METRIC_NAMES = ("GAUC", "nDCG@5")


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------


def load_splits(data_dir: Path, cache: Path | None) -> dict[str, list[tuple]]:
    """Load the official splits, memoizing the ~9s CSV parse across experiments."""
    if cache is not None and cache.is_file():
        newest = max(p.stat().st_mtime for p in data_dir.glob("*.csv"))
        if cache.stat().st_mtime >= newest:
            with cache.open("rb") as handle:
                return pickle.load(handle)
    splits = load(str(data_dir))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as handle:
            pickle.dump(splits, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return splits


def load_user_features(data_dir: Path) -> dict[str, list[str]]:
    """Side features that are not in the base five-field encoding."""
    wanted = ("user_active_degree", "is_video_author", "register_days_range",
              "follow_user_num_range", "fans_user_num_range")
    table: dict[str, list[str]] = {}
    path = data_dir / "user_features_pure.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(wanted).issubset(reader.fieldnames):
            raise ValueError(f"{path} is missing columns: {sorted(wanted)}")
        for row in reader:
            table[row["user_id"]] = [row[name] for name in wanted]
    return table


def encode(
    splits: dict[str, list[tuple]], extra: Callable[[tuple], list[str]] | None = None
) -> tuple[dict[str, tuple], int]:
    """Encode categorical fields to contiguous ids, optionally with extra fields.

    Mirrors ``data.encode`` but accepts an extra per-row feature extractor so a
    candidate can add feature domains without editing the fixed official
    pipeline. Vocabularies are fit on train only; unseen values fall into a
    per-domain UNK slot, exactly as the official encoder does.
    """
    train = splits["train"]
    edges = np.quantile(np.asarray([x[5] for x in train]), np.linspace(0, 1, 11)[1:-1])

    def raw(row: tuple) -> list[str]:
        base = [row[1], row[2], row[3], row[4], str(int(np.searchsorted(edges, row[5])))]
        return base + (extra(row) if extra is not None else [])

    width = len(raw(train[0]))
    vocabs: list[dict[str, int]] = [{} for _ in range(width)]
    for row in train:
        for i, value in enumerate(raw(row)):
            if value not in vocabs[i]:
                vocabs[i][value] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    encoded: dict[str, tuple] = {}
    for name, rows in splits.items():
        X = np.empty((len(rows), width), dtype=np.int32)
        y = np.empty(len(rows), dtype=np.float32)
        users = []
        for n, row in enumerate(rows):
            for i, value in enumerate(raw(row)):
                X[n, i] = vocabs[i].get(value, unk[i]) + offsets[i]
            y[n] = row[6]
            users.append(row[1])
        encoded[name] = (X, y, users)
    return encoded, int(sum(field_dims))


# --------------------------------------------------------------------------
# scoring strategies
# --------------------------------------------------------------------------


def popularity_scores(splits: dict[str, list[tuple]], target: str, prior: float = 20.0) -> np.ndarray:
    """Smoothed item long_view rate fit on train, applied to `target`."""
    positives: dict[str, int] = {}
    impressions: dict[str, int] = {}
    total_pos = total = 0
    for row in splits["train"]:
        video = row[2]
        impressions[video] = impressions.get(video, 0) + 1
        positives[video] = positives.get(video, 0) + row[6]
        total_pos += row[6]
        total += 1
    global_rate = total_pos / total
    out = np.empty(len(splits[target]), dtype=np.float64)
    for i, row in enumerate(splits[target]):
        shown = impressions.get(row[2], 0)
        out[i] = (
            (positives.get(row[2], 0) + prior * global_rate) / (shown + prior)
            if shown
            else global_rate
        )
    return out


def train_fm(
    encoded: dict[str, tuple],
    dim: int,
    *,
    target: str,
    k: int,
    lr: float,
    epochs: int,
    seed: int,
    patience: int = 4,
    batch: int = 8192,
) -> np.ndarray:
    """Train one FM and return its scores on `target`.

    Early stopping watches valid primary, which is the official selection rule
    and is what ``baseline.run_fm`` does; the returned model is the best epoch.
    """
    Xtr, ytr, _ = encoded["train"]
    Xva, yva, uva = encoded["valid"]
    model = B.FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        order = rng.permutation(len(ytr))
        for i in range(0, len(order), batch):
            chunk = order[i : i + batch]
            model.step(Xtr[chunk], ytr[chunk])
        primary = evaluate(uva, yva, model.predict(Xva))["primary"]
        if primary > best + 1e-5:
            best, bad = primary, 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("FM training produced no best state")
    model.V, model.W, model.b = best_state
    return model.predict(encoded[target][0])


def zscore(values: np.ndarray) -> np.ndarray:
    spread = float(values.std())
    return (values - float(values.mean())) / spread if spread > 0 else values * 0.0


STRATEGIES: dict[str, dict[str, Any]] = {
    # The gate: exactly the published FM baseline, one seed, base five fields.
    "official_baseline": {"kind": "fm", "k": 16, "lr": 0.001, "seeds": [0]},
    "fm_k32": {"kind": "fm", "k": 32, "lr": 0.001, "seeds": [0]},
    "fm_k8": {"kind": "fm", "k": 8, "lr": 0.001, "seeds": [0]},
    "fm_lr_higher": {"kind": "fm", "k": 16, "lr": 0.003, "seeds": [0]},
    "fm_seed_ensemble": {"kind": "fm", "k": 16, "lr": 0.001, "seeds": [0, 1, 2]},
    "fm_user_features": {"kind": "fm", "k": 16, "lr": 0.001, "seeds": [0], "user_features": True},
    "fm_pop_blend": {"kind": "blend", "k": 16, "lr": 0.001, "seeds": [0], "weight": 0.2},
    "pop_control": {"kind": "pop"},
    "random_control": {"kind": "random", "seeds": [0]},
}


def score_strategy(
    name: str, splits: dict[str, list[tuple]], data_dir: Path, target: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (scores on `target`, diagnostics) for one catalog strategy."""
    spec = STRATEGIES[name]
    kind = spec["kind"]
    diagnostics: dict[str, Any] = {"kind": kind}

    if kind == "pop":
        return popularity_scores(splits, target), diagnostics

    if kind == "random":
        rng = np.random.default_rng(spec["seeds"][0])
        return rng.random(len(splits[target])), diagnostics

    extra = None
    if spec.get("user_features"):
        table = load_user_features(data_dir)
        blank = ["UNK"] * 5
        extra = lambda row: table.get(row[1], blank)  # noqa: E731
    encoded, dim = encode(splits, extra)
    diagnostics["feature_fields"] = len(FIELDS) + (5 if extra is not None else 0)
    diagnostics["encoded_dim"] = dim

    per_seed_primary: list[float] = []
    columns: list[np.ndarray] = []
    _, yva, uva = encoded["valid"]
    for seed in spec["seeds"]:
        scores = train_fm(
            encoded, dim, target=target, k=spec["k"], lr=spec["lr"], epochs=40, seed=seed
        )
        columns.append(scores)
        if target == "valid":
            per_seed_primary.append(float(evaluate(uva, yva, scores)["primary"]))
    fm_scores = np.mean(columns, axis=0) if len(columns) > 1 else columns[0]

    diagnostics["seeds"] = list(spec["seeds"])
    if per_seed_primary:
        diagnostics["per_seed_primary"] = [round(float(v), 6) for v in per_seed_primary]
        if len(per_seed_primary) > 1:
            diagnostics["per_seed_primary_stdev"] = round(
                statistics.stdev(float(v) for v in per_seed_primary), 6
            )

    if kind == "blend":
        # Rank-comparable combination: FM logits and popularity rates live on
        # different scales, so standardize before mixing.
        pop = popularity_scores(splits, target)
        weight = float(spec["weight"])
        diagnostics["popularity_weight"] = weight
        return (1.0 - weight) * zscore(fm_scores) + weight * zscore(pop), diagnostics
    return fm_scores, diagnostics


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def write_submission(path: Path, rows: list[tuple], scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for i, (row, score) in enumerate(zip(rows, scores)):
            value = float(score)
            if not np.isfinite(value):
                raise ValueError(f"Non-finite score at row {i}")
            writer.writerow([i, row[1], row[2], f"{value:.6g}"])


def run(args: argparse.Namespace) -> int:
    if args.strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {args.strategy!r}; expected one of {sorted(STRATEGIES)}")
    data_dir = Path(args.data_dir).resolve()
    started = time.time()
    splits = load_splits(data_dir, Path(args.cache).resolve() if args.cache else None)
    rows = splits[SELECTION_SPLIT]
    scores, diagnostics = score_strategy(args.strategy, splits, data_dir, SELECTION_SPLIT)
    if len(scores) != len(rows):
        raise ValueError(f"Strategy produced {len(scores)} scores for {len(rows)} rows")

    measured = evaluate([r[1] for r in rows], [r[6] for r in rows], scores)
    write_submission(Path(args.submission).resolve(), rows, scores)

    result = {
        "status": "succeeded",
        "strategy": args.strategy,
        "metrics": {"GAUC": float(measured["GAUC"]), "nDCG@5": float(measured["nDCG@5"])},
        "official_selection_score": float(measured["primary"]),
        "resources": {"gpu_hours": 0.0, "llm_tokens": 0},
        "diagnostics": {
            **diagnostics,
            "label": "long_view",
            "selection_split": SELECTION_SPLIT,
            "selection_split_dates": list(SPLITS[SELECTION_SPLIT]),
            "selection_rows": measured["rows"],
            "selection_users": measured["users"],
            "train_rows": len(splits["train"]),
            "evaluation_protocol": "within_user_ranking_over_logged_impressions",
            "hidden_test_touched": False,
            "elapsed_seconds": round(time.time() - started, 2),
        },
    }
    result_path = Path(args.result).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {**result["metrics"], "primary": result["official_selection_score"]},
            sort_keys=True,
        )
    )
    return 0


def validate(args: argparse.Namespace) -> int:
    """Format and alignment check only; never reads labels or emits a score."""
    data_dir = Path(args.data_dir).resolve()
    splits = load_splits(data_dir, Path(args.cache).resolve() if args.cache else None)
    rows = splits[SELECTION_SPLIT]
    seen = 0
    with Path(args.submission).resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != HEADER:
            raise ValueError(f"Header must be {','.join(HEADER)}; got {header}")
        for line, record in enumerate(reader, start=2):
            if len(record) != 4:
                raise ValueError(f"Line {line} has {len(record)} fields; expected 4")
            row_id, user_id, video_id, raw_score = record
            if seen >= len(rows):
                raise ValueError(f"Submission has more than {len(rows)} evaluation rows")
            if int(row_id) != seen:
                raise ValueError(f"Line {line} row_id={row_id}; expected {seen}")
            if user_id != rows[seen][1] or video_id != rows[seen][2]:
                raise ValueError(
                    f"Line {line} misaligned: submission ({user_id},{video_id}) but "
                    f"row {seen} is ({rows[seen][1]},{rows[seen][2]})"
                )
            value = float(raw_score)
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"Line {line} score is NaN/Inf")
            seen += 1
    if seen != len(rows):
        raise ValueError(f"Submission has {seen} rows; evaluation set has {len(rows)}")
    print(json.dumps({"validated_rows": seen, "split": SELECTION_SPLIT}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    common = {"--data-dir": "KuaiRand-Pure/data", "--cache": ""}
    run_parser = sub.add_parser("run", help="Train, score the valid split, emit result and submission")
    run_parser.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    run_parser.add_argument("--result", required=True)
    run_parser.add_argument("--submission", required=True)
    validate_parser = sub.add_parser("validate", help="Check submission format and row alignment")
    validate_parser.add_argument("--submission", required=True)
    for target in (run_parser, validate_parser):
        for flag, default in common.items():
            target.add_argument(flag, default=default or None)

    args = parser.parse_args(argv)
    try:
        return run(args) if args.command == "run" else validate(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
