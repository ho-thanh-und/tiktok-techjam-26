from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import AgentError
from .io_utils import read_json, utc_timestamp
from .storage import RUN_ID_PATTERN


def _safe_run_dir(run_root: Path, run_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise AgentError(f"Unsafe run ID: {run_id!r}")
    root = run_root.resolve()
    candidate = (root / run_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise AgentError("Run path escapes run root") from exc
    return candidate


def load_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "events.jsonl"
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AgentError(f"Invalid event JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise AgentError(f"Event at {path}:{line_number} is not an object")
            events.append(value)
    return events


def load_run(run_root: Path, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = _safe_run_dir(run_root, run_id)
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        raise AgentError(f"Run does not exist: {run_id}")
    return read_json(state_path), load_events(run_dir)


def _metric_delta(state: dict[str, Any], name: str) -> float | None:
    baseline = state.get("baseline") or {}
    final = state.get("final") or {}
    baseline_value = (baseline.get("metrics") or {}).get(name)
    final_value = (final.get("metrics") or {}).get(name)
    if not isinstance(baseline_value, (int, float)) or not isinstance(final_value, (int, float)):
        return None
    return float(final_value) - float(baseline_value)


def run_summary(state: dict[str, Any]) -> dict[str, Any]:
    baseline = state.get("baseline") or {}
    final = state.get("final") or {}
    baseline_score = baseline.get("selection_score")
    final_score = final.get("selection_score")
    selection_delta = None
    if isinstance(baseline_score, (int, float)) and isinstance(final_score, (int, float)):
        selection_delta = float(final_score) - float(baseline_score)
    return {
        "run_id": state.get("run_id"),
        "benchmark": state.get("benchmark"),
        "profile": state.get("profile"),
        "status": state.get("status"),
        "started_at": state.get("started_at"),
        "updated_at": state.get("updated_at"),
        "stop_reason": state.get("stop_reason"),
        "iterations_used": state.get("iterations_used", 0),
        "baseline_score": baseline_score,
        "final_score": final_score,
        "selection_delta": selection_delta,
        "final_experiment": final.get("experiment_id"),
        "manual_interventions": len(state.get("manual_interventions", [])),
    }


def list_runs(run_root: Path) -> list[dict[str, Any]]:
    if not run_root.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for directory in run_root.iterdir():
        if not directory.is_dir() or not RUN_ID_PATTERN.fullmatch(directory.name):
            continue
        state_path = directory / "state.json"
        if not state_path.is_file():
            continue
        try:
            summaries.append(run_summary(read_json(state_path)))
        except (OSError, ValueError, json.JSONDecodeError):
            summaries.append(
                {
                    "run_id": directory.name,
                    "status": "unreadable",
                    "benchmark": None,
                    "profile": None,
                    "updated_at": None,
                }
            )
    return sorted(summaries, key=lambda item: item.get("started_at") or "", reverse=True)


def run_detail(state: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = state.get("baseline") or {}
    final = state.get("final") or {}
    metric_names = sorted(set((baseline.get("metrics") or {})) | set((final.get("metrics") or {})))
    metrics = [
        {
            "name": name,
            "baseline": (baseline.get("metrics") or {}).get(name),
            "final": (final.get("metrics") or {}).get(name),
            "delta": _metric_delta(state, name),
        }
        for name in metric_names
    ]
    experiments = []
    for item in state.get("experiments", []):
        experiments.append(
            {
                "iteration": item.get("iteration"),
                "experiment_id": item.get("experiment_id"),
                "family": item.get("family"),
                "status": item.get("status"),
                "selection_score": item.get("selection_score"),
                "failure_class": item.get("failure_class"),
            }
        )
    public_events = [
        {"at": item.get("at"), "event": item.get("event")}
        for item in events[-100:]
    ]
    return {
        **run_summary(state),
        "metrics": metrics,
        "experiments": experiments,
        "resources": state.get("resources", {}),
        "consecutive_small_improvements": state.get("consecutive_small_improvements", 0),
        "best_selection_score": state.get("best_selection_score"),
        "final": {
            "experiment_id": final.get("experiment_id"),
            "selection_score": final.get("selection_score"),
            "elapsed_seconds": final.get("elapsed_seconds"),
            "hidden_test_evaluations": final.get("hidden_test_evaluations"),
            "submission_sha256": final.get("submission_sha256"),
        },
        "events": public_events,
    }


def markdown_report(state: dict[str, Any], events: list[dict[str, Any]]) -> str:
    detail = run_detail(state, events)
    lines = [
        f"# Autonomous ML Run: {detail['run_id']}",
        "",
        f"- Benchmark: `{detail.get('benchmark')}`",
        f"- Profile: `{detail.get('profile')}`",
        f"- Status: `{detail.get('status')}`",
        f"- Stop reason: `{detail.get('stop_reason')}`",
        f"- Iterations: `{detail.get('iterations_used')}`",
        f"- Manual interventions: `{detail.get('manual_interventions')}`",
        "",
        "## Metrics",
        "",
        "| Metric | Baseline | Final | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in detail["metrics"]:
        baseline = metric["baseline"]
        final = metric["final"]
        delta = metric["delta"]
        lines.append(
            f"| {metric['name']} | {baseline:.6f} | {final:.6f} | {delta:+.6f} |"
            if all(isinstance(value, (int, float)) for value in (baseline, final, delta))
            else f"| {metric['name']} | {baseline} | {final} | {delta} |"
        )
    lines.extend(
        [
            "",
            "## Experiments",
            "",
            "| # | Experiment | Family | Status | Selection score |",
            "|---:|---|---|---|---:|",
        ]
    )
    for item in detail["experiments"]:
        score = item["selection_score"]
        score_text = f"{score:.6f}" if isinstance(score, (int, float)) else "—"
        lines.append(
            f"| {item['iteration']} | `{item['experiment_id']}` | {item['family']} | "
            f"{item['status']} | {score_text} |"
        )
    resources = detail["resources"]
    final = detail["final"]
    lines.extend(
        [
            "",
            "## Final artifact",
            "",
            f"- Experiment: `{final.get('experiment_id')}`",
            f"- Selection score: `{final.get('selection_score')}`",
            f"- Submission SHA-256: `{final.get('submission_sha256')}`",
            f"- Hidden-test evaluations during development: `{final.get('hidden_test_evaluations')}`",
            "",
            "## Resource use",
            "",
            f"- Command seconds: `{resources.get('command_seconds', 0)}`",
            f"- GPU hours: `{resources.get('gpu_hours', 0)}`",
            f"- LLM tokens: `{resources.get('llm_tokens', 0)}`",
            "",
            f"Generated at `{utc_timestamp()}`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(run_root: Path, run_id: str, output: Path | None = None) -> Path:
    state, events = load_run(run_root, run_id)
    run_dir = _safe_run_dir(run_root, run_id)
    target = output.resolve() if output else run_dir / "report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(markdown_report(state, events), encoding="utf-8", newline="\n")
    temporary.replace(target)
    return target
