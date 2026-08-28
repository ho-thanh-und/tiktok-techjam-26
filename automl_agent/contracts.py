from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import AgentConfig
from .errors import ContractError
from .io_utils import sha256_file, utc_timestamp


COMPETITION_METRICS = ("NDCG@10", "Recall@50")
REQUIRED_ASSET_ROLES = {
    "official_baseline",
    "official_evaluator",
    "train",
    "validation",
    "submission_schema",
}


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_contract(config: AgentConfig) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if config.profile == "competition":
        if config.benchmark_name != "KuaiRand-Pure":
            errors.append("Required competition benchmark must be KuaiRand-Pure")
        if config.positive_label != "is_click":
            errors.append("Competition positive label must be is_click")
        if config.metric_names != COMPETITION_METRICS:
            errors.append(f"Competition metrics must be {COMPETITION_METRICS!r} in that order")
        if config.iteration_cap != 50:
            errors.append("Competition iteration cap must be exactly 50")
        if config.wall_clock_seconds != 21600:
            errors.append("Competition wall-clock cap must be exactly 21600 seconds")
        if config.epsilon != 0.002 or config.convergence_patience != 3:
            errors.append("Competition convergence must use epsilon=0.002 and patience=3")

    if config.iteration_cap <= 0 or config.iteration_cap > 50:
        errors.append("Iteration cap must be between 1 and 50")
    if config.wall_clock_seconds <= 0 or config.wall_clock_seconds > 21600:
        errors.append("Wall-clock budget must be between 1 and 21600 seconds")
    if config.epsilon < 0 or config.convergence_patience <= 0:
        errors.append("Convergence epsilon/patience are invalid")
    if config.planner_timeout_seconds <= 0 or config.planner_timeout_seconds > config.wall_clock_seconds:
        errors.append("Planner timeout must be positive and within the wall-clock budget")
    api_key_configured = None
    if config.planner_mode == "llm":
        api_key_configured = bool(os.environ.get(config.planner_api_key_env or ""))
        if not api_key_configured:
            message = f"LLM API key environment variable {config.planner_api_key_env} is not set"
            if config.planner_fallback_to_catalog:
                warnings.append(f"{message}; catalog fallback will be used")
            else:
                errors.append(message)
    if config.selection_metric not in config.metric_names and config.selection_metric != "official_selection_score":
        errors.append("Selection metric must be an official metric or official_selection_score")

    roles = [asset.role for asset in config.assets]
    missing_roles = sorted(REQUIRED_ASSET_ROLES - set(roles))
    if missing_roles:
        errors.append(f"Missing required asset roles: {', '.join(missing_roles)}")
    if len(roles) != len(set(roles)):
        errors.append("Asset roles must be unique")

    asset_report: list[dict[str, Any]] = []
    for asset in config.assets:
        exists = asset.path.is_file() if asset.kind == "file" else asset.path.is_dir()
        actual_digest = sha256_file(asset.path) if exists and asset.kind == "file" else None
        if not exists:
            errors.append(f"Missing asset {asset.role}: {asset.path}")
        elif asset.sha256 and asset.sha256.startswith("REPLACE_"):
            errors.append(f"Placeholder checksum was not replaced for asset {asset.role}")
        elif asset.kind == "directory" and asset.sha256:
            errors.append(f"Directory asset {asset.role} cannot use a file checksum; provide a manifest file")
        elif asset.sha256 and actual_digest.lower() != asset.sha256.lower():
            errors.append(f"Checksum mismatch for asset {asset.role}: {asset.path}")
        asset_report.append(
            {
                "role": asset.role,
                "path": str(asset.path),
                "kind": asset.kind,
                "exists": exists,
                "sha256": actual_digest,
            }
        )

    for forbidden in config.forbidden_paths:
        if forbidden.exists():
            errors.append(f"Forbidden hidden-test path is present: {forbidden}")

    ids = [experiment.experiment_id for experiment in config.experiments]
    if len(ids) != len(set(ids)):
        errors.append("Experiment IDs must be unique")
    for experiment in config.experiments:
        if experiment.timeout_seconds <= 0:
            errors.append(f"Experiment {experiment.experiment_id} timeout must be positive")
        if experiment.timeout_seconds > config.wall_clock_seconds:
            errors.append(f"Experiment {experiment.experiment_id} timeout exceeds wall-clock budget")

    if not _is_within(config.run_root, config.workspace):
        warnings.append("Run root is outside the workspace; verify this is intentional")

    report = {
        "checked_at": utc_timestamp(),
        "valid": not errors,
        "benchmark": config.benchmark_name,
        "profile": config.profile,
        "positive_label": config.positive_label,
        "metrics": list(config.metric_names),
        "selection_metric": config.selection_metric,
        "planner": {
            "mode": config.planner_mode,
            "provider": config.planner_provider,
            "model": config.planner_model,
            "api_key_env": config.planner_api_key_env,
            "api_key_configured": api_key_configured,
            "env_file": str(config.planner_env_file) if config.planner_env_file else None,
            "env_file_exists": config.planner_env_file.is_file() if config.planner_env_file else None,
            "fallback_to_catalog": config.planner_fallback_to_catalog,
        },
        "assets": asset_report,
        "errors": errors,
        "warnings": warnings,
    }
    if errors:
        raise ContractError("; ".join(errors))
    return report
